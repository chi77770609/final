# -*- coding: utf-8 -*- 
"""
cnn_letterbox_kfold_roi.py — K-fold + ROI 版（以 ROI 比率為輸出；只訓練 ROI 內的標註）
============================================================================
◆ 模型：3D R(2+1)D-18；輸入 clip 長度 CLIP_LEN 的 224×224 畫布（full-frame letterbox）
◆ 只訓練 ROI：X ∈ [ROI_L, ROI_R], Y ∈ [ROI_T, ROI_B] 的幀；其餘幀不計入 loss
◆ 標註 → ROI 比率：rx = (X - ROI_L)/ROI_W, ry = (Y - ROI_T)/ROI_H（在 [0,1]）
◆ Loss：網路輸出 → ROI 比率 → 還原回原片像素 (X_hat, Y_hat) 與標註做 L1（pixel loss）
◆ K-fold：按「影片」分折（避免同片洩漏），各 fold 皆存檔；同時另存整體最優 best_r2p1d.pth
◆ 指令：
    python cnn_letterbox_kfold_roi.py                 # 訓練（K-fold）
    python cnn_letterbox_kfold_roi.py infer <in> <out>  # 推論（輸出畫紅點影片）
"""

from __future__ import annotations
import os, re, sys, hashlib, math
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from torchvision.transforms.functional import normalize
from tqdm import tqdm

# ============== 超參 ==============
IMG_SIZE   = 224
CLIP_LEN   = 10
STRIDE     = 2
BATCH      = 6            # 若 GPU 記憶體不足可改小（例如 4 或 2）
EPOCHS     = 25
LR         = 2e-4
SEED       = 42
AMP        = True
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# —— K-fold ——
KFOLD      = 5            # 設為 1 則退化為單次訓練（無 K-fold）

# —— ROI ——（只訓練此區域；單位：原片像素）
ROI_L, ROI_R = 604, 981
ROI_T, ROI_B = 108, 855
ROI_W = ROI_R - ROI_L
ROI_H = ROI_B - ROI_T
assert ROI_W > 0 and ROI_H > 0, "ROI 長寬必須為正"

# 前處理（訓練/推論一致，且一律在 RGB 上）
MEAN = (0.5, 0.5, 0.5)
STD  = (0.5, 0.5, 0.5)

# === 資料/標註設定 ===
CSV_ONE_INDEXED       = False  # 若 CSV 的 X,Y 是 (1,1) 起算，改 True；你現在是 0-based → False
FRAMEIDX_ONE_INDEXED  = False  # 若 CSV 的 FrameIndex 從 1 開始，改 True；預設 False

# === 輸出頭與 loss ===
HEAD_ACT                    = 'sigmoid'   # 'sigmoid' 建議；也可用 None / 'tanh01'
CLAMP_RATIO_FOR_PIXEL_LOSS  = True        # 反投影前把比率夾到 [0,1]

# === 混訓樣本加權（若你的資料有幀差/入針片段，可透過檔名自動識別） ===
W_RGB  = 1.0
W_DIFF = 2.0
DIFF_NAME_HINTS = ('_frame_diff', '_diff', '_fd', '_ins')

# 目錄 / 檔名
root  = Path(__file__).parent
V_DIR = root/'videos'
C_DIR = root/'coords'
CACHE = root/'data/clip_cache'; CACHE.mkdir(parents=True, exist_ok=True)
CKPT_DIR  = root/'checkpoints'; CKPT_DIR.mkdir(exist_ok=True)
BEST_ALL  = CKPT_DIR/'best_r2p1d.pth'  # 全 folds 中最佳

# 固定亂數
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ============== 工具 ==============

def open_video(path_str: str) -> cv2.VideoCapture:
    """Windows 多後端容錯，提升開檔成功率。"""
    path_str = str(path_str).replace("\\", "/")
    cap = cv2.VideoCapture(path_str)
    if cap.isOpened(): return cap
    for backend in (cv2.CAP_FFMPEG, cv2.CAP_MSMF, cv2.CAP_DSHOW):
        cap = cv2.VideoCapture(path_str, backend)
        if cap.isOpened(): return cap
    return cv2.VideoCapture()

def _cache_key(p: Path) -> Path:
    """快取鍵：路徑 + IMG_SIZE + 檔案 mtime"""
    st = os.stat(p)
    sig = f"{p.resolve()}|S:{IMG_SIZE}|mtime:{int(st.st_mtime)}"
    h = hashlib.md5(sig.encode()).hexdigest()[:10]
    return CACHE / f'{p.stem}_{h}'

def load_video_letterbox(p: Path) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    逐幀 letterbox 到 (IMG_SIZE, IMG_SIZE)【RGB】。
    回傳：
      frames: list[(H,W,3) uint8, RGB]
      metas : (N,5) float32，每幀 (scale, pad_x, pad_y, W_full, H_full)
    """
    npz = _cache_key(p).with_suffix(".npz")
    if npz.exists():
        z = np.load(npz)
        frames = [f for f in z["frames"]]
        metas  = z["metas"].astype(np.float32)
        return frames, metas

    cap = open_video(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {p}")

    frames, metas = [], []
    while True:
        ok, f = cap.read()
        if not ok: break

        if f.ndim == 2 or (f.ndim == 3 and f.shape[2] == 1):
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

        H_full, W_full = f.shape[:2]
        scale = min(IMG_SIZE / W_full, IMG_SIZE / H_full)
        new_w = int(round(W_full * scale))
        new_h = int(round(H_full * scale))
        pad_x = (IMG_SIZE - new_w) // 2
        pad_y = (IMG_SIZE - new_h) // 2

        resized = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas_bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        canvas_bgr[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)   # ★ 訓練/推論統一用 RGB
        frames.append(canvas_rgb)
        metas.append((scale, float(pad_x), float(pad_y), float(W_full), float(H_full)))

    cap.release()

    frames_arr = np.stack(frames).astype(np.uint8)
    metas_arr  = np.asarray(metas, dtype=np.float32)
    np.savez_compressed(npz, frames=frames_arr, metas=metas_arr)
    return list(frames_arr), metas_arr

def find_label_for_video_stem(stem: str, cdir: Path) -> Path | None:
    """影片 stem 對應座標檔；可把 *_frame_diff* 之類回推到原始名。"""
    def try_with(b: str) -> Path | None:
        for sfx in ('_coords.xlsx','_coords.csv','.xlsx','.csv'):
            p = cdir / f'{b}{sfx}'
            if p.exists(): return p
        return None
    p = try_with(stem)
    if p: return p
    m = re.match(r'(.+?)(?:_frame[_-]?diff(?:[_-]?\d+)?|_diff\d*|_fd\d*|_ins\d*)$', stem, flags=re.I)
    if m:
        p = try_with(m.group(1))
        if p: return p
    m = re.match(r'(.+?)_\d+$', stem)
    if m:
        p = try_with(m.group(1))
        if p: return p
    return None

def read_xy(p: Path) -> Tuple[np.ndarray, np.ndarray]:
    """讀 CSV/XLSX，回傳 (fi, xy_full[Px2])；左上(0,0) 像素座標。"""
    df = pd.read_excel(p) if p.suffix.lower()=='.xlsx' else pd.read_csv(p)
    if {'FrameIndex','X','Y'}.issubset(df.columns):
        fi = df['FrameIndex'].astype(int).to_numpy()
        if FRAMEIDX_ONE_INDEXED: fi = fi - 1
        xy = df[['X','Y']].to_numpy(np.float32)
    else:
        # 只有 X,Y：假設一列對一幀
        xy = df.iloc[:,:2].to_numpy(np.float32)
        fi = np.arange(len(xy), dtype=int)
    if CSV_ONE_INDEXED:
        xy -= 1.0
    return fi, xy

def to_ratio(x: torch.Tensor) -> torch.Tensor:
    if HEAD_ACT == 'sigmoid':
        return x.sigmoid()
    if HEAD_ACT == 'tanh01':
        return (x.tanh() + 1.0) * 0.5
    return x  # linear

# ============== Dataset ==============

class ClipDS(Dataset):
    def __init__(self, vdir: Path, cdir: Path):
        self.sam: list = []     # list of tuples (clip, xy_roi_ratio, xy_full, meta, mask, is_diff, vid)
        self.vids: list[int] = []
        self.video_stems: list[str] = []
        rgb_cnt = diff_cnt = 0
        vid_id = -1

        for v in sorted(vdir.glob('*')):
            if v.suffix.lower() not in {'.mp4','.avi','.mov'}:
                continue
            base = v.stem
            lab  = find_label_for_video_stem(base, cdir)
            if lab is None:
                print(f'[WARN] 找不到標註：{base}')
                continue

            vid_id += 1
            self.video_stems.append(base)
            is_diff_video = any(k in base.lower() for k in DIFF_NAME_HINTS)

            frames, metas = load_video_letterbox(v)   # metas:(N,5)
            N = len(frames)
            fi, xy_full_all = read_xy(lab)

            coords_roi  = np.full((N,2), -1, np.float32)  # ROI 比率標註
            coords_full = np.full((N,2), -1, np.float32)  # 原片像素標註
            mask        = np.zeros((N,),  np.float32)     # 有效幀（且在 ROI 內）

            # 只吃合法幀
            ok = (fi >= 0) & (fi < N)
            fi = fi[ok]; xy_full = xy_full_all[ok]

            if len(fi) > 0:
                # 僅保留在 ROI 內的標註
                inside = (
                    (xy_full[:,0] >= ROI_L) & (xy_full[:,0] <= ROI_R) &
                    (xy_full[:,1] >= ROI_T) & (xy_full[:,1] <= ROI_B)
                )
                if inside.any():
                    fi_in  = fi[inside]
                    xy_in  = xy_full[inside]
                    rx = (xy_in[:,0] - ROI_L) / ROI_W
                    ry = (xy_in[:,1] - ROI_T) / ROI_H
                    coords_roi[fi_in]  = np.stack([rx, ry], 1)
                    coords_full[fi_in] = xy_in
                    mask[fi_in]        = 1.0

            for s in range(0, N-CLIP_LEN+1, STRIDE):
                clip = np.stack(frames[s:s+CLIP_LEN])  # (T,H,W,3) uint8 RGB
                tgt_i = s + CLIP_LEN - 1
                meta_tgt = metas[tgt_i]  # (scale, pad_x, pad_y, W_full, H_full)
                self.sam.append((clip, coords_roi[tgt_i], coords_full[tgt_i], meta_tgt, mask[tgt_i], int(is_diff_video), vid_id))
                self.vids.append(vid_id)
                if is_diff_video: diff_cnt += 1
                else:             rgb_cnt  += 1

        if not self.sam:
            raise RuntimeError('no data')
        print(f'[ClipDS] samples -> RGB: {rgb_cnt} | DIFF: {diff_cnt} | videos: {len(self.video_stems)}')

    def __len__(self): return len(self.sam)

    def __getitem__(self, i):
        clip, xy_roi, xy_full, meta, m, is_diff, vid = self.sam[i]
        # (T,H,W,3 RGB) → (T,C,H,W) → [0,1] → Normalize
        clip = torch.from_numpy(clip).permute(0,3,1,2).float() / 255.0
        clip = normalize(clip, MEAN, STD)
        return (
            clip,                                   # (T,C,H,W)
            torch.from_numpy(xy_roi),               # (2,)  ROI 比率標註
            torch.from_numpy(xy_full),              # (2,)  原片像素標註
            torch.tensor(meta, dtype=torch.float32),# (5,)  scale,pad_x,pad_y,W,H（此版訓練不使用）
            torch.tensor(m),                        # (1,)  mask（有效且在 ROI 內）
            torch.tensor(is_diff, dtype=torch.uint8),
            torch.tensor(vid, dtype=torch.int64),
        )

# ============== 模型 ==============

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-2])
        self.pool     = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc       = nn.Linear(512, 2)

    def forward(self, x):     # x: (B,T,C,H,W)
        x = x.permute(0,2,1,3,4)        # (B,C,T,H,W)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)     # (B,512)
        x = self.fc(x)                  # (B,2)
        return x                        # logits（外面 to_ratio → 比率）

# ============== 訓練 ==============

def _indices_by_vid(ds: ClipDS, vids_in: set[int]) -> List[int]:
    return [i for i, v in enumerate(ds.vids) if v in vids_in]

def _split_kfold(num_videos: int, k: int, seed: int) -> List[List[int]]:
    idx = np.arange(num_videos)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    folds: List[List[int]] = []
    base = num_videos // k
    rem  = num_videos % k
    start = 0
    for f in range(k):
        size = base + (1 if f < rem else 0)
        folds.append(idx[start:start+size].tolist())
        start += size
    return folds

def _one_epoch(net, loader, opt, scaler, l1, train_mode: bool):
    if train_mode:
        net.train()
    else:
        net.eval()
    run = cnt = 0.0

    for batch in loader:
        clip, xy_roi, xy_full, meta, m, is_diff, vid = batch
        keep = m.bool()
        if not keep.any():
            continue
        clip     = clip[keep].to(DEVICE)
        xy_roi   = xy_roi[keep].to(DEVICE)
        xy_full  = xy_full[keep].to(DEVICE)
        w        = (is_diff[keep].float()*(W_DIFF - W_RGB) + W_RGB).to(DEVICE)

        if train_mode:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=AMP and DEVICE=='cuda'):
                out   = net(clip)              # (B,2)
                ratio = to_ratio(out)
                if CLAMP_RATIO_FOR_PIXEL_LOSS:
                    ratio = ratio.clamp(0., 1.)
                # ROI 比率 → 原片像素
                X_hat = ROI_L + ROI_W * ratio[:,0]
                Y_hat = ROI_T + ROI_H * ratio[:,1]
                pred_full = torch.stack([X_hat, Y_hat], dim=1)
                per  = l1(pred_full, xy_full).mean(dim=1)
                loss = (w * per).sum() / w.sum()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        else:
            with torch.no_grad():
                out   = net(clip)
                ratio = to_ratio(out)
                if CLAMP_RATIO_FOR_PIXEL_LOSS:
                    ratio = ratio.clamp(0., 1.)
                X_hat = ROI_L + ROI_W * ratio[:,0]
                Y_hat = ROI_T + ROI_H * ratio[:,1]
                pred_full = torch.stack([X_hat, Y_hat], dim=1)
                per  = l1(pred_full, xy_full).mean(dim=1)
                loss = (w * per).sum() / w.sum()

        run += float(per.mean().item()) * clip.size(0)
        cnt += clip.size(0)

    return run / max(cnt,1)

def train_kfold():
    ds = ClipDS(V_DIR, C_DIR)
    num_videos = len(ds.video_stems)
    folds = _split_kfold(num_videos, KFOLD, SEED)

    best_overall = math.inf
    best_state   = None
    best_fold_id = -1

    for f, val_vids in enumerate(folds):
        train_vids = set(range(num_videos)) - set(val_vids)
        tr_idx = _indices_by_vid(ds, train_vids)
        va_idx = _indices_by_vid(ds, set(val_vids))

        tr_dl = DataLoader(Subset(ds, tr_idx), BATCH, True,  pin_memory=True)
        va_dl = DataLoader(Subset(ds, va_idx), BATCH, False, pin_memory=True)

        net = Net().to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        l1  = nn.L1Loss(reduction='none')
        scaler = torch.cuda.amp.GradScaler(enabled=AMP and DEVICE=='cuda')

        best_val = math.inf
        ckpt_fold = CKPT_DIR / f'best_r2p1d_fold{f}.pth'

        for ep in range(1, EPOCHS+1):
            tr_loss = _one_epoch(net, tr_dl, opt, scaler, l1, train_mode=True)
            va_loss = _one_epoch(net, va_dl, opt, scaler, l1, train_mode=False)
            sch.step()
            print(f'[F{f}] E{ep:02d}/{EPOCHS}  train {tr_loss:.4f} | val {va_loss:.4f}')

            if va_loss < best_val:
                best_val = va_loss
                torch.save(net.state_dict(), ckpt_fold)
                print(f'[save] {ckpt_fold}')

        # 更新整體最佳
        if best_val < best_overall:
            best_overall = best_val
            best_state   = torch.load(ckpt_fold, map_location='cpu')
            best_fold_id = f

    # 另存整體最佳（推論預設讀這個）
    if best_state is not None:
        torch.save(best_state, BEST_ALL)
        print(f'[save] overall best -> {BEST_ALL}  (fold={best_fold_id}, val={best_overall:.4f})')
    else:
        print('[WARN] 未產生 overall best，請檢查資料集')

def train_single():
    """KFOLD=1 時的單次訓練（保留以防需要）。"""
    ds = ClipDS(V_DIR, C_DIR)
    # 9:1 split（影片級）：
    num_videos = len(ds.video_stems)
    folds = _split_kfold(num_videos, 10, SEED)
    val_vids = folds[0]
    train_vids = set(range(num_videos)) - set(val_vids)
    tr_idx = _indices_by_vid(ds, train_vids)
    va_idx = _indices_by_vid(ds, set(val_vids))

    tr_dl = DataLoader(Subset(ds, tr_idx), BATCH, True,  pin_memory=True)
    va_dl = DataLoader(Subset(ds, va_idx), BATCH, False, pin_memory=True)

    net = Net().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    l1  = nn.L1Loss(reduction='none')
    scaler = torch.cuda.amp.GradScaler(enabled=AMP and DEVICE=='cuda')

    best = math.inf
    for ep in range(1, EPOCHS+1):
        tr_loss = _one_epoch(net, tr_dl, opt, scaler, l1, train_mode=True)
        va_loss = _one_epoch(net, va_dl, opt, scaler, l1, train_mode=False)
        sch.step()
        print(f'E{ep:02d} train {tr_loss:.4f} | val {va_loss:.4f}')
        if va_loss < best:
            best = va_loss
            torch.save(net.state_dict(), BEST_ALL)
            print(f'[save] {BEST_ALL}')
    print('best L1', best)

# ============== 推論（ROI 比率 → 原片畫素） ==============

@torch.no_grad()
def infer(inp: str, out: str):
    ckpt_path = BEST_ALL if BEST_ALL.exists() else None
    if ckpt_path is None:
        # fallback：找 fold0
        alt = CKPT_DIR / 'best_r2p1d_fold0.pth'
        if alt.exists():
            ckpt_path = alt
    if ckpt_path is None:
        print('❌ 找不到權重：請先訓練'); return

    net = Net().to(DEVICE).eval()
    net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    cap = open_video(inp)
    if not cap.isOpened():
        print(f'❌ 無法開啟影片：{inp}')
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        print(f'❌ 無法建立輸出影片：{out}')
        return

    buf = []
    n = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        n += 1

        if frame_bgr.ndim == 2 or (frame_bgr.ndim == 3 and frame_bgr.shape[2] == 1):
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

        H_full, W_full = frame_bgr.shape[:2]
        scale = min(IMG_SIZE / W_full, IMG_SIZE / H_full)
        new_w = int(round(W_full * scale))
        new_h = int(round(H_full * scale))
        pad_x = (IMG_SIZE - new_w) // 2
        pad_y = (IMG_SIZE - new_h) // 2

        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas_bgr  = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        canvas_bgr[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
        chw = np.transpose(canvas_rgb, (2,0,1))  # (3,224,224)
        buf.append(chw)
        if len(buf) > CLIP_LEN: buf.pop(0)

        if len(buf) == CLIP_LEN:
            clip = torch.from_numpy(np.stack(buf)).float() / 255.0   # (T,3,224,224)
            clip = normalize(clip, MEAN, STD).unsqueeze(0).to(DEVICE)  # (1,T,C,H,W)
            out_vec = net(clip)[0]
            ratio = to_ratio(out_vec).detach().cpu().numpy()
            rx, ry = float(ratio[0]), float(ratio[1])
            if CLAMP_RATIO_FOR_PIXEL_LOSS:
                rx = max(0.0, min(1.0, rx))
                ry = max(0.0, min(1.0, ry))

            # ROI 比率 → 原片像素
            X_full = ROI_L + ROI_W * rx
            Y_full = ROI_T + ROI_H * ry

            x_px = int(np.clip(X_full, 0, W_full-1))
            y_px = int(np.clip(Y_full, 0, H_full-1))
            cv2.circle(frame_bgr, (x_px, y_px), 5, (0,0,255), -1)
            # 可視化 ROI（若需要，取消註解）
            # cv2.rectangle(frame_bgr, (ROI_L, ROI_T), (ROI_R, ROI_B), (0,255,0), 1)

        writer.write(frame_bgr)

    cap.release(); writer.release()
    print(f'✅ 完成，處理 {n} 幀。輸出：{out}')

# ============== 入口 ==============

if __name__ == '__main__':
    if len(sys.argv) == 1:
        if KFOLD and KFOLD > 1:
            train_kfold()
        else:
            train_single()
    elif len(sys.argv) == 4 and sys.argv[1] == 'infer':
        infer(sys.argv[2], sys.argv[3])
    else:
        print("用法:\n  python cnn_letterbox_kfold_roi.py                 # 訓練（K-fold）\n  python cnn_letterbox_kfold_roi.py infer <in> <out>  # 推論")
