# -*- coding: utf-8 -*-
"""
infer_folds.py — 只做推論；可單獨指定 fold0 / fold1 / fold2 權重
流程與你訓練版完全對齊：letterbox→RGB→normalize→to_ratio→ROI 反投影
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from torchvision.transforms.functional import normalize

# ================== 超參（需與訓練一致） ==================
IMG_SIZE  = 224
CLIP_LEN  = 10
AMP       = True
HEAD_ACT  = 'sigmoid'
CLAMP_RATIO_FOR_PIXEL_LOSS = True

# —— ROI ——（單位：原片像素；與訓練一致）
ROI_L, ROI_R = 604, 981
ROI_T, ROI_B = 108, 855
ROI_W = ROI_R - ROI_L
ROI_H = ROI_B - ROI_T

# 前處理（與訓練一致，RGB 上做）
MEAN = (0.5, 0.5, 0.5)
STD  = (0.5, 0.5, 0.5)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT   = Path(__file__).parent.resolve()
CKPT_DIR = ROOT / 'checkpoints'  # 預期存放 best_r2p1d_fold*.pth

# ================== 模型與前處理 ==================
def to_ratio(x: torch.Tensor) -> torch.Tensor:
    if HEAD_ACT == 'sigmoid':
        return x.sigmoid()
    if HEAD_ACT == 'tanh01':
        return (x.tanh() + 1.0) * 0.5
    return x  # linear

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-2])
        self.pool     = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc       = nn.Linear(512, 2)

    def forward(self, x):   # x: (B,T,C,H,W)
        x = x.permute(0,2,1,3,4)        # -> (B,C,T,H,W)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)     # (B,512)
        x = self.fc(x)                  # (B,2) logits
        return x

def open_video(path_str: str) -> cv2.VideoCapture:
    # 簡單可靠的開檔（Windows 多後端）
    p = str(path_str).replace("\\", "/")
    cap = cv2.VideoCapture(p)
    if cap.isOpened(): return cap
    for backend in (cv2.CAP_FFMPEG, cv2.CAP_MSMF, cv2.CAP_DSHOW):
        cap = cv2.VideoCapture(p, backend)
        if cap.isOpened(): return cap
    return cv2.VideoCapture()

# ================== 推論核心 ==================
@torch.no_grad()
def infer_with_ckpt(ckpt_path: Path, inp: str, out: str, show_roi: bool=False):
    if not ckpt_path.exists():
        print(f'❌ 找不到權重：{ckpt_path}')
        return

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
        cap.release()
        return

    buf = []
    n = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        n += 1

        if frame_bgr.ndim == 2 or (frame_bgr.ndim == 3 and frame_bgr.shape[2] == 1):
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

        # ----- letterbox 到 (IMG_SIZE, IMG_SIZE) 並轉 RGB -----
        H_full, W_full = frame_bgr.shape[:2]
        scale = min(IMG_SIZE / W_full, IMG_SIZE / H_full)
        new_w = int(round(W_full * scale))
        new_h = int(round(H_full * scale))
        pad_x = (IMG_SIZE - new_w) // 2
        pad_y = (IMG_SIZE - new_h) // 2

        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas_bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        canvas_bgr[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

        # ----- 累積 clip -----
        chw = np.transpose(canvas_rgb, (2,0,1))  # (3,224,224)
        buf.append(chw)
        if len(buf) > CLIP_LEN:
            buf.pop(0)

        # ----- 送入網路 -----
        if len(buf) == CLIP_LEN:
            clip = torch.from_numpy(np.stack(buf)).float() / 255.0       # (T,3,224,224)
            clip = normalize(clip, MEAN, STD).unsqueeze(0).to(DEVICE)    # (1,T,3,224,224)
            with torch.cuda.amp.autocast(enabled=AMP and DEVICE=='cuda'):
                out_vec = net(clip)[0]
                ratio   = to_ratio(out_vec)

            rx = float(ratio[0].item())
            ry = float(ratio[1].item())
            if CLAMP_RATIO_FOR_PIXEL_LOSS:
                rx = max(0.0, min(1.0, rx))
                ry = max(0.0, min(1.0, ry))

            # ROI 比率 → 原片像素
            X_full = ROI_L + ROI_W * rx
            Y_full = ROI_T + ROI_H * ry
            x_px   = int(np.clip(X_full, 0, W_full-1))
            y_px   = int(np.clip(Y_full, 0, H_full-1))

            cv2.circle(frame_bgr, (x_px, y_px), 5, (0,0,255), -1)
            if show_roi:
                cv2.rectangle(frame_bgr, (ROI_L, ROI_T), (ROI_R, ROI_B), (0,255,0), 1)

        writer.write(frame_bgr)

    cap.release(); writer.release()
    print(f'✅ 完成，處理 {n} 幀。輸出：{out}')

# ================== CLI 包裝 ==================
def infer_f0(inp: str, out: str, show_roi=False):
    infer_with_ckpt(CKPT_DIR / 'best_r2p1d_fold0.pth', inp, out, show_roi)

def infer_f1(inp: str, out: str, show_roi=False):
    infer_with_ckpt(CKPT_DIR / 'best_r2p1d_fold1.pth', inp, out, show_roi)

def infer_f2(inp: str, out: str, show_roi=False):
    infer_with_ckpt(CKPT_DIR / 'best_r2p1d_fold2.pth', inp, out, show_roi)

def infer_fold(k: int, inp: str, out: str, show_roi=False):
    infer_with_ckpt(CKPT_DIR / f'best_r2p1d_fold{k}.pth', inp, out, show_roi)

# ================== Main ==================
if __name__ == '__main__':
    """
    用法：
      python infer_folds.py infer_f0 <in> <out>
      python infer_folds.py infer_f1 <in> <out>
      python infer_folds.py infer_f2 <in> <out>
      # 或泛用：
      python infer_folds.py infer_fold <k> <in> <out>

    小提示：想畫出 ROI 框可在命令後面加 --show-roi
    """
    if len(sys.argv) < 4:
        print("用法：\n"
              "  python infer_folds.py infer_f0 <in> <out> [--show-roi]\n"
              "  python infer_folds.py infer_f1 <in> <out> [--show-roi]\n"
              "  python infer_folds.py infer_f2 <in> <out> [--show-roi]\n"
              "  python infer_folds.py infer_fold <k> <in> <out> [--show-roi]")
        sys.exit(1)

    cmd = sys.argv[1]
    show_roi = ('--show-roi' in sys.argv)

    try:
        if cmd == 'infer_f0':
            infer_f0(sys.argv[2], sys.argv[3], show_roi)
        elif cmd == 'infer_f1':
            infer_f1(sys.argv[2], sys.argv[3], show_roi)
        elif cmd == 'infer_f2':
            infer_f2(sys.argv[2], sys.argv[3], show_roi)
        elif cmd == 'infer_fold':
            if len(sys.argv) < 5:
                print("用法：python infer_folds.py infer_fold <k> <in> <out> [--show-roi]")
                sys.exit(1)
            k = int(sys.argv[2])
            infer_fold(k, sys.argv[3], sys.argv[4], show_roi)
        else:
            print("未知指令。請用 infer_f0 / infer_f1 / infer_f2 / infer_fold")
            sys.exit(1)
    except Exception as e:
        print(f'❌ 錯誤：{e}')
        sys.exit(1)
