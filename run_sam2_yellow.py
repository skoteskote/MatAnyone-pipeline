"""SAM 2 mask of the yellow backlight patch on mask-test-2."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import cv2
import imageio
import torch
from sam2.build_sam import build_sam2_video_predictor

FRAMES_DIR = "/tmp/sam2_frames2"
OUT_DIR = "results"
CKPT = "pretrained_models/sam2/sam2.1_hiera_base_plus.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
FPS = 25
SUFFIX = "yellow"

POS = [(520, 80), (650, 100), (780, 130), (550, 180),
       (700, 220), (850, 250), (600, 320), (750, 380)]
NEG = [(200, 200), (900, 600), (200, 600), (500, 700)]

device = "mps" if torch.backends.mps.is_available() else "cpu"
predictor = build_sam2_video_predictor(CONFIG, CKPT, device=device)
state = predictor.init_state(video_path=FRAMES_DIR)
points = np.array(POS + NEG, dtype=np.float32)
labels = np.array([1] * len(POS) + [0] * len(NEG), dtype=np.int32)
predictor.add_new_points_or_box(
    inference_state=state, frame_idx=0, obj_id=1,
    points=points, labels=labels,
)

n_frames = len(sorted(os.listdir(FRAMES_DIR)))
H, W = cv2.imread(os.path.join(FRAMES_DIR, "00000.jpg")).shape[:2]
phas = np.zeros((n_frames, H, W), dtype=np.uint8)
for fi, _, ml in predictor.propagate_in_video(state):
    m = (ml[0] > 0.0).squeeze().cpu().numpy()
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    phas[fi] = m.astype(np.uint8) * 255

# Smooth softly so binary edge looks reasonable
for i in range(n_frames):
    phas[i] = cv2.GaussianBlur(phas[i], (0, 0), sigmaX=1.5, sigmaY=1.5)

# Save mask video (white = yellow patch region), plus a visualization video
os.makedirs(OUT_DIR, exist_ok=True)
imageio.mimwrite(f"{OUT_DIR}/mask-test-2_{SUFFIX}_mask.mp4", phas, fps=FPS, quality=7)

vis_frames = []
for i in range(n_frames):
    bgr = cv2.imread(os.path.join(FRAMES_DIR, f"{i:05d}.jpg"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    m = phas[i] > 0
    overlay[m] = (overlay[m] * 0.4 + np.array([255, 0, 255]) * 0.6).astype(np.uint8)
    vis_frames.append(overlay)
imageio.mimwrite(f"{OUT_DIR}/mask-test-2_{SUFFIX}_overlay.mp4", np.stack(vis_frames), fps=FPS, quality=7)
print(f"wrote {OUT_DIR}/mask-test-2_{SUFFIX}_mask.mp4 and _overlay.mp4")
