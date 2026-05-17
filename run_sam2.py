import os, sys, json
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import cv2
import imageio
import torch
from sam2.build_sam import build_sam2_video_predictor

FRAMES_DIR = "/tmp/sam2_frames"
OUT_DIR = "results"
CKPT = "pretrained_models/sam2/sam2.1_hiera_base_plus.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
FPS = 25
BG = np.array([120, 255, 155], dtype=np.float32) / 255.0  # MatAnyone's green

PROMPTS = {
    1: ("blue_box", [510, 840, 945, 1055]),
    2: ("orange",   [890, 895, 1110, 1035]),
    3: ("hand",     [40, 480, 380, 820]),
    4: ("desk",     [0, 875, 1280, 1280]),
}

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

predictor = build_sam2_video_predictor(CONFIG, CKPT, device=device)
state = predictor.init_state(video_path=FRAMES_DIR)

for obj_id, (name, box) in PROMPTS.items():
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=obj_id,
        box=np.array(box, dtype=np.float32),
    )
    print(f"prompted obj {obj_id} ({name})")

n_frames = len(sorted(os.listdir(FRAMES_DIR)))
H, W = cv2.imread(os.path.join(FRAMES_DIR, "00000.jpg")).shape[:2]
all_alphas = np.zeros((n_frames, H, W), dtype=np.uint8)

for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
    union = np.zeros((H, W), dtype=bool)
    for ml in mask_logits:
        m = (ml > 0.0).squeeze().cpu().numpy()
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        union |= m
    all_alphas[frame_idx] = (union.astype(np.uint8) * 255)

# Compose fgr on green + pha
os.makedirs(OUT_DIR, exist_ok=True)
fgrs, phas = [], []
for i in range(n_frames):
    bgr = cv2.imread(os.path.join(FRAMES_DIR, f"{i:05d}.jpg"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    a = (all_alphas[i].astype(np.float32) / 255.0)[..., None]
    com = rgb * a + BG.reshape(1, 1, 3) * (1 - a)
    com = np.round(np.clip(com * 255.0, 0, 255)).astype(np.uint8)
    fgrs.append(com)
    phas.append(all_alphas[i])

imageio.mimwrite(f"{OUT_DIR}/mask-test_sam2_fgr.mp4", np.stack(fgrs), fps=FPS, quality=7)
imageio.mimwrite(f"{OUT_DIR}/mask-test_sam2_pha.mp4", np.stack(phas), fps=FPS, quality=7)
print("wrote", OUT_DIR)
