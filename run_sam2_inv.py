"""SAM 2 strategy: segment the green backdrop as one object, then invert."""
import os
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
BG = np.array([120, 255, 155], dtype=np.float32) / 255.0
SUFFIX = "sam2_inv"

POS_GREEN = [(640, 100), (200, 300), (1100, 300), (1200, 600)]
NEG_FG = [(180, 620), (720, 940), (1020, 970), (640, 1180)]

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

predictor = build_sam2_video_predictor(CONFIG, CKPT, device=device)
state = predictor.init_state(video_path=FRAMES_DIR)

points = np.array(POS_GREEN + NEG_FG, dtype=np.float32)
labels = np.array([1] * len(POS_GREEN) + [0] * len(NEG_FG), dtype=np.int32)
predictor.add_new_points_or_box(
    inference_state=state,
    frame_idx=0,
    obj_id=1,
    points=points,
    labels=labels,
)

n_frames = len(sorted(os.listdir(FRAMES_DIR)))
H, W = cv2.imread(os.path.join(FRAMES_DIR, "00000.jpg")).shape[:2]
green_masks = np.zeros((n_frames, H, W), dtype=bool)

def clean_green_mask(m):
    m = m.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = np.where(lab == i, 255, 0).astype(np.uint8)
    inv = cv2.bitwise_not(m)
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for i in range(1, n2):
        if stats2[i, cv2.CC_STAT_AREA] < 800:
            m[lab2 == i] = 255
    return m.astype(bool)

for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
    m = (mask_logits[0] > 0.0).squeeze().cpu().numpy()
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    green_masks[frame_idx] = clean_green_mask(m)

def refine_alpha(bgr, sam_fg_bool):
    """SAM 2 positive evidence + HSV strong-green + soft chroma alpha everywhere else."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    bright = np.maximum(np.maximum(R, G), B) + 0.05
    greenness = np.clip((G - np.maximum(R, B)) / bright, 0.0, 1.0)
    alpha_ck = np.clip(1.0 - greenness * 5.0, 0.0, 1.0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green_def = cv2.inRange(hsv, np.array([35, 60, 25]), np.array([95, 255, 255])) > 0
    sam_u8 = sam_fg_bool.astype(np.uint8) * 255
    k = lambda s: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
    inside = cv2.erode(sam_u8, k(15)) > 0
    alpha = np.where(inside, 1.0, np.where(green_def, 0.0, alpha_ck)).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.8, sigmaY=0.8)
    return alpha

fgrs, phas = [], []
for i in range(n_frames):
    bgr = cv2.imread(os.path.join(FRAMES_DIR, f"{i:05d}.jpg"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    a2d = refine_alpha(bgr, ~green_masks[i])
    a = a2d[..., None]
    com = rgb * a + BG.reshape(1, 1, 3) * (1 - a)
    com = np.round(np.clip(com * 255.0, 0, 255)).astype(np.uint8)
    fgrs.append(com)
    phas.append(np.round(np.clip(a2d * 255.0, 0, 255)).astype(np.uint8))

os.makedirs(OUT_DIR, exist_ok=True)
imageio.mimwrite(f"{OUT_DIR}/mask-test_{SUFFIX}_fgr.mp4", np.stack(fgrs), fps=FPS, quality=7)
imageio.mimwrite(f"{OUT_DIR}/mask-test_{SUFFIX}_pha.mp4", np.stack(phas), fps=FPS, quality=7)
print(f"wrote {OUT_DIR}/mask-test_{SUFFIX}_(fgr|pha).mp4")
