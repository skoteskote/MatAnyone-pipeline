"""Propagate user's BW scratches mask from a seed frame across full clip.

The seed frame can be:
  - set manually via SEED_FRAME_IDX (e.g. = 31), or
  - auto-detected from SEED_STILL_PATH (the unedited still you painted the mask on)
    by finding the clip frame with the smallest pixel-difference to the still.

If both are set, the manual index wins and we warn if the auto pick disagrees.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import cv2
import torch
from sam2.build_sam import build_sam2_video_predictor

FRAMES_DIR = "/tmp/fdn_frames"
MASKS_DIR = "/tmp/fdn_masks"
SEED_MASK_PATH  = "/Users/jakob/Desktop/TheFoundation.00_01_23_01.Still001-bw-mask.jpg"
SEED_STILL_PATH = "/Users/jakob/Desktop/TheFoundation.00_01_23_01.Still001.jpg"  # set to None to skip auto-detect
SEED_FRAME_IDX  = None   # set to an int to override auto-detection
CKPT = "pretrained_models/sam2/sam2.1_hiera_base_plus.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def autodetect_seed_frame(still_path, frames_dir, match_size=256):
    """Return clip frame index whose content best matches the still."""
    ref = cv2.imread(still_path)
    if ref is None:
        raise FileNotFoundError(still_path)
    ref_small = cv2.resize(ref, (match_size, match_size)).astype(np.int32)
    frame_paths = sorted(p for p in os.listdir(frames_dir) if p.endswith(".jpg"))
    best_idx, best_mse = -1, float("inf")
    for i, name in enumerate(frame_paths):
        f = cv2.imread(os.path.join(frames_dir, name))
        f_small = cv2.resize(f, (match_size, match_size)).astype(np.int32)
        mse = float(np.mean((f_small - ref_small) ** 2))
        if mse < best_mse:
            best_idx, best_mse = i, mse
    return best_idx, best_mse

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Resolve seed frame index
auto_idx, auto_mse = (None, None)
if SEED_STILL_PATH:
    auto_idx, auto_mse = autodetect_seed_frame(SEED_STILL_PATH, FRAMES_DIR)
    print(f"auto-detect: best match is frame {auto_idx} (mse={auto_mse:.1f})")

if SEED_FRAME_IDX is None:
    if auto_idx is None:
        raise RuntimeError("Set SEED_FRAME_IDX or SEED_STILL_PATH.")
    seed_idx = auto_idx
else:
    seed_idx = SEED_FRAME_IDX
    if auto_idx is not None and auto_idx != seed_idx:
        print(f"warning: manual SEED_FRAME_IDX={seed_idx} disagrees with auto pick {auto_idx}")

print(f"device: {device}, seed frame: {seed_idx}")
SEED_FRAME_IDX = seed_idx

# Load + resize seed mask to clip resolution
frame0 = cv2.imread(os.path.join(FRAMES_DIR, "00000.jpg"))
H, W = frame0.shape[:2]
seed = cv2.imread(SEED_MASK_PATH, cv2.IMREAD_GRAYSCALE)
seed = cv2.resize(seed, (W, H), interpolation=cv2.INTER_NEAREST)
seed_bool = seed > 127
print(f"seed coverage at {W}x{H}: {int(seed_bool.sum())} px")

predictor = build_sam2_video_predictor(CONFIG, CKPT, device=device)
state = predictor.init_state(video_path=FRAMES_DIR)
predictor.add_new_mask(
    inference_state=state, frame_idx=SEED_FRAME_IDX, obj_id=1, mask=seed_bool
)

n_frames = len(sorted(os.listdir(FRAMES_DIR)))
masks = np.zeros((n_frames, H, W), dtype=np.uint8)

# Forward from seed
for fi, _, ml in predictor.propagate_in_video(state, start_frame_idx=SEED_FRAME_IDX):
    m = (ml[0] > 0.0).squeeze().cpu().numpy()
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    masks[fi] = m.astype(np.uint8) * 255

# Backward from seed
for fi, _, ml in predictor.propagate_in_video(state, start_frame_idx=SEED_FRAME_IDX, reverse=True):
    m = (ml[0] > 0.0).squeeze().cpu().numpy()
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    masks[fi] = m.astype(np.uint8) * 255

# Write per-frame PNGs for ProPainter
os.makedirs(MASKS_DIR, exist_ok=True)
for i in range(n_frames):
    cv2.imwrite(f"{MASKS_DIR}/{i:05d}.png", masks[i])
print(f"wrote {n_frames} mask PNGs to {MASKS_DIR}")

# QA overlay video
import imageio
ov = []
for i in range(n_frames):
    bgr = cv2.imread(os.path.join(FRAMES_DIR, f"{i:05d}.jpg"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    m = masks[i] > 0
    o = rgb.copy()
    o[m] = (o[m] * 0.4 + np.array([255, 0, 255]) * 0.6).astype(np.uint8)
    ov.append(o)
imageio.mimwrite("results/scratches_mask_overlay.mp4", np.stack(ov), fps=25, quality=7)
print("wrote results/scratches_mask_overlay.mp4")
