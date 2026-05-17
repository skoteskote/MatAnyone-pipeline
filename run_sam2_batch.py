"""Batch SAM 2 background mask for clips with bad greenscreens (yellow patches).

Strategy: auto-pick prompts from HSV — sample interior points of the largest green
region as positive (wall) prompts, sample interior of the largest non-green region as
negative (foreground) prompts. SAM 2 propagates and includes the yellow patches that
chroma keying misses, since they're spatially connected to the green wall.
"""
import os, sys, glob, argparse, shutil
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import cv2
import imageio
import torch
from sam2.build_sam import build_sam2_video_predictor

CKPT = "pretrained_models/sam2/sam2.1_hiera_base_plus.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
TARGET_SIZE = 960


def chroma_mask(bgr):
    """Permissive HSV chroma mask of the green wall, eroded to be conservative."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([25, 40, 25]), np.array([100, 255, 255]))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k)
    # Erode slightly to stay inside the wall (avoid bleed into edges)
    green = cv2.erode(green, k, iterations=1)
    return green > 0


def process_clip(predictor, src_path, frames_dir, out_prefix, fps=25):
    # Decode at 960x960 (square pad-keep is fine since input is already 1:1)
    os.makedirs(frames_dir, exist_ok=True)
    os.system(
        f'ffmpeg -y -loglevel error -i "{src_path}" -vf "scale={TARGET_SIZE}:{TARGET_SIZE}" '
        f'-q:v 2 -start_number 0 "{frames_dir}/%05d.jpg"'
    )

    frame0 = cv2.imread(os.path.join(frames_dir, "00000.jpg"))
    H, W = frame0.shape[:2]
    seed = chroma_mask(frame0)
    print(f"  seed coverage: {seed.mean()*100:.1f}%")

    # QA preview of seed mask
    vis = frame0.copy()
    vis[seed] = (vis[seed] * 0.4 + np.array([0, 255, 255]) * 0.6).astype(np.uint8)
    cv2.imwrite(f"{out_prefix}_seed.png", vis)

    state = predictor.init_state(video_path=frames_dir)
    predictor.add_new_mask(
        inference_state=state, frame_idx=0, obj_id=1, mask=seed
    )

    n_frames = len(sorted(os.listdir(frames_dir)))
    phas = np.zeros((n_frames, H, W), dtype=np.uint8)
    for fi, _, ml in predictor.propagate_in_video(state):
        m = (ml[0] > 0.0).squeeze().cpu().numpy()
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        phas[fi] = m.astype(np.uint8) * 255
    for i in range(n_frames):
        phas[i] = cv2.GaussianBlur(phas[i], (0, 0), sigmaX=1.5, sigmaY=1.5)

    imageio.mimwrite(f"{out_prefix}_mask.mp4", phas, fps=fps, quality=7)

    vis_frames = []
    for i in range(n_frames):
        bgr = cv2.imread(os.path.join(frames_dir, f"{i:05d}.jpg"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ov = rgb.copy()
        m = phas[i] > 0
        ov[m] = (ov[m] * 0.4 + np.array([255, 0, 255]) * 0.6).astype(np.uint8)
        vis_frames.append(ov)
    imageio.mimwrite(f"{out_prefix}_overlay.mp4", np.stack(vis_frames), fps=fps, quality=7)
    return n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tmp-dir", default="/tmp/sf_frames")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    predictor = build_sam2_video_predictor(CONFIG, CKPT, device=device)
    os.makedirs(args.out_dir, exist_ok=True)

    clips = sorted(glob.glob(os.path.join(args.in_dir, "*.mp4")))
    print(f"found {len(clips)} clips")
    for c in clips:
        name = os.path.splitext(os.path.basename(c))[0]
        print(f"\n=== {name} ===")
        frames_dir = os.path.join(args.tmp_dir, name)
        if os.path.isdir(frames_dir): shutil.rmtree(frames_dir)
        out_prefix = os.path.join(args.out_dir, name)
        n = process_clip(predictor, c, frames_dir, out_prefix)
        print(f"  wrote {out_prefix}_mask.mp4 and _overlay.mp4 ({n} frames)")


if __name__ == "__main__":
    main()
