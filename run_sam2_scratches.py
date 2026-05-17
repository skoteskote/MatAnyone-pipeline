"""Batch-propagate hand-painted scratch masks across video clips.

Looks in --in-dir for triplets:
    <name>.mp4         the clip
    <name>.jpg         the unedited still (used to auto-locate the seed frame)
    <name>_mask.jpg    the BW mask painted on that still (white = inpaint)

For each triplet it decodes the clip, finds the seed frame via MSE-match against
the still, propagates the mask forward + backward with SAM 2, and writes:

    <out-dir>/<name>/frames/         clip frames as JPGs (input to ProPainter)
    <out-dir>/<name>/masks/          propagated mask PNGs (input to ProPainter)
    <out-dir>/<name>/mask_overlay.mp4   magenta-over-source QA video
    <out-dir>/<name>/run_propainter.sh  one-line command to run the inpaint

Then run the generated shell scripts (CUDA box recommended):
    bash <out-dir>/<name>/run_propainter.sh

Single-file mode: point --in-dir at a folder with just one triplet.
"""
import os, sys, argparse, shutil, glob
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import cv2
import imageio
import torch
from sam2.build_sam import build_sam2_video_predictor

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


def decode_clip(src_path, frames_dir, target_size):
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    cmd = (
        f'ffmpeg -y -loglevel error -i "{src_path}" '
        f'-vf "scale={target_size}:{target_size}" -q:v 2 -start_number 0 '
        f'"{frames_dir}/%05d.jpg"'
    )
    if os.system(cmd) != 0:
        raise RuntimeError(f"ffmpeg failed for {src_path}")


def propagate_mask(predictor, frames_dir, masks_dir, mask_path, still_path):
    frame0 = cv2.imread(os.path.join(frames_dir, "00000.jpg"))
    H, W = frame0.shape[:2]
    seed_idx, mse = autodetect_seed_frame(still_path, frames_dir)
    print(f"  seed frame: {seed_idx} (mse={mse:.1f})")

    seed = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    seed = cv2.resize(seed, (W, H), interpolation=cv2.INTER_NEAREST)
    seed_bool = seed > 127
    print(f"  seed coverage at {W}x{H}: {int(seed_bool.sum())} px")

    state = predictor.init_state(video_path=frames_dir)
    predictor.add_new_mask(
        inference_state=state, frame_idx=seed_idx, obj_id=1, mask=seed_bool
    )

    n_frames = len(sorted(os.listdir(frames_dir)))
    masks = np.zeros((n_frames, H, W), dtype=np.uint8)
    for direction_reverse in (False, True):
        for fi, _, ml in predictor.propagate_in_video(
            state, start_frame_idx=seed_idx, reverse=direction_reverse
        ):
            m = (ml[0] > 0.0).squeeze().cpu().numpy()
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            masks[fi] = m.astype(np.uint8) * 255

    if os.path.isdir(masks_dir):
        shutil.rmtree(masks_dir)
    os.makedirs(masks_dir, exist_ok=True)
    for i in range(n_frames):
        cv2.imwrite(f"{masks_dir}/{i:05d}.png", masks[i])
    return masks, seed_idx


def write_overlay(frames_dir, masks, out_path, fps=25):
    n = masks.shape[0]
    ov = []
    for i in range(n):
        bgr = cv2.imread(os.path.join(frames_dir, f"{i:05d}.jpg"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = masks[i] > 0
        o = rgb.copy()
        o[m] = (o[m] * 0.4 + np.array([255, 0, 255]) * 0.6).astype(np.uint8)
        ov.append(o)
    imageio.mimwrite(out_path, np.stack(ov), fps=fps, quality=7)


def write_propainter_script(out_dir, propainter_dir, target_size, mask_dilation,
                            ref_stride, neighbor_length, raft_iter, save_fps):
    """Write a per-clip shell script users can run on their GPU."""
    sh = f"""#!/usr/bin/env bash
# Run ProPainter inpainting using the masks SAM 2 propagated for this clip.
set -euo pipefail
cd "{propainter_dir}"
python inference_propainter.py \\
    -i "{os.path.abspath(out_dir)}/frames" \\
    -m "{os.path.abspath(out_dir)}/masks" \\
    -o "{os.path.abspath(out_dir)}/inpaint" \\
    --width {target_size} --height {target_size} \\
    --mask_dilation {mask_dilation} \\
    --ref_stride {ref_stride} \\
    --neighbor_length {neighbor_length} \\
    --raft_iter {raft_iter} \\
    --save_fps {save_fps}
"""
    path = os.path.join(out_dir, "run_propainter.sh")
    with open(path, "w") as f: f.write(sh)
    os.chmod(path, 0o755)


def find_triplets(in_dir):
    """Return list of (name, mp4_path, still_path, mask_path)."""
    triplets = []
    for mask_path in sorted(glob.glob(os.path.join(in_dir, "*_mask.*"))):
        base = os.path.basename(mask_path)
        name = os.path.splitext(base)[0][:-len("_mask")]
        mp4 = os.path.join(in_dir, name + ".mp4")
        # still can be .jpg, .png, .jpeg
        still = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = os.path.join(in_dir, name + ext)
            if os.path.isfile(cand) and cand != mask_path:
                still = cand
                break
        if not os.path.isfile(mp4):
            print(f"skip {name}: no matching .mp4")
            continue
        if still is None:
            print(f"skip {name}: no matching still")
            continue
        triplets.append((name, mp4, still, mask_path))
    return triplets


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="folder of <name>.mp4 + <name>.jpg + <name>_mask.jpg")
    ap.add_argument("--out-dir", required=True, help="output root; one subfolder per clip")
    ap.add_argument("--target-size", type=int, default=1080, help="processing resolution (square)")
    ap.add_argument("--propainter-dir", default="../ProPainter",
                    help="path to your ProPainter clone (used by generated run_propainter.sh)")
    ap.add_argument("--mask-dilation", type=int, default=10)
    ap.add_argument("--ref-stride", type=int, default=4)
    ap.add_argument("--neighbor-length", type=int, default=16)
    ap.add_argument("--raft-iter", type=int, default=30)
    ap.add_argument("--save-fps", type=int, default=25)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}, target size: {args.target_size}")

    triplets = find_triplets(args.in_dir)
    if not triplets:
        print(f"no <name>.mp4 + <name>.* + <name>_mask.* triplets found in {args.in_dir}")
        sys.exit(1)
    print(f"found {len(triplets)} triplet(s)")

    predictor = build_sam2_video_predictor(CONFIG, CKPT, device=device)
    os.makedirs(args.out_dir, exist_ok=True)

    for name, mp4, still, mask_path in triplets:
        print(f"\n=== {name} ===")
        clip_out = os.path.join(args.out_dir, name)
        frames_dir = os.path.join(clip_out, "frames")
        masks_dir  = os.path.join(clip_out, "masks")
        os.makedirs(clip_out, exist_ok=True)

        decode_clip(mp4, frames_dir, args.target_size)
        masks, _ = propagate_mask(predictor, frames_dir, masks_dir, mask_path, still)
        write_overlay(frames_dir, masks, os.path.join(clip_out, "mask_overlay.mp4"),
                      fps=args.save_fps)
        write_propainter_script(clip_out, args.propainter_dir, args.target_size,
                                args.mask_dilation, args.ref_stride,
                                args.neighbor_length, args.raft_iter, args.save_fps)
        print(f"  wrote {clip_out}/ (frames, masks, mask_overlay.mp4, run_propainter.sh)")

    print(f"\nDone. Run ProPainter per clip:")
    for name, *_ in triplets:
        print(f"  bash {os.path.join(args.out_dir, name, 'run_propainter.sh')}")


if __name__ == "__main__":
    main()
