"""Screen-blend two alpha-matte videos with a source video and write a recomposite."""
import os, sys, argparse
import numpy as np
import cv2
import imageio
import av

BG = np.array([120, 255, 155], dtype=np.float32) / 255.0


def read_video_gray(path):
    c = av.open(path); arr = [f.to_ndarray(format='gray') for f in c.decode(video=0)]; c.close()
    return np.stack(arr)


def read_video_rgb(path):
    c = av.open(path); arr = [f.to_ndarray(format='rgb24') for f in c.decode(video=0)]; c.close()
    return np.stack(arr)


def match_size(arr, hw):
    if arr.shape[1:3] == hw:
        return arr
    return np.stack([cv2.resize(x, (hw[1], hw[0])) for x in arr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--a",   required=True, help="alpha video A (.mp4)")
    ap.add_argument("--b",   required=True, help="alpha video B (.mp4)")
    ap.add_argument("--out-prefix", required=True, help="e.g. results/mask-test-2_combined")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    src = read_video_rgb(args.src).astype(np.float32) / 255.0
    a = read_video_gray(args.a).astype(np.float32) / 255.0
    b = read_video_gray(args.b).astype(np.float32) / 255.0
    n = min(src.shape[0], a.shape[0], b.shape[0])
    src, a, b = src[:n], a[:n], b[:n]

    target_hw = a.shape[1:3]
    src = match_size(src, target_hw)
    b = match_size(b, target_hw)

    alpha = 1.0 - (1.0 - a) * (1.0 - b)

    fgrs, phas = [], []
    for i in range(n):
        al = alpha[i][..., None]
        com = src[i] * al + BG.reshape(1, 1, 3) * (1.0 - al)
        com = np.round(np.clip(com * 255.0, 0, 255)).astype(np.uint8)
        fgrs.append(com)
        phas.append(np.round(np.clip(alpha[i] * 255.0, 0, 255)).astype(np.uint8))

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    imageio.mimwrite(f"{args.out_prefix}_fgr.mp4", np.stack(fgrs), fps=args.fps, quality=7)
    imageio.mimwrite(f"{args.out_prefix}_pha.mp4", np.stack(phas), fps=args.fps, quality=7)
    print(f"wrote {args.out_prefix}_(fgr|pha).mp4 from {n} frames at {target_hw[1]}x{target_hw[0]}")


if __name__ == "__main__":
    main()
