# Pipelines: bad-greenscreen masks + scratch inpainting

Two AI-assisted post-production workflows built on this repo's tooling, MatAnyone, and SAM 2.

- **Bad-greenscreen masks** — clips where the green wall is unevenly lit (yellow patches, hot spots) and a plain chroma key leaves holes. SAM 2 grabs the whole wall region as a single object, including the colour-shifted parts.
- **Scratch / mark inpainting** — small unwanted marks on a moving rigid object (logos, surface defects). SAM 2 propagates a single hand-painted mask across all frames, ProPainter inpaints.

Both pipelines were developed on macOS / Apple Silicon (MPS); the CUDA path on Linux/Windows is identical aside from device selection.

---

## 1. Setup

### Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision           # CUDA: choose the cu12* wheel from pytorch.org
.venv/bin/pip install -e . --no-deps              # this repo (MatAnyone) for the inference utils
.venv/bin/pip install \
    opencv-python tqdm "imageio==2.25.0" "imageio[ffmpeg]" \
    omegaconf hydra-core einops huggingface_hub safetensors av \
    hickle requests gdown gitpython "scipy>=1.7" easydict \
    "git+https://github.com/cheind/py-thin-plate-spline" \
    sam2 \
    matplotlib addict scikit-image timm future
```

### Patch needed for newer torchvision

`torchvision >= 0.22` removed `torchvision.io.read_video`. `matanyone/utils/inference_utils.py` in this repo includes a PyAV fallback for the video read; nothing to do if you're using this fork's copy.

### Model weights

```bash
mkdir -p pretrained_models/sam2
curl -L -o pretrained_models/sam2/sam2.1_hiera_base_plus.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
# MatAnyone (~135 MB) auto-downloads on first run.
```

### Sister repo — ProPainter (only for scratch inpainting)

```bash
cd ..
git clone https://github.com/sczhou/ProPainter.git
# Its checkpoints (RAFT, flow completion, ProPainter ~200 MB total) auto-download on first run.
```

---

## 2. Pipeline A — bad-greenscreen masks

Input: folder of .mp4 clips. Output: a per-clip binary background mask video (white = wall + yellow patches, black = foreground) and a magenta-overlay QA video.

```bash
.venv/bin/python run_sam2_batch.py \
    --in-dir "/path/to/clips folder" \
    --out-dir results/<name>
```

### How it works

1. Decode each clip to JPEG frames (`scale=960:960`).
2. From frame 0, build a permissive HSV chroma-key mask of the green wall (Hue 25–100, S ≥ 40, V ≥ 25), then morph open/close and erode slightly to stay inside the wall.
3. Feed that mask straight into SAM 2's video predictor via `add_new_mask(...)`. SAM 2 propagates and extends — the yellow backlit patches are connected to the green wall in the model's eyes, so they get pulled in for free.
4. Soft-blur the propagated mask, save `<clip>_mask.mp4` + `<clip>_overlay.mp4` + `<clip>_seed.png`.

### Tuning

- **Wider HSV range** (e.g. `Hue 20–105`) if the wall colour varies hard between clips.
- **Bigger erosion** in `chroma_mask()` if SAM seed bleeds into hair / loose foreground edges.
- **Resolution** is hard-coded to 960²; bump `TARGET_SIZE` if you have VRAM headroom and need cleaner edges. SAM 2 base_plus runs comfortably at 1280² on a 12 GB card.

### How to use the output mask

The mask covers the *background*. Union (Screen blend, OR) it with your chroma-key background mask, or invert + multiply onto your foreground key. The combination kills both the green areas (your keyer's job) and the yellow patches (SAM 2's job).

Upscale to your master resolution before unioning:

```bash
ffmpeg -i results/<name>/<clip>_mask.mp4 \
       -vf "scale=2496:2496:flags=lanczos" \
       results/<name>/<clip>_mask_2496.mp4
```

---

## 3. Pipeline B — scratch / mark inpainting

Input: one clip + one binary BW mask painted on a single non-blurry frame. Output: an inpainted clip with the marks gone.

### Step 1 — collect triplets in one folder

For every clip, drop three files in the same folder, all with the same base name:

```
my-folder/
  clip-A.mp4         clip-A.jpg        clip-A_mask.jpg
  clip-B.mp4         clip-B.jpg        clip-B_mask.jpg
  ...
```

- `<name>.mp4` — the clip
- `<name>.jpg` / `.png` / `.jpeg` — an **un-edited** still from somewhere in the clip
- `<name>_mask.jpg` (or `.png`) — the **BW mask** painted on that still: white over scratches/marks, black everywhere else, at the clip's resolution

You don't need to know which frame the still is from — the script auto-locates it by MSE-matching the still against each clip frame at 256² (sub-second per clip).

### Step 2 — propagate masks for all clips with SAM 2

```bash
.venv/bin/python run_sam2_scratches.py \
    --in-dir /path/to/my-folder \
    --out-dir results/scratches \
    --target-size 1080 \
    --propainter-dir ../ProPainter
```

For each triplet you get:

```
results/scratches/clip-A/
    frames/             clip frames (input to ProPainter)
    masks/              propagated mask PNGs
    mask_overlay.mp4    magenta-over-source QA video — open this first
    run_propainter.sh   ready-to-run command for the inpainting step
```

**Check `mask_overlay.mp4`** before running ProPainter — if SAM 2 lost the mask anywhere or grew it onto the hand, repaint with more coverage or paint on a sharper still.

### Step 3 — run ProPainter per clip

```bash
bash results/scratches/clip-A/run_propainter.sh
bash results/scratches/clip-B/run_propainter.sh
```

The generated scripts already include the recommended quality knobs (`--mask_dilation 10 --ref_stride 4 --neighbor_length 16 --raft_iter 30 --width 1080 --height 1080`). To use different values, override at script-generation time:

```bash
.venv/bin/python run_sam2_scratches.py \
    --in-dir ... --out-dir ... \
    --mask-dilation 12 --neighbor-length 20 --target-size 1080
```

Outputs land at `results/scratches/clip-A/inpaint/frames/inpaint_out.mp4`.

### Quality knobs

| Flag | Default | Recommended (GPU) | What it does |
| --- | ---: | ---: | --- |
| `--width / --height` | -1 (no resize) | match master res | Process at full res; mask details survive |
| `--mask_dilation` | 4 | 8–12 | Grows mask by N px; biggest lever for "scratches between letters". Too high erodes legitimate detail. |
| `--ref_stride` | 10 | 4 | Picks reference frames every N frames for global context. Lower = more refs = better texture match. |
| `--neighbor_length` | 10 | 16 | Local temporal window around each frame. Wider = smoother propagation, more VRAM. |
| `--raft_iter` | 20 | 30 | RAFT optical flow iterations. Higher = more accurate flow. |
| `--subvideo_length` | 80 | 40 if OOM | Splits long clips into chunks of N frames. Lower = less VRAM, slight quality cost at chunk seams. |
| `--fp16` | off | on if OOM | Half-precision; small quality hit. |

### Common failure modes

- **Scratch reappears partway through clip** → SAM 2 lost the mask in that segment. Open the overlay and find the bad frame; either repaint at *that* frame as well (run SAM 2 with two seed frames) or increase your original mask's coverage.
- **Inpainting destroys legitimate text/logo** → the mask grew onto the letter. Reduce `--mask_dilation` *or* paint a tighter mask. ProPainter is good but it can't guess what's underneath if the mask covers the entire letter.
- **Texture seams between borrowed regions** → lower `--ref_stride`, raise `--neighbor_length`. Costs VRAM.

### MPS vs CUDA timing (59 frames, 960² → 1080²)

| Stage | M3 Max (MPS) | RTX 3080 Ti (CUDA, est.) |
| --- | ---: | ---: |
| SAM 2 mask propagation | ~30 s | ~5 s |
| ProPainter inference | ~90 min | ~5–10 min |

ProPainter on CUDA is roughly 10× faster than MPS, so most of the knob-twiddling iteration cost goes away on your PC.

---

## 4. Files in this repo

| File | Purpose |
| --- | --- |
| `run_sam2_batch.py` | Pipeline A — batch background masks |
| `run_sam2_scratches.py` | Pipeline B — propagate hand-painted mask across video |
| `run_sam2_inv.py` | Earlier experiment: single-clip inverted SAM 2 with HSV + chroma alpha refinement |
| `run_sam2_yellow.py` | Single-clip variant of Pipeline A (click-driven instead of mask-driven seed) |
| `run_sam2.py`, `run_sam2_mt2.py` | Multi-object SAM 2 examples (track box / hand / desk separately) |
| `combine_mattes.py` | Generic screen-blend (`1 − (1−a)(1−b)`) for two alpha-matte videos with a source |
| `matanyone/utils/inference_utils.py` | Patched to fall back to PyAV when `torchvision.io.read_video` is missing |
