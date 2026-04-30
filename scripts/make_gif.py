"""
2-frame blinking GIF:
  Frame 1: 10x5 grid of all 50 masks      (1000 ms)
  Frame 2: 10x5 grid of all 50 lesions    (1000 ms)
Saves to assets/demo.gif

Usage:
    python scripts/make_gif.py \
        --masks_dir   <dir with mask_*.png> \
        --gen_dir     <dir with mask_*_generated.png> \
        --output      assets/demo.gif
"""
import argparse
import glob
import os
from pathlib import Path
from PIL import Image, ImageDraw

MAX_PAIRS  = 50
COLS       = 10
ROWS       = 5
THUMB      = 128
LABEL_H    = 32
BG_COLOR   = (12, 12, 12)
LABEL_BG   = (24, 24, 24)
TEXT_COLOR = (230, 230, 230)

BORDER    = 1
CELL      = THUMB + BORDER
GRID_W    = COLS * CELL + BORDER
GRID_H    = ROWS * CELL + BORDER
CANVAS_H  = GRID_H + LABEL_H
BORDER_COLOR = (255, 255, 255)


def load_pairs(masks_dir: str, gen_dir: str, limit: int = MAX_PAIRS, exclude=None):
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "mask_*.png")))
    exclude_set = set(exclude or [])
    pairs = []
    for mp in mask_files:
        stem = Path(mp).stem
        if stem in exclude_set:
            continue
        gp = os.path.join(gen_dir, f"{stem}_generated.png")
        if os.path.exists(gp):
            pairs.append((mp, gp))
        if len(pairs) == limit:
            break
    return pairs


def build_grid(images: list[Image.Image], label: str) -> Image.Image:
    canvas = Image.new("RGB", (GRID_W, CANVAS_H), BORDER_COLOR)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, GRID_W, LABEL_H], fill=LABEL_BG)
    draw.text((GRID_W // 2, LABEL_H // 2), label, fill=TEXT_COLOR, anchor="mm")
    for idx, img in enumerate(images):
        row, col = idx // COLS, idx % COLS
        thumb = img.convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
        x = BORDER + col * CELL
        y = LABEL_H + BORDER + row * CELL
        canvas.paste(thumb, (x, y))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--gen_dir",   required=True)
    parser.add_argument("--output",    default="assets/demo.gif")
    parser.add_argument("--n",       type=int,   default=MAX_PAIRS)
    parser.add_argument("--exclude", nargs="*",  default=[], help="Mask stems to skip (e.g. mask_00025)")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args.masks_dir, args.gen_dir, limit=args.n, exclude=args.exclude)
    print(f"{len(pairs)} pairs loaded")

    masks   = [Image.open(mp) for mp, _ in pairs]
    lesions = [Image.open(gp) for _, gp in pairs]

    FADE_STEPS   = 8
    HOLD_MS      = 1200
    FADE_MS      = 80

    base_mask   = build_grid(masks,   "Masks")
    base_lesion = build_grid(lesions, "Generated Lesions")

    frames    = []
    durations = []

    # hold on masks
    frames.append(base_mask.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    durations.append(HOLD_MS)

    # fade mask → lesion
    for i in range(1, FADE_STEPS + 1):
        alpha = i / FADE_STEPS
        blended = Image.blend(base_mask, base_lesion, alpha)
        frames.append(blended.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        durations.append(FADE_MS)

    # hold on lesions
    frames.append(base_lesion.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    durations.append(HOLD_MS)

    # fade lesion → mask
    for i in range(1, FADE_STEPS + 1):
        alpha = i / FADE_STEPS
        blended = Image.blend(base_lesion, base_mask, alpha)
        frames.append(blended.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        durations.append(FADE_MS)

    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved → {args.output}  ({size_mb:.1f} MB, 2 frames, {len(pairs)} pairs)")


if __name__ == "__main__":
    main()
