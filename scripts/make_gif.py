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

GRID_W   = COLS * THUMB
GRID_H   = ROWS * THUMB
CANVAS_H = GRID_H + LABEL_H


def load_pairs(masks_dir: str, gen_dir: str, limit: int = MAX_PAIRS):
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "mask_*.png")))
    pairs = []
    for mp in mask_files:
        stem = Path(mp).stem
        gp = os.path.join(gen_dir, f"{stem}_generated.png")
        if os.path.exists(gp):
            pairs.append((mp, gp))
        if len(pairs) == limit:
            break
    return pairs


def build_grid(images: list[Image.Image], label: str) -> Image.Image:
    canvas = Image.new("RGB", (GRID_W, CANVAS_H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, GRID_W, LABEL_H], fill=LABEL_BG)
    draw.text((GRID_W // 2, LABEL_H // 2), label, fill=TEXT_COLOR, anchor="mm")
    for idx, img in enumerate(images):
        row, col = idx // COLS, idx % COLS
        thumb = img.convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
        canvas.paste(thumb, (col * THUMB, LABEL_H + row * THUMB))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--gen_dir",   required=True)
    parser.add_argument("--output",    default="assets/demo.gif")
    parser.add_argument("--n",         type=int, default=MAX_PAIRS)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args.masks_dir, args.gen_dir, limit=args.n)
    print(f"{len(pairs)} pairs loaded")

    masks   = [Image.open(mp) for mp, _ in pairs]
    lesions = [Image.open(gp) for _, gp in pairs]

    f_mask   = build_grid(masks,   "Masks").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    f_lesion = build_grid(lesions, "Generated Lesions").quantize(colors=256, method=Image.Quantize.MEDIANCUT)

    f_mask.save(
        args.output,
        save_all=True,
        append_images=[f_lesion],
        duration=[1000, 1000],
        loop=0,
        optimize=True,
    )
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved → {args.output}  ({size_mb:.1f} MB, 2 frames, {len(pairs)} pairs)")


if __name__ == "__main__":
    main()
