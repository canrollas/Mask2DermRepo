"""
Histogram Matching — Synthetic → Real
======================================
Matches the RGB histogram of synthetic images to the real dataset's distribution.
Saves corrected images to a new directory, originals untouched.

Usage
-----
python experiments/histogram_match.py \
    --real_dir   data/processed/images \
    --syn_dir    /path/to/synthetic \
    --output_dir outputs/generated_matched \
    --max_ref    2000      # how many real images to build the reference from (0 = all)
    --workers    8
"""

import argparse
import os
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from skimage.exposure import match_histograms
from tqdm import tqdm


def collect_images(directory: str, extensions=(".png", ".jpg", ".jpeg")) -> list[Path]:
    paths = []
    for ext in extensions:
        paths.extend(Path(directory).rglob(f"*{ext}"))
    return sorted(paths)


def build_reference(real_paths: list[Path], max_ref: int, seed: int = 42) -> np.ndarray:
    """
    Build a single representative reference image by averaging pixel samples
    from the real dataset. match_histograms needs one reference image, so we
    construct a 256×256 synthetic mosaic from real pixel samples.
    """
    random.seed(seed)
    if max_ref > 0:
        real_paths = random.sample(real_paths, min(max_ref, len(real_paths)))

    print(f"Building reference from {len(real_paths)} real images...")
    pixels = [[], [], []]  # per-channel pixel pools

    for p in tqdm(real_paths, desc="Sampling real pixels", leave=False):
        try:
            img = np.array(Image.open(p).convert("RGB"))
        except Exception:
            continue
        # Sample a random 32×32 patch to keep memory low
        h, w = img.shape[:2]
        r = random.randint(0, max(0, h - 32))
        c = random.randint(0, max(0, w - 32))
        patch = img[r:r+32, c:c+32]
        for ch in range(3):
            pixels[ch].append(patch[:, :, ch].ravel())

    # Stack all sampled pixels into a reference image (H×W×3)
    ref_channels = []
    for ch in range(3):
        flat = np.concatenate(pixels[ch])
        # Reshape into a square image for match_histograms
        side = int(np.sqrt(len(flat)))
        flat = flat[:side * side]
        ref_channels.append(flat.reshape(side, side))

    min_side = min(c.shape[0] for c in ref_channels)
    ref = np.stack([c[:min_side, :min_side] for c in ref_channels], axis=-1)
    print(f"Reference image shape: {ref.shape}")
    return ref


def process_image(src: Path, dst: Path, reference: np.ndarray) -> str:
    try:
        img = np.array(Image.open(src).convert("RGB"))
        matched = match_histograms(img, reference, channel_axis=-1)
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(matched).save(dst)
        return "ok"
    except Exception as e:
        return f"error: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir",   required=True)
    parser.add_argument("--syn_dir",    required=True)
    parser.add_argument("--output_dir", default="outputs/generated_matched")
    parser.add_argument("--max_ref",    type=int, default=2000,
                        help="Real images used to build the reference (0 = all)")
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    real_paths = collect_images(args.real_dir)
    syn_paths  = collect_images(args.syn_dir)
    print(f"Real images : {len(real_paths)}")
    print(f"Syn  images : {len(syn_paths)}")

    reference = build_reference(real_paths, args.max_ref, args.seed)

    out_root = Path(args.output_dir)
    tasks = []
    for src in syn_paths:
        dst = out_root / src.name
        tasks.append((src, dst))

    print(f"\nMatching histograms → {args.output_dir}")
    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_image, src, dst, reference): src
                   for src, dst in tasks}
        with tqdm(total=len(futures), unit="img") as pbar:
            for fut in as_completed(futures):
                result = fut.result()
                if result != "ok":
                    errors += 1
                pbar.update(1)

    print(f"\nDone. {len(tasks) - errors}/{len(tasks)} images saved to {args.output_dir}")
    if errors:
        print(f"  {errors} errors — check input files.")


if __name__ == "__main__":
    main()
