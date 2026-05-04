"""
Pixel Distribution Comparison
==============================
Plots RGB channel histograms of two image datasets side by side and overlaid.

Usage
-----
python experiments/pixel_distribution.py \
    --dir1  data/processed/images \
    --dir2  outputs/generated \
    --label1 "Real (HAM10000)" \
    --label2 "Synthetic (Mask2Derm)" \
    --output_dir results/pixel_distribution \
    --max_images 2000          # sample size for speed, 0 = all
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


CHANNELS = ["Red", "Green", "Blue"]
COLORS   = ["#e74c3c", "#2ecc71", "#3498db"]


def collect_images(directory: str, extensions=(".png", ".jpg", ".jpeg")) -> list[Path]:
    paths = []
    for ext in extensions:
        paths.extend(Path(directory).rglob(f"*{ext}"))
    return sorted(paths)


def compute_histograms(paths: list[Path], max_images: int, bins: int = 256) -> np.ndarray:
    """Returns histogram array of shape (3, bins) — one row per channel."""
    if max_images > 0:
        paths = random.sample(paths, min(max_images, len(paths)))

    counts = np.zeros((3, bins), dtype=np.float64)
    edges  = np.linspace(0, 256, bins + 1)

    for p in tqdm(paths, desc=f"  {Path(paths[0]).parent.name if paths else '?'}", leave=False):
        try:
            img = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
        except Exception:
            continue
        for c in range(3):
            h, _ = np.histogram(img[:, :, c].ravel(), bins=edges)
            counts[c] += h

    # Normalize to density
    counts = counts / counts.sum(axis=1, keepdims=True)
    return counts, edges[:-1]


def plot_distributions(
    hist1: np.ndarray,
    hist2: np.ndarray,
    bin_centers: np.ndarray,
    label1: str,
    label2: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Pixel Intensity Distribution — Real vs Synthetic", fontsize=14, fontweight="bold")

    for c, (ch_name, color) in enumerate(zip(CHANNELS, COLORS)):

        # ---- Row 0: side by side ----
        ax = axes[0, c]
        ax.fill_between(bin_centers, hist1[c], alpha=0.5, color=color,   label=label1)
        ax.fill_between(bin_centers, hist2[c], alpha=0.5, color="gray",  label=label2)
        ax.set_title(f"{ch_name} Channel", fontweight="bold")
        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 255)
        ax.grid(True, alpha=0.25)

        # ---- Row 1: overlaid with difference ----
        ax2 = axes[1, c]
        diff = hist1[c] - hist2[c]
        ax2.plot(bin_centers, hist1[c], color=color,  linewidth=1.2, label=label1)
        ax2.plot(bin_centers, hist2[c], color="gray", linewidth=1.2, label=label2, linestyle="--")
        ax2.fill_between(bin_centers, diff, 0,
                         where=(diff > 0), color=color, alpha=0.2, label="Real > Synth")
        ax2.fill_between(bin_centers, diff, 0,
                         where=(diff < 0), color="gray", alpha=0.2, label="Synth > Real")
        ax2.axhline(0, color="black", linewidth=0.6)
        ax2.set_title(f"{ch_name} — Overlap & Difference")
        ax2.set_xlabel("Pixel value")
        ax2.set_ylabel("Density / Δ")
        ax2.legend(fontsize=7)
        ax2.set_xlim(0, 255)
        ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    png_path = os.path.join(output_dir, "pixel_distribution.png")
    pdf_path = os.path.join(output_dir, "pixel_distribution.pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path,          bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {png_path}")
    print(f"Saved → {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1",       required=True)
    parser.add_argument("--dir2",       required=True)
    parser.add_argument("--label1",     default="Real")
    parser.add_argument("--label2",     default="Synthetic")
    parser.add_argument("--output_dir", default="results/pixel_distribution")
    parser.add_argument("--max_images", type=int, default=2000,
                        help="Max images per dataset (0 = all)")
    parser.add_argument("--bins",       type=int, default=256)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Dataset 1: {args.dir1}")
    paths1 = collect_images(args.dir1)
    print(f"  Found {len(paths1)} images")

    print(f"Dataset 2: {args.dir2}")
    paths2 = collect_images(args.dir2)
    print(f"  Found {len(paths2)} images")

    print("\nComputing histograms...")
    hist1, bins = compute_histograms(paths1, args.max_images, args.bins)
    hist2, _    = compute_histograms(paths2, args.max_images, args.bins)

    print("\nPlotting...")
    plot_distributions(hist1, hist2, bins, args.label1, args.label2, args.output_dir)

    # Print basic stats
    print("\n--- Channel Mean Pixel Values ---")
    for c, ch in enumerate(CHANNELS):
        mean1 = float(np.average(bins, weights=hist1[c]))
        mean2 = float(np.average(bins, weights=hist2[c]))
        print(f"  {ch:5s}  {args.label1}: {mean1:.1f}   {args.label2}: {mean2:.1f}   Δ={mean2-mean1:+.1f}")


if __name__ == "__main__":
    main()
