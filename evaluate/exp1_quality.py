"""
Experiment 1 — Image Quality Assessment
========================================
FID, SSIM, LPIPS between already-generated synthetic images and real test images.

Usage:
    python evaluate/exp1_quality.py \
        --gen_dir   outputs/generated \
        --real_dir  data/processed/test_images \
        --output_dir results/exp1_quality \
        --device    cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate.metrics import (compute_fid, compute_ssim_paired,
                               compute_lpips_paired, print_results, save_results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_dir",    required=True, help="Directory of generated images")
    p.add_argument("--real_dir",   required=True, help="Directory of real test images")
    p.add_argument("--output_dir", default="results/exp1_quality")
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    gen_dir  = Path(args.gen_dir)
    real_dir = Path(args.real_dir)

    n_gen  = len(list(gen_dir.glob("*.png")) + list(gen_dir.glob("*.jpg")))
    n_real = len(list(real_dir.glob("*.png")) + list(real_dir.glob("*.jpg")))
    print(f"Generated : {n_gen} images")
    print(f"Real      : {n_real} images")

    print("\nComputing FID...")
    fid = compute_fid(real_dir, gen_dir, device=args.device)

    print("Computing SSIM...")
    ssim = compute_ssim_paired(real_dir, gen_dir)

    print("Computing LPIPS...")
    try:
        lpips_score = compute_lpips_paired(real_dir, gen_dir, device=args.device)
    except ImportError:
        print("  lpips not installed — skipping (pip install lpips)")
        lpips_score = float("nan")

    results = {
        "fid":    fid,
        "ssim":   ssim,
        "lpips":  lpips_score,
        "n_gen":  n_gen,
        "n_real": n_real,
        "config": vars(args),
    }

    print_results(results, title="Exp 1 — Image Quality")
    save_results(results, Path(args.output_dir) / "results.json")


if __name__ == "__main__":
    main()
