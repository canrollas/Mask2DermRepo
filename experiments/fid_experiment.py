"""
FID & KID Experiment
====================
Computes FID and KID between a real image directory and a generated image directory.

Usage
-----
python experiments/fid_experiment.py \
    --real_dir  data/processed/images \
    --gen_dir   outputs/generated \
    --output_dir results/fid_experiment \
    --device    cuda
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch_fidelity import calculate_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir",   required=True)
    parser.add_argument("--gen_dir",    required=True)
    parser.add_argument("--output_dir", default="results/fid_experiment")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    n_real = len(list(Path(args.real_dir).rglob("*.png")) + list(Path(args.real_dir).rglob("*.jpg")))
    n_gen  = len(list(Path(args.gen_dir).rglob("*.png"))  + list(Path(args.gen_dir).rglob("*.jpg")))
    print(f"Real images : {n_real}")
    print(f"Gen  images : {n_gen}")
    print("Computing FID & KID...")

    metrics = calculate_metrics(
        input1=args.real_dir,
        input2=args.gen_dir,
        cuda=(args.device == "cuda"),
        fid=True,
        kid=True,
        kid_subset_size=min(1000, n_real, n_gen),
        verbose=True,
    )

    fid      = metrics["frechet_inception_distance"]
    kid_mean = metrics["kernel_inception_distance_mean"]
    kid_std  = metrics["kernel_inception_distance_std"]

    print(f"\nFID : {fid:.4f}")
    print(f"KID : {kid_mean:.4f} ± {kid_std:.4f}")

    summary = {
        "n_real":    n_real,
        "n_gen":     n_gen,
        "fid":       fid,
        "kid_mean":  kid_mean,
        "kid_std":   kid_std,
        "config":    vars(args),
    }

    out_path = os.path.join(args.output_dir, "fid_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
