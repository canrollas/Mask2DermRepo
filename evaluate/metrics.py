"""
Distributional metrics: FID (with bias-corrected extrapolation) and KID.

FID extrapolation method (from the paper §3.2):
  Compute FID at subset fractions [0.25, 0.50, 0.75, 1.00],
  fit a linear regression on (1/N, FID), then extrapolate to 1/N → 0
  to obtain the bias-corrected "global FID".

Usage:
    python evaluate/metrics.py \\
        --real_dir    data/processed/images \\
        --generated_dir outputs/generated \\
        --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# FID helpers
# ---------------------------------------------------------------------------

def _compute_fid_at_n(real_dir: str, gen_dir: str, n: int, device: str) -> float:
    """Compute FID between `n` randomly sampled real and generated images."""
    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        raise ImportError("Install clean-fid: pip install clean-fid")

    score = cleanfid.compute_fid(
        real_dir,
        gen_dir,
        mode="clean",
        num_workers=4,
        batch_size=64,
        device=torch.device(device),
        num_gen=n,
    )
    return float(score)


def compute_fid_extrapolated(
    real_dir: str,
    gen_dir: str,
    fractions: list[float] | None = None,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """Compute bias-corrected FID by linear extrapolation to infinite sample size.

    Returns a dict with keys:
        "fid_at_fractions": list of (n, fid) pairs
        "global_fid":       extrapolated FID at 1/N → 0
        "r2":               R² of the linear fit
    """
    if fractions is None:
        fractions = [0.25, 0.50, 0.75, 1.00]

    real_paths = list(Path(real_dir).glob("*.jpg")) + list(Path(real_dir).glob("*.png"))
    total_n = len(real_paths)
    if total_n == 0:
        raise FileNotFoundError(f"No images found in {real_dir}")

    ns = [max(10, int(total_n * f)) for f in fractions]
    ns = sorted(set(ns))  # unique, ascending

    results = []
    for n in ns:
        fid = _compute_fid_at_n(real_dir, gen_dir, n, device)
        results.append((n, fid))
        print(f"  FID (n={n:>5d}): {fid:.4f}")

    # Linear regression: FID ≈ a/N + b
    inv_ns = np.array([1.0 / n for n, _ in results]).reshape(-1, 1)
    fids = np.array([f for _, f in results])
    reg = LinearRegression().fit(inv_ns, fids)
    global_fid = float(reg.intercept_)
    r2 = float(reg.score(inv_ns, fids))

    return {
        "fid_at_fractions": results,
        "global_fid": global_fid,
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# KID helper
# ---------------------------------------------------------------------------

def compute_kid(
    real_dir: str,
    gen_dir: str,
    device: str = "cuda",
) -> dict:
    """Compute KID (unbiased MMD with polynomial kernel) via torch-fidelity.

    Returns a dict with keys "kid_mean" and "kid_std".
    """
    try:
        import torch_fidelity
    except ImportError:
        raise ImportError("Install torch-fidelity: pip install torch-fidelity")

    metrics = torch_fidelity.calculate_metrics(
        input1=real_dir,
        input2=gen_dir,
        cuda=device == "cuda",
        kid=True,
        kid_subset_size=1000,
        verbose=False,
    )
    return {
        "kid_mean": metrics["kernel_inception_distance_mean"],
        "kid_std": metrics["kernel_inception_distance_std"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID and KID")
    parser.add_argument("--real_dir", required=True, help="Directory of real images")
    parser.add_argument("--generated_dir", required=True, help="Directory of generated images")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_fid", action="store_true", help="Skip FID")
    parser.add_argument("--no_kid", action="store_true", help="Skip KID")
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.25, 0.5, 0.75, 1.0],
                        help="Subset fractions for FID extrapolation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.no_fid:
        print("\n=== FID (extrapolated) ===")
        fid_results = compute_fid_extrapolated(
            args.real_dir, args.generated_dir,
            fractions=args.fractions, device=args.device,
        )
        print(f"  Global (extrapolated) FID: {fid_results['global_fid']:.4f}")
        print(f"  Linear fit R²:             {fid_results['r2']:.4f}")

    if not args.no_kid:
        print("\n=== KID ===")
        kid_results = compute_kid(args.real_dir, args.generated_dir, device=args.device)
        print(f"  KID mean: {kid_results['kid_mean']:.4f}")
        print(f"  KID std:  {kid_results['kid_std']:.4f}")


if __name__ == "__main__":
    main()
