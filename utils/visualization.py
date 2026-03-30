"""
Visualization utilities for Mask2Derm.

Functions:
  plot_loss_curve        — parse training log and plot loss over steps/epochs
  plot_iou_histogram     — distribution of per-image IoU scores
  make_comparison_grid   — real vs synthetic side-by-side grid
  plot_fid_extrapolation — FID vs 1/N with regression line
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(
    losses: list[float] | None = None,
    log_file: str | Path | None = None,
    output_path: str | Path = "loss_curve.png",
    title: str = "Training Loss",
    smooth_window: int = 50,
) -> None:
    """Plot and save a training loss curve.

    Provide either `losses` (list of floats) or `log_file` (CSV / plain text).
    """
    if losses is None and log_file is not None:
        losses = _parse_log_file(log_file)

    if not losses:
        raise ValueError("Provide either losses list or a valid log_file.")

    steps = np.arange(1, len(losses) + 1)
    smoothed = _smooth(losses, smooth_window)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, alpha=0.25, color="steelblue", linewidth=0.8, label="raw")
    ax.plot(steps, smoothed, color="steelblue", linewidth=1.8, label=f"smoothed (w={smooth_window})")
    ax.set_xlabel("Training step")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved → {output_path}")


def _smooth(values: list[float], window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _parse_log_file(log_file: str | Path) -> list[float]:
    """Parse a log file with lines like '... loss: 0.0123 ...'."""
    import re
    losses = []
    pattern = re.compile(r"loss[:\s=]+([0-9]+\.[0-9]+)")
    with open(log_file) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                losses.append(float(m.group(1)))
    return losses


# ---------------------------------------------------------------------------
# IoU histogram
# ---------------------------------------------------------------------------

def plot_iou_histogram(
    ious: list[float],
    output_path: str | Path = "iou_distribution.png",
    title: str = "Shape Consistency — IoU Distribution",
    bins: int = 30,
) -> None:
    """Plot and save a histogram of per-image IoU scores."""
    ious = np.array(ious)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ious, bins=bins, color="teal", edgecolor="white", linewidth=0.5)
    ax.axvline(ious.mean(), color="red", linewidth=1.5,
               label=f"mean = {ious.mean():.3f}")
    ax.axvline(np.median(ious), color="orange", linestyle="--", linewidth=1.5,
               label=f"median = {np.median(ious):.3f}")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"IoU histogram saved → {output_path}")


# ---------------------------------------------------------------------------
# Comparison grid
# ---------------------------------------------------------------------------

def make_comparison_grid(
    real_paths: list[str | Path],
    synthetic_paths: list[str | Path],
    output_path: str | Path = "comparison_grid.png",
    n_cols: int = 4,
    img_size: int = 256,
) -> None:
    """Create a two-row comparison grid: real (top) vs synthetic (bottom)."""
    from PIL import Image

    n = min(len(real_paths), len(synthetic_paths), n_cols * 2)
    real_paths = real_paths[:n]
    synthetic_paths = synthetic_paths[:n]

    rows = []
    for paths in (real_paths, synthetic_paths):
        row_imgs = []
        for p in paths:
            img = Image.open(p).convert("RGB").resize((img_size, img_size), Image.LANCZOS)
            row_imgs.append(np.array(img))
        rows.append(np.hstack(row_imgs))

    grid = np.vstack(rows)
    from PIL import Image as PILImage
    PILImage.fromarray(grid).save(output_path)
    print(f"Comparison grid saved → {output_path}")


# ---------------------------------------------------------------------------
# FID extrapolation plot
# ---------------------------------------------------------------------------

def plot_fid_extrapolation(
    fid_results: dict,
    output_path: str | Path = "fid_extrapolation_curve.png",
) -> None:
    """Plot FID vs 1/N with regression line and extrapolated intercept.

    fid_results is the dict returned by evaluate.metrics.compute_fid_extrapolated().
    """
    from sklearn.linear_model import LinearRegression

    pairs = fid_results["fid_at_fractions"]  # [(n, fid), ...]
    global_fid = fid_results["global_fid"]

    ns = np.array([p[0] for p in pairs])
    fids = np.array([p[1] for p in pairs])
    inv_ns = (1.0 / ns).reshape(-1, 1)

    reg = LinearRegression().fit(inv_ns, fids)
    x_range = np.linspace(0, inv_ns.max() * 1.1, 200).reshape(-1, 1)
    y_pred = reg.predict(x_range)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(inv_ns, fids, color="steelblue", zorder=5, label="FID at subset size")
    ax.plot(x_range, y_pred, color="red", linewidth=1.5, linestyle="--", label="Linear fit")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.scatter([0], [global_fid], color="green", zorder=6, s=80,
               label=f"Global FID = {global_fid:.2f}")
    ax.set_xlabel("1 / N (inverse sample size)")
    ax.set_ylabel("FID score")
    ax.set_title("FID Extrapolation to Infinite Sample Size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"FID extrapolation plot saved → {output_path}")
