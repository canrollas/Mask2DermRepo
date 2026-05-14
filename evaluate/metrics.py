"""
Shared metric utilities for Mask2Derm evaluation.

Provides:
  - FID via torch-fidelity
  - SSIM / LPIPS for paired image sets
  - Dice coefficient and IoU for binary masks
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(real_dir: str | Path, gen_dir: str | Path,
                device: str = "cuda") -> float:
    from torch_fidelity import calculate_metrics
    m = calculate_metrics(
        input1=str(real_dir),
        input2=str(gen_dir),
        cuda=(device == "cuda"),
        fid=True,
        verbose=False,
    )
    return float(m["frechet_inception_distance"])


# ---------------------------------------------------------------------------
# SSIM (paired)
# ---------------------------------------------------------------------------

def compute_ssim_paired(real_dir: str | Path,
                        gen_dir: str | Path,
                        size: int = 256) -> float:
    from skimage.metrics import structural_similarity as ssim
    import cv2

    real_dir, gen_dir = Path(real_dir), Path(gen_dir)
    exts = {".png", ".jpg", ".jpeg"}

    real_paths = sorted(p for p in real_dir.iterdir() if p.suffix.lower() in exts)
    scores = []
    for rp in real_paths:
        gp = gen_dir / rp.name
        if not gp.exists():
            continue
        r = np.array(Image.open(rp).convert("RGB").resize((size, size)))
        g = np.array(Image.open(gp).convert("RGB").resize((size, size)))
        s = ssim(r, g, channel_axis=-1, data_range=255)
        scores.append(s)
    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# LPIPS (paired)
# ---------------------------------------------------------------------------

def compute_lpips_paired(real_dir: str | Path,
                         gen_dir: str | Path,
                         device: str = "cuda",
                         size: int = 256) -> float:
    import lpips
    from torchvision import transforms

    loss_fn = lpips.LPIPS(net="alex").to(device)
    to_tensor = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    real_dir, gen_dir = Path(real_dir), Path(gen_dir)
    exts = {".png", ".jpg", ".jpeg"}
    real_paths = sorted(p for p in real_dir.iterdir() if p.suffix.lower() in exts)

    scores = []
    with torch.no_grad():
        for rp in real_paths:
            gp = gen_dir / rp.name
            if not gp.exists():
                continue
            r = to_tensor(Image.open(rp).convert("RGB")).unsqueeze(0).to(device)
            g = to_tensor(Image.open(gp).convert("RGB")).unsqueeze(0).to(device)
            scores.append(loss_fn(r, g).item())

    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# Dice / IoU  (binary masks, pixel-level)
# ---------------------------------------------------------------------------

def dice_iou(pred: np.ndarray, gt: np.ndarray,
             threshold: float = 0.5) -> tuple[float, float]:
    pred_bin = (pred > threshold).astype(bool).ravel()
    gt_bin   = (gt   > threshold).astype(bool).ravel()

    tp = (pred_bin & gt_bin).sum()
    fp = (pred_bin & ~gt_bin).sum()
    fn = (~pred_bin & gt_bin).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou  = tp / (tp + fp + fn + 1e-8)
    return float(dice), float(iou)


def batch_dice_iou(pred_dir: str | Path,
                   gt_dir: str | Path,
                   size: int = 512) -> dict:
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    exts = {".png", ".jpg", ".jpeg"}
    pred_paths = sorted(p for p in pred_dir.iterdir() if p.suffix.lower() in exts)

    dices, ious = [], []
    for pp in pred_paths:
        gp = gt_dir / pp.name
        if not gp.exists():
            gp = gt_dir / (pp.stem + "_segmentation.png")
        if not gp.exists():
            base = pp.stem.removesuffix("_generated")
            gp = gt_dir / f"{base}.png"
        if not gp.exists():
            gp = gt_dir / f"{base}_segmentation.png"
        if not gp.exists():
            continue

        pred = np.array(Image.open(pp).convert("L").resize((size, size))) / 255.0
        gt   = np.array(Image.open(gp).convert("L").resize((size, size))) / 255.0
        d, i = dice_iou(pred, gt)
        dices.append(d)
        ious.append(i)

    return {
        "dice_mean": float(np.mean(dices)),
        "dice_std":  float(np.std(dices)),
        "iou_mean":  float(np.mean(ious)),
        "iou_std":   float(np.std(ious)),
        "n":         len(dices),
    }


# ---------------------------------------------------------------------------
# Save / print helpers
# ---------------------------------------------------------------------------

def save_results(results: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {path}")


def print_results(results: dict, title: str = "") -> None:
    if title:
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")
        else:
            print(f"  {k:<30} {v}")
