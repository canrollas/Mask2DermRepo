"""
Shape Consistency Evaluation (§3.2 — Geometric Fidelity)

Measures how well the ControlNet preserved the input lesion geometry.

Protocol:
  1. For each generated image, run a pre-trained DeepLabV3+ segmentation model
     to predict the lesion mask.
  2. Compare the predicted mask against the ground-truth input mask.
  3. Report mean IoU (Jaccard) and mean Dice (DSC) ± std.

The segmentation model used is DeepLabV3+ with ResNet-101 backbone pre-trained
on PASCAL VOC (torchvision). We use the foreground class (index 15 = person is
repurposed — here we simply threshold the max-class prediction).

Usage:
    python evaluate/shape_consistency.py \\
        --generated_dir outputs/generated \\
        --mask_dir      data/processed/masks \\
        --output_csv    results/shape_consistency.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Segmentation model loader
# ---------------------------------------------------------------------------

def load_segmentation_model(device: str = "cuda") -> torch.nn.Module:
    """Load DeepLabV3+ ResNet-101 pretrained on PASCAL VOC."""
    from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)
    model.eval()
    model.to(device)
    return model


def segment_image(model: torch.nn.Module, pil_img: Image.Image,
                  device: str = "cuda", threshold: float = 0.5) -> np.ndarray:
    """Predict a binary foreground mask for a PIL image.

    Returns:
        Binary mask (H×W) as uint8 numpy array (0 or 255).
    """
    img_t = TF.to_tensor(pil_img.convert("RGB")).unsqueeze(0).to(device)
    # Normalize to ImageNet mean/std (DeepLabV3 expects this)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img_t = (img_t - mean) / std

    with torch.no_grad():
        out = model(img_t)["out"]   # (1, num_classes, H, W)

    # Softmax and take the non-background max class
    probs = torch.softmax(out[0], dim=0)   # (C, H, W)
    # Background = class 0; foreground probability = 1 - p_bg
    fg_prob = 1.0 - probs[0].cpu().numpy()
    binary = (fg_prob >= threshold).astype(np.uint8) * 255
    return binary


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 127
    gt_b = gt > 127
    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return float(intersection) / float(union + 1e-8)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 127
    gt_b = gt > 127
    intersection = np.logical_and(pred_b, gt_b).sum()
    return 2.0 * float(intersection) / (float(pred_b.sum() + gt_b.sum()) + 1e-8)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    generated_dir: str | Path,
    mask_dir: str | Path,
    output_csv: str | Path | None = None,
    device: str = "cuda",
    threshold: float = 0.5,
) -> dict:
    """Run shape consistency evaluation.

    Args:
        generated_dir: Directory of generated images (*.png / *.jpg).
        mask_dir:      Directory of ground-truth masks (matched by stem).
        output_csv:    Optional path to save per-image results.
        device:        CUDA or CPU.
        threshold:     Foreground probability threshold for segmentation.

    Returns:
        Dict with keys "mean_iou", "std_iou", "mean_dice", "std_dice", "n".
    """
    generated_dir = Path(generated_dir)
    mask_dir = Path(mask_dir)

    gen_paths = sorted(generated_dir.glob("*.png")) + sorted(generated_dir.glob("*.jpg"))
    if not gen_paths:
        raise FileNotFoundError(f"No generated images found in {generated_dir}")

    model = load_segmentation_model(device)

    ious, dices = [], []
    rows = []

    for gen_path in tqdm(gen_paths, desc="Shape consistency", unit="img"):
        # Match ground-truth mask — strip "_generated" suffix if present
        stem = gen_path.stem.replace("_generated", "")
        gt_path = mask_dir / f"{stem}.png"
        if not gt_path.exists():
            gt_path = mask_dir / f"{stem}_segmentation.png"
        if not gt_path.exists():
            continue

        gen_img = Image.open(gen_path).convert("RGB")
        gt_mask = np.array(Image.open(gt_path).convert("L"))

        # Resize GT mask to match generated image size
        h, w = gen_img.size[1], gen_img.size[0]
        gt_mask = np.array(Image.fromarray(gt_mask).resize((w, h), Image.NEAREST))

        pred_mask = segment_image(model, gen_img, device=device, threshold=threshold)

        iou = compute_iou(pred_mask, gt_mask)
        dice = compute_dice(pred_mask, gt_mask)
        ious.append(iou)
        dices.append(dice)
        rows.append({"image": gen_path.name, "iou": iou, "dice": dice})

    if not ious:
        raise RuntimeError("No valid image-mask pairs found.")

    results = {
        "mean_iou": float(np.mean(ious)),
        "std_iou": float(np.std(ious)),
        "mean_dice": float(np.mean(dices)),
        "std_dice": float(np.std(dices)),
        "n": len(ious),
    }

    print(f"\n=== Shape Consistency Results (n={results['n']}) ===")
    print(f"  mIoU  = {results['mean_iou']:.4f} ± {results['std_iou']:.4f}")
    print(f"  mDice = {results['mean_dice']:.4f} ± {results['std_dice']:.4f}")

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "iou", "dice"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Per-image results → {output_csv}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shape consistency evaluation")
    parser.add_argument("--generated_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.generated_dir, args.mask_dir, args.output_csv,
             args.device, args.threshold)
