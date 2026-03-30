"""
Train-on-Synthetic, Test-on-Real (TSTR) Evaluation (§3.3)

Protocol:
  M_synth : DeepLabV3+ fine-tuned ONLY on synthetic data
  M_base  : DeepLabV3+ with ImageNet weights, NO dermoscopy training
  Both are evaluated on the held-out real HAM10000 test set.
  Delta = IoU(M_synth) - IoU(M_base)

Usage:
    # Step 1: Fine-tune on synthetic data
    python evaluate/tstr.py train \\
        --synthetic_images outputs/generated \\
        --synthetic_masks  data/processed/masks \\
        --output_dir       outputs/tstr/checkpoints \\
        --epochs 50

    # Step 2: Evaluate both models on real test data
    python evaluate/tstr.py eval \\
        --real_images    data/processed/images \\
        --real_masks     data/processed/masks \\
        --checkpoint     outputs/tstr/checkpoints/best.pth \\
        --output_csv     results/tstr_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Segmentation Dataset
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    def __init__(self, img_dir: str | Path, mask_dir: str | Path,
                 size: int = 256) -> None:
        img_dir = Path(img_dir)
        mask_dir = Path(mask_dir)
        self.size = size

        exts = {".jpg", ".jpeg", ".png"}
        img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)

        self.pairs = []
        for ip in img_paths:
            stem = ip.stem.replace("_generated", "")
            mp = mask_dir / f"{stem}.png"
            if not mp.exists():
                mp = mask_dir / f"{stem}_segmentation.png"
            if mp.exists():
                self.pairs.append((ip, mp))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        ip, mp = self.pairs[idx]
        img = Image.open(ip).convert("RGB").resize((self.size, self.size), Image.LANCZOS)
        mask = Image.open(mp).convert("L").resize((self.size, self.size), Image.NEAREST)

        img_t = TF.to_tensor(img)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        mask_t = torch.tensor(np.array(mask) > 127, dtype=torch.long)  # (H, W)
        return {"image": img_t, "mask": mask_t}


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_deeplabv3(pretrained: bool = True, num_classes: int = 2,
                   device: str = "cuda") -> nn.Module:
    from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
    if pretrained:
        model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    else:
        model = deeplabv3_resnet101(weights=None)

    # Replace classifier head for binary segmentation
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    nn.init.kaiming_normal_(model.classifier[4].weight)
    model.to(device)
    return model


def _compute_iou_batch(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds_b = preds == 1
    targets_b = targets == 1
    intersection = (preds_b & targets_b).float().sum()
    union = (preds_b | targets_b).float().sum()
    return (intersection / (union + 1e-8)).item()


def _compute_dice_batch(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds_b = preds == 1
    targets_b = targets == 1
    inter = (preds_b & targets_b).float().sum()
    return (2.0 * inter / (preds_b.float().sum() + targets_b.float().sum() + 1e-8)).item()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_segmentation_model(
    synthetic_images: str,
    synthetic_masks: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cuda",
) -> str:
    """Fine-tune DeepLabV3+ on synthetic data. Returns path to best checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SegmentationDataset(synthetic_images, synthetic_masks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    model = load_deeplabv3(pretrained=True, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_path = str(output_dir / "best.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            out = model(imgs)["out"]  # (B, 2, H, W)
            loss = F.cross_entropy(out, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:3d}/{epochs} | loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_path)

    print(f"Best checkpoint → {best_path}")
    return best_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    real_images: str,
    real_masks: str,
    device: str = "cuda",
) -> dict:
    dataset = SegmentationDataset(real_images, real_masks)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    model.eval()
    ious, dices = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            out = model(imgs)["out"]
            preds = out.argmax(dim=1)  # (B, H, W)

            for pred, gt in zip(preds, masks):
                ious.append(_compute_iou_batch(pred, gt))
                dices.append(_compute_dice_batch(pred, gt))

    return {
        "mean_iou": float(np.mean(ious)),
        "std_iou": float(np.std(ious)),
        "mean_dice": float(np.mean(dices)),
        "std_dice": float(np.std(dices)),
        "n": len(ious),
    }


def run_tstr(
    real_images: str,
    real_masks: str,
    checkpoint: str,
    output_csv: str | None = None,
    device: str = "cuda",
) -> dict:
    """Compare M_synth vs M_base on the real test set."""
    print("\n=== TSTR Evaluation ===")

    # M_synth: fine-tuned on synthetic data
    print("Loading M_synth…")
    m_synth = load_deeplabv3(pretrained=False, device=device)
    state = torch.load(checkpoint, map_location=device)
    m_synth.load_state_dict(state)

    print("Evaluating M_synth on real data…")
    synth_results = evaluate_model(m_synth, real_images, real_masks, device)

    # M_base: ImageNet init, no dermoscopy training
    print("\nLoading M_base (ImageNet only)…")
    m_base = load_deeplabv3(pretrained=True, device=device)
    # Replace head for 2-class prediction but leave backbone frozen
    in_ch = m_base.classifier[4].in_channels
    m_base.classifier[4] = nn.Conv2d(in_ch, 2, kernel_size=1).to(device)
    nn.init.kaiming_normal_(m_base.classifier[4].weight)

    print("Evaluating M_base on real data…")
    base_results = evaluate_model(m_base, real_images, real_masks, device)

    delta_iou = synth_results["mean_iou"] - base_results["mean_iou"]
    delta_dice = synth_results["mean_dice"] - base_results["mean_dice"]

    print(f"\n{'Model':<15} {'mIoU':>10} {'mDice':>10}")
    print(f"{'M_base':<15} {base_results['mean_iou']:>10.4f} {base_results['mean_dice']:>10.4f}")
    print(f"{'M_synth':<15} {synth_results['mean_iou']:>10.4f} {synth_results['mean_dice']:>10.4f}")
    print(f"{'Δ (↑ better)':<15} {delta_iou:>+10.4f} {delta_dice:>+10.4f}")

    summary = {
        "m_base_iou": base_results["mean_iou"],
        "m_base_dice": base_results["mean_dice"],
        "m_synth_iou": synth_results["mean_iou"],
        "m_synth_dice": synth_results["mean_dice"],
        "delta_iou": delta_iou,
        "delta_dice": delta_dice,
    }

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
        print(f"Results saved → {output_csv}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TSTR evaluation")
    sub = parser.add_subparsers(dest="command")

    # train sub-command
    train_p = sub.add_parser("train", help="Fine-tune segmentation model on synthetic data")
    train_p.add_argument("--synthetic_images", required=True)
    train_p.add_argument("--synthetic_masks", required=True)
    train_p.add_argument("--output_dir", default="outputs/tstr/checkpoints")
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--batch_size", type=int, default=8)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--device", default="cuda")

    # eval sub-command
    eval_p = sub.add_parser("eval", help="Compare M_synth vs M_base on real test set")
    eval_p.add_argument("--real_images", required=True)
    eval_p.add_argument("--real_masks", required=True)
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--output_csv", default=None)
    eval_p.add_argument("--device", default="cuda")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "train":
        train_segmentation_model(
            args.synthetic_images, args.synthetic_masks,
            args.output_dir, args.epochs, args.batch_size, args.lr, args.device,
        )
    elif args.command == "eval":
        run_tstr(args.real_images, args.real_masks, args.checkpoint,
                 args.output_csv, args.device)
    else:
        print("Specify subcommand: train | eval")
