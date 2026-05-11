"""
Experiment 3 — Downstream Segmentation Utility
================================================
Trains two identical U-Nets:
  A) real training data only
  B) real + Mask2Derm synthetic data (already generated)

Both evaluated on the real held-out test set.

Usage:
    python evaluate/exp3_downstream.py \
        --real_train_img  data/processed/train_images \
        --real_train_mask data/processed/train_masks \
        --syn_img         outputs/generated \
        --syn_mask        data/processed/masks \
        --test_img        data/processed/test_images \
        --test_mask       data/processed/test_masks \
        --output_dir      results/exp3_downstream \
        --epochs          30 \
        --device          cuda
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate.seg_model import (train_seg_model, load_seg_model,
                                 _get_pairs, _SegDataset, _eval_dice)
from evaluate.metrics   import print_results, save_results

import torch
from torch.utils.data import DataLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_train_img",  required=True)
    p.add_argument("--real_train_mask", required=True)
    p.add_argument("--syn_img",         required=True,
                   help="Already-generated synthetic images")
    p.add_argument("--syn_mask",        required=True,
                   help="Masks used to generate synthetic images")
    p.add_argument("--test_img",        required=True)
    p.add_argument("--test_mask",       required=True)
    p.add_argument("--output_dir",      default="results/exp3_downstream")
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--resolution",      type=int,   default=256)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--device",          default="cuda")
    args = p.parse_args()

    out_dir = Path(args.output_dir)

    # --- Model A: real only ---
    print("\n--- Training Model A: real data only ---")
    model_a_path = out_dir / "model_A_real_only.pth"
    train_seg_model(
        image_dir=args.real_train_img, mask_dir=args.real_train_mask,
        save_path=model_a_path,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, size=args.resolution, device=args.device,
    )

    # --- Model B: real + synthetic ---
    print("\n--- Training Model B: real + synthetic data ---")
    model_b_path = out_dir / "model_B_augmented.pth"

    combined_img  = out_dir / "combined" / "images"
    combined_mask = out_dir / "combined" / "masks"
    combined_img.mkdir(parents=True, exist_ok=True)
    combined_mask.mkdir(parents=True, exist_ok=True)

    for src_img, src_mask in [
        (Path(args.real_train_img), Path(args.real_train_mask)),
        (Path(args.syn_img),        Path(args.syn_mask)),
    ]:
        for f in src_img.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                dst = combined_img / f.name
                if not dst.exists():
                    shutil.copy(f, dst)
        for f in src_mask.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                dst = combined_mask / f.name
                if not dst.exists():
                    shutil.copy(f, dst)

    train_seg_model(
        image_dir=combined_img, mask_dir=combined_mask,
        save_path=model_b_path,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, size=args.resolution, device=args.device,
    )

    # --- Evaluate both on real test set ---
    print("\nEvaluating on real test set...")
    test_pairs = _get_pairs(Path(args.test_img), Path(args.test_mask))
    test_ds = _SegDataset(
        [p[0] for p in test_pairs], [p[1] for p in test_pairs],
        size=args.resolution, augment=False,
    )
    test_dl = DataLoader(test_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=4)

    dice_a = _eval_dice(load_seg_model(model_a_path, args.device), test_dl, args.device)
    dice_b = _eval_dice(load_seg_model(model_b_path, args.device), test_dl, args.device)

    n_real = len(_get_pairs(Path(args.real_train_img), Path(args.real_train_mask)))
    n_syn  = len(list(Path(args.syn_img).glob("*.png")) +
                  list(Path(args.syn_img).glob("*.jpg")))

    results = {
        "model_A_real_only":       {"dice": dice_a},
        "model_B_real_plus_synth": {"dice": dice_b},
        "delta_dice":              dice_b - dice_a,
        "n_real_train":            n_real,
        "n_synthetic_train":       n_syn,
        "n_test":                  len(test_pairs),
        "config":                  vars(args),
    }

    print_results({
        "Dice (real only)":  dice_a,
        "Dice (real+synth)": dice_b,
        "Δ Dice":            dice_b - dice_a,
    }, title="Exp 3 — Downstream Segmentation Utility")
    save_results(results, out_dir / "results.json")


if __name__ == "__main__":
    main()
