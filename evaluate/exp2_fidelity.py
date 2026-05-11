"""
Experiment 2 — Mask Fidelity
==============================
Passes already-generated images through a pretrained seg U-Net,
compares predicted masks to original conditioning masks via Dice / IoU.

Prerequisite: train the evaluation U-Net first:
    python evaluate/seg_model.py train \
        --image_dir data/processed/images \
        --mask_dir  data/processed/masks \
        --save_path outputs/eval_seg/unet.pth

Usage:
    python evaluate/exp2_fidelity.py \
        --gen_dir   outputs/generated \
        --mask_dir  data/processed/masks \
        --seg_model outputs/eval_seg/unet.pth \
        --output_dir results/exp2_fidelity \
        --device    cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate.seg_model import predict_masks
from evaluate.metrics   import batch_dice_iou, print_results, save_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_dir",    required=True, help="Directory of generated images")
    p.add_argument("--mask_dir",   required=True, help="Ground-truth binary mask directory")
    p.add_argument("--seg_model",  required=True, help="Trained evaluation U-Net (.pth)")
    p.add_argument("--output_dir", default="results/exp2_fidelity")
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    out_dir  = Path(args.output_dir)
    pred_dir = out_dir / "predicted_masks"

    print("Predicting masks on generated images...")
    predict_masks(args.seg_model, args.gen_dir, pred_dir, device=args.device)

    print("Computing Dice / IoU...")
    m = batch_dice_iou(pred_dir, args.mask_dir)

    results = {**m, "config": vars(args)}
    print_results(results, title="Exp 2 — Mask Fidelity")
    save_results(results, out_dir / "results.json")


if __name__ == "__main__":
    main()
