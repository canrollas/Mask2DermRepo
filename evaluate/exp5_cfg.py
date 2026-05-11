"""
Ablation 5 — CFG Guidance Scale
==================================
Evaluates image quality and mask fidelity at w ∈ {5.0, 9.0, 12.0}.

Usage:
    python evaluate/exp5_cfg.py \
        --controlnet  outputs/mask2derm/controlnet-final \
        --seg_model   outputs/eval_seg/unet.pth \
        --data_dir    data/processed \
        --output_dir  results/exp5_cfg \
        --device      cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate.generate  import generate_test_set
from evaluate.seg_model import predict_masks
from evaluate.metrics   import compute_fid, batch_dice_iou, print_results, save_results

CFG_VALUES = [5.0, 9.0, 12.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controlnet",  required=True)
    p.add_argument("--seg_model",   required=True)
    p.add_argument("--base_model",  default="SG161222/RealVisXL_V4.0")
    p.add_argument("--data_dir",    default="data/processed")
    p.add_argument("--output_dir",  default="results/exp5_cfg")
    p.add_argument("--steps",       type=int,   default=30)
    p.add_argument("--cn_scale",    type=float, default=0.8)
    p.add_argument("--resolution",  type=int,   default=512)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--overwrite",   action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    per_cfg = {}

    for cfg_val in CFG_VALUES:
        tag = f"cfg_{cfg_val:.1f}".replace(".", "_")
        print(f"\n=== CFG w={cfg_val} ===")

        gen_dir, real_dir = generate_test_set(
            controlnet=args.controlnet, base_model=args.base_model,
            data_dir=args.data_dir, out_dir=out_dir / tag,
            steps=args.steps, guidance_scale=cfg_val,
            controlnet_conditioning_scale=args.cn_scale,
            resolution=args.resolution, device=args.device,
            overwrite=args.overwrite,
        )
        gt_mask_dir = out_dir / tag / "masks"

        fid = compute_fid(real_dir, gen_dir, device=args.device)

        pred_dir = out_dir / tag / "pred_masks"
        predict_masks(args.seg_model, gen_dir, pred_dir, device=args.device)
        m = batch_dice_iou(pred_dir, gt_mask_dir)

        per_cfg[str(cfg_val)] = {"fid": fid, **m}
        print(f"  FID={fid:.3f}  Dice={m['dice_mean']:.4f}  IoU={m['iou_mean']:.4f}")

    results = {"per_cfg": per_cfg, "config": vars(args)}
    print_results({f"w={k}  FID": v["fid"] for k, v in per_cfg.items()},
                  title="Ablation 5 — CFG Scale")
    save_results(results, out_dir / "results.json")


if __name__ == "__main__":
    main()
