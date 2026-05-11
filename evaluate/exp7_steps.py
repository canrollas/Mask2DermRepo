"""
Ablation 7 — Denoising Steps
==============================
Evaluates quality-efficiency trade-off at S ∈ {20, 30, 50}.
Records FID, Dice, and wall-clock inference time per image.

Usage:
    python evaluate/exp7_steps.py \
        --controlnet  outputs/mask2derm/controlnet-final \
        --seg_model   outputs/eval_seg/unet.pth \
        --data_dir    data/processed \
        --output_dir  results/exp7_steps \
        --device      cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate.generate  import get_test_pairs
from evaluate.seg_model import predict_masks
from evaluate.metrics   import compute_fid, batch_dice_iou, print_results, save_results

STEP_VALUES = [20, 30, 50]


def _generate_timed(controlnet, base_model, test_pairs, out_dir, *,
                    steps, cfg, cn_scale, device, resolution, overwrite):
    from inference import load_pipeline, load_mask, generate_image
    from evaluate.generate import _EVAL_PROMPT, _NEGATIVE_PROMPT
    from PIL import Image
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    real_dir = out_dir.parent / "real"
    real_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(controlnet, base_model, device=device)

    times = []
    for sample in tqdm(test_pairs, desc=f"S={steps}"):
        stem     = sample["image"].stem
        out_path = out_dir / f"{stem}.png"

        if not out_path.exists() or overwrite:
            mask_img = load_mask(sample["mask"], size=resolution)
            t0 = time.perf_counter()
            img = generate_image(
                pipe, mask_img, _EVAL_PROMPT, _NEGATIVE_PROMPT,
                num_inference_steps=steps,
                guidance_scale=cfg,
                controlnet_conditioning_scale=cn_scale,
                seed=0,
            )
            times.append(time.perf_counter() - t0)
            img.save(out_path)

        real_copy = real_dir / f"{stem}.png"
        if not real_copy.exists():
            Image.open(sample["image"]).convert("RGB").resize(
                (resolution, resolution)).save(real_copy)

    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controlnet",  required=True)
    p.add_argument("--seg_model",   required=True)
    p.add_argument("--base_model",  default="SG161222/RealVisXL_V4.0")
    p.add_argument("--data_dir",    default="data/processed")
    p.add_argument("--output_dir",  default="results/exp7_steps")
    p.add_argument("--cfg",         type=float, default=9.0)
    p.add_argument("--cn_scale",    type=float, default=0.8)
    p.add_argument("--resolution",  type=int,   default=512)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--overwrite",   action="store_true")
    args = p.parse_args()

    out_dir    = Path(args.output_dir)
    test_pairs = get_test_pairs(args.data_dir)
    real_dir   = out_dir / "real"
    per_steps  = {}

    for S in STEP_VALUES:
        tag     = f"steps_{S}"
        gen_dir = out_dir / tag / "generated"
        gt_mask_dir_shared = out_dir / "gt_masks"

        print(f"\n=== S={S} denoising steps ===")

        times = _generate_timed(
            args.controlnet, args.base_model, test_pairs, gen_dir,
            steps=S, cfg=args.cfg, cn_scale=args.cn_scale,
            device=args.device, resolution=args.resolution,
            overwrite=args.overwrite,
        )

        # Copy GT masks once
        if not gt_mask_dir_shared.exists():
            from PIL import Image
            gt_mask_dir_shared.mkdir(parents=True, exist_ok=True)
            for sample in test_pairs:
                dst = gt_mask_dir_shared / (sample["image"].stem + ".png")
                if not dst.exists():
                    Image.open(sample["mask"]).convert("L").resize(
                        (args.resolution, args.resolution),
                        Image.NEAREST).save(dst)

        fid = compute_fid(real_dir, gen_dir, device=args.device)

        pred_dir = out_dir / tag / "pred_masks"
        predict_masks(args.seg_model, gen_dir, pred_dir, device=args.device)
        m = batch_dice_iou(pred_dir, gt_mask_dir_shared)

        import numpy as np
        time_mean = float(np.mean(times)) if times else float("nan")
        per_steps[S] = {
            "fid": fid, **m,
            "time_per_image_sec": time_mean,
        }
        print(f"  FID={fid:.3f}  Dice={m['dice_mean']:.4f}  "
              f"time/img={time_mean:.2f}s")

    results = {"per_steps": {str(k): v for k, v in per_steps.items()},
               "config": vars(args)}
    print_results({f"S={k}  FID": v["fid"] for k, v in per_steps.items()},
                  title="Ablation 7 — Denoising Steps")
    save_results(results, out_dir / "results.json")


if __name__ == "__main__":
    main()
