"""
Ablation 4 — Mask Generator vs Real Mask
==========================================
Compares two inference configurations:
  A) Real HAM10000 masks → ControlNet → generated images
  B) DDPM mask generator → masks → ControlNet → generated images

Metrics: FID (vs real test set) + Dice/IoU (via pretrained seg model).

Usage:
    python evaluate/exp4_mask_gen.py \
        --controlnet      outputs/mask2derm/controlnet-final \
        --mask_gen_ckpt   outputs/mask_diffusion/best.pt \
        --seg_model       outputs/eval_seg/unet.pth \
        --data_dir        data/processed \
        --output_dir      results/exp4_mask_gen \
        --device          cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate.generate   import generate_test_set, generate_batch_from_masks
from evaluate.seg_model  import predict_masks
from evaluate.metrics    import compute_fid, batch_dice_iou, print_results, save_results


def _generate_masks_from_ddpm(
    ckpt_path: str, n: int, out_dir: Path, device: str, steps: int = 50,
) -> list[Path]:
    """Use mask_diffusion.py generate logic to produce n binary masks."""
    import torch
    from mask_diffusion import UNet, GaussianDiffusion, ddim_sample, postprocess_mask

    out_dir.mkdir(parents=True, exist_ok=True)
    raw = torch.load(ckpt_path, map_location=device)
    cfg = raw.get("config", {})
    base_ch    = cfg.get("base_ch", 64)
    resolution = cfg.get("resolution", 128)
    timesteps  = cfg.get("timesteps", 1000)

    model = UNet(base_ch=base_ch).to(device)
    model.load_state_dict(raw["model"])
    model.eval()

    diff = GaussianDiffusion(T=timesteps)

    paths = []
    for i in range(n):
        out_path = out_dir / f"gen_mask_{i:04d}.png"
        if out_path.exists():
            paths.append(out_path)
            continue
        mask = ddim_sample(model, diff, resolution=resolution,
                           steps=steps, device=device)
        mask = postprocess_mask(mask)
        mask.save(out_path)
        paths.append(out_path)

    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controlnet",    required=True)
    p.add_argument("--mask_gen_ckpt", required=True,
                   help="Path to mask_diffusion best.pt checkpoint")
    p.add_argument("--seg_model",     required=True)
    p.add_argument("--base_model",    default="SG161222/RealVisXL_V4.0")
    p.add_argument("--data_dir",      default="data/processed")
    p.add_argument("--output_dir",    default="results/exp4_mask_gen")
    p.add_argument("--steps",         type=int,   default=30)
    p.add_argument("--cfg",           type=float, default=9.0)
    p.add_argument("--cn_scale",      type=float, default=0.8)
    p.add_argument("--mask_gen_steps",type=int,   default=50)
    p.add_argument("--resolution",    type=int,   default=512)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--overwrite",     action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    from evaluate.generate import get_test_pairs
    test_pairs = get_test_pairs(args.data_dir)
    n = len(test_pairs)

    # --- Config A: real masks ---
    print(f"\n=== Config A: real masks (n={n}) ===")
    gen_a, real_dir = generate_test_set(
        controlnet=args.controlnet, base_model=args.base_model,
        data_dir=args.data_dir, out_dir=out_dir / "config_A",
        steps=args.steps, guidance_scale=args.cfg,
        controlnet_conditioning_scale=args.cn_scale,
        resolution=args.resolution, device=args.device,
        overwrite=args.overwrite,
    )
    gt_mask_dir = out_dir / "config_A" / "masks"

    # --- Config B: generated masks ---
    print(f"\n=== Config B: DDPM-generated masks (n={n}) ===")
    ddpm_mask_dir = out_dir / "config_B" / "ddpm_masks"
    mask_paths = _generate_masks_from_ddpm(
        args.mask_gen_ckpt, n, ddpm_mask_dir, args.device, args.mask_gen_steps)

    gen_b = generate_batch_from_masks(
        controlnet=args.controlnet, base_model=args.base_model,
        mask_paths=mask_paths, out_dir=out_dir / "config_B" / "generated",
        steps=args.steps, guidance_scale=args.cfg,
        controlnet_conditioning_scale=args.cn_scale,
        seed=0, device=args.device, resolution=args.resolution,
        overwrite=args.overwrite,
    )

    # --- Metrics ---
    print("\nComputing FID...")
    fid_a = compute_fid(real_dir, gen_a, device=args.device)
    fid_b = compute_fid(real_dir, gen_b, device=args.device)

    print("Predicting masks for config A...")
    pred_a = out_dir / "pred_masks_A"
    predict_masks(args.seg_model, gen_a, pred_a, device=args.device)
    m_a = batch_dice_iou(pred_a, gt_mask_dir)

    print("Predicting masks for config B (using GT masks as reference)...")
    pred_b = out_dir / "pred_masks_B"
    predict_masks(args.seg_model, gen_b, pred_b, device=args.device)
    m_b = batch_dice_iou(pred_b, ddpm_mask_dir)

    results = {
        "config_A_real_masks":      {"fid": fid_a, **m_a},
        "config_B_generated_masks": {"fid": fid_b, **m_b},
        "delta_fid":                fid_b - fid_a,
        "delta_dice":               m_b["dice_mean"] - m_a["dice_mean"],
        "config":                   vars(args),
    }

    print_results({
        "FID (real masks)":  fid_a,
        "FID (gen masks)":   fid_b,
        "Dice (real masks)": m_a["dice_mean"],
        "Dice (gen masks)":  m_b["dice_mean"],
    }, title="Ablation 4 — Mask Generator vs Real Mask")
    save_results(results, out_dir / "results.json")


if __name__ == "__main__":
    main()
