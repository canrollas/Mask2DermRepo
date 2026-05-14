"""
Mask2Derm — Full Evaluation Runner
====================================
Exp 1-3 direkt mevcut görüntüler üzerinde çalışır (generation yok).
Ablasyonlar (5-7) farklı parametrelerle yeniden üretir.

Usage (single real dir, split internally):
    python evaluate/run_all.py \
        --gen_dir     outputs/generated \
        --real_dir    data/processed/images \
        --mask_dir    data/processed/masks \
        --seg_model   outputs/eval_seg/unet.pth \
        --results_dir results \
        --device      cuda

Usage (pre-split dirs):
    python evaluate/run_all.py \
        --gen_dir         outputs/generated \
        --real_dir        data/processed/test_images \
        --mask_dir        data/processed/test_masks \
        --real_train_img  data/processed/train_images \
        --real_train_mask data/processed/train_masks \
        --seg_model       outputs/eval_seg/unet.pth \
        --controlnet      outputs/mask2derm/controlnet-final \
        --results_dir     results \
        --device          cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    ok = subprocess.run(cmd, check=False).returncode == 0
    print(f"  [{'OK' if ok else 'FAIL'}] {label}")
    return ok


def main():
    p = argparse.ArgumentParser()
    # Pre-generated images (Exp 1-3)
    p.add_argument("--gen_dir",          required=True)
    p.add_argument("--real_dir",         required=True)
    p.add_argument("--mask_dir",         required=True)
    # Train/test split: either provide pre-split dirs OR let exp3 split internally
    p.add_argument("--real_train_img",   default=None)
    p.add_argument("--real_train_mask",  default=None)
    p.add_argument("--test_img",         default=None)
    p.add_argument("--test_mask",        default=None)
    p.add_argument("--test_split",       type=float, default=0.2,
                   help="Fraction held out for test when no pre-split dirs given")
    p.add_argument("--seg_model",        required=True)
    # ControlNet (only needed for ablations 4-7)
    p.add_argument("--controlnet",       default=None)
    p.add_argument("--base_model",       default="SG161222/RealVisXL_V4.0")
    p.add_argument("--mask_gen_ckpt",    default=None)
    # Generation params (ablations)
    p.add_argument("--steps",            type=int,   default=30)
    p.add_argument("--cfg",              type=float, default=9.0)
    p.add_argument("--cn_scale",         type=float, default=0.8)
    p.add_argument("--resolution",       type=int,   default=512)
    p.add_argument("--results_dir",      default="results")
    p.add_argument("--device",           default="cuda")
    p.add_argument("--skip_ablations",   action="store_true",
                   help="Skip experiments 4-7 (require --controlnet)")
    args = p.parse_args()

    py = sys.executable
    rd = Path(args.results_dir)

    status = {}

    # Exp 1 — FID / SSIM / LPIPS
    status["exp1"] = _run([
        py, "evaluate/exp1_quality.py",
        "--gen_dir",    args.gen_dir,
        "--real_dir",   args.real_dir,
        "--output_dir", str(rd / "exp1_quality"),
        "--device",     args.device,
    ], "Exp 1 — Image Quality (FID / SSIM / LPIPS)")

    # Exp 2 — Mask Fidelity
    status["exp2"] = _run([
        py, "evaluate/exp2_fidelity.py",
        "--gen_dir",    args.gen_dir,
        "--mask_dir",   args.mask_dir,
        "--seg_model",  args.seg_model,
        "--output_dir", str(rd / "exp2_fidelity"),
        "--device",     args.device,
    ], "Exp 2 — Mask Fidelity (Dice / IoU)")

    # Exp 3 — Downstream Utility
    if args.real_train_img and args.test_img:
        exp3_real_flags = [
            "--real_train_img",  args.real_train_img,
            "--real_train_mask", args.real_train_mask,
            "--test_img",        args.test_img,
            "--test_mask",       args.test_mask,
        ]
    else:
        exp3_real_flags = [
            "--real_img",    args.real_dir,
            "--real_mask",   args.mask_dir,
            "--test_split",  str(args.test_split),
        ]
    status["exp3"] = _run([
        py, "evaluate/exp3_downstream.py",
        *exp3_real_flags,
        "--syn_img",    args.gen_dir,
        "--syn_mask",   args.mask_dir,
        "--output_dir", str(rd / "exp3_downstream"),
        "--device",     args.device,
    ], "Exp 3 — Downstream Segmentation Utility")

    if args.skip_ablations or not args.controlnet:
        print("\n[SKIP] Ablations 4-7 — pass --controlnet to enable")
    else:
        cn_flags = [
            "--controlnet", args.controlnet,
            "--base_model", args.base_model,
            "--data_dir",   str(Path(args.real_dir).parent),
            "--steps",      str(args.steps),
            "--cfg",        str(args.cfg),
            "--cn_scale",   str(args.cn_scale),
            "--resolution", str(args.resolution),
            "--device",     args.device,
        ]

        if args.mask_gen_ckpt:
            status["exp4"] = _run([
                py, "evaluate/exp4_mask_gen.py",
                "--mask_gen_ckpt", args.mask_gen_ckpt,
                "--seg_model",     args.seg_model,
                "--output_dir",    str(rd / "exp4_mask_gen"),
                *cn_flags,
            ], "Ablation 4 — Mask Generator vs Real Mask")

        status["exp5"] = _run([
            py, "evaluate/exp5_cfg.py",
            "--seg_model",  args.seg_model,
            "--output_dir", str(rd / "exp5_cfg"),
            *cn_flags,
        ], "Ablation 5 — CFG Scale")

        status["exp6"] = _run([
            py, "evaluate/exp6_cn_scale.py",
            "--seg_model",  args.seg_model,
            "--output_dir", str(rd / "exp6_cn_scale"),
            *cn_flags,
        ], "Ablation 6 — ControlNet Conditioning Scale")

        status["exp7"] = _run([
            py, "evaluate/exp7_steps.py",
            "--seg_model",  args.seg_model,
            "--output_dir", str(rd / "exp7_steps"),
            *cn_flags,
        ], "Ablation 7 — Denoising Steps")

    # --- Summary ---
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for exp, ok in status.items():
        print(f"  [{'OK  ' if ok else 'FAIL'}] {exp}")

    all_results = {}
    for exp in status:
        rp = rd / exp / "results.json"
        if rp.exists():
            all_results[exp] = json.load(open(rp))

    out = rd / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, open(out, "w"), indent=2)
    print(f"\nFull summary → {out}")


if __name__ == "__main__":
    main()
