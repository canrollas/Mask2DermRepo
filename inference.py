"""
Mask2Derm — Inference Script

Generate synthetic dermoscopic images from segmentation masks.

Single image:
    python inference.py \\
        --controlnet outputs/mask2derm/controlnet-final \\
        --base_model SG161222/Realistic_Vision_V5.1_noVAE \\
        --mask assets/sample_mask.png \\
        --prompt "dermoscopy image of a benign skin lesion, clinical photography" \\
        --output generated.png

Batch (directory → directory):
    python inference.py \\
        --controlnet outputs/mask2derm/controlnet-final \\
        --mask_dir data/processed/masks \\
        --output_dir outputs/generated \\
        --batch_size 4 \\
        --save_grid
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionXLControlNetPipeline,
)
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data.preprocessing import apply_barrel_distortion, apply_vignetting, apply_circular_mask
from data.dataset import _BENIGN_PROMPTS, _MALIGNANT_PROMPTS, _GENERIC_PROMPTS

_ALL_PROMPTS = _BENIGN_PROMPTS + _MALIGNANT_PROMPTS + _GENERIC_PROMPTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pipeline(
    controlnet_path: str,
    base_model: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> StableDiffusionXLControlNetPipeline:
    """Load the Mask2Derm SDXL inference pipeline."""
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch_dtype)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
    )
    # DPM-Solver++: DDIM'den ~35% hızlı, 15 adımda yeterli kalite
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    if device == "cuda" and torch.cuda.is_available():
        pipe.to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("xformers aktif")
        except Exception:
            pipe.enable_attention_slicing()
        try:
            import torch._dynamo
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            print("torch.compile aktif (ilk batch yavaş, sonrası hızlı)")
        except Exception:
            pass
    return pipe


def load_mask(mask_path: str | Path, size: int = 256) -> Image.Image:
    """Load and resize a mask image to RGB (ControlNet expects RGB)."""
    mask = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    return mask.convert("RGB")


def apply_optics(img: Image.Image) -> Image.Image:
    """Apply dermoscopy optics: barrel distortion → vignetting → circular aperture."""
    arr = np.array(img.convert("RGB"))
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    arr_bgr = apply_barrel_distortion(arr_bgr, k1=-0.3)
    arr_bgr = apply_vignetting(arr_bgr, alpha=0.5, gamma=2.0)
    arr_bgr = apply_circular_mask(arr_bgr)
    return Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))


def generate_image(
    pipe: StableDiffusionXLControlNetPipeline,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    seed: int | None,
) -> Image.Image:
    generator = torch.manual_seed(seed) if seed is not None else None
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator,
    )
    return apply_optics(result.images[0])


def save_comparison_grid(masks: list[Image.Image], generated: list[Image.Image],
                          out_path: str | Path, cols: int = 4) -> None:
    """Save a side-by-side grid: top row = masks, bottom row = generated."""
    from torchvision.utils import make_grid
    import numpy as np
    import torchvision.transforms.functional as TF

    pairs = []
    for m, g in zip(masks, generated):
        pairs.append(TF.to_tensor(m.convert("RGB")))
        pairs.append(TF.to_tensor(g))

    grid = make_grid(pairs, nrow=cols * 2, padding=2)
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
    Image.fromarray(grid_np).save(out_path)
    print(f"Grid saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask2Derm inference")

    # Model
    parser.add_argument("--controlnet", required=True, help="Path to trained ControlNet")
    parser.add_argument("--base_model", default="SG161222/RealVisXL_V4.0",
                        help="Base SDXL model")

    # Input
    parser.add_argument("--mask", type=str, default=None, help="Single mask image path")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Directory of mask images for batch generation")

    # Prompt
    parser.add_argument("--prompt",
                        default="dermoscopy image of a skin lesion, clinical photography, "
                                "high quality, realistic",
                        help="Generation prompt")
    parser.add_argument("--negative_prompt",
                        default="blurry, low quality, cartoon, painting, sketch, "
                                "unrealistic, artifacts",
                        help="Negative prompt")

    # Generation params
    parser.add_argument("--num_steps", type=int, default=50, help="DDIM inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_scale", type=float, default=1.0,
                        help="ControlNet conditioning scale")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output", type=str, default="generated.png",
                        help="Output path (single image mode)")
    parser.add_argument("--output_dir", type=str, default="outputs/generated",
                        help="Output directory (batch mode)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for generation (set >1 if VRAM allows)")
    parser.add_argument("--save_grid", action="store_true",
                        help="Also save a mask/generated comparison grid")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp32", action="store_true",
                        help="Use FP32 instead of FP16 (slower but more precise)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float32 if args.fp32 else torch.float16

    print(f"Loading pipeline from {args.controlnet}…")
    pipe = load_pipeline(args.controlnet, args.base_model, device=args.device, torch_dtype=dtype)

    # ---- Single image mode ----
    if args.mask:
        mask = load_mask(args.mask, size=args.resolution)
        print(f"Generating from {args.mask}…")
        img = generate_image(pipe, mask, args.prompt, args.negative_prompt,
                             args.num_steps, args.guidance_scale,
                             args.controlnet_scale, args.seed)
        img.save(args.output)
        print(f"Saved → {args.output}")

        if args.save_grid:
            grid_path = Path(args.output).with_suffix(".grid.png")
            save_comparison_grid([mask], [img], grid_path, cols=1)
        return

    # ---- Batch mode ----
    if args.mask_dir:
        mask_dir = Path(args.mask_dir)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        mask_paths = sorted(mask_dir.glob("*.png")) + sorted(mask_dir.glob("*.jpg"))
        print(f"Found {len(mask_paths)} masks in {mask_dir}. Generating…")

        # Filter already-generated
        pending = [(i, mp) for i, mp in enumerate(mask_paths)
                   if not (out_dir / f"{mp.stem}_generated.png").exists()]
        print(f"{len(mask_paths) - len(pending)} already done, {len(pending)} remaining.")

        all_masks, all_generated = [], []

        for batch_start in tqdm(range(0, len(pending), args.batch_size), unit="batch"):
            batch = pending[batch_start: batch_start + args.batch_size]
            idxs, paths = zip(*batch)

            masks   = [load_mask(mp, size=args.resolution) for mp in paths]
            prompts = [random.choice(_ALL_PROMPTS) for _ in masks]
            neg     = [args.negative_prompt] * len(masks)
            generator = torch.Generator(device=args.device).manual_seed(args.seed + batch_start)

            results = pipe(
                prompt=prompts,
                negative_prompt=neg,
                image=masks,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_scale,
                generator=generator,
            ).images

            for mp, img in zip(paths, results):
                img = apply_optics(img)
                img.save(out_dir / f"{mp.stem}_generated.png")
                if args.save_grid:
                    all_masks.append(load_mask(mp, size=args.resolution))
                    all_generated.append(img)

        if args.save_grid and all_generated:
            save_comparison_grid(all_masks[:16], all_generated[:16],
                                 out_dir / "comparison_grid.png", cols=4)
        print(f"Batch generation complete → {out_dir}")
        return

    print("Provide --mask (single) or --mask_dir (batch).")


if __name__ == "__main__":
    main()
