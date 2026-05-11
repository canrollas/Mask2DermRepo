"""
Generation utilities for evaluation experiments.

Wraps inference.py to generate synthetic images for the test split
with configurable parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

_EVAL_PROMPT = (
    "dermoscopy image of a skin lesion, photorealistic, clinical photography, high quality"
)
_NEGATIVE_PROMPT = (
    "blurry, low quality, cartoon, painting, sketch, unrealistic, artifacts, "
    "oversaturated, neon colors, illustration, digital art"
)


def get_test_pairs(data_dir: str | Path, seed: int = 42,
                   val_fraction: float = 0.1) -> list[dict]:
    """Return test split sample dicts (image, mask, label) matching DermoscopyDataset logic."""
    from data.dataset import _dx_to_label
    import pandas as pd

    root = Path(data_dir)
    img_dir   = root / "images"
    mask_dir  = root / "masks"
    meta_path = root / "metadata.csv"

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)

    label_map: dict[str, str] = {}
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        if "benign_malignant" in meta.columns:
            label_map = dict(zip(meta["image_id"].astype(str),
                                  meta["benign_malignant"].astype(str)))
        elif "dx" in meta.columns:
            label_map = {str(r["image_id"]): _dx_to_label(r["dx"])
                         for _, r in meta.iterrows()}

    samples = []
    for ip in img_paths:
        mp = mask_dir / f"{ip.stem}.png"
        if not mp.exists():
            mp = mask_dir / f"{ip.stem}_segmentation.png"
        if not mp.exists():
            continue
        samples.append({"image": ip, "mask": mp,
                         "label": label_map.get(ip.stem, None)})

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples)).tolist()
    n_val  = max(1, int(len(samples) * val_fraction))
    n_test = n_val
    test_idx = idx[n_val: n_val + n_test]
    return [samples[i] for i in test_idx]


def generate_test_set(
    controlnet:  str,
    base_model:  str,
    data_dir:    str | Path,
    out_dir:     str | Path,
    *,
    prompt:      str   = _EVAL_PROMPT,
    steps:       int   = 30,
    guidance_scale:            float = 9.0,
    controlnet_conditioning_scale: float = 0.8,
    seed:        int   = 0,
    device:      str   = "cuda",
    resolution:  int   = 512,
    overwrite:   bool  = False,
) -> tuple[Path, Path]:
    """Generate one synthetic image per test-split mask.

    Returns (out_img_dir, out_mask_copy_dir) — both dirs contain
    identically-named files so paired metrics can be computed.
    """
    from inference import load_pipeline, load_mask, generate_image

    out_dir = Path(out_dir)
    gen_dir  = out_dir / "generated"
    real_dir = out_dir / "real"
    mask_dir = out_dir / "masks"
    for d in (gen_dir, real_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    pairs = get_test_pairs(data_dir)
    print(f"Test split: {len(pairs)} samples")

    pipe = load_pipeline(controlnet, base_model, device=device)

    for sample in tqdm(pairs, desc="Generating"):
        stem = sample["image"].stem
        gen_path = gen_dir / f"{stem}.png"
        if gen_path.exists() and not overwrite:
            continue

        mask_img = load_mask(sample["mask"], size=resolution)
        img = generate_image(
            pipe, mask_img, prompt, _NEGATIVE_PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            seed=seed,
        )
        img.save(gen_path)

        # Copy real image and mask with same stem for paired metrics
        real_copy = real_dir / f"{stem}.png"
        if not real_copy.exists():
            Image.open(sample["image"]).convert("RGB").resize(
                (resolution, resolution)).save(real_copy)

        mask_copy = mask_dir / f"{stem}.png"
        if not mask_copy.exists():
            Image.open(sample["mask"]).convert("L").resize(
                (resolution, resolution), Image.NEAREST).save(mask_copy)

    return gen_dir, real_dir


def generate_batch_from_masks(
    controlnet:  str,
    base_model:  str,
    mask_paths:  list[Path],
    out_dir:     Path,
    *,
    prompt:      str   = _EVAL_PROMPT,
    steps:       int   = 30,
    guidance_scale:            float = 9.0,
    controlnet_conditioning_scale: float = 0.8,
    seed:        int   = 0,
    device:      str   = "cuda",
    resolution:  int   = 512,
    overwrite:   bool  = False,
) -> Path:
    from inference import load_pipeline, load_mask, generate_image

    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(controlnet, base_model, device=device)

    for mp in tqdm(mask_paths, desc="Generating"):
        out_path = out_dir / f"{mp.stem}.png"
        if out_path.exists() and not overwrite:
            continue
        mask_img = load_mask(mp, size=resolution)
        img = generate_image(
            pipe, mask_img, prompt, _NEGATIVE_PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            seed=seed,
        )
        img.save(out_path)

    return out_dir
