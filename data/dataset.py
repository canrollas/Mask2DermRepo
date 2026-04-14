"""
DermoscopyDataset — PyTorch Dataset for (image, mask, prompt) triplets.

Supports HAM10000 and ISIC-2018 directory layouts.
Applies optics-inspired standardization from preprocessing.py.

Expected directory structure:
    data/processed/
    ├── images/          # *.jpg / *.png dermoscopy images
    ├── masks/           # binary segmentation masks (same stem, *.png)
    └── metadata.csv     # columns: image_id, dx, benign_malignant (optional)

metadata.csv benign_malignant column rules:
    - "benign"    → benign prompt
    - "malignant" → malignant prompt
    - absent / NaN → generic prompt

HAM10000 dx → malignancy mapping (from published literature):
    melanoma (mel)         → malignant
    basal cell carcinoma   → malignant
    squamous cell carcinoma→ malignant
    all others             → benign
"""

from __future__ import annotations

import random

import torch
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.preprocessing import standardize_pil

# ---------------------------------------------------------------------------
# HAM10000 dx → malignancy
# ---------------------------------------------------------------------------

_HAM_MALIGNANT_DX = {"mel", "bcc", "scc", "akiec"}  # actinic keratosis borderline

_BENIGN_PROMPTS = [
    "dermoscopy image of a benign skin lesion, clinical photography, high quality",
    "dermoscopic view of a benign nevus, uniform pigmentation, clinical setting",
    "high quality dermoscopy of a benign melanocytic lesion",
    "dermoscopy of a benign compound nevus, regular pigment network, brown tones",
    "dermoscopic image of a seborrheic keratosis, stuck-on appearance, tan and brown",
    "clinical dermoscopy of a benign junctional nevus, homogeneous pattern",
    "dermoscopy of a dermatofibroma, central white scar-like patch, light brown halo",
    "dermoscopic view of a benign lesion with regular globules, brownish coloration",
    "high resolution dermoscopy of a benign skin lesion, well-defined borders, even tone",
    "dermoscopy image of a benign melanocytic nevus, symmetric dots and globules",
]

_MALIGNANT_PROMPTS = [
    "dermoscopy image of a malignant skin lesion, irregular borders, clinical photography",
    "dermoscopic view of melanoma with asymmetric pigmentation, clinical setting",
    "high quality dermoscopy of a malignant skin lesion with variegated color",
    "dermoscopy of melanoma with blue-white veil, irregular pigmentation, dark brown",
    "clinical dermoscopy of basal cell carcinoma, arborizing vessels, translucent",
    "dermoscopic image of melanoma, atypical network, regression structures",
    "dermoscopy of a malignant lesion with multiple colors, brown black and gray tones",
    "high quality dermoscopy of melanoma with radial streaks and irregular border",
    "dermoscopic view of malignant melanoma, shiny white structures, red milky areas",
    "dermoscopy of actinic keratosis, strawberry pattern, erythematous background",
]

_GENERIC_PROMPTS = [
    "dermoscopy image of a skin lesion, clinical photography, high quality",
    "dermoscopic view of a skin lesion, clinical setting",
    "clinical dermoscopy photograph of a skin lesion, realistic texture",
    "dermoscopy image of a pigmented skin lesion, natural skin tones",
    "high quality dermoscopic image, realistic dermoscopy photography",
]


def _get_prompt(label: str | None, randomize: bool = True) -> str:
    if label == "malignant":
        pool = _MALIGNANT_PROMPTS
    elif label == "benign":
        pool = _BENIGN_PROMPTS
    else:
        pool = _GENERIC_PROMPTS
    return random.choice(pool) if randomize else pool[0]


def _dx_to_label(dx: str) -> str:
    return "malignant" if str(dx).strip().lower() in _HAM_MALIGNANT_DX else "benign"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DermoscopyDataset(Dataset):
    """Dataset yielding (pixel_values, conditioning_pixel_values, prompt) tuples.

    Args:
        root:              Root of the processed data directory.
        split:             "train", "val", or "test".
        resolution:        Target image size (square).
        apply_lens:        Whether to apply optics standardization.
        randomize_prompt:  Use random prompt from the pool vs. first entry.
        augment:           Apply random horizontal/vertical flip and colour jitter.
        val_fraction:      Fraction of data held out for validation (if no split file).
        seed:              Random seed for train/val split.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        resolution: int = 256,
        apply_lens: bool = True,
        randomize_prompt: bool = True,
        augment: bool = True,
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.resolution = resolution
        self.apply_lens = apply_lens
        self.randomize_prompt = randomize_prompt
        self.augment = augment and (split == "train")

        # Build sample list
        self.samples = self._build_samples(val_fraction, seed)

        # Augmentation pipeline (applied after optics / resize)
        # Geometric transforms — applied identically to image AND mask
        self._aug_geom = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
        ]) if self.augment else None
        # Color jitter — applied to image ONLY (mask must stay binary)
        # Stronger jitter forces the model to learn color-invariant features,
        # preventing mode collapse to a single hue (e.g. pink/magenta).
        self._aug_color = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.4, hue=0.08,
        ) if self.augment else None

        # Normalisation to [-1, 1] for diffusion model input
        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),                   # [0, 1]
            transforms.Normalize([0.5], [0.5]),       # [-1, 1]
        ])
        self._mask_to_tensor = transforms.ToTensor()  # mask stays [0, 1]

    # ------------------------------------------------------------------
    def _build_samples(self, val_fraction: float, seed: int) -> list[dict]:
        img_dir = self.root / "images"
        mask_dir = self.root / "masks"
        meta_path = self.root / "metadata.csv"

        # Collect image paths
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)

        if not img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")

        # Load metadata if present
        label_map: dict[str, str] = {}
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            if "benign_malignant" in meta.columns:
                label_map = dict(zip(meta["image_id"].astype(str),
                                     meta["benign_malignant"].astype(str)))
            elif "dx" in meta.columns:
                label_map = {str(row["image_id"]): _dx_to_label(row["dx"])
                             for _, row in meta.iterrows()}

        # Build sample dicts
        samples = []
        for img_path in img_paths:
            stem = img_path.stem
            mask_path = mask_dir / f"{stem}.png"
            if not mask_path.exists():
                # Try alternate naming (HAM10000 uses _segmentation suffix)
                mask_path = mask_dir / f"{stem}_segmentation.png"
            if not mask_path.exists():
                continue  # skip images without masks

            label = label_map.get(stem, None)
            samples.append({"image": img_path, "mask": mask_path, "label": label})

        # Train / val / test split
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(samples)).tolist()
        n_val = max(1, int(len(samples) * val_fraction))
        n_test = n_val

        if self.split == "train":
            keep = indices[n_val + n_test:]
        elif self.split == "val":
            keep = indices[:n_val]
        else:  # test
            keep = indices[n_val:n_val + n_test]

        return [samples[i] for i in keep]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load image
        pil_img = Image.open(sample["image"]).convert("RGB")
        # Load mask (binary → RGB for ControlNet conditioning)
        pil_mask = Image.open(sample["mask"]).convert("L")

        # Optics standardization
        if self.apply_lens:
            pil_img = standardize_pil(pil_img, size=self.resolution)
        else:
            pil_img = pil_img.resize((self.resolution, self.resolution), Image.LANCZOS)

        pil_mask = pil_mask.resize((self.resolution, self.resolution), Image.NEAREST)
        # Binarize: ara gri değerleri temizle
        pil_mask = pil_mask.point(lambda p: 255 if p > 127 else 0)

        # Convert mask to 3-channel (ControlNet expects RGB)
        pil_mask_rgb = pil_mask.convert("RGB")

        # Data augmentation — geometric transforms synced via torch seed
        if self._aug_geom is not None:
            seed = random.randint(0, 2 ** 32 - 1)
            torch.manual_seed(seed)
            pil_img = self._aug_geom(pil_img)
            torch.manual_seed(seed)
            pil_mask_rgb = self._aug_geom(pil_mask_rgb)
            # Color jitter on image only — mask must stay binary
            pil_img = self._aug_color(pil_img)

        prompt = _get_prompt(sample["label"], randomize=self.randomize_prompt)

        return {
            "pixel_values": self._to_tensor(pil_img),
            "conditioning_pixel_values": self._mask_to_tensor(pil_mask_rgb),
            "prompt": prompt,
        }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    ds = DermoscopyDataset(root, split="train", apply_lens=False)
    print(f"Dataset size (train): {len(ds)}")
    if len(ds):
        item = ds[0]
        print(f"  pixel_values:              {item['pixel_values'].shape}")
        print(f"  conditioning_pixel_values: {item['conditioning_pixel_values'].shape}")
        print(f"  prompt: '{item['prompt']}'")
