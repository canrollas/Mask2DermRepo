<p align="center" style="background-color: #0f1117; padding: 20px; margin: 0;">
  <img src="assets/mask2derm_logo.svg" alt="Mask2Derm" width="5000"/>
</p>

<p align="center">
  <em>Photorealistic dermoscopic image synthesis from binary lesion masks via ControlNet + SDXL</em>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
</p>

<p align="center">
  Can Rollas · Mehmet Kemal Güllü · İbrahim Onur Alıcı<br/>
  <sub>Amatis IT R&D — IZTECH · İzmir Bakırçay University</sub>
</p>

---

<p align="center">
  <img src="assets/demo.gif" alt="Mask2Derm Demo — masks blinking to generated lesions" width="900"/>
</p>

---

## Overview

Mask2Derm is a ControlNet-based latent diffusion framework that synthesizes photorealistic dermoscopic images conditioned solely on a binary lesion segmentation mask. It fine-tunes a ControlNet module on top of a frozen **SDXL (RealVisXL V4.0)** backbone while keeping all other components frozen.

**Key results (HAM10000):**
| Metric | Score |
|---|---|
| Shape consistency — mIoU | 0.866 ± 0.119 |
| Shape consistency — mDice | 0.922 ± 0.096 |
| Distributional alignment — FID (extrapolated) | 62.33 |
| Distributional alignment — KID | 0.07 ± 0.01 |
| TSTR ΔIoU vs ImageNet baseline | **+0.134** |

![Architecture](paper/architecture_preview.png)

---

## Architecture

```
Binary Mask ──► ControlNet (trainable, init from UNet) ──► residuals ──►
                                                                          U-Net (frozen, SDXL) ──► latent ──► VAE Decoder ──► Image
Text Prompt ──► CLIP ×2 (frozen) ──► [B, 77, 2048] + pooled [B, 1280] ──►
```

- **VAE** compresses 512×512 images to 64×64 latent space (factor-8).
- **U-Net** iteratively denoises in latent space using dual CLIP conditioning.
- **ControlNet** is initialized from the frozen U-Net encoder weights (no pretrained segmentation bias) and injects mask-derived spatial residuals into the U-Net decoder.
- **Dual CLIP**: SDXL requires both text encoders — hidden states concatenated to `[B, 77, 2048]`, pooled embeds from encoder 2 only.

---

## Setup

```bash
git clone https://github.com/canrollas/Mask2DermRepo.git
cd Mask2DermRepo
pip install -r requirements.txt
accelerate config  # first time only
```

**HuggingFace access** — accept the license on the Hub before first run:
- [SG161222/RealVisXL_V4.0](https://huggingface.co/SG161222/RealVisXL_V4.0)

---

## Data Preparation

```bash
# ISIC 2018 Task 1 (~2600 segmentation pairs)
python data/download.py --download-isic

# HAM10000 (requires ~/.kaggle/kaggle.json)
python data/download.py --download-ham

# Merge, resize to 512×512, write metadata.csv
python data/download.py --prepare --size 512
```

Every image is standardized through a physics-based optics simulation (circular FOV, vignetting, barrel distortion) to match real dermatoscope characteristics:

```python
from data.preprocessing import standardize_pil
from PIL import Image

pil_out = standardize_pil(Image.open("input.jpg"), size=512)
```

---

## Training

```bash
accelerate launch train.py --config configs/train_config.yaml

# Resume from checkpoint
accelerate launch train.py --config configs/train_config.yaml \
    --resume-from-checkpoint outputs/checkpoints/checkpoint-N
```

Key hyperparameters (`configs/train_config.yaml`):

| Parameter | Value |
|---|---|
| Base model | `SG161222/RealVisXL_V4.0` |
| ControlNet init | from frozen U-Net encoder weights |
| Resolution | 512 × 512 |
| Effective batch size | 64 (grad. accum. = 8) |
| Learning rate | 1e-5 (cosine, 1000 warmup steps) |
| Optimizer | AdamW (β₁=0.9, β₂=0.999, wd=1e-2) |
| Epochs | 150 |
| Precision | BF16 |

---

## Inference

```bash
# Single mask → image
python inference.py \
    --controlnet outputs/mask2derm/controlnet-final \
    --base_model SG161222/RealVisXL_V4.0 \
    --mask assets/sample_mask.png \
    --prompt "dermoscopy image of a benign skin lesion, clinical photography" \
    --output generated.png

# Batch: directory of masks → directory of images
python inference.py \
    --controlnet outputs/mask2derm/controlnet-final \
    --mask_dir   data/processed/masks \
    --output_dir outputs/generated \
    --save_grid
```

---

## Mask Generation (optional)

To generate novel masks without real segmentation masks, a lightweight DDPM (22M param UNet) is included:

```bash
# Train mask diffusion model
python mask_diffusion.py train \
    --mask_dir data/processed/masks \
    --base_ch 48 --epochs 50 --timesteps 500

# Generate new masks
python mask_diffusion.py generate \
    --checkpoint outputs/mask_diffusion/best.pt \
    --n 64 --steps 200
```

---

## Evaluation

```bash
# FID & KID (bias-corrected via linear extrapolation)
python evaluate/metrics.py \
    --real_dir      data/processed/images \
    --generated_dir outputs/generated

# Shape Consistency (IoU & Dice via DeepLabV3+)
python evaluate/shape_consistency.py \
    --generated_dir outputs/generated \
    --mask_dir      data/processed/masks \
    --output_csv    results/shape_consistency.csv

# TSTR — train on synthetic, test on real
python evaluate/tstr.py train \
    --synthetic_images outputs/generated \
    --synthetic_masks  data/processed/masks \
    --output_dir       outputs/tstr/checkpoints --epochs 50

python evaluate/tstr.py eval \
    --real_images  data/processed/images \
    --real_masks   data/processed/masks \
    --checkpoint   outputs/tstr/checkpoints/best.pth \
    --output_csv   results/tstr_results.csv
```

---

## Repository Structure

```
Mask2DermRepo/
├── configs/
│   ├── train_config.yaml       # Default training hyperparameters (A100 / Colab)
│   └── train_config_8gb.yaml   # Low-VRAM config (8 GB GPUs, fp16 + grad checkpointing)
├── data/
│   ├── preprocessing.py        # Optics simulation (vignetting, barrel distortion)
│   ├── dataset.py              # DermoscopyDataset (image, mask, prompt triplets)
│   ├── download.py             # HAM10000 + ISIC 2018 download & preparation
│   └── prepare_hf_dataset.py   # Push to HuggingFace Hub
├── evaluate/
│   ├── metrics.py              # FID extrapolation, KID
│   ├── shape_consistency.py    # IoU / Dice via DeepLabV3+
│   └── tstr.py                 # TSTR experiment
├── scripts/
│   └── make_gif.py             # Generate demo GIF (mask grid ↔ lesion grid)
├── utils/
│   └── visualization.py        # Loss curve, IoU histogram, comparison grid
├── notebooks/
│   └── Mask2DermRepo.ipynb     # Colab training notebook
├── assets/                     # Logo, demo GIF
├── paper/                      # LaTeX source of the accompanying paper
├── train.py                    # Main training script (accelerate + SDXL)
├── inference.py                # Mask → image generation (SDXL pipeline)
├── mask_diffusion.py           # DDPM for novel mask generation
└── requirements.txt
```

---

## Citation

```bibtex
@article{rollas2025mask2derm,
  title   = {Mask2Derm: Photorealistic and Controllable Skin Lesion Synthesis via Latent Diffusion},
  author  = {Rollas, Can and G{\"u}ll{\"u}, Mehmet Kemal and Al{\i}c{\i}, {\.I}brahim Onur},
  year    = {2025},
}
```
