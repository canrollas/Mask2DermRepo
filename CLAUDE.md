# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mask2Derm is a ControlNet-based latent diffusion framework for photorealistic synthesis of dermoscopic skin lesion images from binary segmentation masks. It fine-tunes a ControlNet module on top of a frozen SDXL (RealVisXL V4.0) backbone.

## Key Commands

### Setup
```bash
pip install -r requirements.txt
accelerate config  # first-time only
```

### Data Preparation
```bash
# Download ISIC 2018 Task 1 dataset
python data/download.py --download-isic

# Download HAM10000 (requires ~/.kaggle/kaggle.json)
python data/download.py --download-ham

# Merge, resize to 256×256, generate metadata.csv
python data/download.py --prepare --size 256
```

### Training
```bash
accelerate launch train.py --config configs/train_config.yaml

# Multi-GPU
accelerate launch --multi_gpu train.py --config configs/train_config.yaml
```

### Inference
```bash
# Single image
python inference.py \
    --controlnet outputs/mask2derm/controlnet-final \
    --base_model SG161222/RealVisXL_V4.0 \
    --mask path/to/mask.png \
    --prompt "dermoscopy image of a skin lesion, clinical photography, high quality" \
    --output generated.png

# Batch
python inference.py \
    --controlnet outputs/mask2derm/controlnet-final \
    --mask_dir data/processed/masks \
    --output_dir outputs/generated \
    --batch_size 4 --save_grid
```

### Evaluation
```bash
# FID & KID
python evaluate/metrics.py --real_dir data/processed/images --generated_dir outputs/generated --device cuda

# Shape consistency (IoU & Dice)
python evaluate/shape_consistency.py --generated_dir outputs/generated --mask_dir data/processed/masks --output_csv results/shape_consistency.csv

# Train-on-Synthetic, Test-on-Real (TSTR)
python evaluate/tstr.py train --synthetic_images outputs/generated --synthetic_masks data/processed/masks --output_dir outputs/tstr/checkpoints --epochs 50
python evaluate/tstr.py eval --real_images data/processed/images --real_masks data/processed/masks --checkpoint outputs/tstr/checkpoints/best.pth --output_csv results/tstr_results.csv
```

### Mask VAE (optional — generates novel masks)
```bash
python mask_vae.py train --mask_dir data/processed/masks --epochs 100
python mask_vae.py generate --checkpoint outputs/mask_vae/best.pt --n 16 --out_dir outputs/masks_generated
```

## Architecture

### Inference Pipeline
```
Binary Mask ──► ControlNet (trainable) ──► residuals ──►
                                                         U-Net (frozen, SDXL) ──► latent ──► VAE Decoder ──► Synthetic Image
Text Prompt ──► CLIP Encoders (dual, frozen) ──► embeddings ──►
```

### Training Strategy
- Only ControlNet weights are trained; VAE, U-Net, and both CLIP encoders remain frozen.
- Loss is MSE between predicted noise and actual noise, backpropagated through ControlNet only.
- Mixed precision BF16, AdamW optimizer, cosine LR scheduler with 1000 warmup steps.
- Effective batch size ~64 via gradient accumulation (train_batch_size=4, grad_accum=8).
- `torch_compile: false` in config — noted as sometimes problematic with SDXL.

### SDXL Dual Text Encoding
SDXL requires both CLIP text encoders. Their outputs are concatenated:
- Encoder 1 hidden states: `[B, 77, 768]`
- Encoder 2 hidden states: `[B, 77, 1280]`
- Concatenated prompt_embeds: `[B, 77, 2048]`
- Pooled embeds (encoder 2): `[B, 1280]`

### Optics Simulation (`data/preprocessing.py`)
Physics-based preprocessing applied to standardize raw dermoscopy images:
1. `apply_circular_mask()` — circular aperture with feathering
2. `apply_vignetting()` — radial falloff: `V(d) = (1-α) + α * (1-(d/R)²)^γ`
3. `apply_barrel_distortion()` — lens distortion correction

This same simulation is optionally applied at inference time for consistency.

### FID Computation (`evaluate/metrics.py`)
Bias-corrected FID: computed at subset fractions [0.25, 0.50, 0.75, 1.0], then linearly extrapolated to infinite samples using `FID ≈ a/N + b`.

### Dataset (`data/dataset.py`)
- Expects `data/processed/images/`, `data/processed/masks/`, `data/processed/metadata.csv`.
- Assigns benign/malignant prompts based on `metadata.csv` `dx`/`benign_malignant` column.
- Augmentation: horizontal/vertical flips, color jitter (training only).
- Returns `pixel_values` (images, normalized to [-1, 1]) and `conditioning_pixel_values` (masks, RGB, [0, 1]).

## Configuration

All training hyperparameters live in `configs/train_config.yaml`. Key values:
- `base_model`: `SG161222/RealVisXL_V4.0` (requires accepting HuggingFace license)
- `controlnet_model`: `xinsir/controlnet-union-sdxl-1.0`
- `resolution`: 512
- `learning_rate`: 5.0e-6
- `num_train_epochs`: 150
- `validation_steps`: 500
- Output paths default to `outputs/mask2derm/`, `outputs/checkpoints/`, `outputs/samples/`

### WandB
Training logs to WandB. Runs offline by default:
```bash
WANDB_MODE=offline accelerate launch train.py --config configs/train_config.yaml
```

## External Model Dependencies

Both models are auto-downloaded from HuggingFace on first run:
- `SG161222/RealVisXL_V4.0` — SDXL base, requires license acceptance on HF Hub
- `xinsir/controlnet-union-sdxl-1.0` — ControlNet init weights
