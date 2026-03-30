"""
Mask2Derm — ControlNet Fine-tuning Script

Trains a ControlNet module conditioned on lesion segmentation masks
on top of a frozen Realistic Vision V5.1 backbone.

Usage (single GPU):
    accelerate launch train.py --config configs/train_config.yaml

Usage (multi-GPU):
    accelerate launch --multi_gpu train.py --config configs/train_config.yaml

Key design decisions (from the paper):
  - VAE, U-Net backbone, and CLIP text encoder are fully frozen.
  - Only ControlNet parameters receive gradient updates.
  - Training precision: FP32.
  - Optimizer: AdamW (β1=0.9, β2=0.999, wd=1e-2, lr=1e-5)
  - Scheduler: Cosine with 500 warmup steps.
  - Resolution: 256×256, batch size 16 + 4 grad-accum steps.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from data.dataset import DermoscopyDataset

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask2Derm ControlNet training")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="Path to YAML training config")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint directory to resume from")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run one batch forward pass and exit (smoke test)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Tokenization helper
# ---------------------------------------------------------------------------

def tokenize_prompts(tokenizer: CLIPTokenizer, prompts: list[str]) -> torch.Tensor:
    return tokenizer(
        prompts,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids


# ---------------------------------------------------------------------------
# Validation: generate a few samples and log to tracker
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_validation(
    vae, text_encoder, tokenizer, unet, controlnet, scheduler,
    cfg, accelerator, step: int,
) -> None:
    from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler
    from torchvision.utils import make_grid

    logger.info("Running validation…")
    pipeline = StableDiffusionControlNetPipeline(
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        controlnet=accelerator.unwrap_model(controlnet),
        scheduler=DDIMScheduler.from_config(scheduler.config),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline.set_progress_bar_config(disable=True)

    # Use a simple circular white mask as validation condition
    import numpy as np
    from PIL import Image, ImageDraw

    h = w = cfg.resolution
    mask = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    r = int(min(w, h) * 0.3)
    cx, cy = w // 2, h // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))

    images = pipeline(
        prompt=[cfg.validation_prompt] * cfg.num_validation_images,
        image=[mask] * cfg.num_validation_images,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        generator=torch.manual_seed(42),
    ).images

    grid = make_grid([torch.tensor(np.array(img)).permute(2, 0, 1) for img in images],
                     nrow=4, normalize=False)
    accelerator.log({"validation": grid.float() / 255.0}, step=step)
    del pipeline
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    # Accelerator
    logging_dir = Path(cfg.output_dir) / "logs"
    project_config = ProjectConfiguration(project_dir=cfg.output_dir,
                                           logging_dir=str(logging_dir))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.get("tracker", "tensorboard"),
        project_config=project_config,
    )
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state)

    set_seed(42)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    tokenizer = CLIPTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(cfg.base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.base_model, subfolder="scheduler")

    # Load / init ControlNet
    if args.resume_from_checkpoint:
        controlnet = ControlNetModel.from_pretrained(args.resume_from_checkpoint)
        logger.info(f"Resumed ControlNet from {args.resume_from_checkpoint}")
    else:
        controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model)
        logger.info(f"Initialized ControlNet from {cfg.controlnet_model}")

    # ------------------------------------------------------------------
    # Freeze everything except ControlNet
    # ------------------------------------------------------------------
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    train_dataset = DermoscopyDataset(
        root=cfg.data_dir,
        split="train",
        resolution=cfg.resolution,
        apply_lens=cfg.apply_lens_simulation,
        augment=True,
    )
    val_dataset = DermoscopyDataset(
        root=cfg.data_dir,
        split="val",
        resolution=cfg.resolution,
        apply_lens=cfg.apply_lens_simulation,
        augment=False,
    )

    def collate_fn(examples):
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        conditioning = torch.stack([e["conditioning_pixel_values"] for e in examples])
        prompts = [e["prompt"] for e in examples]
        return {"pixel_values": pixel_values,
                "conditioning_pixel_values": conditioning,
                "prompts": prompts}

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # LR scheduler
    # ------------------------------------------------------------------
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / cfg.gradient_accumulation_steps
    )
    max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=max_train_steps * cfg.gradient_accumulation_steps,
    )

    # ------------------------------------------------------------------
    # Prepare with Accelerator
    # ------------------------------------------------------------------
    controlnet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_loader, lr_scheduler
    )
    vae.to(accelerator.device)
    unet.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # ------------------------------------------------------------------
    # Tracker init
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        accelerator.init_trackers(cfg.get("tracker_project", "mask2derm"))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    logger.info(f"  Num train samples   = {len(train_dataset)}")
    logger.info(f"  Num epochs          = {cfg.num_train_epochs}")
    logger.info(f"  Batch size          = {cfg.train_batch_size}")
    logger.info(f"  Grad accum steps    = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total update steps  = {max_train_steps}")

    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")

    for epoch in range(cfg.num_train_epochs):
        controlnet.train()
        train_loss = 0.0

        for batch in train_loader:
            with accelerator.accumulate(controlnet):
                # Encode images to latent space
                latents = vae.encode(batch["pixel_values"].to(vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                          (bsz,), device=latents.device).long()

                # Add noise (forward diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode text
                input_ids = tokenize_prompts(tokenizer, batch["prompts"]).to(latents.device)
                encoder_hidden_states = text_encoder(input_ids)[0]

                # ControlNet forward
                controlnet_image = batch["conditioning_pixel_values"].to(
                    dtype=controlnet.dtype if hasattr(controlnet, "dtype") else latents.dtype
                )
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # U-Net forward with ControlNet residuals
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # MSE loss on noise prediction
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                train_loss += loss.detach().item()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), cfg.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Sync step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                avg_loss = train_loss / cfg.gradient_accumulation_steps
                train_loss = 0.0

                logs = {"loss": avg_loss, "lr": lr_scheduler.get_last_lr()[0]}
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

                # Validation samples
                if (cfg.validation_steps and global_step % cfg.validation_steps == 0
                        and accelerator.is_main_process):
                    log_validation(vae, text_encoder, tokenizer, unet, controlnet,
                                   noise_scheduler, cfg, accelerator, global_step)

            if args.dry_run:
                logger.info("Dry run complete — exiting after one batch.")
                accelerator.end_training()
                return

        # ----- End of epoch -----
        if accelerator.is_main_process:
            save_path = Path(cfg.output_dir) / f"checkpoint-epoch-{epoch:04d}"
            accelerator.unwrap_model(controlnet).save_pretrained(str(save_path))
            logger.info(f"Saved checkpoint → {save_path}")

    # Final save
    if accelerator.is_main_process:
        final_path = Path(cfg.output_dir) / "controlnet-final"
        accelerator.unwrap_model(controlnet).save_pretrained(str(final_path))
        logger.info(f"Training complete. Final model → {final_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
