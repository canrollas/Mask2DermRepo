"""
Mask2Derm — ControlNet Fine-tuning Script (SDXL)

Trains a ControlNet module conditioned on lesion segmentation masks
on top of a frozen RealVisXL V4.0 (SDXL) backbone.

Usage (single GPU):
    accelerate launch train.py --config configs/train_config.yaml

Usage (multi-GPU):
    accelerate launch --multi_gpu train.py --config configs/train_config.yaml

Key design decisions:
  - VAE, U-Net backbone, and both CLIP text encoders are fully frozen.
  - Only ControlNet parameters receive gradient updates.
  - Training precision: BF16.
  - Optimizer: AdamW (β1=0.9, β2=0.999, wd=1e-2, lr=1e-5)
  - Scheduler: Cosine with 500 warmup steps.
  - Resolution: 1024×1024, effective batch size 64 (4 x 16 grad-accum).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")

for _noisy in ("httpx", "httpcore", "huggingface_hub", "huggingface_hub.utils._http",
               "huggingface_hub.utils._validators", "wandb", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from data.dataset import DermoscopyDataset

logger = get_logger(__name__, log_level="WARNING")
console = Console(force_terminal=False, force_jupyter=False, highlight=False)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask2Derm ControlNet SDXL training")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--resume-from-checkpoint", dest="resume_from_checkpoint",
                        type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# SDXL text encoding helpers
# ---------------------------------------------------------------------------

def encode_prompts_xl(
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    prompts: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompts with both SDXL text encoders.

    Returns:
        prompt_embeds:  [B, 77, 2048]  — concatenated hidden states
        pooled_embeds:  [B, 1280]      — pooled output from encoder 2
    """
    def _tokenize(tok, texts):
        return tok(texts, padding="max_length", max_length=tok.model_max_length,
                   truncation=True, return_tensors="pt").input_ids.to(device)

    ids_1 = _tokenize(tokenizer_1, prompts)
    ids_2 = _tokenize(tokenizer_2, prompts)

    out_1 = text_encoder_1(ids_1, output_hidden_states=True)
    out_2 = text_encoder_2(ids_2, output_hidden_states=True)

    hidden_1 = out_1.hidden_states[-2]   # [B, 77, 768]
    hidden_2 = out_2.hidden_states[-2]   # [B, 77, 1280]
    pooled   = out_2[0]                  # [B, 1280]

    prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1)  # [B, 77, 2048]
    return prompt_embeds, pooled


def make_add_time_ids(resolution: int, batch_size: int,
                      device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build SDXL add_time_ids: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]."""
    ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        dtype=dtype, device=device,
    )
    return ids.repeat(batch_size, 1)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_validation(
    vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2,
    unet, controlnet, scheduler, cfg, accelerator, step: int,
) -> None:
    from diffusers import StableDiffusionXLControlNetPipeline, DDIMScheduler
    from torchvision.utils import make_grid
    import numpy as np
    from PIL import Image, ImageDraw

    console.log("[dim]Running validation...[/dim]")
    pipeline = StableDiffusionXLControlNetPipeline(
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder_1),
        text_encoder_2=accelerator.unwrap_model(text_encoder_2),
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        unet=accelerator.unwrap_model(unet),
        controlnet=accelerator.unwrap_model(controlnet),
        scheduler=DDIMScheduler.from_config(scheduler.config),
    )
    pipeline.set_progress_bar_config(disable=True)

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


@torch.no_grad()
def save_epoch_samples(
    vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2,
    unet, controlnet, scheduler, cfg, accelerator, val_dataset, epoch: int,
) -> None:
    from diffusers import StableDiffusionXLControlNetPipeline, DDIMScheduler
    from torchvision.utils import save_image
    import numpy as np
    from PIL import Image

    n     = cfg.get("num_epoch_samples", 4)
    steps = cfg.get("epoch_samples_steps", 20)
    out_dir = Path(cfg.get("samples_dir", "outputs/samples"))
    out_dir.mkdir(parents=True, exist_ok=True)

    malignant = [s for s in val_dataset.samples if s.get("label") == "malignant"]
    pool = malignant if len(malignant) >= n else val_dataset.samples
    chosen = random.sample(pool, min(n, len(pool)))

    masks_pil = []
    for s in chosen:
        m = Image.open(s["mask"]).convert("L").resize(
            (cfg.resolution, cfg.resolution), Image.NEAREST
        ).convert("RGB")
        masks_pil.append(m)

    pipeline = StableDiffusionXLControlNetPipeline(
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder_1),
        text_encoder_2=accelerator.unwrap_model(text_encoder_2),
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        unet=accelerator.unwrap_model(unet),
        controlnet=accelerator.unwrap_model(controlnet),
        scheduler=DDIMScheduler.from_config(scheduler.config),
    )
    pipeline.set_progress_bar_config(disable=True)

    prompt = "dermoscopy image of a malignant skin lesion, irregular borders, clinical photography, high quality"
    images = pipeline(
        prompt=[prompt] * len(masks_pil),
        image=masks_pil,
        num_inference_steps=steps,
        guidance_scale=cfg.guidance_scale,
        generator=torch.Generator().manual_seed(epoch),
    ).images

    to_tensor = lambda img: torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    pairs = []
    for mask_pil, gen_pil in zip(masks_pil, images):
        pairs.extend([to_tensor(mask_pil), to_tensor(gen_pil)])

    save_path = out_dir / f"epoch_{epoch:04d}.png"
    save_image(pairs, save_path, nrow=2, padding=4, pad_value=0.5)
    console.log(f"[green]Epoch {epoch} samples saved →[/] {save_path}")

    del pipeline
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Rich UI helpers
# ---------------------------------------------------------------------------

def print_step(epoch, total_epochs, step, total_steps, loss, lr):
    pct = step / total_steps * 100
    bar_len = 30
    filled = int(bar_len * step / total_steps)
    bar = "█" * filled + "░" * (bar_len - filled)
    console.print(
        f"  [cyan]E[bold]{epoch:>3}/{total_epochs}[/bold][/cyan]"
        f"  [blue]{bar}[/blue]"
        f"  [white]{step:>4}/{total_steps}[/white] ({pct:5.1f}%)"
        f"  loss=[yellow]{loss:.4f}[/yellow]"
        f"  lr=[dim]{lr:.2e}[/dim]",
        highlight=False,
    )


def print_epoch_summary(epoch, total_epochs, avg_loss, best_loss, elapsed):
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n{'='*60}")
    print(f"  EPOCH {epoch}/{total_epochs} DONE  |  {mins}m {secs}s")
    print(f"  Avg loss: {avg_loss:.5f}  |  Best: {best_loss:.5f}")
    print(f"{'='*60}\n", flush=True)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    logging.basicConfig(level=logging.ERROR)
    project_config = ProjectConfiguration(
        project_dir=cfg.output_dir,
        logging_dir=str(Path(cfg.output_dir) / "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.get("tracker", "tensorboard"),
        project_config=project_config,
    )
    set_seed(42)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    with console.status("[bold green]Loading SDXL models...", spinner="dots"):
        tokenizer_1   = CLIPTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer")
        tokenizer_2   = CLIPTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer_2")
        text_encoder_1 = CLIPTextModel.from_pretrained(cfg.base_model, subfolder="text_encoder")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            cfg.base_model, subfolder="text_encoder_2"
        )
        vae            = AutoencoderKL.from_pretrained(cfg.base_model, subfolder="vae")
        unet           = UNet2DConditionModel.from_pretrained(cfg.base_model, subfolder="unet")
        noise_scheduler = DDPMScheduler.from_pretrained(cfg.base_model, subfolder="scheduler")

        if args.resume_from_checkpoint:
            controlnet = ControlNetModel.from_pretrained(args.resume_from_checkpoint)
            console.log(f"[green]Resumed ControlNet from[/] {args.resume_from_checkpoint}")
        else:
            controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model)
            console.log(f"[green]Loaded ControlNet from[/] {cfg.controlnet_model}")

    # ------------------------------------------------------------------
    # Freeze everything except ControlNet
    # ------------------------------------------------------------------
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
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
        root=cfg.data_dir, split="train", resolution=cfg.resolution,
        apply_lens=cfg.apply_lens_simulation, augment=True,
    )
    val_dataset = DermoscopyDataset(
        root=cfg.data_dir, split="val", resolution=cfg.resolution,
        apply_lens=cfg.apply_lens_simulation, augment=False,
    )

    def collate_fn(examples):
        return {
            "pixel_values":              torch.stack([e["pixel_values"] for e in examples]),
            "conditioning_pixel_values": torch.stack([e["conditioning_pixel_values"] for e in examples]),
            "prompts":                   [e["prompt"] for e in examples],
        }

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
        num_workers=cfg.get("dataloader_num_workers", 4),
        pin_memory=True, collate_fn=collate_fn,
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
    text_encoder_1.to(accelerator.device)
    text_encoder_2.to(accelerator.device)

    if accelerator.is_main_process:
        accelerator.init_trackers(cfg.get("tracker_project", "mask2derm"))

    # ------------------------------------------------------------------
    # Print training summary
    # ------------------------------------------------------------------
    console.rule("[bold magenta]Mask2Derm SDXL Training")
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    summary.add_row("Base model",  cfg.base_model)
    summary.add_row("Resolution",  f"{cfg.resolution}×{cfg.resolution}")
    summary.add_row("Samples",     str(len(train_dataset)))
    summary.add_row("Epochs",      str(cfg.num_train_epochs))
    summary.add_row("Batch size",  str(cfg.train_batch_size))
    summary.add_row("Grad accum",  str(cfg.gradient_accumulation_steps))
    summary.add_row("Eff. batch",  str(cfg.train_batch_size * cfg.gradient_accumulation_steps))
    summary.add_row("Total steps", str(max_train_steps))
    summary.add_row("Precision",   cfg.mixed_precision)
    summary.add_row("Device",      str(accelerator.device))
    console.print(Panel(summary, title="Config", border_style="magenta"))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    global_step = 0
    best_loss   = float("inf")
    start_epoch = 0
    log_every   = cfg.get("logging_steps", 50)

    if args.resume_from_checkpoint:
        ckpt_path  = Path(args.resume_from_checkpoint)
        opt_file   = ckpt_path / "optimizer.pt"
        sch_file   = ckpt_path / "scheduler.pt"
        state_file = ckpt_path / "training_state.json"
        if opt_file.exists():
            optimizer.load_state_dict(torch.load(opt_file, map_location=accelerator.device))
            console.log(f"[green]Optimizer state restored[/]")
        if sch_file.exists():
            lr_scheduler.load_state_dict(torch.load(sch_file, map_location="cpu"))
            console.log(f"[green]LR scheduler state restored[/]")
        if state_file.exists():
            state       = json.loads(state_file.read_text())
            start_epoch = state["epoch"]
            global_step = state["global_step"]
            best_loss   = state["best_loss"]
            console.log(f"[green]Resuming from epoch {start_epoch + 1}[/]")

    import time

    for epoch in range(start_epoch, cfg.num_train_epochs):
        controlnet.train()
        epoch_loss = 0.0
        step_count = 0
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch + 1}/{cfg.num_train_epochs} ---", flush=True)

        for batch in train_loader:
            with accelerator.accumulate(controlnet):
                # Encode images to latents
                latents = vae.encode(
                    batch["pixel_values"].to(vae.dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                noise     = torch.randn_like(latents)
                bsz       = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # SDXL text embeddings
                prompt_embeds, pooled_embeds = encode_prompts_xl(
                    tokenizer_1, tokenizer_2,
                    text_encoder_1, text_encoder_2,
                    batch["prompts"], latents.device,
                )

                # SDXL additional conditioning
                add_time_ids = make_add_time_ids(
                    cfg.resolution, bsz, latents.device, prompt_embeds.dtype
                )
                added_cond_kwargs = {
                    "text_embeds": pooled_embeds,
                    "time_ids":    add_time_ids,
                }

                controlnet_image = batch["conditioning_pixel_values"].to(
                    dtype=controlnet.dtype if hasattr(controlnet, "dtype") else latents.dtype
                )

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=controlnet_image,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )

                noise_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                epoch_loss += loss.detach().item()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), cfg.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                step_count  += 1
                current_lr   = lr_scheduler.get_last_lr()[0]
                avg_loss     = epoch_loss / step_count

                accelerator.log({"loss": avg_loss, "lr": current_lr}, step=global_step)

                if step_count % log_every == 0 or step_count == num_update_steps_per_epoch:
                    print_step(epoch + 1, cfg.num_train_epochs,
                               step_count, num_update_steps_per_epoch,
                               avg_loss, current_lr)

                if (cfg.validation_steps and global_step % cfg.validation_steps == 0
                        and accelerator.is_main_process):
                    log_validation(
                        vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2,
                        unet, controlnet, noise_scheduler, cfg, accelerator, global_step,
                    )

            if args.dry_run:
                console.log("[yellow]Dry run complete — exiting.[/yellow]")
                accelerator.end_training()
                return

        # ----- End of epoch -----
        avg_epoch_loss = epoch_loss / max(step_count, 1)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss

        print_epoch_summary(epoch + 1, cfg.num_train_epochs,
                            avg_epoch_loss, best_loss,
                            time.time() - epoch_start)

        if accelerator.is_main_process:
            ckpt_dir  = Path(cfg.get("checkpoint_dir", cfg.output_dir))
            save_path = ckpt_dir / f"checkpoint-epoch-{epoch:04d}"
            accelerator.unwrap_model(controlnet).save_pretrained(str(save_path))
            torch.save(optimizer.state_dict(), save_path / "optimizer.pt")
            torch.save(lr_scheduler.state_dict(), save_path / "scheduler.pt")
            (save_path / "training_state.json").write_text(
                json.dumps({"epoch": epoch, "global_step": global_step,
                            "best_loss": best_loss})
            )
            save_epoch_samples(
                vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2,
                unet, controlnet, noise_scheduler, cfg, accelerator, val_dataset, epoch + 1,
            )

    if accelerator.is_main_process:
        final_path = Path(cfg.get("checkpoint_dir", cfg.output_dir)) / "controlnet-final"
        accelerator.unwrap_model(controlnet).save_pretrained(str(final_path))
        console.rule("[bold green]Training complete")
        console.print(f"[green]Final model →[/] {final_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
