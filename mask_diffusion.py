"""
Mask2Derm — Diffusion-based Binary Mask Generator

Small DDPM (UNet + cosine schedule + DDIM sampling).
Replaces mask_vae.py — sharper shapes, no posterior collapse.

Train:
    python mask_diffusion.py train --mask_dir data/processed/masks

Generate:
    python mask_diffusion.py generate \
        --checkpoint outputs/mask_diffusion/best.pt --n 64
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm


# ── device ────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── dataset ───────────────────────────────────────────────────────────────────

class MaskDataset(Dataset):
    def __init__(self, mask_dir: str, size: int = 256):
        self.paths = sorted(Path(mask_dir).glob("*.png")) + \
                     sorted(Path(mask_dir).glob("*.jpg"))
        assert len(self.paths), f"No masks found in {mask_dir}"
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("L").resize(
            (self.size, self.size), Image.NEAREST
        )
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
        x = torch.from_numpy(arr).unsqueeze(0)
        if torch.rand(1).item() > 0.5:
            x = x.flip(-1)
        if torch.rand(1).item() > 0.5:
            x = x.flip(-2)
        return x


# ── time embedding ────────────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
        )
        emb = t.float()[:, None] * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ── UNet blocks ───────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1     = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1     = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2     = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)
        self.drop      = nn.Dropout(dropout)
        self.skip      = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(F.silu(t_emb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[..., None, None]) + shift[..., None, None]
        return self.conv2(self.drop(F.silu(h))) + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        assert ch % heads == 0
        self.heads    = heads
        self.head_dim = ch // heads
        self.norm     = nn.GroupNorm(min(8, ch), ch)
        self.to_qkv   = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.proj     = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        def sh(t):
            return t.reshape(B, self.heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        q, k, v = sh(q), sh(k), sh(v)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.head_dim ** -0.5, dim=-1)
        out  = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(out)


# ── UNet ─────────────────────────────────────────────────────────────────────
#
# Channel layout (base_ch=64):
#   Encoder: 256→128→64→32→16  (channels: 64, 64, 128, 256, 256)
#   Bottleneck: 16×16, 256ch + attention
#   Decoder: mirror with skip connections

class UNet(nn.Module):
    def __init__(self, base_ch: int = 64, dropout: float = 0.1):
        super().__init__()
        c  = [base_ch, base_ch * 2, base_ch * 4, base_ch * 4]  # 64 128 256 256
        td = base_ch * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, td), nn.SiLU(),
            nn.Linear(td, td),
        )
        self.init_conv = nn.Conv2d(1, c[0], 3, padding=1)

        # Encoder
        self.enc1  = nn.ModuleList([ResBlock(c[0], c[0], td, dropout),
                                     ResBlock(c[0], c[0], td, dropout)])
        self.down1 = nn.Conv2d(c[0], c[0], 3, stride=2, padding=1)

        self.enc2  = nn.ModuleList([ResBlock(c[0], c[1], td, dropout),
                                     ResBlock(c[1], c[1], td, dropout)])
        self.down2 = nn.Conv2d(c[1], c[1], 3, stride=2, padding=1)

        self.enc3     = nn.ModuleList([ResBlock(c[1], c[2], td, dropout),
                                        ResBlock(c[2], c[2], td, dropout)])
        self.attn_enc = Attention(c[2])
        self.down3    = nn.Conv2d(c[2], c[2], 3, stride=2, padding=1)

        self.enc4  = nn.ModuleList([ResBlock(c[2], c[3], td, dropout),
                                     ResBlock(c[3], c[3], td, dropout)])
        self.down4 = nn.Conv2d(c[3], c[3], 3, stride=2, padding=1)

        # Bottleneck 16×16
        self.mid1     = ResBlock(c[3], c[3], td, dropout)
        self.mid_attn = Attention(c[3])
        self.mid2     = ResBlock(c[3], c[3], td, dropout)

        # Decoder
        self.up4  = nn.ConvTranspose2d(c[3], c[3], 4, stride=2, padding=1)
        self.dec4 = nn.ModuleList([ResBlock(c[3] * 2,      c[3], td, dropout),
                                    ResBlock(c[3],          c[3], td, dropout)])

        self.up3      = nn.ConvTranspose2d(c[3], c[3], 4, stride=2, padding=1)
        self.dec3     = nn.ModuleList([ResBlock(c[3] + c[2], c[2], td, dropout),
                                        ResBlock(c[2],        c[2], td, dropout)])
        self.attn_dec = Attention(c[2])

        self.up2  = nn.ConvTranspose2d(c[2], c[2], 4, stride=2, padding=1)
        self.dec2 = nn.ModuleList([ResBlock(c[2] + c[1], c[1], td, dropout),
                                    ResBlock(c[1],        c[1], td, dropout)])

        self.up1  = nn.ConvTranspose2d(c[1], c[1], 4, stride=2, padding=1)
        self.dec1 = nn.ModuleList([ResBlock(c[1] + c[0], c[0], td, dropout),
                                    ResBlock(c[0],        c[0], td, dropout)])

        self.out_norm = nn.GroupNorm(8, c[0])
        self.out_conv = nn.Conv2d(c[0], 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        for b in self.enc1: x = b(x, t_emb)
        s1 = x;  x = self.down1(x)

        for b in self.enc2: x = b(x, t_emb)
        s2 = x;  x = self.down2(x)

        for b in self.enc3: x = b(x, t_emb)
        x = self.attn_enc(x)
        s3 = x;  x = self.down3(x)

        for b in self.enc4: x = b(x, t_emb)
        s4 = x;  x = self.down4(x)

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        x = self.up4(x);  x = torch.cat([x, s4], 1)
        for b in self.dec4: x = b(x, t_emb)

        x = self.up3(x);  x = torch.cat([x, s3], 1)
        for b in self.dec3: x = b(x, t_emb)
        x = self.attn_dec(x)

        x = self.up2(x);  x = torch.cat([x, s2], 1)
        for b in self.dec2: x = b(x, t_emb)

        x = self.up1(x);  x = torch.cat([x, s1], 1)
        for b in self.dec1: x = b(x, t_emb)

        return self.out_conv(F.silu(self.out_norm(x)))


# ── diffusion ─────────────────────────────────────────────────────────────────

class GaussianDiffusion:
    """Cosine-schedule DDPM + deterministic DDIM sampling."""

    def __init__(self, T: int = 1000):
        self.T = T
        betas = self._cosine_betas(T)
        acp   = torch.cumprod(1.0 - betas, dim=0)
        self.betas          = betas
        self.alphas_cumprod = acp
        self.sqrt_acp       = acp.sqrt()
        self.sqrt_one_minus = (1.0 - acp).sqrt()

    @staticmethod
    def _cosine_betas(T: int) -> torch.Tensor:
        s = 0.008
        t = torch.linspace(0, T, T + 1)
        f = torch.cos((t / T + s) / (1 + s) * math.pi * 0.5) ** 2
        acp = f / f[0]
        return (1 - acp[1:] / acp[:-1]).clamp(1e-5, 0.999)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x0)
        t_cpu = t.cpu()
        sa = self.sqrt_acp[t_cpu].to(x0.device)[:, None, None, None]
        sm = self.sqrt_one_minus[t_cpu].to(x0.device)[:, None, None, None]
        return sa * x0 + sm * noise, noise

    def loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        t  = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        xt, noise = self.q_sample(x0, t)
        return F.mse_loss(model(xt, t.float()), noise)

    @torch.no_grad()
    def ddim_sample(self, model: nn.Module, n: int, size: int,
                    device: str, steps: int = 50) -> torch.Tensor:
        model.eval()
        seq = torch.linspace(self.T - 1, 0, steps + 1).long().tolist()
        x   = torch.randn(n, 1, size, size, device=device)
        for i in range(steps):
            t_now, t_prev = seq[i], seq[i + 1]
            t_b  = torch.full((n,), t_now, device=device, dtype=torch.long)
            eps  = model(x, t_b.float())
            a    = self.alphas_cumprod[t_now].to(device)
            a_p  = self.alphas_cumprod[t_prev].to(device)
            x0p  = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            x    = a_p.sqrt() * x0p + (1 - a_p).sqrt() * eps
        return x  # [-1, 1]


# ── post-processing ───────────────────────────────────────────────────────────

def to_binary(samples: torch.Tensor) -> list[np.ndarray]:
    arr = samples.squeeze(1).cpu().numpy()
    return [(m >= 0.0).astype(np.uint8) * 255 for m in arr]


def clean_mask(mask: np.ndarray) -> np.ndarray:
    binary = mask > 127
    labeled, n = ndimage.label(binary)
    if n == 0:
        return mask
    sizes   = ndimage.sum(binary, labeled, range(1, n + 1))
    largest = labeled == (int(np.argmax(sizes)) + 1)
    largest = ndimage.binary_fill_holes(largest)
    r = 4;  y, xg = np.ogrid[-r:r+1, -r:r+1]
    disk    = (xg**2 + y**2 <= r**2).astype(np.uint8)
    largest = ndimage.binary_opening(largest, structure=disk)
    smoothed = ndimage.gaussian_filter(largest.astype(np.float32), sigma=3)
    return ((smoothed >= 0.4).astype(np.uint8) * 255)


# ── train ─────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device  = get_device()
    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    dataset = MaskDataset(args.mask_dir, size=args.resolution)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=4, pin_memory=(device == "cuda"))
    print(f"Dataset: {len(dataset)} masks  |  batch: {args.batch_size}")

    model     = UNet(base_ch=args.base_ch, dropout=args.dropout).to(device)
    diffusion = GaussianDiffusion(T=args.timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 10
    )
    print(f"UNet params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    start_epoch = 1
    best_loss   = float("inf")

    if args.resume and (out_dir / "last.pt").exists():
        ckpt = torch.load(out_dir / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_loss   = ckpt.get("best_loss", best_loss)
        print(f"Resumed from epoch {ckpt['epoch']}  (best loss {best_loss:.5f})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = 0.0
        for x in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x    = x.to(device)
            loss = diffusion.loss(model, x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        avg = total / len(loader)
        scheduler.step()
        print(f"Epoch {epoch:>4}  loss={avg:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "best_loss": best_loss}
        torch.save(ckpt, out_dir / "last.pt")

        if avg < best_loss:
            best_loss = avg
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  ↳ best (loss={best_loss:.5f})")

        if epoch % args.sample_every == 0:
            samples = diffusion.ddim_sample(model, n=16, size=args.resolution,
                                            device=device, steps=50)
            save_image((samples + 1) / 2, out_dir / f"samples_{epoch:04d}.png",
                       nrow=4, pad_value=0.5)
            model.train()

    print(f"Done. Best loss: {best_loss:.5f}  →  {out_dir}/best.pt")


# ── generate ──────────────────────────────────────────────────────────────────

def generate(args: argparse.Namespace) -> None:
    device  = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_ckpt = torch.load(args.checkpoint, map_location=device)
    state    = raw_ckpt["model"] if "model" in raw_ckpt else raw_ckpt
    model    = UNet(base_ch=args.base_ch).to(device)
    model.load_state_dict(state)
    model.eval()

    diffusion = GaussianDiffusion(T=1000)
    saved     = 0
    grid_done = False

    for start in tqdm(range(0, args.n, args.gen_batch_size), desc="Generating"):
        n_batch = min(args.gen_batch_size, args.n - start)
        raw     = diffusion.ddim_sample(model, n=n_batch, size=args.resolution,
                                        device=device, steps=args.steps)
        if not grid_done:
            save_image(((raw[:16] + 1) / 2).clamp(0, 1),
                       out_dir / "grid.png", nrow=4, pad_value=0.5)
            grid_done = True

        for m in to_binary(raw):
            clean = clean_mask(m)
            if clean.sum() == 0:
                continue
            Image.fromarray(clean).save(out_dir / f"mask_{saved:05d}.png")
            saved += 1

    print(f"{saved} masks saved → {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p   = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--mask_dir",       required=True)
    t.add_argument("--resolution",     type=int,   default=256)
    t.add_argument("--base_ch",        type=int,   default=64)
    t.add_argument("--timesteps",      type=int,   default=1000)
    t.add_argument("--epochs",         type=int,   default=200)
    t.add_argument("--batch_size",     type=int,   default=16)
    t.add_argument("--lr",             type=float, default=2e-4)
    t.add_argument("--dropout",        type=float, default=0.1)
    t.add_argument("--sample_every",   type=int,   default=10)
    t.add_argument("--checkpoint_dir", default="outputs/mask_diffusion")
    t.add_argument("--resume",         action="store_true")

    g = sub.add_parser("generate")
    g.add_argument("--checkpoint",     required=True)
    g.add_argument("--n",              type=int,   default=64)
    g.add_argument("--resolution",     type=int,   default=256)
    g.add_argument("--base_ch",        type=int,   default=64)
    g.add_argument("--steps",          type=int,   default=50,
                   help="DDIM denoising steps (50 = fast & high quality)")
    g.add_argument("--gen_batch_size", type=int,   default=16)
    g.add_argument("--out_dir",        default="outputs/masks_diffusion")

    args = p.parse_args()
    (train if args.cmd == "train" else generate)(args)


if __name__ == "__main__":
    main()
