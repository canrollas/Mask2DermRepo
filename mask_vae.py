"""
Mask2Derm — Binary Mask VAE

Trains a convolutional VAE on segmentation masks.
After training, generates novel binary lesion masks from random latent vectors.

Train:
    python mask_vae.py train --mask_dir data/processed/masks --epochs 100

Generate:
    python mask_vae.py generate --checkpoint outputs/mask_vae/best.pt --n 16 --out_dir outputs/masks_generated
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MaskDataset(Dataset):
    def __init__(self, mask_dir: str, size: int = 256):
        self.paths = sorted(Path(mask_dir).glob("*.png")) + \
                     sorted(Path(mask_dir).glob("*.jpg"))
        assert len(self.paths) > 0, f"No masks found in {mask_dir}"
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("L").resize(
            (self.size, self.size), Image.NEAREST
        )
        x = torch.from_numpy(np.array(img)).float() / 255.0
        return x.unsqueeze(0)  # (1, H, W)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,  32,  4, 2, 1), nn.LeakyReLU(0.2),   # 128
            nn.Conv2d(32, 64,  4, 2, 1), nn.LeakyReLU(0.2),   # 64
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),   # 32
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),  # 16
        )
        self.fc_mu     = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),  # 32
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.ReLU(),  # 64
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.ReLU(),  # 128
            nn.ConvTranspose2d(32,   1,  4, 2, 1),              # 256 — raw logits
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 256, 16, 16)
        return self.net(h)


class MaskVAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device: str = "cuda", threshold: float = 0.5) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        soft = torch.sigmoid(self.decoder(z))
        return (soft >= threshold).float()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def vae_loss(recon_logits: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> tuple[torch.Tensor, float, float]:
    bce = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="sum") / x.size(0)
    kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kl, bce.item(), kl.item()


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = MaskDataset(args.mask_dir, size=args.resolution)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=4, pin_memory=True)

    model = MaskVAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_bce = total_kl = 0.0

        # KL annealing: 0'dan args.beta'ya ilk %30 epoch'ta lineer artış
        warmup_epochs = max(1, int(args.epochs * 0.30))
        beta = args.beta * min(1.0, epoch / warmup_epochs)

        for x in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, bce, kl = vae_loss(recon, x, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_bce  += bce
            total_kl   += kl

        n = len(loader)
        print(f"Epoch {epoch:>4} | beta={beta:.3f}  loss={total_loss/n:.4f}  "
              f"bce={total_bce/n:.4f}  kl={total_kl/n:.4f}")

        # Save best
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), out_dir / "best.pt")

        # Sample grid every 10 epochs
        if epoch % 10 == 0:
            samples = model.sample(16, device=device)
            save_image(samples, out_dir / f"samples_epoch{epoch:04d}.png",
                       nrow=4, pad_value=0.5)

    torch.save(model.state_dict(), out_dir / "last.pt")
    print(f"Training complete. Checkpoints → {out_dir}")


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def generate(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = MaskVAE(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    masks = model.sample(args.n, device=device, threshold=args.threshold)

    for i, m in enumerate(masks):
        img = Image.fromarray((m.squeeze().cpu().numpy() * 255).astype(np.uint8), mode="L")
        img.save(out_dir / f"mask_{i:04d}.png")

    save_image(masks, out_dir / "grid.png", nrow=4, pad_value=0.5)
    print(f"{args.n} masks saved → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- train ---
    t = sub.add_parser("train")
    t.add_argument("--mask_dir",       required=True)
    t.add_argument("--resolution",     type=int,   default=256)
    t.add_argument("--latent_dim",     type=int,   default=128)
    t.add_argument("--epochs",         type=int,   default=100)
    t.add_argument("--batch_size",     type=int,   default=32)
    t.add_argument("--lr",             type=float, default=1e-3)
    t.add_argument("--beta",           type=float, default=1.0,
                   help="KL weight (beta-VAE, >1 → more disentangled latent)")
    t.add_argument("--checkpoint_dir", default="outputs/mask_vae")

    # --- generate ---
    g = sub.add_parser("generate")
    g.add_argument("--checkpoint",  required=True, help="Path to best.pt or last.pt")
    g.add_argument("--n",           type=int,   default=16,  help="Number of masks to generate")
    g.add_argument("--latent_dim",  type=int,   default=128)
    g.add_argument("--threshold",   type=float, default=0.5, help="Binarization threshold")
    g.add_argument("--out_dir",     default="outputs/masks_generated")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
