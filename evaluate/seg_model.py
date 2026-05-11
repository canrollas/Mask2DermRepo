"""
Lightweight U-Net for mask fidelity evaluation.

Used in:
  - Exp 2: predict masks on generated images → compare to GT masks
  - Exp 3: train on real / real+synthetic → evaluate on real test set
  - Ablations 4-7: mask fidelity via Dice/IoU

Architecture: 4-level encoder-decoder U-Net, ~7M params.

Usage:
    # Train
    python evaluate/seg_model.py train \
        --data_dir data/processed \
        --save_path outputs/eval_seg/unet.pth \
        --epochs 30 --device cuda

    # Predict (generates mask PNGs into out_dir)
    python evaluate/seg_model.py predict \
        --model_path outputs/eval_seg/unet.pth \
        --image_dir outputs/generated/exp1 \
        --out_dir outputs/eval_seg/pred_masks \
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class _Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = _DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class LightweightUNet(nn.Module):
    """~7M param U-Net for binary dermoscopy lesion segmentation."""

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        b = base_ch
        self.inc  = _DoubleConv(in_ch, b)       # → b
        self.d1   = _Down(b,     b*2)            # → b*2
        self.d2   = _Down(b*2,   b*4)            # → b*4
        self.d3   = _Down(b*4,   b*8)            # → b*8
        self.neck = _Down(b*8,   b*8)            # → b*8
        self.u1   = _Up(b*8, b*8, b*4)          # neck + skip_d3
        self.u2   = _Up(b*4, b*4, b*2)          # u1   + skip_d2
        self.u3   = _Up(b*2, b*2, b)            # u2   + skip_d1
        self.u4   = _Up(b,   b,   b)            # u3   + skip_inc
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.neck(x4)
        x  = self.u1(x5, x4)
        x  = self.u2(x,  x3)
        x  = self.u3(x,  x2)
        x  = self.u4(x,  x1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _SegDataset(Dataset):
    def __init__(self, image_paths: list[Path], mask_paths: list[Path],
                 size: int = 256, augment: bool = False):
        self.pairs   = list(zip(image_paths, mask_paths))
        self.size    = size
        self.augment = augment

        self.img_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
        self.aug_tf = transforms.RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img  = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("L").resize((self.size, self.size), Image.NEAREST)

        if self.augment and torch.rand(1) > 0.5:
            img  = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)
        if self.augment and torch.rand(1) > 0.5:
            img  = transforms.functional.vflip(img)
            mask = transforms.functional.vflip(mask)

        img_t  = self.img_tf(img)
        mask_t = (transforms.functional.to_tensor(mask) > 0.5).float()
        return img_t, mask_t, ip.name


# ---------------------------------------------------------------------------
# Train / Predict
# ---------------------------------------------------------------------------

def _get_pairs(image_dir: Path, mask_dir: Path):
    exts = {".jpg", ".jpeg", ".png"}
    pairs = []
    for ip in sorted(image_dir.iterdir()):
        if ip.suffix.lower() not in exts:
            continue
        mp = mask_dir / f"{ip.stem}.png"
        if not mp.exists():
            mp = mask_dir / f"{ip.stem}_segmentation.png"
        if mp.exists():
            pairs.append((ip, mp))
    return pairs


def train_seg_model(
    image_dir: str | Path,
    mask_dir:  str | Path,
    save_path: str | Path,
    *,
    val_image_dir: str | Path | None = None,
    val_mask_dir:  str | Path | None = None,
    epochs:        int   = 30,
    batch_size:    int   = 8,
    lr:            float = 1e-3,
    size:          int   = 256,
    device:        str   = "cuda",
) -> dict:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_pairs = _get_pairs(Path(image_dir), Path(mask_dir))
    train_ds    = _SegDataset([p[0] for p in train_pairs],
                               [p[1] for p in train_pairs],
                               size=size, augment=True)
    train_dl    = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=(device == "cuda"))

    val_dl = None
    if val_image_dir and val_mask_dir:
        val_pairs = _get_pairs(Path(val_image_dir), Path(val_mask_dir))
        val_ds    = _SegDataset([p[0] for p in val_pairs],
                                 [p[1] for p in val_pairs],
                                 size=size, augment=False)
        val_dl    = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=(device == "cuda"))

    model = LightweightUNet().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce   = nn.BCEWithLogitsLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"LightweightUNet: {n_params/1e6:.1f}M params")

    best_val_dice = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, masks, _ in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss   = bce(logits, masks)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        sched.step()
        avg_loss = total_loss / len(train_dl)

        val_dice = _eval_dice(model, val_dl, device) if val_dl else float("nan")
        history.append({"epoch": epoch, "loss": avg_loss, "val_dice": val_dice})
        print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | val_dice {val_dice:.4f}")

        if val_dl is None or val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_dice": val_dice,
                        "config": {"size": size}}, save_path)

    print(f"Best model saved → {save_path}  (val_dice={best_val_dice:.4f})")
    return {"best_val_dice": best_val_dice, "history": history}


def _eval_dice(model: LightweightUNet, dl: DataLoader, device: str) -> float:
    model.eval()
    dices = []
    with torch.no_grad():
        for imgs, masks, _ in dl:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = torch.sigmoid(model(imgs)) > 0.5
            tp = (preds & masks.bool()).float().sum(dim=(1, 2, 3))
            fp = (preds & ~masks.bool()).float().sum(dim=(1, 2, 3))
            fn = (~preds & masks.bool()).float().sum(dim=(1, 2, 3))
            d  = (2 * tp / (2 * tp + fp + fn + 1e-8)).cpu().tolist()
            dices.extend(d)
    return float(np.mean(dices))


def predict_masks(
    model_path: str | Path,
    image_dir:  str | Path,
    out_dir:    str | Path,
    *,
    device: str = "cuda",
    size:   int = 256,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt  = torch.load(model_path, map_location=device)
    model = LightweightUNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])

    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts)

    with torch.no_grad():
        for ip in tqdm(paths, desc="Predicting masks"):
            img  = tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(device)
            pred = torch.sigmoid(model(img)).squeeze().cpu().numpy()
            mask = Image.fromarray((pred > 0.5).astype(np.uint8) * 255)
            mask.save(out_dir / (ip.stem + ".png"))

    print(f"Predicted {len(paths)} masks → {out_dir}")


def load_seg_model(model_path: str | Path, device: str = "cuda") -> LightweightUNet:
    ckpt  = torch.load(model_path, map_location=device)
    model = LightweightUNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Split-aware training from a single data_dir (mirrors DermoscopyDataset)
# ---------------------------------------------------------------------------

def train_seg_model_from_data_dir(
    data_dir:  str | Path,
    save_path: str | Path,
    *,
    val_fraction: float = 0.1,
    seed:         int   = 42,
    epochs:       int   = 30,
    batch_size:   int   = 8,
    lr:           float = 1e-3,
    size:         int   = 256,
    device:       str   = "cuda",
) -> dict:
    """Train using the same train/val split logic as DermoscopyDataset."""
    root     = Path(data_dir)
    img_dir  = root / "images"
    mask_dir = root / "masks"

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    all_pairs = _get_pairs(img_dir, mask_dir)

    rng   = _np.random.default_rng(seed)
    idx   = rng.permutation(len(all_pairs)).tolist()
    n_val = max(1, int(len(all_pairs) * val_fraction))
    n_test = n_val

    train_pairs = [all_pairs[i] for i in idx[n_val + n_test:]]
    val_pairs   = [all_pairs[i] for i in idx[:n_val]]

    print(f"Split → train: {len(train_pairs)}  val: {len(val_pairs)}  "
          f"test (held-out): {n_test}")

    train_ds = _SegDataset([p[0] for p in train_pairs],
                            [p[1] for p in train_pairs], size=size, augment=True)
    val_ds   = _SegDataset([p[0] for p in val_pairs],
                            [p[1] for p in val_pairs],   size=size, augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    # Reuse core training loop
    return train_seg_model(
        image_dir=[p[0] for p in train_pairs],   # pass as lists (duck-typed below)
        mask_dir =[p[1] for p in train_pairs],
        save_path=save_path,
        epochs=epochs, batch_size=batch_size,
        lr=lr, size=size, device=device,
        _train_dl=train_dl, _val_dl=val_dl,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    # Option A: single data_dir (recommended — uses same split as DermoscopyDataset)
    t.add_argument("--data_dir",      default=None,
                   help="Root of processed data (images/ + masks/ subdirs). "
                        "Uses same 80/10/10 split as training.")
    # Option B: explicit dirs
    t.add_argument("--image_dir",     default=None)
    t.add_argument("--mask_dir",      default=None)
    t.add_argument("--val_image_dir", default=None)
    t.add_argument("--val_mask_dir",  default=None)

    t.add_argument("--save_path",  default="outputs/eval_seg/unet.pth")
    t.add_argument("--epochs",     type=int,   default=30)
    t.add_argument("--batch_size", type=int,   default=8)
    t.add_argument("--lr",         type=float, default=1e-3)
    t.add_argument("--size",       type=int,   default=256)
    t.add_argument("--device",     default="cuda")

    pr = sub.add_parser("predict")
    pr.add_argument("--model_path", required=True)
    pr.add_argument("--image_dir",  required=True)
    pr.add_argument("--out_dir",    required=True)
    pr.add_argument("--device",     default="cuda")
    pr.add_argument("--size",       type=int, default=256)

    args = p.parse_args()

    if args.cmd == "train":
        if args.data_dir:
            _train_from_data_dir_cli(args)
        elif args.image_dir and args.mask_dir:
            train_seg_model(
                image_dir=args.image_dir, mask_dir=args.mask_dir,
                save_path=args.save_path,
                val_image_dir=args.val_image_dir,
                val_mask_dir=args.val_mask_dir,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, size=args.size, device=args.device,
            )
        else:
            print("Provide --data_dir OR both --image_dir and --mask_dir")
            sys.exit(1)
    else:
        predict_masks(
            model_path=args.model_path, image_dir=args.image_dir,
            out_dir=args.out_dir, device=args.device, size=args.size,
        )


def _train_from_data_dir_cli(args):
    import numpy as np

    root     = Path(args.data_dir)
    img_dir  = root / "images"
    mask_dir = root / "masks"

    all_pairs = _get_pairs(img_dir, mask_dir)
    rng   = np.random.default_rng(42)
    idx   = rng.permutation(len(all_pairs)).tolist()
    n_val = max(1, int(len(all_pairs) * 0.1))
    n_test = n_val

    train_pairs = [all_pairs[i] for i in idx[n_val + n_test:]]
    val_pairs   = [all_pairs[i] for i in idx[:n_val]]

    print(f"Split → train: {len(train_pairs)}  val: {len(val_pairs)}  "
          f"test (held-out, not used here): {n_test}")

    train_ds = _SegDataset([p[0] for p in train_pairs],
                            [p[1] for p in train_pairs],
                            size=args.size, augment=True)
    val_ds   = _SegDataset([p[0] for p in val_pairs],
                            [p[1] for p in val_pairs],
                            size=args.size, augment=False)
    pin = args.device == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=pin)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = LightweightUNet().to(args.device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce   = nn.BCEWithLogitsLoss()

    print(f"LightweightUNet: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for imgs, masks, _ in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            imgs, masks = imgs.to(args.device), masks.to(args.device)
            loss = bce(model(imgs), masks)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        sched.step()
        val_dice = _eval_dice(model, val_dl, args.device)
        print(f"Epoch {epoch:3d} | loss {total/len(train_dl):.4f} | val_dice {val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_dice": val_dice, "config": {"size": args.size}},
                       save_path)

    print(f"Best model saved → {save_path}  (val_dice={best_dice:.4f})")


if __name__ == "__main__":
    _cli()
