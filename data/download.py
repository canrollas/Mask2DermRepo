"""
Dataset download helpers for Mask2Derm.

Supported datasets:
  - ISIC 2018 Task 1 (segmentation): via ISIC Archive API
  - HAM10000: via Kaggle API (requires ~/.kaggle/kaggle.json)

After downloading, run `python data/download.py --prepare` to resize and
organize into data/processed/{images,masks,metadata.csv}.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, urlopen

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# ISIC 2018 Task 1
# ---------------------------------------------------------------------------

ISIC_API = "https://api.isic-archive.com/api/v2"


def _isic_fetch_json(url: str) -> dict:
    with urlopen(url) as resp:
        return json.loads(resp.read().decode())


def download_isic2018(limit: int | None = None, out_dir: Path = RAW_DIR / "isic2018") -> None:
    """Download ISIC 2018 Task 1 images and masks via the ISIC REST API.

    Args:
        limit:   Maximum number of images to download (None = all ~2600).
        out_dir: Target directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    img_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    print("Fetching ISIC 2018 Task 1 image list…")
    page_size = 100
    offset = 0
    all_images = []

    while True:
        url = f"{ISIC_API}/images/?limit={page_size}&offset={offset}&collections=81"
        data = _isic_fetch_json(url)
        results = data.get("results", [])
        all_images.extend(results)
        if limit and len(all_images) >= limit:
            all_images = all_images[:limit]
            break
        if not data.get("next"):
            break
        offset += page_size

    print(f"Found {len(all_images)} images. Downloading…")
    for item in tqdm(all_images, unit="img"):
        isic_id = item["isic_id"]
        img_path = img_dir / f"{isic_id}.jpg"
        mask_path = mask_dir / f"{isic_id}_segmentation.png"

        if not img_path.exists():
            img_url = f"https://isic-archive.com/api/v1/image/{isic_id}/download"
            try:
                urlretrieve(img_url, img_path)
            except Exception as e:
                print(f"  ✗ image {isic_id}: {e}")

        if not mask_path.exists():
            seg_url = f"https://isic-archive.com/api/v1/segmentation?imageId={isic_id}&limit=1"
            try:
                seg_data = _isic_fetch_json(seg_url)
                if seg_data.get("results"):
                    seg_id = seg_data["results"][0]["_id"]
                    mask_url = f"https://isic-archive.com/api/v1/segmentation/{seg_id}/mask"
                    urlretrieve(mask_url, mask_path)
            except Exception as e:
                print(f"  ✗ mask {isic_id}: {e}")

    print(f"ISIC 2018 download complete → {out_dir}")


# ---------------------------------------------------------------------------
# HAM10000 via Kaggle
# ---------------------------------------------------------------------------

def download_ham10000(out_dir: Path = RAW_DIR / "ham10000") -> None:
    """Download HAM10000 via the Kaggle API.

    Requires:
        - kaggle Python package: pip install kaggle
        - ~/.kaggle/kaggle.json with your API credentials
          (download from https://www.kaggle.com/settings)
    """
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise ImportError(
            "Install the kaggle package: pip install kaggle\n"
            "And place your API key at ~/.kaggle/kaggle.json"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading HAM10000 from Kaggle…")
    os.system(
        f"kaggle datasets download -d surajghuwalewala/ham1000-segmentation-and-classification "
        f"-p {out_dir} --unzip"
    )
    print(f"HAM10000 download complete → {out_dir}")


# ---------------------------------------------------------------------------
# Prepare: organize into data/processed/
# ---------------------------------------------------------------------------

def _resize_and_save(src: Path, dst: Path, size: int, is_mask: bool = False) -> None:
    img = Image.open(src)
    if is_mask:
        img = img.convert("L")
        img = img.resize((size, size), Image.NEAREST)
        # Binarise
        arr = (np.asarray(img) > 127).astype("uint8") * 255
        Image.fromarray(arr).save(dst)
    else:
        img = img.convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        img.save(dst, quality=95)


def prepare_dataset(
    size: int = 256,
    include_isic: bool = True,
    include_ham: bool = True,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    """Merge and resize downloaded datasets into a unified processed directory.

    Output layout:
        data/processed/
        ├── images/     *.jpg
        ├── masks/      *_segmentation.png  (binary, 0 or 255)
        └── metadata.csv
    """
    import numpy as np  # noqa: F401

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)

    records = []

    # --- ISIC 2018 ---
    if include_isic:
        isic_raw = RAW_DIR / "isic2018"
        imgs = sorted((isic_raw / "images").glob("*.jpg"))
        print(f"Processing {len(imgs)} ISIC 2018 images…")
        for img_path in tqdm(imgs, unit="img"):
            stem = img_path.stem
            mask_src = isic_raw / "masks" / f"{stem}_segmentation.png"
            if not mask_src.exists():
                continue
            dst_img = out_dir / "images" / f"isic_{stem}.jpg"
            dst_mask = out_dir / "masks" / f"isic_{stem}_segmentation.png"
            if not dst_img.exists():
                _resize_and_save(img_path, dst_img, size, is_mask=False)
            if not dst_mask.exists():
                _resize_and_save(mask_src, dst_mask, size, is_mask=True)
            records.append({"image_id": f"isic_{stem}", "source": "isic2018",
                            "benign_malignant": "unknown"})

    # --- HAM10000 ---
    if include_ham:
        ham_raw = RAW_DIR / "ham10000"
        # Support both HAM10000_metadata.csv (kmader) and GroundTruth.csv (surajghuwalewala)
        meta_files = list(ham_raw.rglob("HAM10000_metadata.csv")) or list(ham_raw.rglob("GroundTruth.csv"))
        if not meta_files:
            print("⚠ HAM10000 metadata CSV not found, skipping.")
        else:
            meta = pd.read_csv(meta_files[0])
            malignant_dx = {"mel", "bcc", "scc", "akiec", "MEL", "BCC", "AKIEC"}
            # GroundTruth.csv uses one-hot columns; convert to dx string
            if "dx" not in meta.columns and "image" in meta.columns:
                dx_cols = [c for c in meta.columns if c != "image"]
                meta["image_id"] = meta["image"]
                meta["dx"] = meta[dx_cols].idxmax(axis=1).str.lower()
            malignant_dx_lower = {"mel", "bcc", "scc", "akiec"}
            meta["benign_malignant"] = meta["dx"].apply(
                lambda d: "malignant" if str(d).lower() in malignant_dx_lower else "benign"
            )
            # Find image dirs
            img_dirs = [p for p in ham_raw.rglob("*") if p.is_dir()
                        and any(p.glob("*.jpg"))]
            print(f"Processing {len(meta)} HAM10000 entries…")
            for _, row in tqdm(meta.iterrows(), total=len(meta), unit="img"):
                img_id = row["image_id"]
                # Find image in any sub-directory
                img_src = None
                for d in img_dirs:
                    candidate = d / f"{img_id}.jpg"
                    if candidate.exists():
                        img_src = candidate
                        break
                if img_src is None:
                    continue
                mask_src = ham_raw / "masks" / f"{img_id}_segmentation.png"
                if not mask_src.exists():
                    continue
                dst_img = out_dir / "images" / f"ham_{img_id}.jpg"
                dst_mask = out_dir / "masks" / f"ham_{img_id}_segmentation.png"
                if not dst_img.exists():
                    _resize_and_save(img_src, dst_img, size, is_mask=False)
                if not dst_mask.exists():
                    _resize_and_save(mask_src, dst_mask, size, is_mask=True)
                records.append({
                    "image_id": f"ham_{img_id}",
                    "source": "ham10000",
                    "dx": row.get("dx", ""),
                    "benign_malignant": row["benign_malignant"],
                })

    # Write metadata CSV
    if records:
        df = pd.DataFrame(records)
        df.to_csv(out_dir / "metadata.csv", index=False)
        print(f"\nDataset prepared: {len(df)} samples → {out_dir}")
        print(df["benign_malignant"].value_counts().to_string())
    else:
        print("⚠ No samples processed. Check that raw data was downloaded first.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask2Derm dataset download & preparation")
    parser.add_argument("--download-isic", action="store_true", help="Download ISIC 2018 Task 1")
    parser.add_argument("--download-ham", action="store_true", help="Download HAM10000 (needs Kaggle API)")
    parser.add_argument("--prepare", action="store_true", help="Merge & resize into data/processed/")
    parser.add_argument("--size", type=int, default=256, help="Target resolution (default 256)")
    parser.add_argument("--isic-limit", type=int, default=None, help="Limit ISIC downloads")
    args = parser.parse_args()

    if args.download_isic:
        download_isic2018(limit=args.isic_limit)
    if args.download_ham:
        download_ham10000()
    if args.prepare:
        prepare_dataset(size=args.size)
    if not any([args.download_isic, args.download_ham, args.prepare]):
        parser.print_help()
