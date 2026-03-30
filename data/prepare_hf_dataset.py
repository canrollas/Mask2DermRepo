"""
Push the processed Mask2Derm dataset to HuggingFace Hub.

Usage:
    python data/prepare_hf_dataset.py \
        --data_dir data/processed \
        --repo_id your-hf-username/mask2derm-dataset \
        --private          # omit to make public
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, Image as HFImage, Features, Value
from huggingface_hub import login


def build_hf_dataset(data_dir: Path, val_fraction: float = 0.1, seed: int = 42) -> DatasetDict:
    """Build a HuggingFace DatasetDict from data/processed/."""
    meta_path = data_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at {meta_path}. Run download.py --prepare first.")

    df = pd.read_csv(meta_path)

    records = []
    for _, row in df.iterrows():
        img_path = data_dir / "images" / f"{row['image_id']}.jpg"
        mask_path = data_dir / "masks" / f"{row['image_id']}_segmentation.png"
        if img_path.exists() and mask_path.exists():
            records.append({
                "image_id": str(row["image_id"]),
                "image": str(img_path),
                "mask": str(mask_path),
                "source": str(row.get("source", "")),
                "benign_malignant": str(row.get("benign_malignant", "unknown")),
                "dx": str(row.get("dx", "")),
            })

    full_ds = Dataset.from_list(records).cast_column("image", HFImage()).cast_column("mask", HFImage())

    # Train / val split
    split = full_ds.train_test_split(test_size=val_fraction, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Push Mask2Derm dataset to HuggingFace Hub")
    parser.add_argument("--data_dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--repo_id", required=True, help="HF repo id: username/dataset-name")
    parser.add_argument("--private", action="store_true", help="Create private HF repo")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    if args.token:
        login(token=args.token)
    else:
        login()  # Interactive login / reads HF_TOKEN env var

    print(f"Building dataset from {args.data_dir}…")
    ds = build_hf_dataset(Path(args.data_dir))
    print(ds)

    print(f"Pushing to {args.repo_id} (private={args.private})…")
    ds.push_to_hub(args.repo_id, private=args.private)
    print("Done.")


if __name__ == "__main__":
    main()
