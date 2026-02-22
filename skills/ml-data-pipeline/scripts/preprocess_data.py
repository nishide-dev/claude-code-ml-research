#!/usr/bin/env python3
"""Preprocess dataset and save to disk."""

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
import torch
from tqdm import tqdm


def preprocess_dataset(
    raw_dir: Path,
    processed_dir: Path,
    transform: Callable,
    num_workers: int = 4,
):
    """Preprocess dataset and save to disk."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    raw_files = list(raw_dir.glob("**/*.jpg"))
    print(f"Found {len(raw_files)} images")

    # Process in parallel
    def process_file(file_path: Path):
        # Load and transform
        data = transform(file_path)
        # Save processed
        out_path = processed_dir / f"{file_path.stem}.pt"
        torch.save(data, out_path)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_file, raw_files), total=len(raw_files), desc="Processing"))

    print(f"Preprocessed data saved to {processed_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # Define your transform
    def transform(file_path: Path):
        from torchvision import transforms as T

        img = Image.open(file_path).convert("RGB")
        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return transform(img)

    preprocess_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        transform=transform,
        num_workers=args.num_workers,
    )
