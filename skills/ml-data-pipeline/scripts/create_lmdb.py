#!/usr/bin/env python3
"""Convert dataset to LMDB format for faster loading."""

from pathlib import Path
import pickle

import lmdb
from tqdm import tqdm


def create_lmdb(dataset, output_path: str, write_frequency: int = 5000):
    """Convert dataset to LMDB format.

    Args:
        dataset: PyTorch dataset with __getitem__ and __len__
        output_path: Directory to save LMDB database
        write_frequency: Commit transactions every N samples
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Estimate map size (adjust as needed)
    map_size = len(dataset) * 1024 * 1024  # 1MB per sample estimate

    env = lmdb.open(str(output_path), map_size=map_size)

    with env.begin(write=True) as txn:
        for idx in tqdm(range(len(dataset)), desc="Creating LMDB"):
            image, label = dataset[idx]
            sample = {"image": image, "label": label}
            txn.put(
                f"{idx}".encode(),
                pickle.dumps(sample),
            )

    env.close()
    print(f"LMDB created at {output_path}")


if __name__ == "__main__":
    import sys

    from torchvision import datasets, transforms

    if len(sys.argv) < 3:
        print("Usage: python create_lmdb.py <dataset_dir> <output_lmdb_dir>")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Example: Create LMDB from ImageFolder dataset
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    create_lmdb(dataset, output_dir)
