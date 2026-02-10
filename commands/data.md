---
description: Create and manage data loading, preprocessing, and augmentation pipelines (DataModule, transforms, data loaders)
---

# ML Data Pipeline Management

Create and manage data loading, preprocessing, and augmentation pipelines for machine learning projects.

## Process

### 1. Data Pipeline Type Selection

Ask user about data type and task:

- **Computer Vision**: Image classification, object detection, segmentation
- **NLP**: Text classification, generation, translation
- **Graph ML**: Node classification, graph classification, link prediction
- **Tabular**: Structured data, regression, classification
- **Time Series**: Forecasting, anomaly detection
- **Multimodal**: Vision + text, audio + visual

### 2. Create DataModule

Generate PyTorch Lightning DataModule based on task type:

**Computer Vision DataModule:**

```python
# src/data/vision_datamodule.py
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class VisionDataModule(pl.LightningDataModule):
    """DataModule for computer vision tasks."""

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 128,
        num_workers: int = 4,
        image_size: int = 224,
        train_val_test_split: tuple[float, float, float] = (0.8, 0.1, 0.1),
        augment: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_val_test_split = train_val_test_split
        self.augment = augment

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        # Training transforms with augmentation
        train_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]) if self.augment else self._base_transforms()

        # Validation/test transforms (no augmentation)
        eval_transforms = self._base_transforms()

        if stage == "fit" or stage is None:
            full_dataset = datasets.ImageFolder(
                root=f"{self.data_dir}/train",
                transform=train_transforms
            )

            # Split into train/val
            train_size = int(self.train_val_test_split[0] * len(full_dataset))
            val_size = len(full_dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )

            # Apply eval transforms to val
            self.val_dataset.dataset.transform = eval_transforms

        if stage == "test" or stage is None:
            self.test_dataset = datasets.ImageFolder(
                root=f"{self.data_dir}/test",
                transform=eval_transforms
            )

    def _base_transforms(self):
        """Base transforms without augmentation."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
```

**Graph ML DataModule (PyTorch Geometric):**

```python
# src/data/graph_datamodule.py
import pytorch_lightning as pl
from torch_geometric.data import LightningDataset
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures


class GraphDataModule(pl.LightningDataModule):
    """DataModule for graph neural networks."""

    def __init__(
        self,
        dataset_name: str = "Cora",
        data_dir: str = "data/graphs/",
        batch_size: int = 32,
        num_workers: int = 4,
        use_sampling: bool = False,
        num_neighbors: list[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_sampling = use_sampling
        self.num_neighbors = num_neighbors or [15, 10, 5]

    def setup(self, stage: Optional[str] = None):
        """Load graph dataset."""
        # Node classification datasets (e.g., Cora, PubMed)
        if self.dataset_name in ["Cora", "CiteSeer", "PubMed"]:
            self.dataset = Planetoid(
                root=self.data_dir,
                name=self.dataset_name,
                transform=NormalizeFeatures(),
            )
            self.data = self.dataset[0]  # Single graph
            self.task_type = "node_classification"

        # Graph classification datasets (e.g., PROTEINS, ENZYMES)
        elif self.dataset_name in ["PROTEINS", "ENZYMES", "MUTAG"]:
            self.dataset = TUDataset(
                root=self.data_dir,
                name=self.dataset_name,
                transform=NormalizeFeatures(),
            )
            self.task_type = "graph_classification"

            # Split dataset
            train_size = int(0.8 * len(self.dataset))
            val_size = int(0.1 * len(self.dataset))
            test_size = len(self.dataset) - train_size - val_size

            self.train_dataset, self.val_dataset, self.test_dataset = \
                random_split(self.dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        if self.task_type == "node_classification":
            # For node classification, typically use full graph
            return DataLoader([self.data], batch_size=1)
        else:
            # For graph classification, batch multiple graphs
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

    def val_dataloader(self):
        if self.task_type == "node_classification":
            return DataLoader([self.data], batch_size=1)
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )

    def test_dataloader(self):
        if self.task_type == "node_classification":
            return DataLoader([self.data], batch_size=1)
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
```

### 3. Data Augmentation Strategies

**Computer Vision Augmentations:**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Heavy augmentation for small datasets
heavy_aug = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(p=0.2),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# MixUp / CutMix (in training_step)
def training_step(self, batch, batch_idx):
    x, y = batch

    # MixUp
    if random.random() < 0.5:
        lam = np.random.beta(1.0, 1.0)
        idx = torch.randperm(x.size(0))
        x = lam * x + (1 - lam) * x[idx]
        y = lam * y + (1 - lam) * y[idx]

    logits = self(x)
    loss = self.criterion(logits, y)
    return loss
```

**Graph Augmentations:**

```python
from torch_geometric.transforms import (
    RandomNodeSplit,
    AddRandomEdge,
    RemoveRandomEdge,
    Compose as PyGCompose,
)

graph_aug = PyGCompose([
    AddRandomEdge(p=0.1),
    RemoveRandomEdge(p=0.1),
])
```

### 4. Data Preprocessing Pipeline

**Create Preprocessing Script:**

```python
# scripts/preprocess_data.py
import argparse
from pathlib import Path
from typing import Callable
import torch
from torch.utils.data import Dataset
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
    from concurrent.futures import ThreadPoolExecutor

    def process_file(file_path: Path):
        # Load and transform
        data = transform(file_path)
        # Save processed
        out_path = processed_dir / f"{file_path.stem}.pt"
        torch.save(data, out_path)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(
            executor.map(process_file, raw_files),
            total=len(raw_files),
            desc="Processing"
        ))

    print(f"Preprocessed data saved to {processed_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # Define your transform
    def transform(file_path: Path):
        from PIL import Image
        import torchvision.transforms as T

        img = Image.open(file_path).convert("RGB")
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transform(img)

    preprocess_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        transform=transform,
        num_workers=args.num_workers,
    )
```

### 5. Data Validation

**Create Data Validation Script:**

```python
# scripts/validate_data.py
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def validate_dataset(data_dir: Path):
    """Validate dataset for common issues."""
    print("=" * 50)
    print("DATA VALIDATION REPORT")
    print("=" * 50)

    issues = []

    # 1. Check directory structure
    required_dirs = ["train", "val", "test"]
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            issues.append(f"❌ Missing directory: {dir_name}")
        else:
            print(f"✅ Found {dir_name} directory")

    # 2. Check number of samples
    for split in required_dirs:
        split_dir = data_dir / split
        if split_dir.exists():
            num_samples = len(list(split_dir.glob("**/*.jpg")))
            print(f"   - {split}: {num_samples} samples")
            if num_samples == 0:
                issues.append(f"❌ No samples in {split}")

    # 3. Check class balance
    print("\nClass Distribution:")
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
            class_counts = {
                cls: len(list((split_dir / cls).glob("*.jpg")))
                for cls in classes
            }
            print(f"\n{split}:")
            for cls, count in class_counts.items():
                print(f"  {cls}: {count}")

            # Check imbalance
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            if max_count / min_count > 5:
                issues.append(f"⚠️  High class imbalance in {split} (ratio: {max_count/min_count:.1f}:1)")

    # 4. Check image properties
    print("\nChecking image properties...")
    sample_images = list((data_dir / "train").glob("**/*.jpg"))[:100]

    sizes = []
    for img_path in sample_images:
        from PIL import Image
        img = Image.open(img_path)
        sizes.append(img.size)

    # Analyze sizes
    widths, heights = zip(*sizes)
    print(f"Image sizes - Width: [{min(widths)}, {max(widths)}], Height: [{min(heights)}, {max(heights)}]")

    if len(set(sizes)) > 10:
        issues.append("⚠️  High variance in image sizes - consider resizing")

    # 5. Check for corrupted files
    print("\nChecking for corrupted files...")
    corrupted = []
    for img_path in sample_images:
        try:
            from PIL import Image
            img = Image.open(img_path)
            img.verify()
        except Exception as e:
            corrupted.append(str(img_path))
            issues.append(f"❌ Corrupted file: {img_path}")

    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    if not issues:
        print("✅ All checks passed!")
    else:
        print(f"Found {len(issues)} issues:")
        for issue in issues:
            print(issue)

    return len(issues) == 0


if __name__ == "__main__":
    import sys
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/")
    validate_dataset(data_dir)
```

### 6. Efficient Data Loading

**Use LMDB for Fast Loading:**

```python
import lmdb
import pickle
from torch.utils.data import Dataset


class LMDBDataset(Dataset):
    """Fast dataset using LMDB."""

    def __init__(self, lmdb_path: str):
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin() as txn:
            self.length = txn.stat()["entries"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = txn.get(f"{idx}".encode())
            sample = pickle.loads(data)
        return sample["image"], sample["label"]
```

**Create LMDB from Dataset:**

```python
# scripts/create_lmdb.py
import lmdb
import pickle
from pathlib import Path
from tqdm import tqdm


def create_lmdb(dataset, output_path: str, write_frequency: int = 5000):
    """Convert dataset to LMDB format."""
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
```

### 7. Data Configuration

Generate Hydra config for data:

```yaml
# configs/data/custom.yaml
_target_: src.data.custom_datamodule.CustomDataModule

# Paths
data_dir: "data/"
processed_dir: "data/processed/"

# DataLoader settings
batch_size: 128
num_workers: 4
pin_memory: true
persistent_workers: true

# Splits
train_val_test_split: [0.8, 0.1, 0.1]

# Preprocessing
image_size: 224
normalize: true
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]

# Augmentation
augment_train: true
augmentation:
  horizontal_flip: true
  random_crop: true
  color_jitter: true
  rotation_degrees: 15

# Caching
use_lmdb: false  # Set to true for faster loading
cache_in_memory: false  # Set to true if dataset fits in RAM
```

### 8. Data Analysis Tools

**Generate Data Report:**

```bash
python scripts/analyze_data.py data/ --output data_report.html
```

**Visualize Data Distribution:**

```python
# In notebook or script
import matplotlib.pyplot as plt
import seaborn as sns

dm = MyDataModule()
dm.setup()

# Sample from dataloader
batch = next(iter(dm.train_dataloader()))
images, labels = batch

# Plot grid of samples
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].permute(1, 2, 0))
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis("off")
plt.savefig("data_samples.png")

# Plot label distribution
label_counts = pd.Series([labels[i].item() for i in range(len(labels))]).value_counts()
plt.figure(figsize=(10, 6))
label_counts.plot(kind="bar")
plt.title("Label Distribution")
plt.savefig("label_distribution.png")
```

## Success Criteria

- [ ] DataModule created and tested
- [ ] Data loads without errors
- [ ] Shapes are correct
- [ ] No corrupted files
- [ ] Reasonable class balance
- [ ] Augmentation works correctly
- [ ] DataLoader performance is good (high GPU utilization)
- [ ] Data validation passes

Data pipeline is ready for training!
