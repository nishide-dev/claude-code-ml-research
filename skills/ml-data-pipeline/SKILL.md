---
name: ml-data-pipeline
description: Create and manage data loading, preprocessing, and augmentation pipelines (DataModule, transforms, data loaders). Use when implementing DataModules, setting up data loaders, or optimizing data pipelines for computer vision, NLP, or graph ML tasks.
disable-model-invocation: true
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

Generate PyTorch Lightning DataModule based on task type.

See `templates/vision_datamodule.py` for computer vision implementation.

See `templates/graph_datamodule.py` for graph ML implementation.

**Basic DataModule Structure:**

```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = ...
            self.val_dataset = ...

        if stage == "test" or stage is None:
            self.test_dataset = ...

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
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

See `scripts/preprocess_data.py` for preprocessing implementation.

**Key preprocessing steps:**

1. Load raw data
2. Apply transforms (resize, normalize, etc.)
3. Save processed data
4. Verify integrity

### 5. Data Validation

See `scripts/validate_data.py` for validation implementation.

**Validation checks:**

1. Directory structure
2. Number of samples per split
3. Class balance
4. Image properties (sizes, formats)
5. Corrupted files detection

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

See `scripts/create_lmdb.py` for LMDB creation.

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
