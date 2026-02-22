# Image Classification Training Example

Complete example for training an image classification model with PyTorch Lightning.

## Setup

```bash
# Install dependencies
uv add torch torchvision pytorch-lightning hydra-core

# Download dataset (using CIFAR-10 as example)
python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='data', download=True)"
```

## Project Structure

```text
project/
├── configs/
│   ├── config.yaml
│   ├── model/
│   │   └── resnet.yaml
│   ├── data/
│   │   └── cifar10.yaml
│   └── experiment/
│       └── cifar10_resnet.yaml
├── src/
│   ├── models/
│   │   └── image_classifier.py
│   ├── data/
│   │   └── image_datamodule.py
│   └── train.py
└── data/
```

## Model Implementation

**`src/models/image_classifier.py`:**

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics import Accuracy


class ImageClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained model
        self.model = getattr(models, model_name)(pretrained=True)

        # Replace final layer for our number of classes
        if "resnet" in model_name:
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)

        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log metrics
        self.train_acc(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log metrics
        self.val_acc(logits, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
```

## DataModule Implementation

**`src/data/image_datamodule.py`:**

```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 128,
        num_workers: int = 4,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        # Image transformations
        self.train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def prepare_data(self):
        # Download dataset
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Load training data
            full_train = CIFAR10(
                self.data_dir, train=True, transform=self.train_transforms
            )

            # Split into train and validation
            val_size = int(len(full_train) * self.val_split)
            train_size = len(full_train) - val_size
            self.train_dataset, self.val_dataset = random_split(
                full_train, [train_size, val_size]
            )

        if stage == "test" or stage is None:
            self.test_dataset = CIFAR10(
                self.data_dir, train=False, transform=self.test_transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
```

## Configuration

**`configs/experiment/cifar10_resnet.yaml`:**

```yaml
# @package _global_

defaults:
  - override /model: resnet
  - override /data: cifar10
  - override /trainer: default
  - override /logger: wandb

# Experiment name
experiment_name: cifar10_resnet18

# Model settings
model:
  model_name: resnet18
  num_classes: 10
  learning_rate: 1e-3
  weight_decay: 1e-4

# Data settings
data:
  data_dir: data
  batch_size: 128
  num_workers: 4
  val_split: 0.1

# Training settings
trainer:
  max_epochs: 100
  accelerator: gpu
  devices: 1
  precision: 16-mixed

# Logging
logger:
  wandb:
    project: image-classification
    tags: ["cifar10", "resnet18"]
```

## Training Commands

**Basic training:**

```bash
python src/train.py experiment=cifar10_resnet
```

**With hyperparameter overrides:**

```bash
python src/train.py experiment=cifar10_resnet \
  model.learning_rate=5e-4 \
  data.batch_size=256 \
  trainer.max_epochs=50
```

**Multi-GPU training:**

```bash
python src/train.py experiment=cifar10_resnet \
  trainer.devices=4 \
  trainer.strategy=ddp
```

**Hyperparameter search:**

```bash
python src/train.py experiment=cifar10_resnet --multirun \
  model.learning_rate=1e-4,5e-4,1e-3 \
  model.weight_decay=1e-5,1e-4,1e-3
```

## Expected Results

**CIFAR-10 with ResNet-18:**

- Training accuracy: ~95%
- Validation accuracy: ~93%
- Training time: ~10 minutes on single GPU (V100)

**Metrics to monitor:**

- Training/validation loss should decrease steadily
- Training/validation accuracy should increase
- No large gap between train and val metrics (overfitting)

## Common Issues

**Low accuracy:**

- Check data augmentation is applied correctly
- Verify normalization values match ImageNet statistics
- Try different learning rates (1e-4 to 1e-3)

**Overfitting:**

- Add dropout to model
- Increase weight decay
- Add more data augmentation
- Use early stopping

**Slow training:**

- Increase num_workers for data loading
- Use mixed precision (16-mixed)
- Increase batch size if GPU memory allows
- Enable persistent_workers=True
