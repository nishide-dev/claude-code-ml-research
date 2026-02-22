# Image Classification Configuration Example

Complete Hydra configuration for CIFAR-10 image classification with ResNet.

## Directory Structure

```text
configs/
├── config.yaml                    # Main config
├── model/
│   └── resnet18.yaml             # Model config
├── data/
│   └── cifar10.yaml              # Data config
├── trainer/
│   └── gpu_single.yaml           # Trainer config
├── logger/
│   └── wandb.yaml                # Logger config
└── experiment/
    └── resnet18_cifar10.yaml     # Complete experiment
```

## Main Config (configs/config.yaml)

```yaml
defaults:
  - model: resnet18
  - data: cifar10
  - trainer: gpu_single
  - logger: wandb
  - _self_

# Seed for reproducibility
seed: 42

# Paths
paths:
  root_dir: ${oc.env:PROJECT_ROOT,.}
  data_dir: ${paths.root_dir}/data
  log_dir: ${paths.root_dir}/logs
  output_dir: ${hydra:runtime.output_dir}

# Task name
task_name: "image_classification"

# Hydra output
hydra:
  run:
    dir: ${paths.log_dir}/runs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: ${paths.log_dir}/multiruns/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
```

## Model Config (configs/model/resnet18.yaml)

```yaml
_target_: src.models.resnet.ResNet18LightningModule

# Architecture
num_classes: 10
pretrained: false

# Training
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 0.0001
  betas: [0.9, 0.999]
  eps: 1e-8

scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 100
  eta_min: 1e-6

# Loss function
loss_fn:
  _target_: torch.nn.CrossEntropyLoss
  label_smoothing: 0.1

# Metrics
metrics:
  - accuracy
  - top5_accuracy
  - f1_score
```

## Data Config (configs/data/cifar10.yaml)

```yaml
_target_: src.data.cifar10_datamodule.CIFAR10DataModule

# Dataset
data_dir: ${paths.data_dir}/cifar10
download: true

# DataLoader
batch_size: 128
num_workers: 4
pin_memory: true
persistent_workers: true
prefetch_factor: 2
drop_last: false

# Splits
train_val_split: 0.9
shuffle_train: true
shuffle_val: false

# Augmentation (training)
train_transforms:
  random_crop:
    size: 32
    padding: 4
  random_horizontal_flip:
    p: 0.5
  normalize:
    mean: [0.4914, 0.4822, 0.4465]
    std: [0.2470, 0.2435, 0.2616]
  random_erasing:
    p: 0.1
    scale: [0.02, 0.33]

# Transforms (validation/test)
val_transforms:
  normalize:
    mean: [0.4914, 0.4822, 0.4465]
    std: [0.2470, 0.2435, 0.2616]
```

## Trainer Config (configs/trainer/gpu_single.yaml)

```yaml
_target_: pytorch_lightning.Trainer

# Training duration
max_epochs: 100
min_epochs: 1

# Hardware
accelerator: gpu
devices: 1
precision: 16-mixed

# Distributed
strategy: auto
sync_batchnorm: false

# Optimization
gradient_clip_val: 1.0
gradient_clip_algorithm: norm
accumulate_grad_batches: 1

# Validation
check_val_every_n_epoch: 1
val_check_interval: 1.0
num_sanity_val_steps: 2

# Logging
log_every_n_steps: 10
enable_progress_bar: true
enable_model_summary: true

# Checkpointing
enable_checkpointing: true
default_root_dir: ${paths.log_dir}

# Reproducibility
deterministic: false  # Set to true for full reproducibility (slower)
benchmark: true  # Enable cudnn.benchmark for speed

# Callbacks
callbacks:
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    dirpath: ${paths.output_dir}/checkpoints
    filename: "epoch_{epoch:03d}"
    monitor: "val/acc"
    mode: "max"
    save_top_k: 3
    save_last: true
    save_on_train_epoch_end: false
    auto_insert_metric_name: false
    every_n_epochs: 1

  - _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: "val/acc"
    patience: 15
    mode: "max"
    min_delta: 0.001
    check_on_train_epoch_end: false

  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: "step"
    log_momentum: false

  - _target_: pytorch_lightning.callbacks.RichProgressBar
    refresh_rate: 1
    leave: true
```

## Logger Config (configs/logger/wandb.yaml)

```yaml
_target_: pytorch_lightning.loggers.WandbLogger

# Project
project: "cifar10-classification"
name: null  # Auto-generated from config
save_dir: ${paths.log_dir}

# Logging
log_model: true  # Log checkpoints to W&B
save_code: true  # Save code snapshot
offline: false

# Additional settings
tags: ["resnet18", "cifar10", "baseline"]
notes: "Baseline ResNet-18 on CIFAR-10"
```

## Experiment Config (configs/experiment/resnet18_cifar10.yaml)

```yaml
# @package _global_

# Experiment metadata
name: "resnet18_cifar10_baseline"
description: "Baseline ResNet-18 on CIFAR-10 with standard augmentation"
tags: ["baseline", "resnet18", "cifar10"]

# Compose from existing configs
defaults:
  - override /model: resnet18
  - override /data: cifar10
  - override /trainer: gpu_single
  - override /logger: wandb

# Seed
seed: 42

# Model overrides
model:
  optimizer:
    lr: 0.001
    weight_decay: 0.0001
  scheduler:
    T_max: 100

# Data overrides
data:
  batch_size: 128
  num_workers: 4

# Trainer overrides
trainer:
  max_epochs: 100
  precision: 16-mixed

# Logger overrides
logger:
  wandb:
    project: "cifar10-experiments"
    tags: ${tags}
    notes: ${description}
```

## Usage

### Basic Training

```bash
# Use default config
python src/train.py

# Use specific experiment
python src/train.py experiment=resnet18_cifar10

# Override parameters
python src/train.py model.optimizer.lr=0.01 data.batch_size=256
```

### Hyperparameter Sweep

```bash
# Grid search over learning rates and batch sizes
python src/train.py --multirun \
  model.optimizer.lr=0.0001,0.001,0.01 \
  data.batch_size=64,128,256

# Random search with different architectures
python src/train.py --multirun \
  model=resnet18,resnet34,resnet50 \
  model.optimizer.lr=choice(0.0001,0.001,0.01)
```

### Multi-GPU Training

```bash
# Use 4 GPUs with DDP
python src/train.py trainer=gpu_multi trainer.devices=4

# With experiment config
python src/train.py experiment=resnet18_cifar10 trainer.devices=4
```

## Implementation

### LightningModule (src/models/resnet.py)

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics import Accuracy, F1Score


class ResNet18LightningModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = False,
        optimizer: dict = None,
        scheduler: dict = None,
        loss_fn: dict = None,
        metrics: list[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Loss
        self.criterion = self._build_loss(loss_fn or {})

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def _build_loss(self, loss_config: dict) -> nn.Module:
        if "_target_" in loss_config:
            import hydra
            return hydra.utils.instantiate(loss_config)
        return nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Metrics
        acc = self.train_acc(logits, y)

        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Metrics
        acc = self.val_acc(logits, y)

        # Logging
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        import hydra

        # Optimizer
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer,
            params=self.parameters(),
        )

        # Scheduler
        if self.hparams.scheduler is None:
            return optimizer

        scheduler = hydra.utils.instantiate(
            self.hparams.scheduler,
            optimizer=optimizer,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
```

### DataModule (src/data/cifar10_datamodule.py)

```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        train_val_split: float = 0.9,
        train_transforms: dict = None,
        val_transforms: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str = None):
        # Transforms
        train_transform = self._build_transforms(self.hparams.train_transforms)
        val_transform = self._build_transforms(self.hparams.val_transforms)

        if stage == "fit" or stage is None:
            full_dataset = CIFAR10(
                self.hparams.data_dir,
                train=True,
                download=True,
                transform=train_transform,
            )

            # Split
            train_size = int(len(full_dataset) * self.hparams.train_val_split)
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )

        if stage == "test" or stage is None:
            self.test_dataset = CIFAR10(
                self.hparams.data_dir,
                train=False,
                download=True,
                transform=val_transform,
            )

    def _build_transforms(self, transform_config: dict):
        transform_list = []

        if transform_config is None:
            transform_config = {}

        # Random crop
        if "random_crop" in transform_config:
            cfg = transform_config["random_crop"]
            transform_list.append(
                transforms.RandomCrop(cfg["size"], padding=cfg.get("padding", 0))
            )

        # Horizontal flip
        if "random_horizontal_flip" in transform_config:
            p = transform_config["random_horizontal_flip"].get("p", 0.5)
            transform_list.append(transforms.RandomHorizontalFlip(p=p))

        # To tensor
        transform_list.append(transforms.ToTensor())

        # Normalize
        if "normalize" in transform_config:
            cfg = transform_config["normalize"]
            transform_list.append(
                transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
            )

        # Random erasing
        if "random_erasing" in transform_config:
            cfg = transform_config["random_erasing"]
            transform_list.append(
                transforms.RandomErasing(
                    p=cfg.get("p", 0.5),
                    scale=cfg.get("scale", (0.02, 0.33)),
                )
            )

        return transforms.Compose(transform_list)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
```

## Expected Results

- **Training time**: ~10 minutes per epoch on single GPU (V100)
- **Validation accuracy**: ~92-94% after 100 epochs
- **Test accuracy**: ~91-93%
- **Model size**: ~45MB
- **GPU memory**: ~2GB with batch_size=128

## Troubleshooting

**OOM errors:**

- Reduce batch_size: `data.batch_size=64`
- Use gradient accumulation: `trainer.accumulate_grad_batches=2`
- Reduce num_workers: `data.num_workers=2`

**Slow training:**

- Increase num_workers: `data.num_workers=8`
- Enable benchmark: `trainer.benchmark=true`
- Use larger batch size: `data.batch_size=256`

**Poor accuracy:**

- Increase epochs: `trainer.max_epochs=200`
- Adjust learning rate: `model.optimizer.lr=0.01`
- Enable label smoothing: `model.loss_fn.label_smoothing=0.1`
