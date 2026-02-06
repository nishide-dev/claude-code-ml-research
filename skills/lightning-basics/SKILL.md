# PyTorch Lightning Basics

Comprehensive guide to PyTorch Lightning for ML research and experimentation.

## Overview

PyTorch Lightning is a high-performance framework that organizes PyTorch code to decouple research from engineering, making ML development faster and more reproducible.

## Core Concepts

### 1. LightningModule

The `LightningModule` is your model + training logic:

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn


class MyModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
        super().__init__()
        # IMPORTANT: Save hyperparameters for checkpointing
        self.save_hyperparameters()

        # Define model
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """Forward pass - just the model logic."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training logic for one batch."""
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)

        # Log metrics (automatically sent to loggers)
        self.log("train/loss", loss)

        return loss  # MUST return loss for backprop

    def validation_step(self, batch, batch_idx):
        """Validation logic."""
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test logic (optional)."""
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()

        self.log("test/loss", loss)
        self.log("test/acc", acc)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or "step"
                "frequency": 1,
            },
        }
```

### 2. LightningDataModule

Organizes all data loading logic:

```python
class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        """Download data (called once, on a single GPU)."""
        # Download dataset if needed
        # DON'T set state here (self.x = y)
        pass

    def setup(self, stage=None):
        """Load data (called on every GPU)."""
        # stage is 'fit', 'validate', 'test', or 'predict'

        if stage == "fit" or stage is None:
            # Load training data
            self.train_dataset = ...
            self.val_dataset = ...

        if stage == "test" or stage is None:
            self.test_dataset = ...

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
```

### 3. Trainer

The Trainer handles all training loop logic:

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Create trainer
trainer = Trainer(
    # Hardware
    accelerator="gpu",  # or "cpu", "tpu", "mps"
    devices=1,  # Number of GPUs/TPUs
    precision="16-mixed",  # Mixed precision training

    # Training duration
    max_epochs=100,
    min_epochs=1,

    # Validation
    check_val_every_n_epoch=1,
    val_check_interval=1.0,  # Check val after every epoch

    # Logging
    log_every_n_steps=50,
    enable_progress_bar=True,

    # Callbacks
    callbacks=[
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            filename="epoch_{epoch:02d}-loss_{val/loss:.4f}",
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=10,
            mode="min",
        ),
    ],

    # Debugging
    fast_dev_run=False,  # Set to True for quick test
    overfit_batches=0,  # Overfit on N batches for debugging

    # Reproducibility
    deterministic=True,  # Makes training reproducible
    benchmark=False,  # Set to True for faster training if input size is constant
)

# Train
trainer.fit(model, datamodule=dm)

# Test
trainer.test(model, datamodule=dm, ckpt_path="best")
```

## Common Patterns

### Multiple Optimizers

```python
def configure_optimizers(self):
    # Different LR for different parts
    optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=0.001)
    optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)

    return [optimizer_gen, optimizer_disc], []
```

### Manual Optimization

```python
def __init__(self):
    super().__init__()
    self.automatic_optimization = False  # Disable automatic

def training_step(self, batch, batch_idx):
    opt = self.optimizers()

    # Manual forward and backward
    loss = self.compute_loss(batch)

    opt.zero_grad()
    self.manual_backward(loss)
    opt.step()

    return loss
```

### Gradient Accumulation

```python
trainer = Trainer(
    accumulate_grad_batches=4,  # Accumulate 4 batches
)
# Effective batch size = batch_size * 4
```

### Multi-GPU Training

```python
# Data Parallel (simplest)
trainer = Trainer(accelerator="gpu", devices=4, strategy="ddp")

# For very large models
trainer = Trainer(accelerator="gpu", devices=8, strategy="fsdp")
```

## Callbacks

### Built-in Callbacks

```python
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    ModelSummary,
)

callbacks = [
    # Save best models
    ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=3),

    # Stop early if not improving
    EarlyStopping(monitor="val/loss", patience=10, mode="min"),

    # Log learning rate
    LearningRateMonitor(logging_interval="step"),

    # Rich progress bar
    RichProgressBar(),

    # Model summary
    ModelSummary(max_depth=2),
]

trainer = Trainer(callbacks=callbacks)
```

### Custom Callback

```python
from pytorch_lightning.callbacks import Callback


class MyCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started!")

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch} finished")

    def on_validation_end(self, trainer, pl_module):
        # Access metrics
        val_loss = trainer.callback_metrics.get("val/loss")
        print(f"Validation loss: {val_loss}")
```

## Logging

### Multiple Loggers

```python
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

wandb_logger = WandbLogger(project="my-project", name="experiment-1")
tb_logger = TensorBoardLogger("logs/", name="my_model")

trainer = Trainer(logger=[wandb_logger, tb_logger])
```

### Log Everything

```python
def training_step(self, batch, batch_idx):
    # ... compute loss ...

    # Log scalars
    self.log("train/loss", loss)

    # Log multiple metrics at once
    self.log_dict({
        "train/loss": loss,
        "train/acc": acc,
        "train/f1": f1,
    })

    # Log on specific conditions
    self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    return loss
```

## Best Practices

1. **Always use `self.save_hyperparameters()`** in `__init__`
2. **Use DataModule** for all data loading logic
3. **Log with `self.log()`** for automatic aggregation
4. **Use callbacks** for checkpointing and early stopping
5. **Enable mixed precision** with `precision="16-mixed"`
6. **Use `pin_memory=True`** and `persistent_workers=True` in DataLoader
7. **Set seeds** for reproducibility
8. **Use `fast_dev_run=True`** for quick testing

## Common Pitfalls

- ❌ Not calling `self.save_hyperparameters()`
- ❌ Setting state in `prepare_data()` instead of `setup()`
- ❌ Not returning loss from `training_step()`
- ❌ Forgetting `self.training` check for dropout/batchnorm
- ❌ Not using `prog_bar=True` for key metrics
- ❌ Hardcoding paths instead of using `self.hparams`

## Quick Reference

```python
# Minimal working example
class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Train
trainer = Trainer(max_epochs=10)
trainer.fit(model, train_dataloader)
```

For more info: https://lightning.ai/docs/pytorch/stable/
