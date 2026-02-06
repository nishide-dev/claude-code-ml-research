---
name: ml-lightning-basics
description: Comprehensive guide for PyTorch Lightning - LightningModule, Trainer, distributed training, PyTorch 2.0 torch.compile integration, Lightning Fabric, and production best practices
---

# PyTorch Lightning for ML Research

## Overview

PyTorch Lightning is the industry-standard framework that organizes PyTorch code to decouple research from engineering. It eliminates boilerplate while maintaining full PyTorch flexibility, enabling researchers to focus on model logic rather than training infrastructure.

**Key Benefits:**

- Eliminates 90% of boilerplate code
- Automatic distributed training (DDP, FSDP, DeepSpeed)
- Hardware agnostic (CPU, GPU, TPU, MPS)
- Built-in best practices (checkpointing, logging, profiling)
- Full PyTorch 2.0 compatibility with torch.compile
- Production-ready code from day one

**Resources:**

- Official docs: <https://lightning.ai/docs/pytorch/stable/>
- Style guide: <https://lightning.ai/docs/pytorch/stable/starter/style_guide.html>
- Performance guide: <https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html>

---

## Core Concepts

### 1. LightningModule

The `LightningModule` encapsulates model + training logic in a self-contained class.

**Complete example:**

```python
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageClassifier(L.LightningModule):
    def __init__(self, backbone="resnet18", num_classes=10, lr=1e-3):
        super().__init__()

        # CRITICAL: Save all hyperparameters for checkpointing
        self.save_hyperparameters()

        # Define model architecture
        if backbone == "resnet18":
            from torchvision.models import resnet18
            self.model = resnet18(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Metrics (use TorchMetrics for efficiency)
        from torchmetrics import Accuracy
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        """Forward pass - inference only."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training logic for one batch."""
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Update and log metrics
        acc = self.train_acc(y_hat, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss  # MUST return loss

    def validation_step(self, batch, batch_idx):
        """Validation logic."""
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        acc = self.val_acc(y_hat, y)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """Test logic (optional)."""
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()

        self.log("test/loss", loss)
        self.log("test/acc", acc)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
```

**Key methods:**

| Method | Purpose | Required |
|--------|---------|----------|
| `__init__` | Model architecture, hyperparameters | Yes |
| `forward` | Inference logic (no training code) | Yes |
| `training_step` | Training logic for one batch | Yes |
| `validation_step` | Validation logic | Recommended |
| `test_step` | Test logic | Optional |
| `configure_optimizers` | Optimizer and scheduler setup | Yes |

### 2. LightningDataModule

Organizes all data loading logic in a reusable, reproducible way.

```python
class ImageDataModule(L.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=32, num_workers=4):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        """Download data (runs once, on single GPU)."""
        # Download datasets
        # DON'T set instance variables here (no self.x = y)
        from torchvision.datasets import CIFAR10
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Load data (runs on every GPU in distributed)."""
        from torchvision.datasets import CIFAR10
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if stage == "fit" or stage is None:
            full_dataset = CIFAR10(
                self.hparams.data_dir,
                train=True,
                transform=transform
            )
            # Split into train/val
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )

        if stage == "test" or stage is None:
            self.test_dataset = CIFAR10(
                self.hparams.data_dir,
                train=False,
                transform=transform
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,  # Keeps workers alive between epochs
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
```

### 3. Trainer

The Trainer automates the training loop, hardware management, and logging.

```python
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

# Create trainer
trainer = Trainer(
    # Hardware
    accelerator="auto",  # Auto-detect: GPU, CPU, TPU, MPS
    devices="auto",  # Use all available devices
    precision="16-mixed",  # Mixed precision (faster, less memory)

    # Training duration
    max_epochs=100,
    min_epochs=10,

    # Validation
    check_val_every_n_epoch=1,
    val_check_interval=1.0,  # Fraction of training epoch

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
        LearningRateMonitor(logging_interval="step"),
    ],

    # Logger
    logger=WandbLogger(project="my-project", name="experiment-1"),

    # Debugging
    fast_dev_run=False,  # Set to True for 1 batch test
    overfit_batches=0,  # Overfit on N batches for debugging

    # Reproducibility
    deterministic=True,
    benchmark=False,  # Set True if input size is constant
)

# Train
model = ImageClassifier()
datamodule = ImageDataModule()
trainer.fit(model, datamodule=datamodule)

# Test
trainer.test(model, datamodule=datamodule, ckpt_path="best")
```

---

## PyTorch 2.0 Integration

PyTorch 2.0's `torch.compile` provides massive speedups (40%+ on average) through graph compilation.

### Using torch.compile with Lightning

#### Method 1: Compile the entire model

```python
class MyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.compile(
            YourModel(),
            mode="default"  # or "reduce-overhead", "max-autotune"
        )
```

#### Method 2: Configure in Trainer (recommended)

```python
model = MyModel()

# Compile automatically
trainer = Trainer(max_epochs=10)
compiled_model = torch.compile(model, mode="default")
trainer.fit(compiled_model, datamodule=dm)
```

### torch.compile Modes

| Mode | Optimization Level | Compile Time | Use Case |
|------|-------------------|--------------|----------|
| `default` | Balanced | Fast | Development, general use |
| `reduce-overhead` | Minimize kernel launch overhead | Medium | Small batch sizes, CPU bottlenecks |
| `max-autotune` | Maximum performance | Slow | Production, long training runs |

**Performance example:**

```python
import torch

# Standard model
model = MyModel()
# 40% faster on average with compilation
compiled_model = torch.compile(model, mode="max-autotune")
```

### Compilation Best Practices

**DO:**

- Use `mode="default"` during development
- Use `mode="max-autotune"` for production
- Profile first to identify bottlenecks
- Keep model architecture static (no dynamic shapes)

**DON'T:**

- Compile with highly dynamic models (RNNs with variable length)
- Expect speedups on CPU (torch.compile is GPU-focused)
- Mix compiled and non-compiled modules

### Graph Breaks (Performance Issue)

Graph breaks occur when PyTorch can't compile a section:

```python
def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)

    # AVOID: Python control flow breaks graph
    if batch_idx % 100 == 0:
        print(f"Batch {batch_idx}")  # Breaks compilation

    loss = F.cross_entropy(y_hat, y)
    return loss
```

**Check graph breaks:**

```python
import torch._dynamo as dynamo

# Reset and enable logging
dynamo.reset()
dynamo.config.verbose = True

model = torch.compile(model, mode="default")
# Warnings will show graph break locations
```

---

## Distributed Training

Lightning makes distributed training trivial - change one argument.

### DDP (Distributed Data Parallel)

**Standard multi-GPU training:**

```python
# Single GPU
trainer = Trainer(accelerator="gpu", devices=1)

# Multi-GPU (automatic DDP)
trainer = Trainer(accelerator="gpu", devices=4, strategy="ddp")

# All GPUs
trainer = Trainer(accelerator="gpu", devices="auto", strategy="ddp")
```

**DDP spawn (Windows compatibility):**

```python
trainer = Trainer(accelerator="gpu", devices=4, strategy="ddp_spawn")
```

### FSDP (Fully Sharded Data Parallel)

For models that don't fit in single GPU memory:

```python
from lightning.pytorch.strategies import FSDPStrategy

trainer = Trainer(
    accelerator="gpu",
    devices=8,
    strategy=FSDPStrategy(
        sharding_strategy="FULL_SHARD",  # Shard params, gradients, optimizer
        auto_wrap_policy={nn.Linear},  # Auto-wrap Linear layers
    ),
)
```

### DeepSpeed

For extreme-scale models (billions of parameters):

```python
from lightning.pytorch.strategies import DeepSpeedStrategy

trainer = Trainer(
    accelerator="gpu",
    devices=8,
    strategy=DeepSpeedStrategy(
        stage=3,  # ZeRO Stage 3 (most memory efficient)
        offload_optimizer=True,  # Offload optimizer to CPU
        offload_parameters=True,  # Offload params to CPU
    ),
    precision="16-mixed",
)
```

**DeepSpeed config file:**

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "fp16": {
    "enabled": true
  }
}
```

---

## Lightning Fabric

Fabric is Lightning's lightweight abstraction - more control than Trainer, less boilerplate than raw PyTorch.

### When to use Fabric

- Custom training loops with Lightning benefits
- Gradual migration from PyTorch
- Research requiring fine-grained control

**Example:**

```python
import lightning as L
from lightning.fabric import Fabric

# Initialize Fabric
fabric = L.Fabric(
    accelerator="cuda",
    devices=2,
    precision="16-mixed"
)
fabric.launch()

# Setup model and optimizer
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())
model, optimizer = fabric.setup(model, optimizer)

# Setup data
train_loader = fabric.setup_dataloaders(train_loader)

# Custom training loop
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model(batch)
        fabric.backward(loss)
        optimizer.step()

        # Automatic logging
        fabric.log("train_loss", loss)
```

---

## Advanced Patterns

### Multiple Optimizers

```python
def configure_optimizers(self):
    # Different LR for different parts
    opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.001)
    opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)
    return [opt_g, opt_d], []
```

### Manual Optimization

```python
class GANModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False  # Disable automatic

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        # Train generator
        loss_g = self.generator_loss(batch)
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()

        # Train discriminator
        loss_d = self.discriminator_loss(batch)
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()

        self.log_dict({"loss_g": loss_g, "loss_d": loss_d})
```

### Gradient Accumulation

```python
# Effective batch size = batch_size * accumulate_grad_batches
trainer = Trainer(
    accumulate_grad_batches=4,  # Accumulate 4 batches before update
)
```

### Gradient Clipping

```python
trainer = Trainer(
    gradient_clip_val=1.0,  # Clip gradients to max norm 1.0
    gradient_clip_algorithm="norm",  # or "value"
)
```

---

## Callbacks

### Built-in Callbacks

```python
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    ModelSummary,
    TQDMProgressBar,
)

callbacks = [
    # Save best models
    ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        filename="best-{epoch:02d}-{val_loss:.2f}",
    ),

    # Early stopping
    EarlyStopping(
        monitor="val/loss",
        patience=10,
        mode="min",
        verbose=True,
    ),

    # Log learning rate
    LearningRateMonitor(logging_interval="step"),

    # Rich progress bar
    RichProgressBar(),

    # Model summary
    ModelSummary(max_depth=2),
]
```

### Custom Callback

```python
from lightning.pytorch.callbacks import Callback

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started!")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get("train/loss")
        val_loss = trainer.callback_metrics.get("val/loss")
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    def on_validation_end(self, trainer, pl_module):
        # Save custom artifacts
        pass
```

---

## Logging

### Multiple Loggers

```python
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

wandb_logger = WandbLogger(project="my-project", name="run-1")
tb_logger = TensorBoardLogger("logs/", name="my_model")

trainer = Trainer(logger=[wandb_logger, tb_logger])
```

### Advanced Logging

```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    # Log scalars
    self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    # Log multiple metrics
    self.log_dict({
        "train/loss": loss,
        "train/acc": acc,
        "train/f1": f1,
    }, on_epoch=True)

    # Log histograms (for TensorBoard/W&B)
    if batch_idx % 100 == 0:
        self.logger.experiment.add_histogram(
            "gradients/layer1",
            self.model.layer1.weight.grad,
            self.global_step
        )

    return loss
```

---

## Best Practices

### ✅ DO

1. **Always call `self.save_hyperparameters()`** in `__init__` for reproducibility
2. **Use DataModule** to encapsulate all data logic
3. **Log with `self.log()`** for automatic aggregation and sync
4. **Use callbacks** for checkpointing, early stopping, and monitoring
5. **Enable mixed precision** with `precision="16-mixed"` for speedup
6. **Use `pin_memory=True`** and `persistent_workers=True` in DataLoader
7. **Set `deterministic=True`** for reproducibility
8. **Use `fast_dev_run=True`** for quick sanity checks
9. **Use TorchMetrics** for efficient metric computation
10. **Compile models** with `torch.compile` for PyTorch 2.0+

### ❌ DON'T

1. **Don't forget to return loss** from `training_step`
2. **Don't set state in `prepare_data()`** (use `setup()` instead)
3. **Don't manually move tensors** with `.to(device)` (Lightning handles this)
4. **Don't use `print()`** for logging (use `self.log()`)
5. **Don't hardcode hyperparameters** (use `self.hparams`)
6. **Don't ignore `sync_dist=True`** for distributed metrics
7. **Don't mix Lightning and raw PyTorch training loops**

---

## Common Issues and Solutions

### Issue 1: NaN Loss

```python
# Solution 1: Gradient clipping
trainer = Trainer(gradient_clip_val=1.0)

# Solution 2: Lower learning rate
optimizer = torch.optim.Adam(params, lr=1e-4)  # Instead of 1e-3

# Solution 3: Full precision
trainer = Trainer(precision=32)  # Instead of 16-mixed
```

### Issue 2: Out of Memory

```python
# Solution 1: Reduce batch size
datamodule = MyDataModule(batch_size=16)  # Instead of 32

# Solution 2: Gradient accumulation
trainer = Trainer(accumulate_grad_batches=4)

# Solution 3: Mixed precision
trainer = Trainer(precision="16-mixed")
```

### Issue 3: Slow Training

```python
# Solution 1: Compile model (PyTorch 2.0+)
model = torch.compile(model, mode="max-autotune")

# Solution 2: Profile bottlenecks
trainer = Trainer(profiler="simple")  # or "advanced"

# Solution 3: Increase num_workers
datamodule = MyDataModule(num_workers=8)  # Match CPU cores
```

---

## Essential Resources

### Official Documentation

- **Lightning Docs**: <https://lightning.ai/docs/pytorch/stable/>
- **API Reference**: <https://lightning.ai/docs/pytorch/stable/api_references.html>
- **Style Guide**: <https://lightning.ai/docs/pytorch/stable/starter/style_guide.html>

### Tutorials

- **15min to Lightning**: <https://lightning.ai/docs/pytorch/stable/starter/introduction.html>
- **From PyTorch to Lightning**: <https://lightning.ai/docs/pytorch/stable/starter/converting.html>
- **Distributed Training**: <https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html>

### PyTorch 2.0

- **torch.compile Overview**: <https://pytorch.org/get-started/pytorch-2-0/>
- **torch.compile Tutorial**: <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>
- **Performance Tips**: <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>

### Community

- **Lightning Bolts**: <https://lightning-bolts.readthedocs.io/> (Model implementations)
- **GitHub Discussions**: <https://github.com/Lightning-AI/pytorch-lightning/discussions>

---

## Quick Reference

**Minimal working example:**

```python
import lightning as L
import torch

class MinimalModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Train
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, train_dataloader)
```

---

## Summary

PyTorch Lightning provides:

- **Simplicity**: Eliminate boilerplate, focus on research
- **Scale**: From laptop to supercomputer with one line
- **Speed**: PyTorch 2.0 integration, automatic optimizations
- **Flexibility**: Full PyTorch control when needed
- **Production**: Code is deployment-ready from day one

Combined with PyTorch 2.0's torch.compile, Lightning delivers maximum performance with minimal code.
