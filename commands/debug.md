---
name: debug
description: Debug common ML training issues (NaN loss, OOM, slow training, convergence problems) and provide solutions
---

# ML Training Debugging

Debug common machine learning training issues, diagnose problems, and provide solutions.

## Process

### 1. Identify Problem Category

Ask user about the symptom:
- **Loss issues**: NaN loss, exploding/vanishing gradients
- **Performance issues**: Poor accuracy, overfitting, underfitting
- **Speed issues**: Slow training, data loading bottleneck
- **Memory issues**: OOM errors, memory leaks
- **Convergence issues**: Loss not decreasing, plateau
- **Data issues**: Wrong shapes, missing data, incorrect labels

### 2. Loss Issues Debugging

**NaN or Inf Loss:**

Check these in order:

1. **Learning Rate Too High:**
```bash
# Test with lower learning rate
python src/train.py model.optimizer.lr=0.0001

# Or use learning rate finder
python -c "
from pytorch_lightning import Trainer
trainer = Trainer(auto_lr_find=True)
trainer.tune(model, datamodule=dm)
"
```

2. **Gradient Clipping:**
```yaml
# In trainer config
trainer:
  gradient_clip_val: 1.0
  gradient_clip_algorithm: "norm"
```

3. **Numerical Stability:**
```python
# In model code
# Add epsilon to divisions
output = numerator / (denominator + 1e-8)

# Use stable softmax
F.log_softmax(..., dim=-1)  # Instead of log(softmax(...))

# Check for inf/nan in forward pass
def forward(self, x):
    out = self.layer(x)
    assert not torch.isnan(out).any(), "NaN detected after layer"
    assert not torch.isinf(out).any(), "Inf detected after layer"
    return out
```

4. **Mixed Precision Issues:**
```bash
# Try full precision first
python src/train.py trainer.precision=32

# If NaN only in mixed precision, use gradient scaling
trainer:
  precision: "16-mixed"
  # Lightning handles gradient scaling automatically
```

**Exploding Gradients:**
```python
# Enable gradient tracking
trainer = Trainer(
    gradient_clip_val=1.0,
    track_grad_norm=2,  # Log L2 norm of gradients
    log_every_n_steps=10
)

# Check gradient norms in W&B
# If gradients >100, reduce learning rate or increase clipping
```

**Vanishing Gradients:**
```python
# Solutions:
# 1. Use ReLU instead of Sigmoid/Tanh
# 2. Add skip connections (ResNet-style)
# 3. Use batch normalization
# 4. Reduce network depth
# 5. Use better initialization (Kaiming, Xavier)

# In model
def __init__(self):
    self.layers = nn.ModuleList([
        nn.Linear(in_features, out_features)
        for _ in range(num_layers)
    ])
    # Initialize with Kaiming
    for layer in self.layers:
        nn.init.kaiming_normal_(layer.weight)
```

### 3. Performance Issues Debugging

**Overfitting (Val Loss > Train Loss):**

Diagnosis script:
```python
import matplotlib.pyplot as plt
import pandas as pd

metrics = pd.read_csv("logs/metrics.csv")

plt.plot(metrics["train_loss"], label="Train")
plt.plot(metrics["val_loss"], label="Val")
plt.legend()
plt.title("Overfit Detection: Val Loss Diverging")
plt.savefig("overfit_analysis.png")

# Calculate overfit ratio
overfit_ratio = metrics["val_loss"].iloc[-1] / metrics["train_loss"].iloc[-1]
print(f"Overfit Ratio: {overfit_ratio:.2f}")
if overfit_ratio > 1.5:
    print("⚠️  SEVERE OVERFITTING DETECTED")
```

Solutions:
```yaml
# 1. Add regularization
model:
  dropout: 0.3  # Increase dropout
  weight_decay: 0.0001  # Increase weight decay

# 2. Data augmentation
data:
  augmentation:
    random_crop: true
    horizontal_flip: true
    color_jitter: true
    mixup_alpha: 0.2  # Mixup augmentation

# 3. Early stopping
callbacks:
  early_stopping:
    monitor: "val/loss"
    patience: 10
    mode: "min"

# 4. Get more data or reduce model capacity
model:
  hidden_dims: [256, 128]  # Smaller model
```

**Underfitting (Both Losses High):**

Solutions:
```yaml
# 1. Increase model capacity
model:
  hidden_dims: [2048, 1024, 512, 256]  # Larger model
  num_layers: 6

# 2. Train longer
trainer:
  max_epochs: 200

# 3. Increase learning rate
model:
  optimizer:
    lr: 0.01  # Higher learning rate

# 4. Remove excessive regularization
model:
  dropout: 0.1  # Less dropout
  weight_decay: 0.00001  # Less weight decay
```

**Slow Convergence:**
```python
# Try different optimizers
# AdamW (default, good for most cases)
optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)

# SGD with momentum (better for some vision tasks)
optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True)

# RMSprop (good for RNNs)
optimizer = torch.optim.RMSprop(params, lr=0.001)

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# or
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=num_steps)
```

### 4. Speed Issues Debugging

**Profile Training:**
```bash
# Use PyTorch Lightning profiler
python src/train.py \
  trainer.profiler="advanced" \
  trainer.max_epochs=1
```

**Data Loading Bottleneck:**
```python
# Check if GPU is waiting for data
# If GPU utilization < 80%, data loading is bottleneck

# Solutions:
data:
  num_workers: 8  # Increase workers
  pin_memory: true  # Pin memory for faster transfer
  persistent_workers: true  # Keep workers alive

# Use faster data formats
# HDF5, LMDB, or preprocessed tensors instead of images
```

**Slow Forward/Backward Pass:**
```python
# 1. Use torch.compile (PyTorch 2.0+)
def configure_model(self):
    self.model = torch.compile(self.model, mode="reduce-overhead")

# 2. Use more efficient operations
# Bad: Python loops
for i in range(batch_size):
    result[i] = self.layer(x[i])

# Good: Vectorized
result = self.layer(x)

# 3. Mixed precision
trainer:
  precision: "16-mixed"

# 4. Gradient accumulation to reduce syncing
trainer:
  accumulate_grad_batches: 4
```

### 5. Memory Issues Debugging

**Out of Memory (OOM):**

Diagnose:
```bash
# Check memory usage
nvidia-smi

# Profile memory
python -c "
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.profilers import AdvancedProfiler

profiler = AdvancedProfiler(filename='memory_profile.txt')
trainer = Trainer(profiler=profiler, max_epochs=1)
trainer.fit(model, dm)
"
```

Solutions in priority order:
```yaml
# 1. Reduce batch size
data:
  batch_size: 32  # Was 128

# 2. Enable gradient checkpointing
# In model.__init__
self.model.gradient_checkpointing_enable()

# 3. Mixed precision
trainer:
  precision: "16-mixed"

# 4. Gradient accumulation (simulate larger batch)
trainer:
  accumulate_grad_batches: 4  # Effective batch = 32 * 4 = 128

# 5. Reduce model size
model:
  hidden_dims: [512, 256]  # Smaller dimensions

# 6. Clear cache periodically
# In training_step
if self.global_step % 100 == 0:
    torch.cuda.empty_cache()
```

**Memory Leak:**
```python
# Common causes:
# 1. Storing tensors in lists
bad_list = []
for batch in dataloader:
    loss = model(batch)
    bad_list.append(loss)  # ❌ Keeps computation graph

# Fix: Detach from graph
good_list = []
for batch in dataloader:
    loss = model(batch)
    good_list.append(loss.item())  # ✅ Only stores scalar

# 2. Not releasing old checkpoints
# Configure checkpoint callback
callbacks:
  model_checkpoint:
    save_top_k: 3  # Only keep 3 best, delete others
```

### 6. PyTorch Geometric Specific Issues

**Over-smoothing in GNNs:**
```python
# Symptoms: All node representations become similar
# Solutions:
# 1. Reduce number of layers
model:
  num_layers: 2  # Instead of 5+

# 2. Add skip connections
def forward(self, x, edge_index):
    x_orig = x
    for conv in self.convs:
        x = conv(x, edge_index)
        x = x + x_orig  # Skip connection
    return x

# 3. Use jumping knowledge
from torch_geometric.nn import JumpingKnowledge
self.jk = JumpingKnowledge(mode='cat')
```

**Large Graph OOM:**
```yaml
# Use mini-batch training with sampling
data:
  use_sampling: true
  num_neighbors: [15, 10, 5]  # Sample neighbors per layer
  batch_size: 1024  # Mini-batches of nodes

# Or use graph sampling strategies
from torch_geometric.loader import NeighborSampler, ClusterData, ClusterLoader
```

### 7. Data Issues Debugging

**Check Data Shapes:**
```python
# Add to DataModule
def setup(self, stage=None):
    self.train_dataset = ...

    # Validate shapes
    sample = self.train_dataset[0]
    print(f"Input shape: {sample[0].shape}")
    print(f"Label shape: {sample[1].shape}")
    print(f"Label range: [{sample[1].min()}, {sample[1].max()}]")

    # Check for NaN
    assert not torch.isnan(sample[0]).any(), "NaN in input data"

    # Check label distribution
    labels = [self.train_dataset[i][1] for i in range(min(1000, len(self.train_dataset)))]
    print(f"Label distribution: {pd.Series(labels).value_counts()}")
```

**Visualize Data:**
```python
# In notebook
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dm = MyDataModule()
dm.setup()
loader = dm.train_dataloader()

# Show a batch
batch = next(iter(loader))
inputs, labels = batch

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(inputs[i].permute(1, 2, 0))  # CHW -> HWC
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis('off')
plt.savefig("data_samples.png")
```

### 8. Debugging Checklist

Run through this checklist systematically:

**Data:**
- [ ] Data loads without errors
- [ ] Shapes are correct (batch_size, channels, height, width)
- [ ] No NaN or Inf values
- [ ] Labels are in correct range (0 to num_classes-1)
- [ ] Data augmentation works correctly
- [ ] Train/val/test splits are correct

**Model:**
- [ ] Model forward pass works with dummy data
- [ ] Output shape matches expected
- [ ] Number of parameters is reasonable
- [ ] Gradients flow through all layers

**Training:**
- [ ] Loss decreases in first epoch
- [ ] Validation is run correctly
- [ ] Checkpoints are saved
- [ ] Metrics are logged
- [ ] GPU utilization is high

**Config:**
- [ ] Learning rate is appropriate
- [ ] Batch size fits in memory
- [ ] Number of epochs is sufficient

### 9. Common Error Messages and Solutions

**"CUDA out of memory":**
- Reduce batch_size
- Enable gradient_checkpointing
- Use precision="16-mixed"

**"RuntimeError: Expected all tensors to be on the same device":**
```python
# Move tensors to model device
def forward(self, x):
    x = x.to(self.device)
    return self.model(x)
```

**"ValueError: Target size must be the same as input size":**
- Check loss function matches task (CrossEntropyLoss for classification)
- Verify output dimensions match num_classes
- For CrossEntropyLoss, labels should be class indices (Long), not one-hot

**"RuntimeError: Sizes of tensors must match":**
- Check batch dimensions are consistent
- Ensure data augmentation doesn't break shapes

### 10. Generate Debug Report

```python
def generate_debug_report(log_dir: Path):
    """Generate comprehensive debug report."""

    print("=" * 50)
    print("ML TRAINING DEBUG REPORT")
    print("=" * 50)

    # 1. System info
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # 2. Memory usage
    print(f"\nGPU Memory:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # 3. Training metrics
    metrics = pd.read_csv(log_dir / "metrics.csv")
    print(f"\nTraining Progress:")
    print(f"Epochs: {len(metrics)}")
    print(f"Latest train loss: {metrics['train_loss'].iloc[-1]:.4f}")
    print(f"Latest val loss: {metrics['val_loss'].iloc[-1]:.4f}")
    print(f"Best val loss: {metrics['val_loss'].min():.4f}")

    # 4. Loss trends
    recent_losses = metrics['train_loss'].tail(10)
    if recent_losses.is_monotonic_decreasing:
        print("✅ Training loss is decreasing")
    else:
        print("⚠️  Training loss is not decreasing consistently")

    # 5. Overfit check
    overfit_ratio = metrics['val_loss'].iloc[-1] / metrics['train_loss'].iloc[-1]
    if overfit_ratio > 1.5:
        print(f"⚠️  Possible overfitting (ratio: {overfit_ratio:.2f})")

    print("\n" + "=" * 50)
```

## Success Criteria

- [ ] Problem identified and categorized
- [ ] Root cause determined
- [ ] Solution applied
- [ ] Training resumes successfully
- [ ] Metrics improve
- [ ] No more errors in logs

Debugging complete - training is back on track!
