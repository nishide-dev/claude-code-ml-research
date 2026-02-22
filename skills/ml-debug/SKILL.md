---
name: ml-debug
description: Debug common ML training issues (NaN loss, OOM, slow training, convergence problems) and provide solutions. Use when training fails, metrics don't improve, or encountering errors like NaN loss, CUDA OOM, or slow convergence.
argument-hint: [log_file] [issue_type]
disable-model-invocation: true
---

# ML Training Debugging

Systematic debugging guide for machine learning training issues with PyTorch Lightning.

## Quick Diagnosis

Identify your problem category:

| Symptom | Category | Quick Check |
|---------|----------|-------------|
| `NaN` or `Inf` in loss | **Loss Issues** | Check learning rate, gradient clipping |
| Training loss >> validation loss | **Overfitting** | Add regularization, data augmentation |
| Both losses high | **Underfitting** | Increase model capacity, train longer |
| GPU utilization <80% | **Data Loading** | Increase `num_workers`, use faster storage |
| CUDA out of memory | **Memory Issues** | Reduce `batch_size`, use gradient checkpointing |
| Loss plateau | **Convergence Issues** | Adjust learning rate, try different optimizer |
| Wrong predictions | **Data Issues** | Check labels, verify preprocessing |

## 1. Loss Issues

### NaN or Inf Loss

**Diagnosis:**

```bash
# Check training logs for NaN
grep -i "nan\|inf" logs/train.log

# Test with lower learning rate
python src/train.py model.optimizer.lr=0.0001
```

**Solutions (try in order):**

1. **Reduce learning rate**

```yaml
model:
  optimizer:
    lr: 0.0001  # Start small
```

1. **Enable gradient clipping**

```yaml
trainer:
  gradient_clip_val: 1.0
  gradient_clip_algorithm: "norm"
```

1. **Fix numerical stability**

```python
# In model forward pass
# Bad: Division without epsilon
output = x / y

# Good: Add epsilon
output = x / (y + 1e-8)

# Bad: Manual log(softmax)
output = torch.log(F.softmax(x, dim=-1))

# Good: Use stable version
output = F.log_softmax(x, dim=-1)
```

1. **Try full precision**

```bash
python src/train.py trainer.precision=32
```

1. **Check for inf/nan in data**

```python
# In DataModule.setup()
sample = self.train_dataset[0]
assert not torch.isnan(sample[0]).any(), "NaN in input data"
assert not torch.isinf(sample[0]).any(), "Inf in input data"
```

See `examples/nan-loss-debugging.md` for detailed guide.

### Exploding Gradients

**Diagnosis:**

```python
# Enable gradient tracking
trainer = Trainer(
    gradient_clip_val=1.0,
    track_grad_norm=2,  # Log L2 norm
    log_every_n_steps=10
)
```

**Solutions:**

- If gradient norm >100: reduce learning rate or increase clipping
- If gradients explode in specific layer: check weight initialization
- Use gradient accumulation to reduce per-step updates

### Vanishing Gradients

**Symptoms:** Gradients close to zero, no learning in early layers.

**Solutions:**

```python
# 1. Use ReLU instead of Sigmoid/Tanh
self.activation = nn.ReLU()

# 2. Add skip connections (ResNet-style)
def forward(self, x):
    identity = x
    out = self.layer(x)
    return out + identity  # Skip connection

# 3. Use batch normalization
self.bn = nn.BatchNorm1d(hidden_dim)

# 4. Better initialization
for layer in self.layers:
    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
```

## 2. Performance Issues

### Overfitting (val_loss > train_loss)

**Diagnosis:**

```python
# Plot losses
import matplotlib.pyplot as plt
import pandas as pd

metrics = pd.read_csv("logs/metrics.csv")
plt.plot(metrics["train_loss"], label="Train")
plt.plot(metrics["val_loss"], label="Val")
plt.legend()
plt.savefig("overfit_analysis.png")

# Overfit ratio
ratio = metrics["val_loss"].iloc[-1] / metrics["train_loss"].iloc[-1]
if ratio > 1.5:
    print("⚠️  SEVERE OVERFITTING")
```

**Solutions:**

```yaml
# 1. Add regularization
model:
  dropout: 0.3  # Increase dropout
  optimizer:
    weight_decay: 0.0001  # L2 regularization

# 2. Data augmentation
data:
  augmentation:
    random_crop: true
    horizontal_flip: true
    mixup_alpha: 0.2

# 3. Early stopping
callbacks:
  early_stopping:
    monitor: "val/loss"
    patience: 10
    mode: "min"

# 4. Reduce model size
model:
  hidden_dims: [256, 128]  # Smaller model
```

### Underfitting (both losses high)

**Solutions:**

```yaml
# 1. Increase model capacity
model:
  hidden_dims: [2048, 1024, 512, 256]
  num_layers: 6

# 2. Train longer
trainer:
  max_epochs: 200

# 3. Increase learning rate
model:
  optimizer:
    lr: 0.01

# 4. Remove excessive regularization
model:
  dropout: 0.1
  optimizer:
    weight_decay: 0.00001
```

### Slow Convergence

**Try different optimizers:**

```python
# AdamW (default, good for most)
optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)

# SGD with momentum (good for vision)
optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True)

# RMSprop (good for RNNs)
optimizer = torch.optim.RMSprop(params, lr=0.001)
```

**Add learning rate scheduler:**

```yaml
model:
  scheduler:
    # Cosine annealing
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    T_max: 100
    eta_min: 1e-6

    # Or OneCycle (often fastest convergence)
    # _target_: torch.optim.lr_scheduler.OneCycleLR
    # max_lr: 0.01
    # total_steps: ${trainer.max_steps}
```

## 3. Speed Issues

### Profile Training

```bash
# Use Lightning profiler
python src/train.py \
  trainer.profiler="advanced" \
  trainer.max_epochs=1

# Check GPU utilization
watch -n 1 nvidia-smi
```

### Data Loading Bottleneck

**Symptoms:** GPU utilization <80%, slow batch iteration.

**Solutions:**

```yaml
data:
  num_workers: 8  # Increase workers
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2

# Use faster data formats
# HDF5, LMDB, or preprocessed tensors
```

### Slow Forward/Backward

**Solutions:**

```python
# 1. Use torch.compile (PyTorch 2.0+)
def configure_model(self):
    self.model = torch.compile(self.model, mode="reduce-overhead")

# 2. Avoid Python loops - use vectorization
# Bad
for i in range(batch_size):
    result[i] = self.layer(x[i])

# Good
result = self.layer(x)

# 3. Mixed precision
# trainer.precision = "16-mixed"

# 4. Gradient accumulation
# trainer.accumulate_grad_batches = 4
```

See `scripts/profile_training.py` for profiling script.

## 4. Memory Issues

### Out of Memory (OOM)

**Diagnosis:**

```bash
# Check GPU memory
nvidia-smi

# Profile memory usage
python src/train.py trainer.profiler="pytorch" trainer.max_epochs=1
```

**Solutions (in order):**

```yaml
# 1. Reduce batch size
data:
  batch_size: 32  # Was 128

# 2. Mixed precision
trainer:
  precision: "16-mixed"

# 3. Gradient accumulation (maintains effective batch size)
trainer:
  accumulate_grad_batches: 4  # Effective: 32 * 4 = 128

# 4. Gradient checkpointing (in model)
# self.model.gradient_checkpointing_enable()

# 5. Reduce model size
model:
  hidden_dims: [512, 256]
```

### Memory Leak

**Common causes:**

```python
# Bad: Keeps computation graph
loss_history = []
for batch in dataloader:
    loss = model(batch)
    loss_history.append(loss)  # ❌

# Good: Only store scalar
loss_history = []
for batch in dataloader:
    loss = model(batch)
    loss_history.append(loss.item())  # ✅

# Clear cache periodically
if self.global_step % 100 == 0:
    torch.cuda.empty_cache()
```

**Checkpoint management:**

```yaml
callbacks:
  model_checkpoint:
    save_top_k: 3  # Only keep 3 best
```

## 5. Data Issues

### Check Data Shapes

```python
# In DataModule.setup()
def setup(self, stage=None):
    self.train_dataset = ...

    # Validate
    sample = self.train_dataset[0]
    print(f"Input shape: {sample[0].shape}")
    print(f"Label shape: {sample[1].shape}")
    print(f"Label range: [{sample[1].min()}, {sample[1].max()}]")

    # Check for NaN
    assert not torch.isnan(sample[0]).any(), "NaN in input"

    # Label distribution
    labels = [self.train_dataset[i][1] for i in range(min(1000, len(self.train_dataset)))]
    print(f"Distribution: {pd.Series(labels).value_counts()}")
```

### Visualize Data

```python
# Show a batch
import matplotlib.pyplot as plt

dm = MyDataModule()
dm.setup()
loader = dm.train_dataloader()
batch = next(iter(loader))
inputs, labels = batch

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(inputs[i].permute(1, 2, 0))  # CHW -> HWC
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis('off')
plt.savefig("data_samples.png")
```

## 6. PyTorch Geometric Specific Issues

### Over-smoothing in GNNs

**Symptoms:** All node representations become similar after many layers.

**Solutions:**

```yaml
# 1. Reduce layers
model:
  num_layers: 2  # Instead of 5+

# 2. Add skip connections
# In model forward:
# x = conv(x, edge_index) + x_orig

# 3. Use jumping knowledge
model:
  jk_mode: "cat"  # or "max", "lstm"
```

### Large Graph OOM

```yaml
# Use mini-batch training with sampling
data:
  use_sampling: true
  num_neighbors: [15, 10, 5]  # Per-layer sampling
  batch_size: 1024  # Mini-batches of nodes
```

See `examples/gnn-debugging.md` for GNN-specific guide.

## Debugging Checklist

**Data:**

- [ ] Loads without errors
- [ ] Shapes are correct
- [ ] No NaN or Inf values
- [ ] Labels in correct range (0 to num_classes-1)
- [ ] Augmentation works
- [ ] Splits are correct

**Model:**

- [ ] Forward pass works with dummy data
- [ ] Output shape matches expected
- [ ] Reasonable number of parameters
- [ ] Gradients flow through all layers

**Training:**

- [ ] Loss decreases in first epoch
- [ ] Validation runs correctly
- [ ] Checkpoints save
- [ ] Metrics logged
- [ ] GPU utilization high (>80%)

**Config:**

- [ ] Learning rate appropriate (0.0001-0.01)
- [ ] Batch size fits in memory
- [ ] Enough epochs

## Common Error Messages

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch_size, enable gradient checkpointing, use fp16 |
| `Expected all tensors on same device` | `x = x.to(self.device)` in forward |
| `Target size must match input size` | Check loss function, verify output dims |
| `Sizes of tensors must match` | Check batch dimensions |

## Generate Debug Report

Use the debug report script:

```bash
python scripts/debug_report.py --log-dir logs/
```

See `scripts/debug_report.py` for implementation.

## Success Criteria

- [ ] Problem identified and categorized
- [ ] Root cause determined
- [ ] Solution applied
- [ ] Training resumes successfully
- [ ] Metrics improve
- [ ] No errors in logs

Debugging complete - training is back on track!
