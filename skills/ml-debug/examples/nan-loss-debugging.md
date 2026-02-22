# NaN Loss Debugging Guide

Comprehensive guide to diagnosing and fixing NaN (Not a Number) or Inf (Infinity) loss in PyTorch training.

## Quick Diagnosis

```bash
# Check for NaN in logs
grep -i "nan\|inf" logs/train.log

# Check for NaN in checkpoints
python -c "
import torch
ckpt = torch.load('checkpoints/last.ckpt')
for name, param in ckpt['state_dict'].items():
    if torch.isnan(param).any():
        print(f'NaN in {name}')
    if torch.isinf(param).any():
        print(f'Inf in {name}')
"
```

## Root Causes and Solutions

### 1. Learning Rate Too High

**Symptoms:**

- NaN appears within first few steps
- Loss explodes then becomes NaN
- Gradients very large before NaN

**Diagnosis:**

```python
# Check gradient norms
trainer = Trainer(track_grad_norm=2, log_every_n_steps=10)

# If grad_norm > 100 in first steps, LR too high
```

**Solutions:**

```yaml
# Option 1: Lower learning rate
model:
  optimizer:
    lr: 0.0001  # Start with 10x smaller

# Option 2: Use learning rate finder
# In train.py:
# trainer = Trainer(auto_lr_find=True)
# trainer.tune(model, datamodule=dm)
```

**Learning rate finder:**

```python
from pytorch_lightning.tuner import Tuner

trainer = Trainer()
tuner = Tuner(trainer)

# Automatically find optimal learning rate
lr_finder = tuner.lr_find(model, datamodule=dm)

# Plot
fig = lr_finder.plot(suggest=True)
fig.savefig("lr_finder.png")

# Get suggestion
new_lr = lr_finder.suggestion()
print(f"Suggested LR: {new_lr}")
```

### 2. No Gradient Clipping

**Symptoms:**

- NaN appears after several epochs
- Gradient norms gradually increase
- Particular batches cause NaN

**Solution:**

```yaml
trainer:
  gradient_clip_val: 1.0  # Start with 1.0
  gradient_clip_algorithm: "norm"  # L2 norm clipping
```

**Advanced gradient clipping:**

```python
# In LightningModule
def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
    # Custom gradient clipping
    self.clip_gradients(
        optimizer,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
    )

    # Log gradient norms for debugging
    grad_norm = self.get_gradient_norm()
    self.log("train/grad_norm", grad_norm, on_step=True)

def get_gradient_norm(self):
    total_norm = 0.0
    for p in self.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
```

### 3. Numerical Instability

**Common unstable operations:**

```python
# Bad: Division without epsilon
output = x / y

# Good: Add epsilon
output = x / (y + 1e-8)

# Bad: Manual log(softmax)
output = torch.log(F.softmax(x, dim=-1))

# Good: Use stable version
output = F.log_softmax(x, dim=-1)

# Bad: Manual log(sigmoid)
output = torch.log(torch.sigmoid(x))

# Good: Use logsigmoid
output = F.logsigmoid(x)

# Bad: Sqrt without clamp
output = torch.sqrt(x)

# Good: Clamp before sqrt
output = torch.sqrt(torch.clamp(x, min=1e-8))

# Bad: Variance calculation
var = ((x - mean) ** 2).mean()

# Good: Use stable variance
var = torch.var(x, unbiased=False)
```

**Numerical stability in custom loss:**

```python
class StableCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        # Bad: Manual softmax + log + nll
        # probs = F.softmax(logits, dim=-1)
        # log_probs = torch.log(probs + 1e-8)
        # loss = F.nll_loss(log_probs, targets)

        # Good: Use stable cross_entropy
        loss = F.cross_entropy(logits, targets)
        return loss
```

### 4. Mixed Precision Issues

**Symptoms:**

- NaN only with `precision="16-mixed"`
- Works fine with `precision=32`
- Loss scale becomes 0

**Diagnosis:**

```bash
# Test with full precision
python src/train.py trainer.precision=32

# If works with fp32 but not fp16, it's a precision issue
```

**Solutions:**

```python
# Option 1: Use automatic loss scaling (default in Lightning)
trainer = Trainer(precision="16-mixed")

# Option 2: Manually adjust loss scale
from pytorch_lightning.plugins import MixedPrecisionPlugin

plugin = MixedPrecisionPlugin(
    precision="16-mixed",
    # Start with smaller scale
    scaler_kwargs={"init_scale": 2.**10, "growth_interval": 2000}
)
trainer = Trainer(plugins=[plugin])

# Option 3: Use bfloat16 (more stable than float16)
trainer = Trainer(precision="bf16-mixed")
```

**Gradient overflow detection:**

```python
# In training_step
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    # Check for inf/nan before backward
    if not torch.isfinite(loss):
        print(f"Non-finite loss at step {self.global_step}: {loss}")
        # Skip this batch
        return None

    return loss
```

### 5. Bad Weight Initialization

**Symptoms:**

- NaN in first forward pass
- Activations explode immediately
- Some layers have huge weights

**Check initialization:**

```python
# In LightningModule.__init__
def __init__(self):
    super().__init__()
    self.model = MyModel()

    # Check initial weights
    self._check_initialization()

def _check_initialization(self):
    for name, param in self.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN in initialized weights: {name}")
        if torch.isinf(param).any():
            print(f"Inf in initialized weights: {name}")
        # Check scale
        std = param.std().item()
        if std > 10.0 or std < 0.001:
            print(f"Warning: {name} has unusual std: {std:.6f}")
```

**Proper initialization:**

```python
def __init__(self):
    super().__init__()

    # Linear layers
    self.layers = nn.ModuleList([
        nn.Linear(in_dim, out_dim)
        for in_dim, out_dim in zip(dims[:-1], dims[1:])
    ])

    # Initialize
    self._initialize_weights()

def _initialize_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Linear):
            # Kaiming initialization for ReLU
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
```

### 6. NaN in Input Data

**Diagnosis:**

```python
# In DataModule.setup()
def setup(self, stage=None):
    self.train_dataset = ...

    # Check for NaN in data
    print("Checking dataset for NaN/Inf...")
    num_samples = min(1000, len(self.train_dataset))

    for i in range(num_samples):
        sample, label = self.train_dataset[i]

        if torch.isnan(sample).any():
            print(f"❌ NaN found in sample {i}")
            print(f"Indices: {torch.where(torch.isnan(sample))}")

        if torch.isinf(sample).any():
            print(f"❌ Inf found in sample {i}")

    print("✅ Data check complete")
```

**Fix data preprocessing:**

```python
class MyTransform:
    def __call__(self, x):
        # Normalize
        x = (x - self.mean) / (self.std + 1e-8)  # Add epsilon

        # Clip outliers
        x = torch.clamp(x, min=-10, max=10)

        # Check for NaN
        if torch.isnan(x).any():
            print("Warning: NaN after transform, replacing with zeros")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        return x
```

### 7. NaN in Custom Layers

**Add NaN detection:**

```python
class MyCustomLayer(nn.Module):
    def forward(self, x):
        # Input check
        if torch.isnan(x).any():
            raise ValueError(f"NaN in input to {self.__class__.__name__}")

        # Forward pass
        out = self.some_operation(x)

        # Output check
        if torch.isnan(out).any():
            print(f"NaN detected in {self.__class__.__name__}")
            print(f"Input stats: mean={x.mean():.4f}, std={x.std():.4f}")
            print(f"Output stats: mean={out[~torch.isnan(out)].mean():.4f}")
            raise ValueError("NaN in layer output")

        return out
```

**Register hooks for automatic checking:**

```python
def check_nan_hook(module, input, output):
    """Hook to check for NaN in layer outputs."""
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            print(f"❌ NaN in {module.__class__.__name__}")
            raise ValueError("NaN detected")

# Register hook on all modules
for name, module in model.named_modules():
    module.register_forward_hook(check_nan_hook)
```

## Systematic Debugging Process

### Step 1: Isolate the Source

```python
# Minimal reproduction
python src/train.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10 \
  trainer.limit_val_batches=5 \
  trainer.fast_dev_run=true
```

### Step 2: Enable Anomaly Detection

```python
# In train.py, before trainer.fit()
import torch

torch.autograd.set_detect_anomaly(True)

# This will show exact operation causing NaN
# WARNING: Very slow, only use for debugging
```

### Step 3: Add Gradient/Activation Logging

```python
# In LightningModule
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    # Log intermediate values
    if self.global_step % 10 == 0:
        self.log_dict({
            "debug/loss": loss,
            "debug/loss_isnan": torch.isnan(loss).any(),
            "debug/grad_norm": self.get_gradient_norm(),
        })

    return loss

def on_before_optimizer_step(self, optimizer):
    # Log gradient statistics
    for name, param in self.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            if torch.isnan(grad_norm):
                print(f"NaN gradient in {name}")
```

### Step 4: Binary Search

If NaN appears after some steps, find the exact step:

```bash
# Test with early stopping
python src/train.py trainer.max_steps=100  # Works?
python src/train.py trainer.max_steps=200  # NaN?
python src/train.py trainer.max_steps=150  # Works?
python src/train.py trainer.max_steps=175  # NaN?
# ... narrow down to exact step
```

### Step 5: Checkpoint Analysis

```python
# Load checkpoint before NaN
ckpt = torch.load("checkpoints/epoch_005.ckpt")

# Check weight statistics
for name, param in ckpt['state_dict'].items():
    print(f"{name}: mean={param.mean():.6f}, std={param.std():.6f}, max={param.abs().max():.6f}")

# Compare with checkpoint after NaN
ckpt_nan = torch.load("checkpoints/epoch_006.ckpt")
for name in ckpt['state_dict']:
    before = ckpt['state_dict'][name]
    after = ckpt_nan['state_dict'][name]
    diff = (after - before).abs().max()
    print(f"{name}: max_change={diff:.6f}")
```

## Prevention Checklist

**Model Design:**

- [ ] Use stable operations (log_softmax, not log(softmax))
- [ ] Add epsilon to divisions
- [ ] Proper weight initialization (Kaiming, Xavier)
- [ ] Gradient clipping enabled

**Training Configuration:**

- [ ] Reasonable learning rate (<0.01)
- [ ] Gradient clipping (1.0-10.0)
- [ ] Warmup for first few epochs
- [ ] Full precision for debugging

**Data Pipeline:**

- [ ] No NaN in dataset
- [ ] Normalization uses epsilon
- [ ] Clip extreme values
- [ ] Verify preprocessing

**Monitoring:**

- [ ] Log gradient norms
- [ ] Track loss scale (for mixed precision)
- [ ] Alert on first NaN
- [ ] Save checkpoints frequently

## Emergency Fixes

If you need to continue training despite occasional NaN:

```python
# In training_step
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    # Skip NaN batches (NOT RECOMMENDED for production)
    if not torch.isfinite(loss):
        print(f"⚠️  Skipping NaN batch at step {self.global_step}")
        return None

    return loss

# In on_before_optimizer_step
def on_before_optimizer_step(self, optimizer):
    # Clip gradients more aggressively
    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)

    # Check for NaN gradients
    for name, param in self.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"Zeroing NaN gradient in {name}")
            param.grad.zero_()
```

**Note:** These are temporary workarounds. Always fix the root cause!

## Testing for Numerical Stability

```python
def test_numerical_stability(model, datamodule):
    """Test model for numerical stability issues."""
    print("Testing numerical stability...")

    model.eval()
    dm = datamodule
    dm.setup()
    loader = dm.train_dataloader()

    # Test on multiple batches
    for i, batch in enumerate(loader):
        if i >= 10:
            break

        x, y = batch
        x = x.to(model.device)

        # Forward pass
        with torch.no_grad():
            output = model(x)

        # Check for NaN/Inf
        assert torch.isfinite(x).all(), f"NaN/Inf in input (batch {i})"
        assert torch.isfinite(output).all(), f"NaN/Inf in output (batch {i})"

        # Check for extreme values
        assert output.abs().max() < 1e6, f"Extreme values in output (batch {i})"

    print("✅ Numerical stability test passed")
```

Run before training:

```python
# In train.py
if __name__ == "__main__":
    model = MyModel()
    dm = MyDataModule()

    # Test stability
    test_numerical_stability(model, dm)

    # Train
    trainer = Trainer()
    trainer.fit(model, dm)
```
