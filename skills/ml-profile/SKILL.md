---
name: ml-profile
description: Profile ML training performance to identify bottlenecks (data loading, compute, memory usage) and optimize GPU utilization. Use when training is slow, GPU utilization is low, or experiencing memory issues.
argument-hint: [profiler_type] [profile_duration]
disable-model-invocation: true
---

# ML Training Performance Profiling

Profile ML training to identify bottlenecks in data loading, model computation, and memory usage.

## Quick Start

```bash
# Simple profiling (built-in)
python src/train.py trainer.profiler=simple trainer.max_epochs=1

# Detailed profiling (PyTorch Profiler)
python src/train.py trainer.profiler=pytorch trainer.max_steps=100

# View in TensorBoard
tensorboard --logdir=./tb_logs/profiler

# Monitor GPU utilization
watch -n 1 nvidia-smi
```

## Profiling Tools

### 1. PyTorch Lightning Profilers

**Simple Profiler (quick overview):**

```python
from pytorch_lightning import Trainer

trainer = Trainer(
    profiler="simple",  # Built-in simple profiler
    max_epochs=1,
)
trainer.fit(model, datamodule)

# Output: timing summary printed at end
```

**Advanced Profiler (detailed breakdown):**

```python
from pytorch_lightning.profilers import AdvancedProfiler

profiler = AdvancedProfiler(
    dirpath="./profiler_logs",
    filename="advanced_profile",
)

trainer = Trainer(profiler=profiler, max_epochs=1)
trainer.fit(model, datamodule)
```

**PyTorch Profiler (most detailed, visualizable):**

```python
import torch
from pytorch_lightning.profilers import PyTorchProfiler

profiler = PyTorchProfiler(
    dirpath="./profiler_logs",
    filename="pytorch_profile",
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
)

trainer = Trainer(profiler=profiler, max_epochs=1)
trainer.fit(model, datamodule)

# View results:
# tensorboard --logdir=./tb_logs
```

### 2. Data Loading Profiling

**Diagnose data loading bottleneck:**

```python
import time

def profile_dataloader(dataloader, num_batches=100):
    """Measure data loading speed."""
    times = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        start = time.time()
        _ = batch  # Access data
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    throughput = 1.0 / avg_time if avg_time > 0 else 0

    print(f"Data Loading Profile:")
    print(f"  Avg time/batch: {avg_time*1000:.2f}ms")
    print(f"  Throughput: {throughput:.2f} batches/sec")

    return avg_time

# Usage
train_loader = datamodule.train_dataloader()
profile_dataloader(train_loader)
```

**Find optimal num_workers:**

```python
from torch.utils.data import DataLoader

def find_optimal_num_workers(dataset, batch_size=32, max_workers=16):
    """Test different num_workers to find optimal."""
    results = []

    for num_workers in range(0, max_workers + 1):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        avg_time = profile_dataloader(loader, num_batches=50)
        results.append((num_workers, avg_time))
        print(f"num_workers={num_workers}: {avg_time*1000:.2f}ms/batch")

    # Find optimal
    workers, times = zip(*results)
    optimal_idx = times.index(min(times))
    optimal_workers = workers[optimal_idx]

    print(f"\n✓ Optimal num_workers: {optimal_workers}")
    return optimal_workers
```

See `scripts/profile_dataloader.py` for complete implementation.

### 3. GPU Utilization Monitoring

**Real-time monitoring:**

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Log GPU usage during training
nvidia-smi dmon -s um -o TD > gpu_usage.log &
PID=$!
python src/train.py
kill $PID
```

**GPU Monitor (programmatic):**

```python
import subprocess
import threading
import time

class GPUMonitor:
    """Monitor GPU utilization during training."""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.utilizations = []
        self.memories = []

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        while self.running:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                )
                util, mem = result.stdout.strip().split(", ")
                self.utilizations.append(float(util))
                self.memories.append(float(mem))
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        self.thread.join()

    def report(self):
        avg_util = sum(self.utilizations) / len(self.utilizations)
        avg_mem = sum(self.memories) / len(self.memories)
        max_mem = max(self.memories)

        print(f"\nGPU Utilization Report:")
        print(f"  Average GPU util: {avg_util:.1f}%")
        print(f"  Average memory: {avg_mem:.0f} MB")
        print(f"  Peak memory: {max_mem:.0f} MB")

        if avg_util < 80:
            print(f"  ⚠️  Low GPU utilization ({avg_util:.1f}%)")
            print("     → Increase num_workers or batch size")

# Use in LightningModule
class MyModel(pl.LightningModule):
    def on_train_start(self):
        self.gpu_monitor = GPUMonitor()
        self.gpu_monitor.start()

    def on_train_end(self):
        self.gpu_monitor.stop()
        self.gpu_monitor.report()
```

### 4. Model Profiling

**Layer-wise profiling:**

```python
import torch
from torch.profiler import profile, ProfilerActivity

def profile_model_layers(model, input_shape=(1, 3, 224, 224)):
    """Profile computation time of each layer."""
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            model(dummy_input)

    # Print layer-wise stats
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=20
    ))

    return prof
```

**Component timing:**

```python
import time
from pytorch_lightning import LightningModule

class ProfiledModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.timings = {"forward": [], "loss": [], "backward": []}

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Time forward pass
        start = time.time()
        y_hat = self(x)
        self.timings["forward"].append(time.time() - start)

        # Time loss computation
        start = time.time()
        loss = self.criterion(y_hat, y)
        self.timings["loss"].append(time.time() - start)

        return loss

    def on_train_epoch_end(self):
        # Report average timings
        for key, times in self.timings.items():
            if times:
                avg = sum(times) / len(times) * 1000
                print(f"  Avg {key}: {avg:.2f}ms")

        # Reset for next epoch
        self.timings = {"forward": [], "loss": [], "backward": []}
```

### 5. Memory Profiling

**Track GPU memory:**

```python
import torch

def print_memory_stats(device=0):
    """Print current GPU memory usage."""
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2

    print(f"\nGPU Memory:")
    print(f"  Allocated: {allocated:.2f} MB")
    print(f"  Reserved: {reserved:.2f} MB")
    print(f"  Max: {max_allocated:.2f} MB")

# Use during training
class MemoryProfiledModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            print_memory_stats()

        # Regular training
        ...
```

**Memory callback:**

```python
from pytorch_lightning.callbacks import Callback

class MemoryProfileCallback(Callback):
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3

            pl_module.log("memory/allocated_gb", allocated)
            pl_module.log("memory/reserved_gb", reserved)

# Use in trainer
trainer = Trainer(callbacks=[MemoryProfileCallback(log_every_n_steps=50)])
```

## Complete Profiling Script

Use the automated profiling script:

```bash
python scripts/profile_training.py --max-steps 100
```

See `scripts/profile_training.py` for implementation.

## Performance Analysis

### Interpreting Results

**Symptoms and fixes:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| GPU util <80% | Data loading bottleneck | Increase `num_workers`, faster storage |
| GPU util >95% | Compute bound (good!) | Increase batch size if memory allows |
| High data time | Slow transforms | Optimize augmentation, use caching |
| OOM errors | Memory bottleneck | Reduce batch size, gradient checkpointing |

### Optimization Recommendations

```python
def analyze_profile(gpu_util, data_time, model_time):
    """Provide optimization recommendations based on profile."""
    recommendations = []

    if gpu_util < 70:
        recommendations.append("🔴 Low GPU utilization")
        recommendations.append("   → Increase num_workers")
        recommendations.append("   → Use pin_memory=True")
        recommendations.append("   → Cache/preprocess data")

    if data_time > model_time * 0.5:
        recommendations.append("🟡 Data loading slow")
        recommendations.append("   → Optimize transforms")
        recommendations.append("   → Use faster format (LMDB, HDF5)")

    if gpu_util > 95:
        recommendations.append("🟢 GPU well utilized")
        recommendations.append("   → Increase batch size if possible")

    return recommendations
```

## Common Bottlenecks

### Data Loading Bottleneck

**Symptoms:**

- GPU utilization <80%
- Slow batch iteration
- `nvidia-smi` shows low GPU usage

**Solutions:**

```yaml
data:
  num_workers: 8  # Increase workers
  pin_memory: true  # Faster GPU transfer
  persistent_workers: true  # Keep workers alive
  prefetch_factor: 2  # Prefetch batches
```

**Advanced:**

- Use faster storage (SSD vs HDD)
- Preprocess data offline
- Use LMDB or HDF5 format
- Cache augmented data

### Compute Bottleneck

**Symptoms:**

- GPU utilization >95%
- Training very slow
- Model forward/backward takes most time

**Solutions:**

```yaml
trainer:
  precision: "16-mixed"  # Mixed precision
  accumulate_grad_batches: 2  # Simulate larger batch

data:
  batch_size: 256  # Increase if memory allows
```

**Model optimizations:**

- Use `torch.compile()` (PyTorch 2.0+)
- Optimize architecture (fewer parameters)
- Use efficient operations

### Memory Bottleneck

**Symptoms:**

- OOM errors
- Memory usage near limit
- Can't increase batch size

**Solutions:**

```yaml
data:
  batch_size: 32  # Reduce

trainer:
  precision: "16-mixed"
  accumulate_grad_batches: 4

model:
  use_checkpoint: true  # Gradient checkpointing
```

## Profiling Workflow

1. **Run simple profiler** - Quick overview

```bash
python src/train.py trainer.profiler=simple trainer.max_epochs=1
```

1. **Monitor GPU** - Check utilization

```bash
watch -n 1 nvidia-smi
```

1. **Profile data loading** - If GPU util <80%

```bash
python scripts/profile_dataloader.py
```

1. **Detailed profiling** - PyTorch Profiler + TensorBoard

```bash
python src/train.py trainer.profiler=pytorch trainer.max_steps=100
tensorboard --logdir=./tb_logs
```

1. **Optimize** - Based on findings
2. **Re-profile** - Verify improvements

## Success Criteria

- [ ] GPU utilization >80% during training
- [ ] Data loading time <10% of total time
- [ ] No memory leaks
- [ ] Bottlenecks identified
- [ ] Optimizations applied
- [ ] Training speed improved

✅ Training optimized for maximum performance!
