# ML Training Performance Profiling

Profile ML training performance to identify bottlenecks in data loading, model forward/backward pass, and optimization steps.

## Overview

Performance profiling helps you:
- Identify training bottlenecks (data loading vs compute)
- Optimize GPU utilization
- Reduce training time
- Fix memory issues
- Improve data pipeline efficiency

## Profiling Tools

### 1. PyTorch Lightning Profiler

**Simple Profiler (built-in):**
```python
from pytorch_lightning.profilers import SimpleProfiler

trainer = Trainer(
    profiler="simple",  # or SimpleProfiler()
    max_epochs=1,
)
trainer.fit(model, datamodule)

# View results
# Automatically printed at end of training
```

**Advanced Profiler:**
```python
from pytorch_lightning.profilers import AdvancedProfiler

profiler = AdvancedProfiler(
    dirpath="./profiler_logs",
    filename="advanced_profile",
)

trainer = Trainer(profiler=profiler, max_epochs=1)
trainer.fit(model, datamodule)
```

**PyTorch Profiler (most detailed):**
```python
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

# View in TensorBoard
# tensorboard --logdir=./tb_logs
```

### 2. Data Loading Profiler

**Diagnose Data Loading Bottleneck:**
```python
import time
import torch
from torch.utils.data import DataLoader

def profile_dataloader(dataloader, num_batches=100):
    """Profile data loading speed."""
    times = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        start = time.time()
        # Simulate accessing data
        _ = batch
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    throughput = 1.0 / avg_time if avg_time > 0 else 0

    print(f"Data Loading Profile:")
    print(f"  Average time per batch: {avg_time*1000:.2f}ms")
    print(f"  Throughput: {throughput:.2f} batches/sec")
    print(f"  Total batches: {len(times)}")

    return avg_time

# Usage
train_loader = datamodule.train_dataloader()
profile_dataloader(train_loader)
```

**Optimize num_workers:**
```python
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def find_optimal_num_workers(dataset, batch_size=32, max_workers=16):
    """Find optimal num_workers for DataLoader."""
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

    # Plot results
    workers, times = zip(*results)
    plt.plot(workers, times)
    plt.xlabel("num_workers")
    plt.ylabel("Time per batch (s)")
    plt.title("DataLoader Performance vs num_workers")
    plt.savefig("num_workers_profile.png")

    # Find optimal
    optimal_idx = times.index(min(times))
    optimal_workers = workers[optimal_idx]
    print(f"\n✓ Optimal num_workers: {optimal_workers}")

    return optimal_workers
```

### 3. GPU Utilization Monitoring

**Monitor GPU Usage:**
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Log GPU usage during training
nvidia-smi dmon -s um -o TD > gpu_usage.log &
PID=$!
python src/train.py
kill $PID
```

**GPU Utilization Tracker:**
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

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def report(self):
        if not self.utilizations:
            print("No GPU data collected")
            return

        avg_util = sum(self.utilizations) / len(self.utilizations)
        avg_mem = sum(self.memories) / len(self.memories)
        max_mem = max(self.memories)

        print(f"\nGPU Utilization Report:")
        print(f"  Average GPU utilization: {avg_util:.1f}%")
        print(f"  Average memory used: {avg_mem:.0f} MB")
        print(f"  Peak memory used: {max_mem:.0f} MB")

        if avg_util < 80:
            print(f"  ⚠️  Low GPU utilization ({avg_util:.1f}%)")
            print("     Consider: larger batch size, more num_workers, or profile data loading")

# Usage in LightningModule
class MyModel(pl.LightningModule):
    def on_train_start(self):
        self.gpu_monitor = GPUMonitor()
        self.gpu_monitor.start()

    def on_train_end(self):
        self.gpu_monitor.stop()
        self.gpu_monitor.report()
```

### 4. Model Forward/Backward Profiling

**Profile Individual Components:**
```python
import torch
from pytorch_lightning import LightningModule

class ProfiledModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.timings = {"forward": [], "loss": [], "backward": []}

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Profile forward pass
        start = time.time()
        y_hat = self(x)
        self.timings["forward"].append(time.time() - start)

        # Profile loss computation
        start = time.time()
        loss = self.criterion(y_hat, y)
        self.timings["loss"].append(time.time() - start)

        return loss

    def on_train_epoch_end(self):
        # Report timings
        for key, times in self.timings.items():
            if times:
                avg = sum(times) / len(times) * 1000
                print(f"  Avg {key} time: {avg:.2f}ms")

        # Clear for next epoch
        self.timings = {"forward": [], "loss": [], "backward": []}
```

**Layer-wise Profiling:**
```python
def profile_model_layers(model, input_shape=(1, 3, 224, 224)):
    """Profile computation time of each layer."""
    import torch
    from torch.profiler import profile, ProfilerActivity

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

### 5. Memory Profiling

**Track Memory Usage:**
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
    print(f"  Max Allocated: {max_allocated:.2f} MB")

# Use in training
class MemoryProfiledModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            print_memory_stats()

        # Regular training code
        ...
```

**Memory Profiler Callback:**
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

### 6. End-to-End Training Profile

**Complete Profiling Script:**
```python
#!/usr/bin/env python3
"""Profile complete training run."""

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.profilers import PyTorchProfiler

def profile_training(config_name="config", max_steps=100):
    """Profile a short training run."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from pathlib import Path

    # Load config
    config_dir = Path.cwd() / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)

    # Create model and data
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)

    # Setup profiler
    profiler = PyTorchProfiler(
        dirpath="./profiler_output",
        filename="training_profile",
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_logs/profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    # Train with profiling
    trainer = Trainer(
        profiler=profiler,
        max_steps=max_steps,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(model, datamodule)

    print(f"\n✓ Profiling complete!")
    print(f"  View results: tensorboard --logdir=./tb_logs/profiler")
    print(f"  Profile saved to: ./profiler_output/")

if __name__ == "__main__":
    profile_training()
```

## Performance Analysis

### Interpreting Results

**Data Loading Bottleneck:**
- GPU utilization < 80%
- Fix: Increase `num_workers`, use faster storage, preprocess data offline

**Compute Bottleneck:**
- GPU utilization > 95%
- Fix: Use mixed precision, optimize model architecture, larger batch size

**Memory Bottleneck:**
- OOM errors or near maximum memory
- Fix: Reduce batch size, use gradient checkpointing, optimize model

### Optimization Recommendations

**Based on Profile Results:**

```python
def analyze_profile(gpu_util, data_time, model_time):
    """Provide optimization recommendations."""
    recommendations = []

    if gpu_util < 70:
        recommendations.append("🔴 Low GPU utilization - data loading bottleneck")
        recommendations.append("   → Increase num_workers in DataLoader")
        recommendations.append("   → Use pin_memory=True")
        recommendations.append("   → Consider data preprocessing/caching")

    if data_time > model_time * 0.5:
        recommendations.append("🟡 Data loading takes significant time")
        recommendations.append("   → Optimize transforms/augmentations")
        recommendations.append("   → Use faster data format (LMDB, HDF5)")

    if gpu_util > 95:
        recommendations.append("🟢 GPU well utilized")
        recommendations.append("   → Consider increasing batch size if memory allows")

    return recommendations
```

## Usage

```bash
# Quick profile
python src/train.py trainer.profiler=simple trainer.max_epochs=1

# Detailed profile
python src/train.py trainer.profiler=pytorch trainer.max_steps=100

# Profile data loading
python scripts/profile_data.py

# Monitor GPU during training
watch -n 1 nvidia-smi
```

## CI/CD Performance Tests

```yaml
# .github/workflows/performance.yml
name: Performance Tests

on: [push]

jobs:
  profile:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Profile training
        run: |
          python scripts/profile_training.py
          python scripts/analyze_profile.py > performance_report.txt
      - name: Upload report
        uses: actions/upload-artifact@v4
        with:
          name: performance-report
          path: performance_report.txt
```

## Success Criteria

- [ ] GPU utilization > 80% during training
- [ ] Data loading time < 10% of total training time
- [ ] No memory leaks detected
- [ ] Bottlenecks identified and documented
- [ ] Optimization recommendations provided

Optimize your training for maximum performance!
