# Training Reference Guide

Detailed reference for advanced training techniques, hyperparameter sweeps, and W&B integration.

## Hyperparameter Sweeps

### Grid Search

Systematically explore all combinations of hyperparameters:

```bash
python src/train.py --multirun \
  model.hidden_dims="[512,256],[1024,512,256]" \
  model.dropout=0.1,0.2,0.3 \
  data.batch_size=64,128,256
```

**When to use:**

- Small number of hyperparameters (< 5)
- Discrete parameter values
- Want to explore all combinations

**Example grid:**

- 2 architectures × 3 dropout values × 3 batch sizes = 18 runs

### Random Search with Optuna

More efficient than grid search for high-dimensional spaces:

```bash
# Using Hydra Optuna plugin
python src/train.py \
  --multirun \
  --config-name=config \
  hydra/sweeper=optuna \
  hydra.sweeper.n_trials=50
```

**Optuna sweep configuration:**

```yaml
# configs/config.yaml
defaults:
  - override hydra/sweeper: optuna
  - override hydra/sweeper/sampler: tpe

hydra:
  sweeper:
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
    direction: minimize
    n_trials: 100
    n_jobs: 1

    params:
      model.learning_rate: tag(log, interval(1e-5, 1e-1))
      model.weight_decay: tag(log, interval(1e-6, 1e-2))
      model.dropout: interval(0.0, 0.5)
      data.batch_size: choice(16, 32, 64, 128, 256)
      trainer.max_epochs: 100
```

**When to use:**

- Large hyperparameter space (> 5 parameters)
- Continuous parameter ranges
- Limited compute budget

### Bayesian Optimization

Smart search using past results to guide future trials:

```yaml
# configs/sweep/bayesian.yaml
defaults:
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    sampler:
      _target_: optuna.samplers.TPESampler
      n_startup_trials: 10  # Random trials before Bayesian
      n_ei_candidates: 24
      seed: 42

    direction: minimize
    study_name: bayesian_sweep
    storage: sqlite:///optuna.db  # Persist trials

    params:
      model.learning_rate: tag(log, interval(1e-5, 1e-2))
      model.hidden_channels: choice(64, 128, 256, 512)
      model.num_layers: range(2, 7)  # 2 to 6 layers
      model.dropout: interval(0.0, 0.5)
```

Run the sweep:

```bash
python src/train.py \
  --config-name=sweep/bayesian \
  --multirun \
  hydra.sweeper.n_trials=100
```

**When to use:**

- Expensive training runs
- Want to find optimal hyperparameters efficiently
- Can afford sequential evaluation

### Multi-Objective Optimization

Optimize for multiple metrics simultaneously:

```yaml
hydra:
  sweeper:
    direction:
      - minimize  # validation loss
      - maximize  # validation accuracy

    params:
      model.learning_rate: tag(log, interval(1e-5, 1e-2))
      model.weight_decay: tag(log, interval(1e-6, 1e-2))
```

Returns Pareto-optimal solutions (trade-offs between objectives).

## W&B Advanced Integration

### Custom Metrics Logging

Log custom metrics during training:

```python
# In LightningModule
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    # Log scalar metrics
    self.log("train/loss", loss)
    self.log("train/acc", accuracy)

    # Log learning rate
    current_lr = self.optimizers().param_groups[0]["lr"]
    self.log("train/lr", current_lr)

    # Log gradient norms
    total_norm = 0.0
    for p in self.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    self.log("train/grad_norm", total_norm)

    return loss

def validation_step(self, batch, batch_idx):
    loss, preds, targets = self.compute_loss_and_predictions(batch)

    # Log validation metrics
    self.log("val/loss", loss)
    self.log("val/acc", accuracy)

    # Store predictions for epoch end
    self.validation_outputs.append({
        "preds": preds,
        "targets": targets,
    })
```

### Logging Artifacts

**Log confusion matrices:**

```python
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def on_validation_epoch_end(self):
    # Aggregate predictions
    all_preds = torch.cat([x["preds"] for x in self.validation_outputs])
    all_targets = torch.cat([x["targets"] for x in self.validation_outputs])

    # Compute confusion matrix
    cm = confusion_matrix(
        all_targets.cpu().numpy(),
        all_preds.cpu().numpy(),
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Log to W&B
    self.logger.experiment.log({
        "val/confusion_matrix": wandb.Image(fig),
        "epoch": self.current_epoch,
    })
    plt.close(fig)

    self.validation_outputs.clear()
```

**Log sample predictions:**

```python
def on_validation_epoch_end(self):
    # Take first batch of validation data
    batch = next(iter(self.trainer.val_dataloaders))
    images, labels = batch
    images = images[:8].to(self.device)  # First 8 images

    # Get predictions
    with torch.no_grad():
        logits = self(images)
        preds = torch.argmax(logits, dim=1)

    # Log images with predictions
    self.logger.experiment.log({
        "val/sample_predictions": [
            wandb.Image(
                img,
                caption=f"True: {label}, Pred: {pred}"
            )
            for img, label, pred in zip(images.cpu(), labels[:8], preds.cpu())
        ],
        "epoch": self.current_epoch,
    })
```

**Log model graphs:**

```python
from pytorch_lightning.callbacks import Callback

class LogModelGraphCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        # Log model architecture
        sample_input = torch.randn(1, 3, 224, 224).to(pl_module.device)

        trainer.logger.experiment.watch(
            pl_module,
            log="all",  # Log gradients and parameters
            log_freq=100,
            log_graph=True,
        )
```

**Log histograms:**

```python
def on_train_epoch_end(self):
    # Log weight distributions
    for name, params in self.named_parameters():
        self.logger.experiment.log({
            f"weights/{name}": wandb.Histogram(params.detach().cpu().numpy()),
            "epoch": self.current_epoch,
        })
```

### Model Checkpointing to W&B

Automatically upload checkpoints:

```yaml
# In config
logger:
  wandb:
    project: my-project
    log_model: true  # Upload checkpoints to W&B
    save_code: true  # Save code snapshot
```

Or programmatically:

```python
import wandb

# After training
artifact = wandb.Artifact("model", type="model")
artifact.add_file("checkpoints/best.ckpt")
wandb.log_artifact(artifact)
```

### W&B Sweeps Configuration

Alternative to Hydra sweeps, using W&B's native sweep functionality:

```yaml
# sweep.yaml
program: src/train.py
method: bayes
metric:
  name: val/loss
  goal: minimize
parameters:
  model.learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-1
  model.weight_decay:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-2
  data.batch_size:
    values: [16, 32, 64, 128, 256]
  model.dropout:
    distribution: uniform
    min: 0.0
    max: 0.5
```

Run the sweep:

```bash
# Initialize sweep
wandb sweep sweep.yaml

# Run agents (can run multiple in parallel)
wandb agent <sweep-id>
```

## Common Training Commands Reference

Quick reference for debugging and profiling:

```bash
# Quick debug run (5 batches only)
python src/train.py trainer.fast_dev_run=5

# Overfit single batch (check model capacity)
python src/train.py \
  trainer.overfit_batches=1 \
  trainer.max_epochs=100

# Profile training (identify bottlenecks)
python src/train.py \
  trainer.profiler=advanced \
  trainer.max_epochs=1

# Sanity check (run validation before training)
python src/train.py \
  trainer.num_sanity_val_steps=2

# Find optimal batch size
python src/train.py \
  trainer.auto_scale_batch_size=true

# Find optimal learning rate
python src/train.py \
  trainer.auto_lr_find=true

# Multi-GPU distributed
python src/train.py \
  trainer.devices=4 \
  trainer.strategy=ddp

# Resume interrupted training
python src/train.py \
  ckpt_path=logs/last.ckpt

# Test model without training
python src/train.py \
  --config-name=test \
  ckpt_path=checkpoints/best.ckpt

# Accumulate gradients (effective larger batch)
python src/train.py \
  data.batch_size=32 \
  trainer.accumulate_grad_batches=4  # Effective batch = 128
```

## Environment Variables

Useful environment variables for training:

```bash
# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use specific GPUs
export CUDA_LAUNCH_BLOCKING=1  # Easier debugging (slower)

# PyTorch settings
export TORCH_USE_CUDA_DSA=1  # Better CUDA error messages
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory fragmentation

# W&B settings
export WANDB_PROJECT=my-project
export WANDB_ENTITY=my-team
export WANDB_MODE=offline  # Log locally, sync later
export WANDB_DIR=/path/to/logs

# Hydra settings
export HYDRA_FULL_ERROR=1  # Show full stack traces

# Debugging
export PYTHONFAULTHANDLER=1  # Better crash reports
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Distributed debugging
```

## Training on Cloud/HPC

### SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=ml-training
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Load modules
module load cuda/11.8
module load python/3.10

# Activate environment
source venv/bin/activate

# Set distributed training env vars
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Run training
srun python src/train.py \
  experiment=large_scale \
  trainer.devices=4 \
  trainer.num_nodes=$SLURM_NNODES \
  trainer.strategy=ddp
```

Submit job:

```bash
sbatch train.slurm
```

### Docker Training

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Install dependencies
COPY pyproject.toml .
RUN pip install uv && uv sync

# Copy code
COPY . .

# Run training
CMD ["python", "src/train.py"]
```

Run container:

```bash
docker run --gpus all -v $(pwd):/workspace ml-training
```

## Performance Benchmarking

Track and compare training performance:

```python
from pytorch_lightning.callbacks import Callback
import time

class ThroughputCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()
        self.samples_seen = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.samples_seen += batch[0].size(0)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start
        throughput = self.samples_seen / epoch_time

        pl_module.log("train/samples_per_sec", throughput)
        pl_module.log("train/epoch_time", epoch_time)
```

Compare across runs:

```bash
# Baseline
python src/train.py experiment=baseline

# With optimization
python src/train.py experiment=optimized \
  trainer.precision=16-mixed \
  data.num_workers=8
```

Check W&B to compare `train/samples_per_sec` across runs.
