---
description: Execute training runs with proper monitoring, checkpointing, and experiment tracking (PyTorch Lightning with Hydra config support)
---

# ML Training Execution

Execute training runs with proper monitoring, checkpointing, and experiment tracking.

## Process

### 1. Pre-Training Validation

Before starting training, verify:

**Environment Check:**

```bash
# Check Python version
python --version  # Should be >= 3.10

# Check CUDA availability (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Check package installation
python -c "import pytorch_lightning as pl; print(f'Lightning: {pl.__version__}')"
```

**Configuration Check:**

```bash
# Validate config without training
python src/train.py --cfg job

# Dry run to check data loading
python src/train.py trainer.fast_dev_run=true
```

**Disk Space:**

- Verify sufficient space for checkpoints and logs
- Estimate: (model_size_mb *save_top_k* num_epochs / checkpoint_freq)

### 2. Training Execution Options

**Basic Training:**

```bash
python src/train.py
```

**With Specific Config:**

```bash
python src/train.py experiment=my_experiment
```

**With CLI Overrides:**

```bash
python src/train.py \
  model.hidden_dims=[1024,512,256] \
  data.batch_size=128 \
  trainer.max_epochs=50 \
  trainer.devices=2
```

**Resume from Checkpoint:**

```bash
python src/train.py \
  ckpt_path="checkpoints/epoch_42.ckpt"
```

**Multi-GPU Training:**

```bash
# DDP (Distributed Data Parallel)
python src/train.py \
  trainer.accelerator=gpu \
  trainer.devices=4 \
  trainer.strategy=ddp

# FSDP (Fully Sharded Data Parallel) for large models
python src/train.py \
  trainer.strategy=fsdp \
  trainer.devices=8
```

### 3. Training Monitoring

**Real-time Monitoring:**

- Lightning automatically shows progress bar with metrics
- Monitor GPU usage: `watch -n 1 nvidia-smi`
- Check W&B dashboard: `wandb sync` (if using W&B)

**Key Metrics to Watch:**

- Training loss (should decrease steadily)
- Validation loss (should decrease without diverging from train)
- Learning rate (check scheduler is working)
- GPU utilization (should be >80%)
- Training speed (samples/sec or batches/sec)

**Red Flags:**

- Loss is NaN or inf → Check learning rate, gradient clipping
- Validation loss increasing while train decreasing → Overfitting
- Very slow training → Check data loading bottleneck (num_workers)
- Low GPU usage → Check batch size, data pipeline
- Out of memory → Reduce batch size or model size

### 4. Experiment Tracking (W&B)

If using Weights & Biases:

**Initialize:**

```bash
wandb login
export WANDB_PROJECT="my-ml-project"
```

**Track Custom Metrics:**

```python
# In LightningModule
def training_step(self, batch, batch_idx):
    loss = ...

    # Log metrics
    self.log("train/loss", loss)
    self.log("train/acc", accuracy)
    self.log("lr", self.optimizers().param_groups[0]["lr"])

    return loss
```

**Log Artifacts:**

- Model checkpoints (automatic with log_model=true)
- Confusion matrices
- Sample predictions
- Model graphs

### 5. Hyperparameter Sweeps

**Grid Search:**

```bash
python src/train.py --multirun \
  model.hidden_dims="[512,256],[1024,512,256]" \
  model.dropout=0.1,0.2,0.3 \
  data.batch_size=64,128,256
```

**Random Search:**

```bash
# Using Hydra Optuna plugin
python src/train.py \
  --multirun \
  --config-name=config \
  hydra/sweeper=optuna \
  hydra.sweeper.n_trials=50
```

**Bayesian Optimization:**

```yaml
# configs/sweep/bayesian.yaml
hydra:
  sweeper:
    sampler:
      _target_: optuna.samplers.TPESampler
```

### 6. Debugging Training Issues

**Gradient Issues:**

```python
# Check for NaN/inf gradients
from pytorch_lightning.callbacks import GradientAccumulationScheduler

trainer = Trainer(
    gradient_clip_val=1.0,  # Clip gradients
    gradient_clip_algorithm="norm",
    track_grad_norm=2,  # Log gradient norms
)
```

**Memory Issues:**

```python
# Reduce memory usage
trainer = Trainer(
    precision="16-mixed",  # Use mixed precision
    accumulate_grad_batches=4,  # Gradient accumulation
)

# In model
def configure_model(self):
    self.model = torch.compile(self.model)  # PyTorch 2.0+ optimization
```

**Slow Data Loading:**

```python
# Profile data loading
from pytorch_lightning.profilers import SimpleProfiler

trainer = Trainer(profiler="simple")
```

**Overfitting:**

```python
# Add regularization
trainer = Trainer(
    callbacks=[
        EarlyStopping(monitor="val/loss", patience=10),
    ]
)

# In model config
model:
  dropout: 0.3
  weight_decay: 0.0001
```

### 7. PyTorch Geometric Specific

**For GNN Training:**

```bash
# Node classification
python src/train.py \
  model=gnn \
  data=graph \
  data.dataset_name=Cora

# Graph classification with batching
python src/train.py \
  model=gnn \
  data=graph \
  data.dataset_name=PROTEINS \
  data.batch_size=32

# Large graph sampling
python src/train.py \
  model=gnn \
  data=graph \
  data.use_sampling=true \
  data.num_neighbors=[15,10,5]
```

**Monitoring GNN-Specific Metrics:**

- Node-level accuracy
- Graph-level accuracy
- Over-smoothing (node representations becoming too similar)
- Graph connectivity statistics

### 8. Advanced Training Techniques

**Mixed Precision Training:**

```bash
python src/train.py trainer.precision=16-mixed
```

**Gradient Checkpointing (for large models):**

```python
# In model
def __init__(self):
    self.model = ...
    self.model.gradient_checkpointing_enable()
```

**Learning Rate Finding:**

```python
# Find optimal learning rate
trainer = Trainer()
lr_finder = trainer.tuner.lr_find(model, datamodule=dm)
fig = lr_finder.plot(suggest=True)
```

**Stochastic Weight Averaging:**

```python
from pytorch_lightning.callbacks import StochasticWeightAveraging

trainer = Trainer(callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)])
```

### 9. Post-Training Analysis

After training completes:

**Load Best Checkpoint:**

```python
# From checkpoint callback
best_model_path = trainer.checkpoint_callback.best_model_path
model = MyModel.load_from_checkpoint(best_model_path)
```

**Evaluate on Test Set:**

```python
trainer.test(model, datamodule=dm, ckpt_path="best")
```

**Generate Predictions:**

```python
predictions = trainer.predict(model, datamodule=dm)
```

**Analyze Results:**

- Plot training curves
- Generate confusion matrix
- Calculate per-class metrics
- Visualize learned representations (t-SNE, UMAP)
- Error analysis on misclassified samples

### 10. Training Run Checklist

Before starting long training runs:

- [ ] Config validated and tested with fast_dev_run
- [ ] Data loading verified (correct shapes, no NaN)
- [ ] Checkpoint directory has sufficient disk space
- [ ] Experiment tracking initialized (W&B login, project set)
- [ ] GPU availability checked
- [ ] Learning rate and batch size tested
- [ ] Early stopping configured to prevent wasted compute
- [ ] Training script runs in tmux/screen for long runs
- [ ] Monitoring dashboard accessible
- [ ] Backup strategy in place for checkpoints

## Common Training Commands

```bash
# Quick debug run
python src/train.py trainer.fast_dev_run=5

# Overfit single batch (check model capacity)
python src/train.py trainer.overfit_batches=1 trainer.max_epochs=100

# Profile training
python src/train.py trainer.profiler=advanced trainer.max_epochs=1

# Multi-GPU distributed
python src/train.py trainer.devices=4 trainer.strategy=ddp

# Resume interrupted training
python src/train.py ckpt_path=logs/last.ckpt

# Hyperparameter sweep
python src/train.py --multirun model.lr=0.001,0.01,0.1
```

## Success Criteria

- [ ] Training starts without errors
- [ ] Metrics logged correctly to W&B/TensorBoard
- [ ] Checkpoints saved at expected intervals
- [ ] Training loss decreases steadily
- [ ] Validation metrics improve
- [ ] No NaN or inf values in loss
- [ ] GPU utilization is high (>80%)
- [ ] Training completes or stops early with best model saved

Happy training!
