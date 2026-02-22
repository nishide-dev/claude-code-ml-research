---
name: ml-train
description: Execute training runs with proper monitoring, checkpointing, and experiment tracking. Use when starting training, resuming training, debugging training issues, or setting up multi-GPU/distributed training with PyTorch Lightning and Hydra.
argument-hint: [config-path]
disable-model-invocation: true
---

# ML Training Execution

Execute training runs with proper monitoring, checkpointing, and experiment tracking using PyTorch Lightning and Hydra.

## Quick Start

Choose a training template based on your setup:

- **[Basic single-GPU](templates/basic-training.yaml)** - Simple training on one GPU
- **[Multi-GPU single-node](templates/multi-gpu-training.yaml)** - Multiple GPUs on one machine
- **[Distributed multi-node](templates/distributed-training.yaml)** - Training across multiple machines
- **[FSDP large models](templates/fsdp-large-model.yaml)** - Very large models with Fully Sharded Data Parallel

## Training Commands

**Basic training:**

```bash
python src/train.py
```

**With specific experiment config:**

```bash
python src/train.py experiment=my_experiment
```

**With CLI overrides:**

```bash
python src/train.py \
  model.learning_rate=1e-3 \
  data.batch_size=64 \
  trainer.max_epochs=100
```

**Resume from checkpoint:**

```bash
python src/train.py ckpt_path="checkpoints/epoch_42.ckpt"
```

**Multi-GPU training:**

```bash
python src/train.py \
  trainer.devices=4 \
  trainer.strategy=ddp
```

**Hyperparameter sweep:**

```bash
python src/train.py --multirun \
  model.learning_rate=1e-4,1e-3,1e-2 \
  data.batch_size=32,64,128
```

## Pre-Training Checklist

Before starting training, verify:

**Environment:**

```bash
# Check Python version
python --version  # Should be >= 3.10

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Check package installation
python -c "import pytorch_lightning as pl; print(f'Lightning: {pl.__version__}')"

# Validate config
python src/train.py --cfg job

# Dry run
python src/train.py trainer.fast_dev_run=5
```

**Disk space:**

- Estimate checkpoint storage: `model_size_mb × save_top_k × num_epochs / checkpoint_freq`
- Ensure logging directory has sufficient space

**Experiment tracking:**

```bash
# Initialize W&B (if using)
wandb login
export WANDB_PROJECT="your-project-name"

# Sync offline runs (if needed)
wandb sync
```

## Monitoring Training

**Real-time monitoring:**

- Progress bar: Lightning shows metrics automatically
- GPU usage: `watch -n 1 nvidia-smi`
- W&B dashboard: Metrics, system stats, model graphs

**Key metrics to watch:**

- **Training loss**: Should decrease steadily
- **Validation loss**: Should decrease without diverging from train loss
- **Learning rate**: Verify scheduler is working
- **GPU utilization**: Should be >80%
- **Training speed**: Samples/sec or batches/sec

**Red flags:**

- **Loss is NaN/inf**: Check learning rate, add gradient clipping
- **Val loss increasing, train decreasing**: Overfitting - add regularization
- **Very slow training**: Check data loading (num_workers), batch size
- **Low GPU usage**: Increase batch size, check data pipeline
- **Out of memory**: Reduce batch size, use mixed precision, gradient accumulation

## Common Training Issues

**Gradient issues:**

```python
# Add to trainer config
trainer = Trainer(
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    track_grad_norm=2,  # Log gradient norms
)
```

**Memory issues:**

```python
# Use mixed precision + gradient accumulation
trainer = Trainer(
    precision="16-mixed",
    accumulate_grad_batches=4,
)

# Compile model (PyTorch 2.0+)
model = torch.compile(model)
```

**Slow data loading:**

```python
# Profile to identify bottleneck
trainer = Trainer(profiler="simple")

# Optimize data loading
data_module = DataModule(
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
```

**Overfitting:**

```python
# Add early stopping
from pytorch_lightning.callbacks import EarlyStopping

trainer = Trainer(
    callbacks=[
        EarlyStopping(monitor="val/loss", patience=10, min_delta=0.001)
    ]
)

# Increase regularization
model:
  dropout: 0.3
  weight_decay: 0.0001
```

## Multi-GPU Strategies

**DDP (Distributed Data Parallel):**

- Best for: Most use cases, models that fit in single GPU memory
- Usage: `trainer.strategy=ddp trainer.devices=4`
- Pros: Fast, stable, well-tested
- Cons: Each GPU holds full model copy

**FSDP (Fully Sharded Data Parallel):**

- Best for: Very large models (>10B parameters)
- Usage: See [FSDP template](templates/fsdp-large-model.yaml)
- Pros: Shard model across GPUs, minimal memory per GPU
- Cons: Slower than DDP, more complex setup

**DeepSpeed:**

- Best for: Extreme scale (100B+ parameters)
- Usage: `trainer.strategy=deepspeed_stage_3`
- Pros: Most memory efficient, supports ZeRO optimization
- Cons: Requires additional configuration

## Advanced Techniques

**Mixed precision training:**

```bash
python src/train.py trainer.precision=16-mixed
```

**Gradient checkpointing (save memory):**

```python
model.gradient_checkpointing_enable()
```

**Learning rate finder:**

```python
trainer = Trainer()
lr_finder = trainer.tuner.lr_find(model, datamodule=dm)
fig = lr_finder.plot(suggest=True)
```

**Stochastic Weight Averaging:**

```python
from pytorch_lightning.callbacks import StochasticWeightAveraging

trainer = Trainer(callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)])
```

## Post-Training

**Load best checkpoint:**

```python
best_model_path = trainer.checkpoint_callback.best_model_path
model = MyModel.load_from_checkpoint(best_model_path)
```

**Evaluate on test set:**

```python
trainer.test(model, datamodule=dm, ckpt_path="best")
```

**Generate predictions:**

```python
predictions = trainer.predict(model, datamodule=dm)
```

## Domain-Specific Training

### PyTorch Geometric (GNN Training)

For graph neural networks, see [GNN training guide](examples/gnn-training.md):

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

**GNN-specific metrics to monitor:**

- Node-level accuracy / Graph-level accuracy
- Over-smoothing (node representations becoming too similar)
- Graph connectivity statistics
- Layer-wise gradient norms

See [complete GNN guide](examples/gnn-training.md) for architectures, sampling strategies, and troubleshooting.

## Hyperparameter Optimization

### Quick Grid Search

```bash
python src/train.py --multirun \
  model.learning_rate=1e-4,1e-3,1e-2 \
  data.batch_size=32,64,128
```

### Bayesian Optimization with Optuna

```bash
python src/train.py \
  --multirun \
  hydra/sweeper=optuna \
  hydra.sweeper.n_trials=50
```

For advanced sweep configurations (random search, Bayesian optimization, multi-objective), see [reference guide](reference.md#hyperparameter-sweeps).

## W&B Integration

### Track Custom Metrics

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

### Log Artifacts

W&B can automatically log:

- Model checkpoints (`log_model=true`)
- Confusion matrices
- Sample predictions
- Model graphs

See [reference guide](reference.md#w&b-advanced-integration) for logging confusion matrices, sample predictions, and custom artifacts.

## Common Debug Commands

```bash
# Quick debug run (5 batches)
python src/train.py trainer.fast_dev_run=5

# Overfit single batch (check model capacity)
python src/train.py trainer.overfit_batches=1 trainer.max_epochs=100

# Profile training (identify bottlenecks)
python src/train.py trainer.profiler=advanced trainer.max_epochs=1
```

For complete command reference, see [reference guide](reference.md#common-training-commands-reference).

## Examples

See complete training examples:

- **[Image classification](examples/image-classification.md)** - CIFAR-10 with ResNet
- **[Text classification](examples/text-classification.md)** - IMDB with BERT
- **[Graph neural networks](examples/gnn-training.md)** - Node/graph classification with PyTorch Geometric

## Troubleshooting

**Training doesn't start:**

- Check config syntax: `python src/train.py --cfg job`
- Verify imports: `python -c "import pytorch_lightning; import hydra"`
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

**Training is unstable:**

- Lower learning rate
- Add gradient clipping
- Check for NaN in data
- Try different optimizer (AdamW → SGD)

**Training is slow:**

- Profile: `trainer.profiler="advanced"`
- Check data loading bottleneck
- Increase batch size
- Use mixed precision
- Enable `torch.compile()` (PyTorch 2.0+)

## Success Criteria

- [ ] Training starts without errors
- [ ] Metrics logged correctly to W&B/TensorBoard
- [ ] Checkpoints saved at expected intervals
- [ ] Training loss decreases steadily
- [ ] Validation metrics improve
- [ ] No NaN or inf values in loss
- [ ] GPU utilization is high (>80%)
- [ ] Training completes or stops early with best model saved

## Additional Resources

For advanced topics, see:

- **[Reference Guide](reference.md)** - Hyperparameter sweeps (Optuna, Bayesian), W&B artifacts, environment variables, cloud/HPC training
- **[GNN Training](examples/gnn-training.md)** - PyTorch Geometric specific guide with sampling strategies and architectures
- **[Image Classification](examples/image-classification.md)** - Complete CIFAR-10 example with ResNet
- **[Text Classification](examples/text-classification.md)** - Complete IMDB example with BERT

Happy training! 🚀
