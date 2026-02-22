---
name: ml-config-manager
description: Generate and manage Hydra configuration files for machine learning experiments. Use when creating new configs (model, data, trainer, logger, experiment, sweep), organizing config hierarchies, or setting up hyperparameter sweeps with Optuna.
disable-model-invocation: true
---

# ML Configuration Management

Generate and manage Hydra configuration files for machine learning experiments with PyTorch Lightning.

## Quick Reference

**Template Files Available:**

- `templates/model-config.yaml` - Model architecture configuration
- `templates/data-config.yaml` - Dataset and DataLoader configuration
- `templates/trainer-config.yaml` - PyTorch Lightning Trainer configuration
- `templates/experiment-config.yaml` - Complete experiment composition
- `templates/gnn-config.yaml` - PyTorch Geometric GNN configuration
- `templates/sweep-config.yaml` - Hyperparameter sweep with Optuna

## Configuration Types

### 1. Model Configuration

**Location:** `configs/model/<name>.yaml`

**What to ask:**

- Model architecture (CNN, Transformer, GNN, MLP, etc.)
- Input/output dimensions
- Hidden dimensions and layers
- Activation functions
- Normalization layers (batch norm, layer norm)
- Dropout rates
- Optimizer type and parameters
- Learning rate scheduler

**Generate from template:**

```yaml
_target_: src.models.<name>.Model

# Architecture
input_dim: 784
hidden_dims: [512, 256, 128]
output_dim: 10
activation: relu
dropout: 0.2
batch_norm: true

# Optimizer
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 0.0001
  betas: [0.9, 0.999]

# Scheduler
scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 100
  eta_min: 1e-6
```

**Common optimizers:**

- `torch.optim.AdamW` - Default choice, good for most tasks
- `torch.optim.Adam` - Classic optimizer
- `torch.optim.SGD` - With momentum for vision tasks
- `torch.optim.RMSprop` - Good for recurrent networks

**Common schedulers:**

- `CosineAnnealingLR` - Smooth cosine decay
- `ReduceLROnPlateau` - Reduce on metric plateau
- `OneCycleLR` - Super-convergence with 1cycle policy
- `StepLR` - Step decay at fixed intervals
- `ExponentialLR` - Exponential decay

See `templates/model-config.yaml` for complete template.

### 2. Data Configuration

**Location:** `configs/data/<name>.yaml`

**What to ask:**

- Dataset name/type
- Batch size
- Number of workers
- Data augmentation strategy
- Train/val/test split ratios
- Preprocessing requirements

**Generate from template:**

```yaml
_target_: src.data.<name>.DataModule

# Dataset
dataset_name: "mnist"
data_dir: "data/"
download: true

# DataLoader
batch_size: 128
num_workers: 4
pin_memory: true
persistent_workers: true
prefetch_factor: 2

# Splits
train_val_test_split: [0.8, 0.1, 0.1]
shuffle_train: true

# Augmentation
augmentation:
  random_crop: true
  horizontal_flip: true
  normalize: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```

See `templates/data-config.yaml` for complete template with all augmentation options.

### 3. Trainer Configuration

**Location:** `configs/trainer/<name>.yaml`

**What to ask:**

- Max epochs
- Precision (32, 16-mixed, bf16-mixed)
- Accelerator (auto, gpu, cpu, mps)
- Number of devices (GPUs)
- Strategy (auto, ddp, fsdp, deepspeed)
- Gradient clipping value
- Accumulation steps
- Validation frequency

**Generate from template:**

```yaml
_target_: pytorch_lightning.Trainer

# Training duration
max_epochs: 100
min_epochs: 1

# Hardware
accelerator: gpu
devices: 1
precision: 16-mixed

# Distributed
strategy: auto

# Optimization
gradient_clip_val: 1.0
accumulate_grad_batches: 1

# Validation
check_val_every_n_epoch: 1
num_sanity_val_steps: 2

# Callbacks
callbacks:
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: "val/loss"
    mode: "min"
    save_top_k: 3
    save_last: true

  - _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: "val/loss"
    patience: 10
    mode: "min"

  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: "step"

  - _target_: pytorch_lightning.callbacks.RichProgressBar
```

See `templates/trainer-config.yaml` for complete template with all callbacks and options.

### 4. Logger Configuration

**Location:** `configs/logger/<name>.yaml`

**W&B (Recommended):**

```yaml
_target_: pytorch_lightning.loggers.WandbLogger

project: "my-ml-project"
name: null  # Auto-generated from config
save_dir: "logs/"
log_model: true
save_code: true
```

**TensorBoard:**

```yaml
_target_: pytorch_lightning.loggers.TensorBoardLogger

save_dir: "logs/"
name: null
version: null
log_graph: true
default_hp_metric: false
```

**MLflow:**

```yaml
_target_: pytorch_lightning.loggers.MLFlowLogger

experiment_name: "my-ml-project"
tracking_uri: "file:./mlruns"
log_model: true
```

### 5. Experiment Configuration

**Location:** `configs/experiment/<name>.yaml`

Compose complete experiments from existing configs:

```yaml
# @package _global_

# Experiment metadata
name: "resnet50_imagenet_baseline"
description: "Baseline ResNet-50 on ImageNet with standard augmentation"
tags: ["baseline", "resnet", "imagenet"]

# Compose from existing configs
defaults:
  - override /model: resnet50
  - override /data: imagenet
  - override /trainer: gpu_ddp
  - override /logger: wandb

# Seed for reproducibility
seed: 42

# Model overrides (specific to this experiment)
model:
  hidden_dims: [2048, 1024, 512]
  dropout: 0.3
  optimizer:
    lr: 0.001

# Data overrides
data:
  batch_size: 256
  num_workers: 8

# Trainer overrides
trainer:
  max_epochs: 200
  devices: 4

# Logger configuration
logger:
  wandb:
    project: "imagenet-experiments"
    tags: ${tags}
    notes: ${description}
```

See `templates/experiment-config.yaml` for complete template.

### 6. Hyperparameter Sweep Configuration

**Location:** `configs/sweep/<name>.yaml`

**What to ask:**

- Parameters to sweep (model, data, training params)
- Search strategy (grid, random, bayesian/Optuna)
- Search space (ranges, choices)
- Optimization metric and direction
- Number of trials

**Generate from template (Optuna Bayesian Optimization):**

```yaml
defaults:
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    direction: minimize  # minimize or maximize
    n_trials: 50
    n_jobs: 1

    study_name: "mlp_optimization"
    storage: null  # null for in-memory, or "sqlite:///optuna.db"

    # Sampler
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
      n_startup_trials: 10

    # Search space
    params:
      model.hidden_dims:
        type: categorical
        choices:
          - [512, 256]
          - [1024, 512, 256]
          - [2048, 1024, 512]

      model.dropout:
        type: float
        low: 0.0
        high: 0.5
        step: 0.05

      model.optimizer.lr:
        type: float
        low: 0.0001
        high: 0.01
        log: true

      data.batch_size:
        type: categorical
        choices: [64, 128, 256]

# Metric to optimize
optimized_metric: "val/loss"
```

**Run sweep:**

```bash
python src/train.py --multirun --config-name sweep/<name>
```

**Alternative: Grid search (no Optuna needed):**

```bash
python src/train.py --multirun \
  model.hidden_dims="[512,256],[1024,512,256]" \
  model.dropout=0.1,0.2,0.3,0.4 \
  model.optimizer.lr=0.001,0.01,0.1
```

See `templates/sweep-config.yaml` for complete template with all options.

### 7. PyTorch Geometric Specific Configs

**Location:** `configs/model/gnn/<name>.yaml`

For Graph Neural Networks with PyTorch Geometric:

```yaml
_target_: src.models.gnn.GNNModel

# GNN architecture
conv_type: GCNConv  # GCNConv, GATConv, SAGEConv, GINConv, TransformerConv
num_layers: 3
hidden_channels: 128
out_channels: 64

# GNN-specific
aggr: "add"  # add, mean, max
normalize: true
dropout: 0.2
jk_mode: null  # null, cat, max, lstm

# Attention (for GAT/TransformerConv)
heads: 8
concat_heads: true

# Global pooling (for graph-level tasks)
global_pool: "mean"  # mean, max, add, attention, set2set

# Task configuration
task: "node_classification"  # node_classification, graph_classification, link_prediction
num_classes: 7

# Optimizer
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.01
  weight_decay: 5e-4  # Higher weight decay for GNNs
```

**Corresponding graph data config:**

```yaml
_target_: src.data.graph_datamodule.GraphDataModule

dataset_name: "Cora"
data_dir: "data/graphs/"
batch_size: 32

# Sampling (for large graphs)
use_sampling: false
num_neighbors: [15, 10, 5]
```

See `templates/gnn-config.yaml` for complete template.

## Configuration Best Practices

### 1. Naming Conventions

- Use descriptive names: `resnet50_pretrained.yaml`, not `model1.yaml`
- Hierarchical naming: `trainer/gpu_single.yaml`, `trainer/gpu_multi.yaml`
- Domain prefixes: `data/vision/`, `data/nlp/`, `data/graph/`

### 2. Modularity

- Create reusable components
- Use defaults composition for experiments
- Override only what's necessary
- Keep configs DRY (Don't Repeat Yourself)

### 3. Documentation

Add comments explaining non-obvious parameters:

```yaml
# Use higher weight decay for GNNs to prevent overfitting
weight_decay: 5e-4

# TPESampler needs startup trials for warm-up
n_startup_trials: 10
```

### 4. Type Safety

Use Pydantic models for validation:

```python
from pydantic import BaseModel

class ModelConfig(BaseModel):
    input_dim: int
    hidden_dims: list[int]
    output_dim: int
    dropout: float
```

### 5. Versioning

- Track all config changes in git
- Tag configs with experiment versions
- Use meaningful commit messages for config changes

### 6. Sensible Defaults

- Provide defaults for all optional parameters
- Use industry-standard hyperparameters as defaults
- Document why defaults were chosen

## Validation Checklist

After generating configs, validate:

- [ ] YAML syntax is valid (no tabs, correct indentation)
- [ ] All `_target_` paths exist and are importable
- [ ] No circular dependencies in defaults
- [ ] Config loads without errors: `python src/train.py --cfg job`
- [ ] Print resolved config: `python src/train.py --cfg job --print-config`
- [ ] All file paths are correct (data_dir, save_dir, etc.)
- [ ] Hyperparameter ranges are reasonable
- [ ] Batch size fits in GPU memory

## Common CLI Overrides

```bash
# Override single parameter
python src/train.py model.lr=0.01

# Override multiple parameters
python src/train.py model.lr=0.01 data.batch_size=256

# Use specific experiment config
python src/train.py experiment=resnet50_imagenet

# Override nested parameters
python src/train.py model.optimizer.weight_decay=1e-4

# Use different config group
python src/train.py model=transformer data=wikitext

# Multirun (grid search)
python src/train.py --multirun model.dropout=0.1,0.2,0.3

# Print resolved config
python src/train.py --cfg job

# Print config with overrides
python src/train.py model.lr=0.01 --cfg job
```

## Debugging Config Issues

**Config doesn't load:**

```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/model/mymodel.yaml'))"

# Validate with Hydra
python src/train.py --cfg job --config-name myconfig
```

**Import errors:**

- Verify `_target_` paths are correct
- Check all modules are importable
- Use absolute imports: `src.models.mlp.MLP` not `models.mlp.MLP`

**Defaults not resolving:**

- Check defaults order (later overrides earlier)
- Use `override` keyword for conflicting groups
- Verify group names match directory structure

**Variable interpolation not working:**

- Use `${var}` syntax for interpolation
- Check variable exists in config
- Use `${oc.env:VAR}` for environment variables

## Integration with Training

Once configs are created, use them in training:

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    # Instantiate components from config
    model = hydra.utils.instantiate(cfg.model)
    datamodule = hydra.utils.instantiate(cfg.data)
    trainer = hydra.utils.instantiate(cfg.trainer)

    # Train
    trainer.fit(model, datamodule=datamodule)
```

## Success Criteria

- [ ] Configuration files created in correct locations
- [ ] YAML syntax is valid
- [ ] All `_target_` paths resolve correctly
- [ ] Config loads without errors
- [ ] Overrides work as expected
- [ ] Documentation updated (if needed)
- [ ] Tested with actual training run

Configuration is ready for experimentation!
