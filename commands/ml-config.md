---
name: ml-config
description: Generate and manage Hydra configuration files for machine learning experiments (model, data, trainer, logger, experiment configs)
---

# ML Configuration Management

Generate and manage Hydra configuration files for machine learning experiments.

## Process

### 1. Determine Configuration Type

Ask user what type of configuration to create:
- **Model config**: New model architecture
- **Data config**: New dataset or data pipeline
- **Trainer config**: Training hyperparameters
- **Logger config**: Experiment tracking setup
- **Experiment config**: Complete experiment composition
- **Sweep config**: Hyperparameter sweep configuration

### 2. Model Configuration

For model configs, ask:
- Model architecture (CNN, Transformer, GNN, MLP, etc.)
- Input/output dimensions
- Hidden dimensions and layers
- Activation functions
- Normalization layers
- Dropout rates

Generate `configs/model/<name>.yaml`:
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

# Scheduler
scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 100
```

### 3. Data Configuration

For data configs, ask:
- Dataset name/type
- Batch size
- Number of workers
- Data augmentation strategy
- Train/val/test split ratios

Generate `configs/data/<name>.yaml`:
```yaml
_target_: src.data.<name>.DataModule

# Dataset
dataset_name: "mnist"
data_dir: "data/"

# DataLoader
batch_size: 128
num_workers: 4
pin_memory: true
persistent_workers: true

# Splits
train_val_test_split: [0.8, 0.1, 0.1]

# Augmentation
augmentation:
  random_crop: true
  horizontal_flip: true
  normalize: true
```

### 4. Trainer Configuration

For trainer configs, ask:
- Max epochs
- Precision (32, 16-mixed, bf16-mixed)
- Accelerator (auto, gpu, cpu, mps)
- Devices (number of GPUs)
- Strategy (ddp, ddp_notebook, fsdp)
- Gradient clipping
- Accumulation steps

Generate `configs/trainer/<name>.yaml`:
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

# Callbacks
callbacks:
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: "val/loss"
    mode: "min"
    save_top_k: 3
    save_last: true
    filename: "epoch_{epoch:02d}-val_loss_{val/loss:.2f}"

  - _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: "val/loss"
    patience: 10
    mode: "min"

  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: "step"

  - _target_: pytorch_lightning.callbacks.RichProgressBar

# Checkpointing
enable_checkpointing: true
default_root_dir: "checkpoints/"

# Logging
log_every_n_steps: 50
```

### 5. Logger Configuration

For logger configs, based on tracking tool:

**W&B (configs/logger/wandb.yaml):**
```yaml
_target_: pytorch_lightning.loggers.WandbLogger

project: "my-ml-project"
name: null  # Auto-generated from config
save_dir: "logs/"
log_model: true
```

**TensorBoard (configs/logger/tensorboard.yaml):**
```yaml
_target_: pytorch_lightning.loggers.TensorBoardLogger

save_dir: "logs/"
name: null
version: null
log_graph: true
```

**MLFlow (configs/logger/mlflow.yaml):**
```yaml
_target_: pytorch_lightning.loggers.MLFlowLogger

experiment_name: "my-ml-project"
tracking_uri: "file:./mlruns"
log_model: true
```

### 6. Experiment Configuration

For complete experiment configs, compose from existing configs:

Generate `configs/experiment/<name>.yaml`:
```yaml
# @package _global_

defaults:
  - override /model: resnet50
  - override /data: imagenet
  - override /trainer: gpu_ddp
  - override /logger: wandb

# Experiment name
name: "resnet50_imagenet_baseline"

# Seed
seed: 42

# Model overrides
model:
  hidden_dims: [2048, 1024, 512]
  dropout: 0.3

# Data overrides
data:
  batch_size: 256
  num_workers: 8

# Trainer overrides
trainer:
  max_epochs: 200
  devices: 4

# Tags for experiment tracking
tags: ["baseline", "resnet", "imagenet"]
```

### 7. Hyperparameter Sweep Configuration

For sweep configs, ask:
- Parameters to sweep
- Search strategy (grid, random, bayesian)
- Search space (ranges, choices)
- Optimization metric

Generate `configs/sweep/<name>.yaml`:
```yaml
# Hydra multirun configuration
defaults:
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper
    direction: minimize
    n_trials: 50
    n_jobs: 1

    study_name: "mlp_optimization"
    storage: null

    # Optimization parameters
    search_space:
      model.hidden_dims:
        type: categorical
        choices: [[512, 256], [1024, 512, 256], [2048, 1024, 512]]

      model.dropout:
        type: float
        low: 0.1
        high: 0.5

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

### 8. PyTorch Geometric Specific Configs

If using PyG, generate GNN-specific configs:

**configs/model/gnn.yaml:**
```yaml
_target_: src.models.gnn.GNNModel

# GNN architecture
conv_type: GCNConv  # GCNConv, GATConv, SAGEConv, GINConv
num_layers: 3
hidden_channels: 128
out_channels: 64

# GNN-specific
aggr: "add"  # add, mean, max
normalize: true
dropout: 0.2

# Global pooling
global_pool: "mean"  # mean, max, add, attention

# Task-specific head
task: "node_classification"  # node_classification, graph_classification, link_prediction
num_classes: 7
```

**configs/data/graph.yaml:**
```yaml
_target_: src.data.graph_datamodule.GraphDataModule

dataset_name: "Cora"  # Cora, PubMed, PROTEINS, etc.
data_dir: "data/graphs/"

# PyG DataLoader settings
batch_size: 32
num_workers: 4
follow_batch: []
exclude_keys: []

# Sampling (for large graphs)
use_sampling: false
num_neighbors: [15, 10, 5]
```

### 9. Validation

After generating configs:
1. Check YAML syntax validity
2. Verify all `_target_` paths exist
3. Ensure no circular dependencies in defaults
4. Test config loading: `python src/train.py --cfg job`

### 10. Documentation

Update project README with:
- New configuration usage
- Override syntax examples:
  ```bash
  # Override from CLI
  python src/train.py model.lr=0.01 data.batch_size=256

  # Use specific experiment
  python src/train.py experiment=resnet50_imagenet

  # Run hyperparameter sweep
  python src/train.py --multirun model.dropout=0.1,0.2,0.3,0.4
  ```

## Configuration Best Practices

- **Naming**: Use descriptive names (not config1, config2)
- **Modularity**: Create reusable components
- **Documentation**: Add comments explaining parameters
- **Validation**: Use Pydantic models for type checking
- **Versioning**: Track config changes in git
- **Defaults**: Provide sensible defaults for all parameters

## Success Criteria

- [ ] Configuration files created in correct locations
- [ ] YAML syntax is valid
- [ ] All _target_ paths resolve correctly
- [ ] Config loads without errors
- [ ] Documentation updated
- [ ] Tested with actual training run

Configuration is ready for experimentation!
