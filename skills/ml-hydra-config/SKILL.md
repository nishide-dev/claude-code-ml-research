---
name: ml-hydra-config
description: Comprehensive guide for Hydra configuration management, hierarchical configs, experiment management, Optuna integration, and Lightning integration patterns
---

# Hydra Configuration Management for ML Research

## Overview

Hydra is a powerful configuration management framework developed by Facebook AI Research (FAIR) that enables hierarchical configuration composition and dynamic parameter overriding. It's the de facto standard for managing complex ML experiments, allowing researchers to focus on science rather than configuration boilerplate.

**Key Capabilities:**

- Hierarchical configuration with composition
- Command-line overrides without touching code
- Multi-run for hyperparameter sweeps
- Dynamic object instantiation from configs
- Automatic experiment directory management
- Integration with Optuna for hyperparameter optimization

**Resources:**

- Official docs: <https://hydra.cc/docs/intro/>
- Lightning-Hydra-Template: <https://github.com/ashleve/lightning-hydra-template>

---

## Core Concepts

### 1. Hierarchical Configuration and Composition

Hydra's main strength is splitting configuration into meaningful units (config groups) that can be composed at runtime.

**Project structure:**

```text
project/
├── configs/
│   ├── config.yaml          # Main config
│   ├── model/
│   │   ├── resnet.yaml
│   │   └── vit.yaml
│   ├── optimizer/
│   │   ├── adam.yaml
│   │   └── sgd.yaml
│   └── dataset/
│       ├── cifar10.yaml
│       └── imagenet.yaml
└── train.py
```

**Main config (configs/config.yaml):**

```yaml
defaults:
  - model: resnet           # Load configs/model/resnet.yaml
  - optimizer: adam         # Load configs/optimizer/adam.yaml
  - dataset: cifar10        # Load configs/dataset/cifar10.yaml

seed: 42
trainer:
  max_epochs: 100
  accelerator: gpu
```

**Model config (configs/model/resnet.yaml):**

```yaml
_target_: torchvision.models.resnet18
pretrained: false
num_classes: 10
```

**Command-line overrides:**

```bash
# Change model and optimizer
python train.py model=vit optimizer=sgd

# Override nested parameters
python train.py model.num_classes=100 trainer.max_epochs=200

# Mix config groups and parameter overrides
python train.py model=resnet optimizer=adam model.pretrained=true
```

### 2. Object Instantiation with `_target_`

The `_target_` key enables dynamic object creation from configs, eliminating conditional logic and adhering to the Open-Closed Principle.

**Config with `_target_`:**

```yaml
# configs/model/custom_model.yaml
_target_: src.models.MyModel
input_dim: 128
hidden_dim: 512
num_layers: 3
dropout: 0.1
```

**Python code:**

```python
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    # Instantiate model directly from config
    model = instantiate(cfg.model)

    # Instantiate optimizer with model parameters
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # Instantiate entire training pipeline
    trainer = instantiate(cfg.trainer)
    datamodule = instantiate(cfg.data)

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    train()
```

**Optimizer config example:**

```yaml
# configs/optimizer/adam.yaml
_target_: torch.optim.Adam
lr: 0.001
betas: [0.9, 0.999]
weight_decay: 0.0001
```

### 3. Multi-Run for Hyperparameter Sweeps

The `-m` (multirun) flag enables running multiple experiments with different parameter combinations.

**Basic multirun:**

```bash
# Run 6 experiments (3 batch sizes × 2 learning rates)
python train.py -m \
  data.batch_size=32,64,128 \
  model.lr=1e-3,1e-4
```

**Parameter sweep with range:**

```bash
# Sweep learning rate with 10 evenly spaced values
python train.py -m model.lr=interval(1e-5,1e-2,10)

# Range notation
python train.py -m seed=range(1,11)  # Seeds 1-10
```

**Multirun with config groups:**

```bash
# Test all model × optimizer combinations
python train.py -m model=resnet,vit optimizer=adam,sgd
```

---

## Integration with PyTorch Lightning

Hydra and Lightning form the standard modern ML stack. The lightning-hydra-template is the community-standard project structure.

### Recommended Project Structure

```text
project/
├── configs/
│   ├── config.yaml                    # Main config with defaults
│   ├── experiment/                     # Experiment-specific configs
│   │   ├── cifar10_resnet.yaml
│   │   └── imagenet_vit.yaml
│   ├── model/                          # LightningModule configs
│   │   ├── resnet.yaml
│   │   └── vit.yaml
│   ├── data/                           # DataModule configs
│   │   ├── cifar10.yaml
│   │   └── imagenet.yaml
│   ├── trainer/                        # Lightning Trainer configs
│   │   ├── default.yaml
│   │   ├── gpu.yaml
│   │   └── ddp.yaml
│   ├── callbacks/                      # Callback configs
│   │   ├── default.yaml
│   │   └── early_stopping.yaml
│   ├── logger/                         # Logger configs
│   │   ├── wandb.yaml
│   │   └── tensorboard.yaml
│   └── hparams_search/                 # HPO configs
│       └── mnist_optuna.yaml
├── src/
│   ├── models/                         # LightningModule implementations
│   │   ├── __init__.py
│   │   └── classifier.py
│   ├── data/                           # DataModule implementations
│   │   ├── __init__.py
│   │   └── image_datamodule.py
│   ├── utils/                          # Utilities
│   │   ├── __init__.py
│   │   └── instantiators.py
│   └── train.py                        # Training entrypoint
└── tests/
    └── test_configs.py
```

### Complete Training Script Example

**src/train.py:**

```python
import lightning as L
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pyrootutils

# Find project root
root = pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> float:
    """Train model using Hydra config.

    Args:
        cfg: Hydra configuration object

    Returns:
        Best validation metric
    """
    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Instantiate datamodule
    datamodule: L.LightningDataModule = instantiate(cfg.data)

    # Instantiate model
    model: L.LightningModule = instantiate(cfg.model)

    # Instantiate callbacks
    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(instantiate(cb_conf))

    # Instantiate logger
    logger = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                logger.append(instantiate(lg_conf))

    # Instantiate trainer
    trainer: L.Trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train
    if cfg.get("train"):
        trainer.fit(model, datamodule=datamodule)

    # Test
    if cfg.get("test"):
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            ckpt_path = None
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Return best metric for hyperparameter optimization
    metric_dict = trainer.callback_metrics
    metric_value = metric_dict.get(cfg.get("optimized_metric", "val/loss"))
    return float(metric_value) if metric_value is not None else None

if __name__ == "__main__":
    train()
```

### Configuration Examples

**configs/config.yaml:**

```yaml
# @package _global_

defaults:
  - model: resnet
  - data: cifar10
  - trainer: default
  - logger: wandb
  - callbacks: default
  - _self_

# Seed for reproducibility
seed: 42

# Enable/disable training and testing
train: true
test: true

# Metric to optimize (for HPO)
optimized_metric: "val/acc"

# Working directory
work_dir: ${hydra:runtime.cwd}
```

**configs/model/resnet.yaml:**

```yaml
_target_: src.models.classifier.ImageClassifier

# Model architecture
backbone:
  _target_: torchvision.models.resnet18
  pretrained: false

num_classes: 10

# Optimizer
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
  weight_decay: 0.0001

# Learning rate scheduler
scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 100
  eta_min: 1e-6
```

**configs/trainer/gpu.yaml:**

```yaml
_target_: lightning.Trainer

accelerator: gpu
devices: 1
precision: 16-mixed

max_epochs: 100
gradient_clip_val: 1.0

# Validation frequency
val_check_interval: 1.0
check_val_every_n_epoch: 1

# Logging
log_every_n_steps: 50

# Debugging
fast_dev_run: false
overfit_batches: 0.0

# Deterministic mode
deterministic: false
```

**configs/callbacks/default.yaml:**

```yaml
model_checkpoint:
  _target_: lightning.pytorch.callbacks.ModelCheckpoint
  dirpath: ${paths.output_dir}/checkpoints
  filename: "epoch_{epoch:03d}"
  monitor: "val/acc"
  mode: "max"
  save_top_k: 3
  save_last: true
  auto_insert_metric_name: false
  verbose: false

early_stopping:
  _target_: lightning.pytorch.callbacks.EarlyStopping
  monitor: "val/acc"
  mode: "max"
  patience: 10
  min_delta: 0.001
  verbose: false

rich_progress_bar:
  _target_: lightning.pytorch.callbacks.RichProgressBar
  refresh_rate: 1
```

---

## Experiment Configs

Experiment configs bundle related configurations into a single file for easy reproducibility.

**configs/experiment/cifar10_resnet.yaml:**

```yaml
# @package _global_

defaults:
  - override /data: cifar10
  - override /model: resnet
  - override /trainer: gpu
  - override /callbacks: default
  - override /logger: wandb

# Experiment tags for filtering in W&B
tags: ["cifar10", "resnet", "baseline"]

# Seed for reproducibility
seed: 12345

# Training parameters
trainer:
  max_epochs: 100

model:
  optimizer:
    lr: 0.01
  num_classes: 10

data:
  batch_size: 128
  num_workers: 4

# W&B config
logger:
  wandb:
    project: "cifar10-classification"
    name: "resnet18-baseline"
```

**Running experiments:**

```bash
# Run predefined experiment
python src/train.py experiment=cifar10_resnet

# Override experiment parameters
python src/train.py experiment=cifar10_resnet model.optimizer.lr=0.001

# Run multiple experiments
python src/train.py -m experiment=cifar10_resnet,imagenet_vit
```

---

## Hyperparameter Optimization with Optuna

Hydra's Optuna Sweeper plugin enables advanced hyperparameter optimization without code changes.

### Installation

```bash
pip install hydra-optuna-sweeper
```

### HPO Configuration

**configs/hparams_search/mnist_optuna.yaml:**

```yaml
# @package _global_

defaults:
  - override /hydra/sweeper: optuna

hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    # Optimization direction
    direction: maximize

    # Study name for persistence
    study_name: mnist_hpo

    # Storage (optional, for persistence)
    storage: null  # e.g., "sqlite:///optuna.db"

    # Number of trials
    n_trials: 50

    # Number of parallel jobs
    n_jobs: 1

    # Sampler
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
      n_startup_trials: 10

    # Define search space
    params:
      model.optimizer.lr: interval(1e-5, 1e-1)
      model.hidden_dim: choice(64, 128, 256, 512)
      data.batch_size: choice(32, 64, 128)
      model.dropout: interval(0.1, 0.5)
      model.optimizer.weight_decay: interval(1e-6, 1e-3)
```

**Running HPO:**

```bash
# Run hyperparameter search
python src/train.py -m hparams_search=mnist_optuna

# With experiment config
python src/train.py -m \
  experiment=cifar10_resnet \
  hparams_search=mnist_optuna
```

### Advanced Optuna Features

**Pruning (early stopping bad trials):**

```yaml
hydra:
  sweeper:
    # Add pruner
    pruner:
      _target_: optuna.pruners.HyperbandPruner
      min_resource: 1
      max_resource: 100
      reduction_factor: 3
```

**Logging integration:**

```python
# In your LightningModule
from lightning.pytorch.callbacks import Callback
from optuna.integration import PyTorchLightningPruningCallback

class OptunaCallback(PyTorchLightningPruningCallback):
    def on_validation_end(self, trainer, pl_module):
        # Report metric to Optuna
        epoch = trainer.current_epoch
        current_score = trainer.callback_metrics.get("val/acc")
        self.check_pruned()
```

---

## Advanced Patterns

### 1. Structured Configs with Dataclasses

Use Python dataclasses for type-safe configs with IDE autocomplete.

```python
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    _target_: str = "src.models.MyModel"
    input_dim: int = 128
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    seed: int = 42
    train: bool = True

# Register with Hydra
cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: Config):  # Now type-safe!
    model = instantiate(cfg.model)
```

### 2. Custom Resolvers

Create custom interpolation functions for configs.

```python
from omegaconf import OmegaConf

# Register custom resolvers
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)

# In config.yaml:
# total_steps: ${multiply:${trainer.max_epochs},${data.steps_per_epoch}}
```

### 3. Partial Instantiation

Useful for callbacks and functions that need runtime arguments.

```python
from functools import partial

# Config
callbacks:
  custom:
    _target_: src.callbacks.CustomCallback
    _partial_: true
    threshold: 0.95

# Usage
callback_fn = instantiate(cfg.callbacks.custom)
callback = callback_fn(model=model)  # Provide runtime args
```

---

## Best Practices

### ✅ DO

1. **Use config groups for modularity**: One config file per logical component (model, data, trainer, etc.)

2. **Leverage `_target_` for instantiation**: Avoid conditional logic; let configs define what to create

3. **Use experiment configs**: Bundle related settings into reusable experiment files

4. **Fix seeds for reproducibility**: Always set `seed_everything(cfg.seed)`

5. **Use relative imports**: Keep configs portable with `${hydra:runtime.cwd}`

6. **Validate configs early**: Use `fast_dev_run=true` and `overfit_batches` to catch errors before long runs

7. **Version control configs**: Track experiment configs in git for full reproducibility

8. **Use meaningful names**: Name experiments descriptively (e.g., `resnet18_cifar10_aug_heavy`)

### ❌ DON'T

1. **Don't pass giant cfg objects everywhere**: Extract and pass only needed parameters to functions

2. **Don't mutate configs at runtime**: Configs should be immutable; use class attributes for state

3. **Don't hardcode values in code**: All varying parameters should be in configs

4. **Don't over-abstract**: Balance flexibility with readability; don't hide control flow entirely

5. **Don't ignore output directories**: Each run gets a unique directory; use it for reproducibility

6. **Don't skip documentation**: Comment complex config compositions and overrides

---

## Debugging and Development

### Fast Development Iteration

```bash
# Run single batch (catches errors fast)
python train.py trainer.fast_dev_run=true

# Overfit on small data (verify model can learn)
python train.py trainer.overfit_batches=10

# Limit training data
python train.py trainer.limit_train_batches=100

# Dry run (show config without running)
python train.py --cfg job --resolve
```

### Config Validation

**tests/test_configs.py:**

```python
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

@pytest.mark.parametrize("model", ["resnet", "vit"])
@pytest.mark.parametrize("data", ["cifar10", "imagenet"])
def test_config_composition(model: str, data: str):
    """Test that all model × data combinations are valid."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[f"model={model}", f"data={data}"]
        )
        assert isinstance(cfg, DictConfig)
        assert cfg.model._target_ is not None
        assert cfg.data._target_ is not None
```

---

## Common Patterns

### Pattern 1: Resume Training from Checkpoint

```yaml
# config.yaml
ckpt_path: null  # Set to checkpoint path to resume

# train.py
if cfg.ckpt_path:
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
else:
    trainer.fit(model, datamodule=datamodule)
```

```bash
# Resume from checkpoint
python train.py ckpt_path=/path/to/checkpoint.ckpt
```

### Pattern 2: Conditional Config Loading

```yaml
defaults:
  - model: resnet
  - data: cifar10
  - trainer: default
  - optional logger: wandb  # Only load if exists
  - override hydra/launcher: joblib  # Override Hydra component
```

### Pattern 3: Environment-Specific Configs

```yaml
# config.yaml
defaults:
  - override /trainer: ${env:TRAINING_ENV,local}  # local/gpu/cluster

# Creates: trainer/local.yaml, trainer/gpu.yaml, trainer/cluster.yaml
```

---

## Essential Resources

- **Official Hydra Docs**: <https://hydra.cc/docs/intro/>
- **Lightning-Hydra-Template**: <https://github.com/ashleve/lightning-hydra-template>
- **Optuna Sweeper Plugin**: <https://hydra.cc/docs/plugins/optuna_sweeper/>
- **Config Store API**: <https://hydra.cc/docs/tutorials/structured_config/config_store/>
- **Best Practices**: <https://hydra.cc/docs/patterns/configuring_experiments/>

---

## Summary

Hydra transforms ML experiment management from chaotic to systematic:

- **Composition over duplication**: Reuse config components across experiments
- **Command-line flexibility**: Override any parameter without touching code
- **Reproducibility by default**: Every run gets a unique directory with config snapshot
- **Scalable experimentation**: From single runs to massive hyperparameter sweeps
- **Type safety**: Structured configs provide IDE support and validation

Combined with PyTorch Lightning, Hydra forms the foundation of modern, scalable ML research infrastructure.
