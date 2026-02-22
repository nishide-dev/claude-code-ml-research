# Hyperparameter Sweep Configuration Example

Complete guide to hyperparameter optimization with Hydra and Optuna.

## Three Sweep Strategies

1. **Grid Search** - Exhaustive search (use for small search spaces)
2. **Random Search** - Random sampling (faster than grid)
3. **Bayesian Optimization** - Optuna TPE sampler (most efficient)

## 1. Grid Search (No Optuna Required)

Simple multirun with Hydra's built-in sweeper.

### Basic Grid Search

```bash
# Search over 3 learning rates and 3 batch sizes = 9 runs
python src/train.py --multirun \
  model.optimizer.lr=0.0001,0.001,0.01 \
  data.batch_size=64,128,256
```

### Grid Search with Config File

**configs/sweep/grid_search.yaml:**

```yaml
# No special sweeper needed - use Hydra multirun

defaults:
  - override /model: resnet18
  - override /data: cifar10
  - override /trainer: gpu_single
  - override /logger: wandb

# Grid parameters (use via CLI)
# python src/train.py --config-name sweep/grid_search --multirun \
#   model.optimizer.lr=0.0001,0.001,0.01 \
#   data.batch_size=64,128,256
```

### Grid Search Advantages

- No additional dependencies
- Simple and predictable
- Good for small search spaces (<100 combinations)

### Grid Search Disadvantages

- Exponential growth with parameters
- Inefficient for large spaces
- No early stopping of bad trials

## 2. Random Search

Random sampling from parameter distributions.

### Optuna Random Sampler

**configs/sweep/random_search.yaml:**

```yaml
defaults:
  - override /model: resnet18
  - override /data: cifar10
  - override /trainer: gpu_single
  - override /logger: wandb
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    # Random search
    sampler:
      _target_: optuna.samplers.RandomSampler
      seed: 42

    # Direction and trials
    direction: maximize
    n_trials: 100
    n_jobs: 1

    study_name: "random_search"
    storage: null

    # Search space
    params:
      model.optimizer.lr:
        type: float
        low: 0.0001
        high: 0.1
        log: true

      model.optimizer.weight_decay:
        type: float
        low: 0.00001
        high: 0.001
        log: true

      data.batch_size:
        type: categorical
        choices: [64, 128, 256, 512]

      model.dropout:
        type: float
        low: 0.0
        high: 0.5

# Metric to optimize
optimized_metric: "val/acc"
```

### Run Random Search

```bash
python src/train.py --multirun --config-name sweep/random_search
```

### Random Search Advantages

- More efficient than grid search
- Good exploration of parameter space
- Simple to understand

### Random Search Disadvantages

- No learning from previous trials
- May waste trials on bad regions
- Fixed number of trials needed

## 3. Bayesian Optimization (Recommended)

Uses Optuna TPE sampler for intelligent search.

### Basic Bayesian Optimization

**configs/sweep/bayesian_optimization.yaml:**

```yaml
defaults:
  - override /model: resnet18
  - override /data: cifar10
  - override /trainer: gpu_single
  - override /logger: wandb
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    # Optimization direction
    direction: maximize  # or minimize for loss
    n_trials: 50
    n_jobs: 1

    # Study configuration
    study_name: "resnet18_cifar10_optimization"
    storage: null  # Use "sqlite:///optuna.db" for persistence

    # TPE Sampler (Tree-structured Parzen Estimator)
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
      n_startup_trials: 10  # Random trials before Bayesian
      n_ei_candidates: 24
      multivariate: true
      warn_independent_sampling: true

    # Search space
    params:
      # Learning rate (log scale)
      model.optimizer.lr:
        type: float
        low: 0.0001
        high: 0.1
        log: true

      # Weight decay (log scale)
      model.optimizer.weight_decay:
        type: float
        low: 0.00001
        high: 0.001
        log: true

      # Batch size (categorical)
      data.batch_size:
        type: categorical
        choices: [64, 128, 256, 512]

      # Dropout (uniform)
      model.dropout:
        type: float
        low: 0.0
        high: 0.5
        step: 0.05

# Metric to optimize
optimized_metric: "val/acc"
```

### Run Bayesian Optimization

```bash
python src/train.py --multirun --config-name sweep/bayesian_optimization
```

### Bayesian Optimization Advantages

- Most sample-efficient
- Learns from previous trials
- Good for expensive evaluations

### Bayesian Optimization Disadvantages

- Requires Optuna dependency
- More complex to configure
- May get stuck in local optima

## Advanced: Multi-Objective Optimization

Optimize multiple metrics simultaneously (e.g., accuracy AND inference speed).

**configs/sweep/multi_objective.yaml:**

```yaml
defaults:
  - override /model: resnet18
  - override /data: cifar10
  - override /trainer: gpu_single
  - override /logger: wandb
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    # Multi-objective: maximize accuracy, minimize latency
    direction: [maximize, minimize]

    n_trials: 100
    n_jobs: 1

    study_name: "multi_objective_optimization"
    storage: "sqlite:///multi_objective.db"

    # NSGA-II sampler for multi-objective
    sampler:
      _target_: optuna.samplers.NSGAIISampler
      seed: 42
      population_size: 50
      mutation_prob: 0.1
      crossover_prob: 0.9

    params:
      model.optimizer.lr:
        type: float
        low: 0.0001
        high: 0.1
        log: true

      data.batch_size:
        type: categorical
        choices: [64, 128, 256, 512]

      # Model size affects latency
      model.hidden_dims:
        type: categorical
        choices:
          - [256, 128]
          - [512, 256]
          - [1024, 512]

# Multiple metrics to optimize
optimized_metric:
  - "val/acc"        # Maximize
  - "val/latency"    # Minimize (need to log this in training)
```

### Run Multi-Objective Optimization

```bash
python src/train.py --multirun --config-name sweep/multi_objective
```

## Advanced: Pruning Unpromising Trials

Stop bad trials early to save compute.

**configs/sweep/with_pruning.yaml:**

```yaml
defaults:
  - override /model: resnet18
  - override /data: cifar10
  - override /trainer: gpu_single
  - override /logger: wandb
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    direction: maximize
    n_trials: 50
    n_jobs: 1

    study_name: "pruning_optimization"
    storage: "sqlite:///pruning.db"

    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
      n_startup_trials: 10

    # Pruner: stop unpromising trials
    pruner:
      _target_: optuna.pruners.MedianPruner
      n_startup_trials: 5      # Don't prune first 5 trials
      n_warmup_steps: 10       # Wait 10 steps before pruning
      interval_steps: 1        # Check every step

    params:
      model.optimizer.lr:
        type: float
        low: 0.0001
        high: 0.1
        log: true

      data.batch_size:
        type: categorical
        choices: [64, 128, 256]

optimized_metric: "val/acc"
```

### Enable Pruning in LightningModule

```python
import optuna
from pytorch_lightning.callbacks import Callback


class OptunaPruningCallback(Callback):
    """Optuna pruning callback for PyTorch Lightning."""

    def __init__(self, trial: optuna.Trial, monitor: str):
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        # Get current metric value
        current_score = trainer.callback_metrics.get(self.monitor)

        if current_score is None:
            return

        # Report to Optuna
        self.trial.report(current_score, step=trainer.current_epoch)

        # Check if trial should be pruned
        if self.trial.should_prune():
            raise optuna.TrialPruned()
```

## Persistent Storage with SQLite

Save all trials to database for later analysis.

**configs/sweep/persistent.yaml:**

```yaml
defaults:
  - override /model: resnet18
  - override /data: cifar10
  - override /trainer: gpu_single
  - override /logger: wandb
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    direction: maximize
    n_trials: 100
    n_jobs: 1

    # Persistent storage
    study_name: "resnet18_cifar10_v1"
    storage: "sqlite:///optuna_studies.db"

    # Load existing study if it exists
    load_if_exists: true

    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42

    params:
      model.optimizer.lr:
        type: float
        low: 0.0001
        high: 0.1
        log: true

optimized_metric: "val/acc"
```

### Continue Existing Study

```bash
# First run: create study
python src/train.py --multirun --config-name sweep/persistent

# Later: continue study with more trials
python src/train.py --multirun --config-name sweep/persistent \
  hydra.sweeper.n_trials=50
```

### Analyze Study

```python
import optuna

# Load study
study = optuna.load_study(
    study_name="resnet18_cifar10_v1",
    storage="sqlite:///optuna_studies.db"
)

# Best trial
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")

# Plot optimization history
optuna.visualization.plot_optimization_history(study).show()

# Plot parameter importances
optuna.visualization.plot_param_importances(study).show()

# Plot parallel coordinates
optuna.visualization.plot_parallel_coordinate(study).show()
```

## Parameter Types Reference

### Float Parameters

```yaml
# Uniform distribution
param:
  type: float
  low: 0.0
  high: 1.0

# Log-uniform (for learning rate, weight decay)
param:
  type: float
  low: 0.0001
  high: 0.1
  log: true

# Discrete float
param:
  type: float
  low: 0.0
  high: 0.5
  step: 0.05  # Values: 0.0, 0.05, 0.1, ..., 0.5
```

### Integer Parameters

```yaml
# Uniform integer
param:
  type: int
  low: 1
  high: 100

# Log-uniform integer
param:
  type: int
  low: 10
  high: 1000
  log: true

# Discrete integer
param:
  type: int
  low: 10
  high: 100
  step: 10  # Values: 10, 20, 30, ..., 100
```

### Categorical Parameters

```yaml
# Simple choices
param:
  type: categorical
  choices: [adam, sgd, adamw]

# Complex nested structures
param:
  type: categorical
  choices:
    - [512, 256]
    - [1024, 512, 256]
    - [2048, 1024, 512]
```

## Best Practices

### 1. Start with Grid Search

For 2-3 parameters, use simple grid search:

```bash
python src/train.py --multirun \
  model.lr=0.001,0.01,0.1 \
  data.batch_size=128,256
```

### 2. Use Random Search for Exploration

When you have 4+ parameters:

```bash
python src/train.py --multirun --config-name sweep/random_search
```

### 3. Use Bayesian Optimization for Fine-Tuning

After finding good region with random search:

```bash
python src/train.py --multirun --config-name sweep/bayesian_optimization
```

### 4. Enable Pruning for Long Training

If each trial takes >1 hour:

- Use MedianPruner or HyperbandPruner
- Set appropriate warmup steps
- Monitor intermediate metrics

### 5. Use Persistent Storage

Always save to database for:

- Resuming interrupted sweeps
- Analyzing results later
- Sharing studies across machines

### 6. Log Everything to W&B

Track all trials in experiment tracking:

```yaml
logger:
  wandb:
    project: "cifar10-sweep"
    group: "resnet18_optimization"
    tags: ["sweep", "optuna"]
```

## Common Patterns

### Small Search Space (Grid)

```bash
# 2-3 parameters, <50 combinations
python src/train.py --multirun \
  model.lr=0.001,0.01 \
  data.batch_size=128,256 \
  model.dropout=0.1,0.2,0.3
```

### Medium Search Space (Random)

```yaml
# 4-6 parameters, 100-200 trials
n_trials: 100
sampler:
  _target_: optuna.samplers.RandomSampler
```

### Large Search Space (Bayesian)

```yaml
# 6+ parameters, 50-100 trials
n_trials: 50
sampler:
  _target_: optuna.samplers.TPESampler
  n_startup_trials: 10
```

### Very Large Search Space (Bayesian + Pruning)

```yaml
# 10+ parameters, expensive evaluation
n_trials: 100
sampler:
  _target_: optuna.samplers.TPESampler
pruner:
  _target_: optuna.pruners.MedianPruner
  n_startup_trials: 5
```

## Troubleshooting

**Optuna not installed:**

```bash
uv add hydra-optuna-sweeper
```

**Study already exists error:**

```yaml
# Set load_if_exists: true
hydra:
  sweeper:
    load_if_exists: true
```

**Too many trials failing:**

- Check parameter ranges are valid
- Verify metric name matches logging
- Enable debug logging: `HYDRA_FULL_ERROR=1`

**Pruning too aggressive:**

```yaml
# Increase warmup steps
pruner:
  n_warmup_steps: 20
```

**Slow optimization:**

- Use parallel jobs: `n_jobs: 4`
- Enable pruning
- Reduce max_epochs for trials
