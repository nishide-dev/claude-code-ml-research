---
name: ml-experiment
description: Manage ML experiments, track results, and compare performance across different configurations. Use when setting up experiment tracking, creating experiment configs, comparing runs, or analyzing experiment results with W&B, TensorBoard, or MLflow.
argument-hint: [experiment_name] [action]
disable-model-invocation: true
---

# ML Experiment Management

Systematic experiment tracking, comparison, and analysis for machine learning research.

## Quick Start

**Directory Structure:**

```text
logs/
├── 2026-02-22/
│   ├── 14-30-22/              # Timestamp of run
│   │   ├── .hydra/
│   │   │   ├── config.yaml    # Full resolved config
│   │   │   ├── overrides.yaml # CLI overrides
│   │   │   └── hydra.yaml
│   │   ├── checkpoints/
│   │   ├── metrics.csv
│   │   └── train.log
│   └── 15-45-10/
└── experiment_registry.json   # Central registry
```

## 1. Create Experiment Config

**Interactive Setup - Ask User:**

- Experiment name and description
- Base configuration to extend
- Key parameters to modify
- Tags (baseline, ablation, optimization, etc.)
- Expected runtime/compute requirements

**Generate:** `configs/experiment/<name>.yaml`

```yaml
# @package _global_

# Metadata
name: "vit_imagenet_finetuning"
description: "Fine-tune Vision Transformer on ImageNet subset"
tags: ["vision-transformer", "transfer-learning", "imagenet"]

# Compose from existing configs
defaults:
  - override /model: vit_base
  - override /data: imagenet
  - override /trainer: gpu_multi
  - override /logger: wandb

# Seed
seed: 42

# Model overrides
model:
  pretrained: true
  freeze_backbone: false
  num_classes: 1000
  optimizer:
    lr: 0.001

# Data overrides
data:
  batch_size: 256
  num_workers: 8
  image_size: 224

# Trainer overrides
trainer:
  max_epochs: 50
  precision: "16-mixed"
  devices: 4
  strategy: "ddp"

# Callbacks
callbacks:
  model_checkpoint:
    monitor: "val/acc"
    mode: "max"
    save_top_k: 3

  early_stopping:
    monitor: "val/loss"
    patience: 10
    mode: "min"

# Logger
logger:
  wandb:
    project: "imagenet-classification"
    tags: ${tags}
    notes: ${description}
```

**Run experiment:**

```bash
python src/train.py experiment=vit_imagenet_finetuning
```

See `templates/experiment-templates.yaml` for common experiment types.

## 2. Track Experiment Results

### Automatic Tracking with Callbacks

```python
# In LightningModule
def on_train_end(self):
    # Log experiment to registry
    from scripts.experiment_registry import log_experiment

    log_experiment(
        name=self.hparams.experiment_name,
        config_path=self.hparams.config_path,
        metrics={
            "best_val_acc": self.trainer.callback_metrics["val/acc"].item(),
            "best_val_loss": self.trainer.checkpoint_callback.best_model_score.item(),
            "epochs_trained": self.trainer.current_epoch,
        },
        hyperparameters={
            "lr": self.hparams.optimizer.lr,
            "batch_size": self.hparams.data.batch_size,
            "optimizer": self.hparams.optimizer._target_,
        },
        tags=self.hparams.tags,
    )
```

### Experiment Registry Format

`logs/experiment_registry.json`:

```json
{
  "experiments": [
    {
      "id": "exp_001",
      "name": "baseline_resnet50",
      "timestamp": "2026-02-22T14:30:22",
      "config": "configs/experiment/baseline.yaml",
      "status": "completed",
      "metrics": {
        "best_val_acc": 0.876,
        "best_val_loss": 0.324,
        "final_train_loss": 0.145,
        "epochs_trained": 45
      },
      "hyperparameters": {
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "AdamW"
      },
      "runtime": "2h 34m",
      "gpu_count": 2,
      "tags": ["baseline", "resnet"]
    }
  ]
}
```

See `scripts/experiment_registry.py` for implementation.

## 3. Compare Experiments

**Compare specific experiments:**

```bash
python scripts/compare_experiments.py exp_001 exp_002 exp_003
```

**Output:**

```text
ID      Name              Val Acc  Val Loss  LR      Batch  Runtime
exp_001 baseline_resnet50 0.876    0.324     0.001   128    2h 34m
exp_002 resnet50_tuned    0.892    0.298     0.005   256    3h 12m
exp_003 resnet50_dropout  0.884    0.312     0.001   128    2h 45m
```

**Comparison plot:**

```python
# Generates logs/experiment_comparison.png
# - Bar charts for accuracy and loss
# - Side-by-side comparison
```

See `scripts/compare_experiments.py` for full implementation.

## 4. Experiment Templates

### Baseline Experiment

```yaml
# configs/experiment/baseline.yaml
name: "baseline"
description: "Baseline with default hyperparameters"
tags: ["baseline"]

# Use defaults from model/data/trainer
model: {}
data: {}
trainer:
  max_epochs: 100
```

### Ablation Study

```yaml
# configs/experiment/ablation_dropout.yaml
name: "ablation_dropout"
description: "Effect of dropout rate"
tags: ["ablation", "regularization"]

# Run with: --multirun model.dropout=0.0,0.1,0.2,0.3,0.4,0.5
model:
  dropout: 0.3
```

### Hyperparameter Optimization

```yaml
# configs/experiment/hp_optimization.yaml
name: "hp_optimization"
description: "Hyperparameter optimization with Optuna"
tags: ["optimization", "tuning"]

defaults:
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    n_trials: 100
    direction: maximize
    study_name: "model_optimization"
    params:
      model.hidden_dims:
        type: categorical
        choices: [[512,256], [1024,512,256]]
      model.optimizer.lr:
        type: float
        low: 0.0001
        high: 0.01
        log: true

optimized_metric: "val/acc"
```

See `templates/` for more experiment types.

## 5. Experiment Reproduction

### Save Full Environment

```bash
# Save package versions
pixi list > logs/exp_001/environment.txt
# or
uv pip freeze > logs/exp_001/requirements.txt

# Save git commit
git rev-parse HEAD > logs/exp_001/commit_hash.txt

# Save system info
python -c "import torch; print(f'PyTorch: {torch.__version__}\nCUDA: {torch.version.cuda}')" > logs/exp_001/system_info.txt
```

### Reproduce Experiment

```bash
# Checkout exact code
git checkout $(cat logs/exp_001/commit_hash.txt)

# Restore environment
pixi install
# or
uv pip install -r logs/exp_001/requirements.txt

# Run with exact config
python src/train.py \
  --config-path ../logs/exp_001/.hydra \
  --config-name config
```

**Reproducibility Checklist:**

- [ ] Set random seeds (Python, NumPy, PyTorch, Lightning)
- [ ] Save exact package versions
- [ ] Document hardware (GPU type, CUDA version)
- [ ] Save data splits/preprocessing
- [ ] Commit code before experiment
- [ ] Don't modify code during run

## 6. Experiment Analysis

### Analyze Single Experiment

```bash
python scripts/analyze_experiment.py logs/2026-02-22/14-30-22/
```

**Generates:**

- `analysis.png` - Training curves (loss, accuracy, LR)
- Summary statistics (best metrics, epochs, final LR)

**Example:**

```text
Experiment Summary:
Best Val Acc:    0.8921
Best Val Loss:   0.2984
Epochs Trained:  45
Final LR:        0.000123
```

### Multi-Experiment Analysis

```bash
# List all experiments
python scripts/list_experiments.py

# Filter by tags
python scripts/list_experiments.py --tags baseline ablation

# Export to CSV
python scripts/export_results.py --output results.csv

# Generate markdown report
python scripts/generate_report.py --format markdown --output report.md
```

See `examples/experiment-analysis.md` for detailed analysis workflows.

## 7. W&B Integration

### Query W&B Runs

```python
import wandb

api = wandb.Api()
runs = api.runs("my-project")

# Filter runs
runs = api.runs("my-project", filters={"tags": "baseline"})

# Get metrics
for run in runs:
    print(f"{run.name}: val_acc={run.summary['val/acc']:.4f}")

# Download artifacts
best_run = runs[0]
best_run.file("model.pt").download()
```

### Compare Runs in W&B

```bash
# Open workspace
wandb workspace

# Generate report
wandb reports create --title "Experiment Comparison"
```

### W&B Sweeps

```bash
# Initialize sweep
wandb sweep configs/sweep/bayesian_optimization.yaml

# Run sweep agent
wandb agent <sweep-id>
```

See `examples/wandb-integration.md` for complete guide.

## 8. Experiment Best Practices

### Naming Conventions

- **Descriptive names:** `vit_large_imagenet_pretrained`
- **Include date for long runs:** `exp_2026_02_baseline`
- **Use prefixes:** `ablation_`, `optimization_`, `baseline_`

### Documentation

- Always add description and notes
- Tag for easy filtering
- Document unexpected results
- Track compute resources

### Version Control

- Commit code before long experiments
- Save git hash with experiment
- Don't modify code during run
- Use branches for experimental features

### Organization

```text
configs/experiment/
├── baselines/
│   ├── resnet_baseline.yaml
│   └── vit_baseline.yaml
├── ablations/
│   ├── ablation_dropout.yaml
│   └── ablation_lr.yaml
└── optimizations/
    └── hp_optimization.yaml
```

## 9. Common Experiment Types

### A. Baseline Experiment

**Purpose:** Establish reference performance.

```yaml
name: "baseline"
tags: ["baseline"]
model: {}  # Use defaults
```

### B. Ablation Study

**Purpose:** Isolate effect of single component.

```yaml
name: "ablation_batch_norm"
tags: ["ablation"]
model:
  use_batch_norm: false  # Remove batch norm
```

### C. Hyperparameter Tuning

**Purpose:** Find optimal hyperparameters.

```yaml
name: "hp_tuning"
tags: ["optimization"]
# Use with --multirun or Optuna sweeper
```

### D. Transfer Learning

**Purpose:** Fine-tune pretrained model.

```yaml
name: "transfer_learning"
tags: ["transfer-learning"]
model:
  pretrained: true
  freeze_backbone: true  # Freeze early layers
```

### E. Architecture Search

**Purpose:** Compare different architectures.

```bash
# Run multiple architectures
python src/train.py --multirun \
  experiment=architecture_search \
  model=resnet18,resnet50,vit_base
```

## 10. Experiment Commands

```bash
# Create new experiment
python src/train.py experiment=<name>

# List experiments
python scripts/list_experiments.py

# Compare experiments
python scripts/compare_experiments.py exp_001 exp_002 exp_003

# Analyze experiment
python scripts/analyze_experiment.py logs/2026-02-22/14-30-22/

# Clean old experiments (keep best 5)
python scripts/clean_experiments.py --keep-best 5

# Export results
python scripts/export_results.py --output results.csv

# Generate report
python scripts/generate_report.py --format markdown --output report.md
```

## Troubleshooting

**Experiment registry not updating:**

- Check permissions on `logs/experiment_registry.json`
- Verify `on_train_end` callback is called
- Check for JSON syntax errors

**Can't reproduce results:**

- Verify exact package versions match
- Check random seeds are set
- Confirm same hardware (GPU model affects results)
- Validate data preprocessing matches

**W&B runs not logging:**

- Check `WANDB_API_KEY` is set
- Verify project name matches
- Check network connectivity
- Try `wandb login` again

**Metrics not saving:**

- Verify `log_every_n_steps` is set
- Check disk space
- Confirm metrics are logged in training_step

## Success Criteria

- [ ] Experiment registry tracks all runs
- [ ] Configs saved with each experiment
- [ ] Metrics logged consistently
- [ ] Easy comparison between experiments
- [ ] Reproducible results
- [ ] Clear documentation
- [ ] Training curves visualized
- [ ] Summary statistics computed

Experiments are well-organized and easily comparable!
