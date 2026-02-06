---
name: experiment
description: Manage ML experiments, track results, and compare performance across different configurations (W&B, TensorBoard, MLflow)
---

# ML Experiment Management

Manage machine learning experiments, track results, and compare performance across different configurations.

## Process

### 1. Experiment Organization

**Directory Structure:**

```text
logs/
├── 2026-02-06/
│   ├── 14-30-22/  # Timestamp of run
│   │   ├── .hydra/
│   │   │   ├── config.yaml      # Full resolved config
│   │   │   ├── overrides.yaml   # CLI overrides
│   │   │   └── hydra.yaml
│   │   ├── checkpoints/
│   │   ├── metrics.csv
│   │   └── train.log
│   └── 15-45-10/
└── experiment_registry.json
```

### 2. Create New Experiment

**Interactive Experiment Setup:**

Ask user for:

- Experiment name and description
- Base configuration to extend
- Key parameters to modify
- Tags for organization (baseline, ablation, optimization, etc.)
- Expected runtime/compute requirements

Generate `configs/experiment/<name>.yaml`:

```yaml
# @package _global_

# Experiment metadata
name: "vit_imagenet_finetuning"
description: "Fine-tune Vision Transformer on ImageNet subset"
tags: ["vision-transformer", "transfer-learning", "imagenet"]

defaults:
  - override /model: vit_base
  - override /data: imagenet
  - override /trainer: gpu_multi
  - override /logger: wandb

# Experiment-specific overrides
seed: 42

model:
  pretrained: true
  freeze_backbone: false
  num_classes: 1000

data:
  batch_size: 256
  num_workers: 8
  image_size: 224

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

# Logging
logger:
  project: "imagenet-classification"
  tags: ${tags}
  notes: ${description}
```

### 3. Track Experiment Results

**Create Experiment Registry:**

Generate `logs/experiment_registry.json`:

```json
{
  "experiments": [
    {
      "id": "exp_001",
      "name": "baseline_resnet50",
      "timestamp": "2026-02-06T14:30:22",
      "config": "configs/experiment/baseline.yaml",
      "status": "completed",
      "metrics": {
        "best_val_acc": 0.876,
        "best_val_loss": 0.324,
        "final_train_loss": 0.145,
        "epochs_trained": 45,
        "early_stopped": true
      },
      "hyperparameters": {
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "AdamW",
        "weight_decay": 0.0001
      },
      "runtime": "2h 34m",
      "gpu_count": 2,
      "tags": ["baseline", "resnet"],
      "notes": "Initial baseline with default hyperparameters"
    }
  ]
}
```

**Update Registry After Each Run:**

```python
import json
from pathlib import Path
from datetime import datetime

def log_experiment(
    name: str,
    config_path: str,
    metrics: dict,
    hyperparameters: dict,
    status: str = "completed",
    tags: list = None,
    notes: str = ""
):
    registry_path = Path("logs/experiment_registry.json")

    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"experiments": []}

    experiment = {
        "id": f"exp_{len(registry['experiments'])+1:03d}",
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "config": config_path,
        "status": status,
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "tags": tags or [],
        "notes": notes
    }

    registry["experiments"].append(experiment)

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
```

### 4. Compare Experiments

**Generate Comparison Report:**

```python
import pandas as pd
import matplotlib.pyplot as plt

def compare_experiments(exp_ids: list[str]):
    """Compare multiple experiments side by side."""
    with open("logs/experiment_registry.json") as f:
        registry = json.load(f)

    experiments = [e for e in registry["experiments"] if e["id"] in exp_ids]

    # Create comparison DataFrame
    comparison = pd.DataFrame([
        {
            "ID": exp["id"],
            "Name": exp["name"],
            "Val Acc": exp["metrics"]["best_val_acc"],
            "Val Loss": exp["metrics"]["best_val_loss"],
            "LR": exp["hyperparameters"]["lr"],
            "Batch Size": exp["hyperparameters"]["batch_size"],
            "Runtime": exp["runtime"]
        }
        for exp in experiments
    ])

    print(comparison.to_string(index=False))

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(comparison["Name"], comparison["Val Acc"])
    axes[0].set_title("Validation Accuracy")
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(comparison["Name"], comparison["Val Loss"])
    axes[1].set_title("Validation Loss")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("logs/experiment_comparison.png")
    print("Comparison plot saved to logs/experiment_comparison.png")
```

### 5. Experiment Templates

**Baseline Experiment:**

```yaml
# configs/experiment/baseline.yaml
name: "baseline"
description: "Baseline experiment with default hyperparameters"
tags: ["baseline"]

model:
  # Use defaults
trainer:
  max_epochs: 100
```

**Ablation Study:**

```yaml
# configs/experiment/ablation_dropout.yaml
name: "ablation_dropout"
description: "Ablation study - effect of dropout rate"
tags: ["ablation", "regularization"]

# Test different dropout values
# Run with: --multirun model.dropout=0.0,0.1,0.2,0.3,0.4,0.5
model:
  dropout: 0.3  # Default for this ablation
```

**Hyperparameter Optimization:**

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

    search_space:
      model.hidden_dims:
        type: categorical
        choices: [[512,256], [1024,512,256], [2048,1024,512]]
      model.dropout:
        type: float
        low: 0.0
        high: 0.5
      model.optimizer.lr:
        type: float
        low: 0.0001
        high: 0.01
        log: true

optimized_metric: "val/acc"
```

### 6. Experiment Reproduction

**Save Full Environment:**

```bash
# Save exact package versions
pixi list > environment_snapshot.txt
# or
uv pip freeze > requirements_exact.txt

# Save git commit hash
git rev-parse HEAD > commit_hash.txt

# Save system info
python -c "import torch; print(f'PyTorch: {torch.__version__}\nCUDA: {torch.version.cuda}')" > system_info.txt
```

**Reproduce Experiment:**

```bash
# Checkout exact code version
git checkout $(cat logs/exp_001/commit_hash.txt)

# Restore environment
pixi install
# or
uv pip install -r logs/exp_001/requirements_exact.txt

# Run with exact config
python src/train.py --config-path ../logs/exp_001/.hydra --config-name config
```

### 7. Experiment Analysis Tools

**Load and Analyze Results:**

```python
from pytorch_lightning import Trainer
from pathlib import Path

def analyze_experiment(exp_dir: Path):
    """Analyze a completed experiment."""
    # Load metrics
    metrics = pd.read_csv(exp_dir / "metrics.csv")

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Training loss
    axes[0, 0].plot(metrics["epoch"], metrics["train_loss"])
    axes[0, 0].set_title("Training Loss")

    # Validation loss
    axes[0, 1].plot(metrics["epoch"], metrics["val_loss"])
    axes[0, 1].set_title("Validation Loss")

    # Learning rate
    axes[1, 0].plot(metrics["epoch"], metrics["lr"])
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_yscale("log")

    # Accuracy
    axes[1, 1].plot(metrics["epoch"], metrics["val_acc"], label="Val")
    axes[1, 1].plot(metrics["epoch"], metrics["train_acc"], label="Train")
    axes[1, 1].set_title("Accuracy")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(exp_dir / "analysis.png")

    # Print summary statistics
    print(f"\nExperiment Summary:")
    print(f"Best Val Acc: {metrics['val_acc'].max():.4f}")
    print(f"Best Val Loss: {metrics['val_loss'].min():.4f}")
    print(f"Epochs Trained: {len(metrics)}")
    print(f"Final LR: {metrics['lr'].iloc[-1]:.6f}")
```

### 8. W&B Integration

**Query W&B Runs:**

```python
import wandb

api = wandb.Api()
runs = api.runs("my-project")

for run in runs:
    print(f"{run.name}: {run.summary['val/acc']:.4f}")
    # Download artifacts
    # run.file("model.pt").download()
```

**Compare Runs in W&B:**

```bash
# Open W&B workspace
wandb workspace

# Generate report
wandb reports create --title "Experiment Comparison"
```

### 9. Experiment Best Practices

**Naming Conventions:**

- Use descriptive names: `vit_large_imagenet_pretrained`
- Include date for long experiments: `exp_2026_02_baseline`
- Use prefixes for experiment types: `ablation_`, `optimization_`, `baseline_`

**Documentation:**

- Always add description and notes
- Tag experiments for easy filtering
- Document unexpected results
- Track compute resources used

**Version Control:**

- Commit code before long experiments
- Save git hash with experiment
- Don't modify code during experiment
- Use branches for experimental features

**Reproducibility:**

- Set seeds everywhere (Python, NumPy, PyTorch, Lightning)
- Save exact package versions
- Document hardware used
- Save data splits/preprocessing

### 10. Experiment Commands

```bash
# List all experiments
python scripts/list_experiments.py

# Compare specific experiments
python scripts/compare_experiments.py exp_001 exp_002 exp_003

# Analyze experiment
python scripts/analyze_experiment.py logs/2026-02-06/14-30-22/

# Clean old experiments (keep only best checkpoints)
python scripts/clean_experiments.py --keep-best 5

# Export results to CSV
python scripts/export_results.py --output results.csv

# Generate experiment report
python scripts/generate_report.py --format markdown --output report.md
```

## Success Criteria

- [ ] Experiment registry tracking all runs
- [ ] Configs saved with each experiment
- [ ] Metrics logged consistently
- [ ] Easy comparison between experiments
- [ ] Reproducible results
- [ ] Clear documentation of all parameters
- [ ] Visualization of training curves
- [ ] Summary statistics computed

Experiments are well-organized and easily comparable!
