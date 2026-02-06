---
name: ml-wandb-tracking
description: Complete guide for Weights & Biases (W&B) - experiment tracking, hyperparameter sweeps, artifact management, model registry, and PyTorch Lightning integration
---

# Weights & Biases for ML Experiment Tracking

## Overview

Weights & Biases (W&B) is the industry-standard platform for ML experiment tracking, visualization, and model management. It provides cloud-based (or self-hosted) infrastructure for logging experiments, running hyperparameter sweeps, managing artifacts, and tracking model lineage from research to production.

**Key Capabilities:**
- Automatic experiment tracking (metrics, hyperparameters, system resources)
- Hyperparameter optimization with W&B Sweeps
- Data and model versioning with Artifacts
- Model registry for deployment lifecycle
- Collaborative reports and visualizations
- Seamless PyTorch Lightning integration

**Resources:**
- Official docs: https://docs.wandb.ai/
- Lightning integration: https://docs.wandb.ai/models/integrations/lightning
- Sweeps guide: https://docs.wandb.ai/models/sweeps

---

## Getting Started

### Installation

```bash
pip install wandb

# Login (creates .netrc credentials)
wandb login
```

### Basic Tracking

```python
import wandb

# Initialize a run
wandb.init(
    project="my-project",
    name="experiment-1",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
    }
)

# Log metrics
for epoch in range(10):
    wandb.log({"loss": 0.5, "accuracy": 0.9, "epoch": epoch})

# Finish the run
wandb.finish()
```

---

## Integration with PyTorch Lightning

W&B integrates seamlessly with Lightning through `WandbLogger`.

### Basic Lightning Integration

```python
import lightning as L
from lightning.pytorch.loggers import WandbLogger

# Create logger
wandb_logger = WandbLogger(
    project="cifar10-classification",
    name="resnet18-baseline",
    log_model="all",  # Log all model checkpoints as artifacts
    save_dir="./logs"
)

# LightningModule
class LitModel(L.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.save_hyperparameters()  # Automatically logs to W&B
        self.model = ...

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        # Log metrics (automatically synced to W&B)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", accuracy, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# Training
trainer = L.Trainer(
    max_epochs=100,
    logger=wandb_logger,
    accelerator="gpu",
    devices=1,
)
trainer.fit(model, datamodule=datamodule)
```

### WandbLogger Configuration

**Important parameters:**
| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `project` | Project name (groups related runs) | Use consistent naming convention |
| `name` | Run name (identifies specific experiment) | Include key experiment details |
| `log_model` | Model checkpoint logging strategy | `"all"` or `True` for full versioning |
| `save_dir` | Local directory for logs | `"./logs"` or project-specific path |
| `tags` | List of tags for filtering runs | `["baseline", "production", "ablation"]` |
| `notes` | Description of the experiment | Document hypothesis or changes |
| `config` | Additional config dict | Supplement Lightning's auto-logging |

**Example with full configuration:**
```python
wandb_logger = WandbLogger(
    project="image-classification",
    name="resnet50-augmented-v2",
    log_model="all",
    save_dir="./wandb_logs",
    tags=["resnet50", "heavy-aug", "baseline"],
    notes="Testing heavy data augmentation with RandAugment",
    config={
        "architecture": "resnet50",
        "augmentation": "randaugment",
        "dataset_version": "v2.1"
    }
)
```

---

## Advanced Logging

### Logging Custom Media

**Images:**
```python
import wandb

class LitModel(L.LightningModule):
    def validation_epoch_end(self, outputs):
        # Log sample predictions as images
        sample_imgs = self.get_sample_images()

        self.logger.experiment.log({
            "predictions": [
                wandb.Image(img, caption=f"Pred: {pred}, True: {true}")
                for img, pred, true in sample_imgs
            ]
        })
```

**Confusion Matrix:**
```python
def validation_epoch_end(self, outputs):
    # Collect all predictions
    all_preds = torch.cat([x["preds"] for x in outputs])
    all_labels = torch.cat([x["labels"] for x in outputs])

    # Log interactive confusion matrix
    self.logger.experiment.log({
        "conf_mat": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels.cpu().numpy(),
            preds=all_preds.cpu().numpy(),
            class_names=self.class_names
        )
    })
```

**Custom Charts:**
```python
# ROC curve
self.logger.experiment.log({
    "roc_curve": wandb.plot.roc_curve(
        y_true, y_probas, labels=class_names
    )
})

# PR curve
self.logger.experiment.log({
    "pr_curve": wandb.plot.pr_curve(
        y_true, y_probas, labels=class_names
    )
})

# Custom tables
table = wandb.Table(
    columns=["id", "prediction", "confidence", "ground_truth"],
    data=[[i, pred, conf, true] for i, (pred, conf, true) in enumerate(results)]
)
self.logger.experiment.log({"predictions_table": table})
```

---

## Hyperparameter Optimization with Sweeps

W&B Sweeps automate hyperparameter search with minimal code changes.

### Sweep Configuration (YAML)

**Complete sweep config:**
```yaml
# sweep_config.yaml
program: train.py
method: bayes  # grid, random, or bayes
metric:
  name: val_loss
  goal: minimize

parameters:
  # Categorical parameters
  optimizer:
    values: ["adam", "adamw", "sgd"]

  # Continuous parameters (log scale)
  learning_rate:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.1

  # Discrete parameters
  batch_size:
    values: [32, 64, 128, 256]

  # Quantized log uniform (for powers of 2)
  hidden_dim:
    distribution: q_log_uniform_values
    min: 64
    max: 512
    q: 64  # Quantization factor

  # Integer uniform
  num_layers:
    distribution: int_uniform
    min: 2
    max: 6

  # Uniform continuous
  dropout:
    distribution: uniform
    min: 0.1
    max: 0.5

# Early termination (Hyperband algorithm)
early_terminate:
  type: hyperband
  min_iter: 3      # Minimum epochs before termination
  eta: 3           # Aggressiveness (higher = more aggressive)
  s: 2             # Number of brackets
```

### Search Strategies

| Method | Algorithm | Pros | Cons | Use Case |
|--------|-----------|------|------|----------|
| `grid` | Exhaustive search | Complete coverage | Exponential compute | Small, discrete spaces (≤3 params) |
| `random` | Random sampling | Better than grid for high dimensions | No learning | Initial exploration, baselines |
| `bayes` | Bayesian optimization (TPE) | Efficient, learns from past trials | Sequential (less parallelizable) | Expensive models, limited budget |

### Running Sweeps

**Step 1: Initialize sweep**
```bash
# Creates sweep on W&B server, returns sweep ID
wandb sweep sweep_config.yaml

# Output: wandb: Created sweep with ID: abc123xyz
# Output: wandb: View sweep at: https://wandb.ai/user/project/sweeps/abc123xyz
```

**Step 2: Launch agents**
```bash
# Single agent (runs forever until stopped)
wandb agent user/project/abc123xyz

# Limited number of runs
wandb agent user/project/abc123xyz --count 10
```

**Step 3: Parallel execution**

**Single machine, multiple GPUs:**
```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 wandb agent SWEEP_ID

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 wandb agent SWEEP_ID

# Terminal 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 wandb agent SWEEP_ID
```

**SLURM cluster:**
```bash
#!/bin/bash
#SBATCH --job-name=wandb-sweep
#SBATCH --array=1-20%4        # 20 jobs, max 4 concurrent
#SBATCH --gres=gpu:1          # 1 GPU per job
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Activate environment
source ~/.bashrc
conda activate ml-env

# Run single trial
wandb agent --count 1 user/project/SWEEP_ID
```

### Training Script for Sweeps

Modify your training script to accept W&B sweep parameters:

```python
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger

def train():
    # Initialize W&B (sweep agent sets config automatically)
    run = wandb.init()
    config = wandb.config

    # Create logger
    wandb_logger = WandbLogger(
        project="sweep-project",
        log_model="all"
    )

    # Create model with sweep parameters
    model = LitModel(
        lr=config.learning_rate,
        batch_size=config.batch_size,
        optimizer=config.optimizer,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )

    # Train
    trainer = L.Trainer(
        max_epochs=50,
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(model, datamodule=datamodule)

    # W&B automatically logs the metric specified in sweep config
    wandb.finish()

if __name__ == "__main__":
    train()
```

---

## Integration with Hydra

For complex projects using Hydra, W&B Sweeps need special configuration.

### Hydra-Compatible Sweep Config

```yaml
# sweep_hydra.yaml
program: train.py
method: bayes
metric:
  name: val/loss
  goal: minimize

parameters:
  # Use dot notation for nested Hydra configs
  model.optimizer.lr:
    min: 0.0001
    max: 0.1
    distribution: log_uniform_values

  data.batch_size:
    values: [32, 64, 128]

  model.hidden_dim:
    values: [128, 256, 512]

# Critical: Use args_no_hyphens for Hydra compatibility
command:
  - ${env}
  - python
  - ${program}
  - ${args_no_hyphens}  # Passes as "key=value" instead of "--key value"
```

**Training script with Hydra:**
```python
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import wandb

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    # Initialize W&B with Hydra config
    wandb.init(
        project=cfg.project_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    # Sweep parameters override Hydra defaults
    if wandb.config:
        # Update config with sweep parameters
        for key, value in wandb.config.items():
            OmegaConf.update(cfg, key, value, merge=False)

    # Instantiate model from updated config
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)

    # Training
    trainer = L.Trainer(**cfg.trainer)
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    train()
```

---

## Artifacts: Data and Model Versioning

W&B Artifacts provide versioning and lineage tracking for datasets and models.

### Logging Artifacts

**Dataset artifact:**
```python
import wandb

run = wandb.init(project="my-project", job_type="data-prep")

# Create artifact
artifact = wandb.Artifact(
    name="cifar10-processed",
    type="dataset",
    description="CIFAR-10 with augmentation pipeline v2",
    metadata={
        "augmentation": "randaugment",
        "normalization": "imagenet",
        "split_ratio": "0.8/0.1/0.1"
    }
)

# Add files
artifact.add_dir("./data/processed")
# Or individual files
artifact.add_file("./data/train.csv")
artifact.add_file("./data/val.csv")

# Log artifact
run.log_artifact(artifact)
run.finish()
```

**Model artifact (automatic with Lightning):**
```python
# WandbLogger automatically logs model checkpoints as artifacts
wandb_logger = WandbLogger(
    project="my-project",
    log_model="all",  # or True (only best), or False
)

trainer = L.Trainer(logger=wandb_logger)
trainer.fit(model, datamodule=datamodule)
# Models saved as artifacts: "model-{run_id}:v0", "v1", etc.
```

### Using Artifacts

**Load dataset artifact:**
```python
run = wandb.init(project="my-project", job_type="training")

# Use specific version
artifact = run.use_artifact("cifar10-processed:v3")
data_dir = artifact.download()  # Downloads to cache

# Or use alias
artifact = run.use_artifact("cifar10-processed:latest")
```

**Load model artifact:**
```python
run = wandb.init(project="my-project", job_type="evaluation")

# Load model from artifact
artifact = run.use_artifact("model-abc123:best")
model_dir = artifact.download()

# Restore Lightning model
model = LitModel.load_from_checkpoint(f"{model_dir}/model.ckpt")
```

### Artifact Lineage

Artifacts automatically track lineage (which data/models produced which outputs):

```python
# Training run
train_run = wandb.init(project="my-project", job_type="train")
data_artifact = train_run.use_artifact("dataset:v2")  # Input
# ... train model ...
model_artifact = wandb.Artifact("model", type="model")
train_run.log_artifact(model_artifact)  # Output
# W&B links dataset:v2 -> model:v0

# Evaluation run
eval_run = wandb.init(project="my-project", job_type="eval")
model_artifact = eval_run.use_artifact("model:v0")  # Input
# ... evaluate ...
results_artifact = wandb.Artifact("results", type="evaluation")
eval_run.log_artifact(results_artifact)  # Output
# W&B links model:v0 -> results:v0
```

---

## Model Registry

W&B Model Registry manages model lifecycle from staging to production.

### Linking Models to Registry

```python
# Option 1: Link from UI (recommended for manual curation)
# Go to W&B UI -> Artifacts -> Select model -> "Link to registry"

# Option 2: Programmatic linking
import wandb

run = wandb.init()
artifact = run.use_artifact("model-abc123:v5")

# Link to registry with alias
run.link_artifact(
    artifact,
    target_path="model-registry/my-classifier",
    aliases=["staging", "candidate-v1"]
)
```

### Loading from Registry

```python
import wandb
from pathlib import Path

# Production inference
run = wandb.init(project="production-inference", job_type="inference")

# Load production model (always gets latest "production" alias)
artifact = run.use_artifact("model-registry/my-classifier:production")
model_dir = artifact.download()

# Restore model
model = LitModel.load_from_checkpoint(Path(model_dir) / "model.ckpt")

# Run inference
predictions = model(new_data)
```

### Registry Workflow

**Typical lifecycle:**
1. **Training**: Models logged as artifacts during training
2. **Evaluation**: Best model identified from sweep/experiments
3. **Staging**: Link to registry with "staging" alias
4. **Testing**: Run integration tests on staging model
5. **Production**: Promote to "production" alias
6. **Monitoring**: Track production metrics, rollback if needed

```python
# Promote model to production
run = wandb.init()
artifact = run.use_artifact("model-registry/my-classifier:staging")

# Update aliases
run.link_artifact(
    artifact,
    target_path="model-registry/my-classifier",
    aliases=["production", "v1.2.0"]  # Add version tag
)
```

---

## Best Practices

### ✅ DO

1. **Use consistent project naming**: Organize by team/application (`team-model-dataset`)

2. **Tag runs systematically**: Use tags for filtering (`["baseline", "ablation", "production"]`)

3. **Log hyperparameters early**: Call `self.save_hyperparameters()` in `__init__`

4. **Enable model logging**: Set `log_model="all"` for full experiment reproducibility

5. **Use artifacts for datasets**: Version datasets the same way you version models

6. **Document experiments**: Use `notes` parameter to record hypotheses and changes

7. **Group related runs**: Use W&B Groups for multi-run experiments (e.g., cross-validation folds)

8. **Set up alerts**: Configure alerts for metric thresholds or training failures

9. **Use sweep early termination**: Save compute with Hyperband pruning

10. **Archive old projects**: Keep workspace clean for active work

### ❌ DON'T

1. **Don't log sensitive data**: Never log API keys, passwords, or PII

2. **Don't log every step**: Use `log_every_n_steps` in Trainer to reduce overhead

3. **Don't ignore offline mode**: Use `wandb.init(mode="offline")` for debugging

4. **Don't hardcode sweep params**: Accept parameters from `wandb.config`

5. **Don't forget `wandb.finish()`**: Especially in notebooks; ensures proper logging

6. **Don't use production project for experiments**: Separate `dev` and `prod` projects

7. **Don't skip artifact versioning**: Always version datasets and models

8. **Don't manually download checkpoints**: Use artifacts for reproducibility

---

## Common Patterns

### Pattern 1: Cross-Validation with W&B

```python
def train_fold(fold_idx, config):
    run = wandb.init(
        project="cv-experiment",
        group="5-fold-cv",  # Groups all folds together
        name=f"fold-{fold_idx}",
        config=config,
    )

    model = LitModel(**config)
    trainer = L.Trainer(
        max_epochs=50,
        logger=WandbLogger()
    )
    trainer.fit(model, train_dataloaders=train_loaders[fold_idx])

    wandb.finish()

# Run all folds
for i in range(5):
    train_fold(i, config={"lr": 0.001, "batch_size": 32})
```

### Pattern 2: Resuming Failed Runs

```python
# Get previous run ID from W&B UI or logs
RUN_ID = "abc123xyz"

run = wandb.init(
    project="my-project",
    id=RUN_ID,
    resume="must"  # Resumes or raises error if not found
)

# Load checkpoint
checkpoint_path = "./checkpoints/last.ckpt"
model = LitModel.load_from_checkpoint(checkpoint_path)

# Continue training
trainer = L.Trainer(logger=WandbLogger())
trainer.fit(model, ckpt_path=checkpoint_path)
```

### Pattern 3: Multi-Run Comparisons

```python
# Log same experiment with different random seeds
for seed in [42, 123, 456, 789, 999]:
    wandb.init(
        project="seed-comparison",
        group="resnet18-baseline",
        config={"seed": seed, "lr": 0.001},
        reinit=True  # Allows multiple inits in same script
    )

    L.seed_everything(seed)
    model = LitModel(lr=0.001)
    trainer = L.Trainer(max_epochs=100)
    trainer.fit(model, datamodule=datamodule)

    wandb.finish()
```

---

## Troubleshooting

### Common Issues

**Issue: Slow logging**
```python
# Solution: Reduce logging frequency
trainer = L.Trainer(
    log_every_n_steps=50,  # Instead of default (1)
)
```

**Issue: Runs not finishing**
```python
# Solution: Always call finish()
try:
    trainer.fit(model)
finally:
    wandb.finish()
```

**Issue: Sweep agents not finding program**
```python
# Solution: Use absolute path in sweep config
program: /absolute/path/to/train.py
# Or ensure script is in current directory
```

**Issue: Out of disk space**
```python
# Solution: Clean W&B cache
import wandb
wandb.agent_sweep_cache.clean()  # Removes old sweep data

# Or manually delete
rm -rf ~/.local/share/wandb/artifacts/*
```

---

## Essential Resources

### Official Documentation
- **W&B Docs**: https://docs.wandb.ai/
- **Lightning Integration**: https://docs.wandb.ai/models/integrations/lightning
- **Sweeps Guide**: https://docs.wandb.ai/models/sweeps
- **Artifacts Guide**: https://docs.wandb.ai/guides/artifacts

### Tutorials
- **W&B + Lightning Tutorial**: https://wandb.ai/site/articles/pytorch-lightning
- **Hyperparameter Tuning**: https://wandb.ai/site/articles/hyperparameter-tuning
- **Model Registry**: https://docs.wandb.ai/guides/model_registry

### Templates
- **Lightning-Hydra-Template**: https://github.com/ashleve/lightning-hydra-template

---

## Summary

Weights & Biases provides:

- **Automatic experiment tracking**: Zero-config metric logging with Lightning
- **Hyperparameter optimization**: Bayesian sweeps with early termination
- **Data versioning**: Artifact system for datasets and models
- **Model lifecycle management**: Registry for staging/production deployment
- **Collaboration**: Shared dashboards, reports, and reproducibility

Combined with PyTorch Lightning, W&B creates a complete MLOps platform from research to production.
