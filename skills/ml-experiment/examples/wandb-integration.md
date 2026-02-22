# W&B (Weights & Biases) Integration Guide

Complete guide for integrating Weights & Biases experiment tracking with PyTorch Lightning.

## Setup

```bash
# Install W&B
uv add wandb

# Login (interactive)
wandb login

# Or set API key
export WANDB_API_KEY=your_api_key_here
```

## Basic Integration

### Configure W&B Logger

```yaml
# configs/logger/wandb.yaml
_target_: pytorch_lightning.loggers.WandbLogger

# Project settings
project: "my-ml-project"
entity: "my-team"  # Optional: team name
name: null  # Auto-generated from config
save_dir: "logs/"

# Logging options
log_model: true  # Save model checkpoints to W&B
save_code: true  # Save code snapshot
offline: false   # Set true for offline mode

# Grouping
group: null      # Group related runs
job_type: null   # Job type (train, eval, etc.)
tags: []         # Tags for organization

# Additional settings
notes: ""        # Run notes/description
```

### Use in Training

```python
# src/train.py
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

# Initialize logger
wandb_logger = WandbLogger(
    project="my-ml-project",
    name="resnet50_baseline",
    save_dir="logs/",
    log_model=True,
)

# Create trainer
trainer = Trainer(
    logger=wandb_logger,
    max_epochs=100,
)

# Train
trainer.fit(model, datamodule=dm)

# Finish run
wandb_logger.experiment.finish()
```

## Advanced Features

### 1. Log Custom Metrics

```python
# In LightningModule
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    # Log to W&B
    self.log("train/loss", loss)
    self.log("train/lr", self.optimizers().param_groups[0]["lr"])

    # Log custom metrics
    self.log("train/gradient_norm", self.get_gradient_norm())

    return loss
```

### 2. Log Images

```python
import wandb

def validation_step(self, batch, batch_idx):
    images, labels = batch
    preds = self(images)

    # Log first 8 images every 10 epochs
    if batch_idx == 0 and self.current_epoch % 10 == 0:
        # Create wandb Images
        wandb_images = [
            wandb.Image(
                images[i],
                caption=f"Pred: {preds[i].argmax()}, True: {labels[i]}",
            )
            for i in range(min(8, len(images)))
        ]

        # Log to W&B
        self.logger.experiment.log({"predictions": wandb_images})

    return loss
```

### 3. Log Confusion Matrix

```python
from pytorch_lightning.callbacks import Callback
import wandb

class LogConfusionMatrix(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Get predictions
        all_preds = []
        all_labels = []

        for batch in trainer.val_dataloaders:
            images, labels = batch
            preds = pl_module(images.to(pl_module.device))
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Log confusion matrix
        trainer.logger.experiment.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=pl_module.class_names,
            )
        })
```

### 4. Log Artifacts

```python
# Save model as artifact
def on_train_end(self):
    # Create artifact
    artifact = wandb.Artifact(
        name="model",
        type="model",
        description="Trained ResNet-50 model",
    )

    # Add file
    artifact.add_file("checkpoints/best.ckpt")

    # Log artifact
    self.logger.experiment.log_artifact(artifact)
```

### 5. Track System Metrics

```python
# In train.py
wandb_logger = WandbLogger(
    project="my-ml-project",
    log_model=True,
)

# Log system metrics (GPU, CPU, memory)
wandb_logger.experiment.config.update({
    "system": {
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "cpu_count": os.cpu_count(),
    }
})
```

## W&B API Usage

### Query Runs

```python
import wandb

# Initialize API
api = wandb.Api()

# Get all runs from project
runs = api.runs("my-project")

# Filter runs by tags
runs = api.runs("my-project", filters={"tags": {"$in": ["baseline"]}})

# Get specific run
run = api.run("my-project/run-id")

# Access metrics
print(f"Best val acc: {run.summary['val/acc']:.4f}")
print(f"Config: {run.config}")

# Download files
run.file("model.pt").download(root="downloads/")
```

### Compare Runs

```python
# Get multiple runs
run_ids = ["run1", "run2", "run3"]
runs = [api.run(f"my-project/{run_id}") for run_id in run_ids]

# Compare metrics
import pandas as pd

comparison = pd.DataFrame([
    {
        "name": run.name,
        "val_acc": run.summary.get("val/acc"),
        "val_loss": run.summary.get("val/loss"),
        "lr": run.config.get("model").get("optimizer").get("lr"),
    }
    for run in runs
])

print(comparison)
```

### Download Artifacts

```python
# Get artifact
artifact = api.artifact("my-project/model:v0")

# Download
artifact_dir = artifact.download()

# Use artifact
import torch
model = torch.load(f"{artifact_dir}/model.ckpt")
```

## W&B Sweeps

### Define Sweep Config

```yaml
# configs/sweep/wandb_sweep.yaml
program: src/train.py
method: bayes  # grid, random, bayes

metric:
  name: val/acc
  goal: maximize

parameters:
  model.optimizer.lr:
    min: 0.0001
    max: 0.1
    distribution: log_uniform

  model.dropout:
    values: [0.1, 0.2, 0.3, 0.4, 0.5]

  data.batch_size:
    values: [64, 128, 256]

  model.hidden_dims:
    values:
      - [512, 256]
      - [1024, 512, 256]
      - [2048, 1024, 512]

early_terminate:
  type: hyperband
  min_iter: 3
  eta: 3
```

### Run Sweep

```bash
# Initialize sweep
wandb sweep configs/sweep/wandb_sweep.yaml

# Output: wandb: Created sweep with ID: abc123

# Run sweep agent
wandb agent my-entity/my-project/abc123

# Run multiple agents in parallel (on different machines/GPUs)
# Terminal 1:
wandb agent my-entity/my-project/abc123

# Terminal 2:
wandb agent my-entity/my-project/abc123
```

### Sweep in Python

```python
import wandb

# Define sweep config
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/acc", "goal": "maximize"},
    "parameters": {
        "lr": {"min": 0.0001, "max": 0.1, "distribution": "log_uniform"},
        "batch_size": {"values": [64, 128, 256]},
    },
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="my-project")

# Train function
def train():
    # Initialize run
    wandb.init()

    # Get hyperparameters from sweep
    config = wandb.config

    # Create model with sweep params
    model = MyModel(lr=config.lr, batch_size=config.batch_size)

    # Train
    trainer.fit(model)

# Run agent
wandb.agent(sweep_id, function=train, count=50)
```

## W&B Reports

### Create Report via API

```python
import wandb

# Initialize run
run = wandb.init(project="my-project")

# Create report
report = wandb.apis.reports.Report(
    project="my-project",
    title="Experiment Results",
    description="Comparison of baseline experiments",
)

# Add run to report
report.blocks = [
    wandb.apis.reports.RunComparer(
        diff_only=False,
        runset=wandb.Runset(
            project="my-project",
            filters={"tags": {"$in": ["baseline"]}},
        ),
    )
]

# Save report
report.save()
print(f"Report URL: {report.url}")
```

### Manual Report Creation

```bash
# Open W&B workspace
wandb workspace

# In browser:
# 1. Navigate to project
# 2. Select runs to compare
# 3. Click "Create report"
# 4. Add visualizations (charts, tables, etc.)
# 5. Publish report
```

## Best Practices

### 1. Organize with Tags

```python
wandb_logger = WandbLogger(
    project="my-ml-project",
    tags=["baseline", "resnet", "imagenet"],
)
```

### 2. Use Groups for Related Runs

```python
# Group hyperparameter sweep runs
wandb_logger = WandbLogger(
    project="my-ml-project",
    group="hp_optimization_v1",
    job_type="train",
)
```

### 3. Log Hyperparameters

```python
# Automatically logged with Hydra integration
# Or manually:
wandb_logger.experiment.config.update({
    "model": {
        "architecture": "resnet50",
        "lr": 0.001,
        "dropout": 0.3,
    },
    "data": {
        "batch_size": 128,
        "num_workers": 4,
    },
})
```

### 4. Save Code Snapshots

```python
wandb_logger = WandbLogger(
    project="my-ml-project",
    save_code=True,  # Saves git diff + uncommitted changes
)
```

### 5. Use Offline Mode

```bash
# For unstable internet or cluster without internet
export WANDB_MODE=offline

# Train normally
python src/train.py

# Sync later
wandb sync logs/wandb/
```

## Troubleshooting

### Issue: "Failed to upload to W&B"

**Solution:**

```bash
# Check API key
wandb login --relogin

# Or set explicitly
export WANDB_API_KEY=your_api_key
```

### Issue: "Too many images logged, slowing down training"

**Solution:**

```python
# Log images less frequently
if batch_idx == 0 and self.current_epoch % 10 == 0:
    self.logger.experiment.log({"images": wandb_images})
```

### Issue: "W&B using too much disk space"

**Solution:**

```bash
# Clean local cache
wandb artifact cache cleanup 10GB

# Or delete all cache
rm -rf ~/.cache/wandb/
```

### Issue: "Can't see runs in dashboard"

**Solution:**

- Check project name matches
- Verify entity (team) name
- Ensure run finished (call `wandb.finish()`)
- Check browser for correct account

## Integration with Hydra

```yaml
# configs/config.yaml
defaults:
  - logger: wandb

# Hydra integration
hydra:
  run:
    dir: logs/runs/${now:%Y-%m-%d}/${now:%H-%M-%S}

# W&B will automatically log Hydra config
```

```python
# src/train.py
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    # Hydra config automatically logged to W&B
    wandb_logger = hydra.utils.instantiate(cfg.logger)

    trainer = Trainer(logger=wandb_logger)
    trainer.fit(model, datamodule)
```

## Resources

- [W&B Documentation](https://docs.wandb.ai/)
- [PyTorch Lightning Integration](https://docs.wandb.ai/guides/integrations/lightning)
- [W&B Sweeps](https://docs.wandb.ai/guides/sweeps)
- [W&B Artifacts](https://docs.wandb.ai/guides/artifacts)
