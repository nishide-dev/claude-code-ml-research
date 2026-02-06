# ML Workflow Constraints

These workflow constraints ensure consistency, reproducibility, and quality in machine learning experiments.

## Configuration Validation

### Always Validate Before Training

- **Validate Hydra configs before starting long training runs**:

  ```bash
  # Validate config without training
  python src/train.py --cfg job

  # Check resolved config
  python src/train.py --cfg hydra
  ```

- **Required config validation checks**:
  - All required fields are present
  - Data paths exist
  - Model hyperparameters are valid
  - Device settings match available hardware

### Quick Sanity Check

- **Always run a quick sanity check** before full training:

  ```bash
  # Run single batch for 5 steps
  python src/train.py trainer.fast_dev_run=5

  # Or limit to 1 epoch with minimal data
  python src/train.py trainer.max_epochs=1 data.train_size=100
  ```

- **Verify during sanity check**:
  - Model forward pass works
  - Loss is computed correctly
  - Backward pass runs without errors
  - Metrics are logged
  - Checkpoints can be saved

## Checkpointing

### Use Checkpointing for All Runs

- **Configure checkpointing for all training runs**:

  ```yaml
  # config.yaml
  callbacks:
    model_checkpoint:
      dirpath: checkpoints/
      filename: "{epoch:02d}-{val_loss:.2f}"
      monitor: val_loss
      mode: min
      save_top_k: 3
      save_last: true
  ```

- **Required checkpointing settings**:
  - Save top-k best models (k >= 3)
  - Save last checkpoint for resuming
  - Monitor appropriate metric (val_loss, val_acc, etc.)
  - Use descriptive filenames with metrics

### Checkpoint Naming Convention

- **Use consistent naming** for checkpoints:

  ```text
  checkpoints/
  ├── {experiment_name}/
  │   ├── epoch=00-val_loss=0.45.ckpt
  │   ├── epoch=05-val_loss=0.32.ckpt
  │   ├── epoch=10-val_loss=0.28.ckpt
  │   └── last.ckpt
  ```

## Experiment Tracking

### Tag All Experiments

- **Use descriptive names and tags** for all experiments:

  ```python
  # Good - descriptive experiment name
  experiment_name = "resnet50_imagenet_lr0.1_bs256"

  # Bad - non-descriptive name
  experiment_name = "experiment_42"
  ```

- **Include in experiment name/tags**:
  - Model architecture
  - Dataset
  - Key hyperparameters
  - Purpose (baseline, ablation, production, etc.)

### Required Experiment Metadata

- **Log the following metadata** for every experiment:

  ```python
  # In LightningModule or config
  metadata = {
      "model": "resnet50",
      "dataset": "imagenet",
      "batch_size": 256,
      "learning_rate": 0.1,
      "optimizer": "sgd",
      "seed": 42,
      "pytorch_version": torch.__version__,
      "lightning_version": pl.__version__,
      "gpu_type": torch.cuda.get_device_name(0),
      "num_gpus": torch.cuda.device_count(),
  }
  ```

### Log to Experiment Tracker

- **Always use an experiment tracker** (W&B, TensorBoard, MLflow):

  ```yaml
  # config.yaml
  logger:
    class_path: pytorch_lightning.loggers.WandbLogger
    init_args:
      project: my-ml-project
      name: ${experiment_name}
      tags: ["baseline", "resnet50"]
      log_model: true
  ```

## Reproducibility

### Set Seeds Everywhere

- **Set all random seeds** at the start of training:

  ```python
  from pytorch_lightning import seed_everything

  # In main training script
  seed_everything(42, workers=True)
  ```

- **Document seed in config**:

  ```yaml
  seed: 42  # For reproducibility
  trainer:
    deterministic: true  # Enable deterministic mode
  ```

### Version Control All Code Changes

- **Create a Git commit** before starting experiments:

  ```bash
  # Before training
  git add .
  git commit -m "feat: add resnet50 baseline"
  git tag experiment-resnet50-baseline

  # Log git commit in experiment metadata
  git_commit=$(git rev-parse HEAD)
  ```

### Document Config Changes

- **Document config changes in commit messages**:

  ```bash
  # Good commit message
  git commit -m "experiment: increase batch size to 256

  - Changed batch_size from 128 to 256
  - Adjusted learning rate proportionally (0.05 -> 0.1)
  - Expected to improve training speed by 2x"

  # Bad commit message
  git commit -m "update config"
  ```

## Code Review

### Review Before Long Runs

- **Have code reviewed** before starting expensive multi-day training runs:
  - Model architecture
  - Data pipeline
  - Training configuration
  - Resource allocation

### Pre-Training Checklist

Before starting training, verify:

- [ ] Config validated (`--cfg job`)
- [ ] Quick sanity check passed (`fast_dev_run=5`)
- [ ] Checkpointing configured
- [ ] Experiment tracker configured
- [ ] Seeds set
- [ ] Git commit created
- [ ] Resource allocation appropriate
- [ ] Data paths exist and accessible
- [ ] Expected runtime estimated
- [ ] Stopping criteria defined

## Resource Management

### Estimate Resource Requirements

- **Estimate before training**:
  - Expected GPU memory usage
  - Expected training time
  - Required disk space for checkpoints and logs
  - Number of GPUs needed

### Monitor Resource Usage

- **Monitor during training**:

  ```python
  # Log GPU utilization
  import torch

  self.log("gpu_memory_allocated", torch.cuda.memory_allocated() / 1e9)
  self.log("gpu_memory_reserved", torch.cuda.memory_reserved() / 1e9)
  ```

### Set Resource Limits

- **Set appropriate limits** to avoid resource exhaustion:

  ```yaml
  # config.yaml
  trainer:
    max_epochs: 100
    max_time: "48:00:00"  # 48 hours
    gradient_clip_val: 1.0
  ```

## Error Handling

### Enable Anomaly Detection During Development

- **Use anomaly detection** to catch training issues early:

  ```python
  # During development
  torch.autograd.set_detect_anomaly(True)

  # Disable in production for performance
  torch.autograd.set_detect_anomaly(False)
  ```

### Configure Failure Recovery

- **Configure automatic resuming** from last checkpoint:

  ```yaml
  trainer:
    resume_from_checkpoint: "checkpoints/last.ckpt"
    auto_resume: true
  ```

## Model Deployment

### Validate Before Deployment

- **Run full validation** before deploying models:

  ```bash
  # Test on full validation set
  python src/eval.py checkpoint=checkpoints/best.ckpt

  # Test on holdout test set
  python src/test.py checkpoint=checkpoints/best.ckpt

  # Export to production format
  python src/export.py checkpoint=checkpoints/best.ckpt format=torchscript
  ```

### Document Model Performance

- **Document performance metrics**:

  ```yaml
  # model_card.yaml
  model: resnet50_imagenet_baseline
  metrics:
    val_accuracy: 0.892
    test_accuracy: 0.885
    inference_time: 23ms  # batch_size=1
    model_size: 98MB
  training:
    epochs: 90
    training_time: "12 hours"
    gpu: "NVIDIA A100"
  ```

## Experiment Cleanup

### Archive Completed Experiments

- **Archive old experiments** to save disk space:

  ```bash
  # Archive to S3, Google Cloud, etc.
  tar -czf experiment-resnet50-baseline.tar.gz \
      checkpoints/experiment-resnet50-baseline/ \
      logs/experiment-resnet50-baseline/

  # Upload to cloud storage
  aws s3 cp experiment-resnet50-baseline.tar.gz s3://my-experiments/
  ```

### Keep Minimal Checkpoints

- **Delete intermediate checkpoints** after training:
  - Keep only best checkpoint
  - Keep last checkpoint for resuming
  - Archive rest or delete

## Summary Workflow

Standard ML workflow:

1. **Plan**: Design experiment, write config
2. **Validate**: Check config (`--cfg job`)
3. **Sanity Check**: Run `fast_dev_run=5`
4. **Commit**: Create Git commit with descriptive message
5. **Train**: Start training with checkpointing and logging
6. **Monitor**: Watch metrics and resource usage
7. **Evaluate**: Test on validation and test sets
8. **Document**: Update model card and experiment notes
9. **Deploy/Archive**: Deploy if successful, archive otherwise
10. **Cleanup**: Remove intermediate checkpoints
