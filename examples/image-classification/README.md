# Image Classification Example

A complete example demonstrating PyTorch Lightning + Hydra for image classification on CIFAR-10.

## Overview

This example demonstrates:
- CNN architecture for image classification
- Data augmentation with torchvision transforms
- Training with PyTorch Lightning
- Configuration management with Hydra
- Experiment tracking with TensorBoard/W&B
- Model checkpointing and early stopping

## Dataset

**CIFAR-10**: 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- 50,000 training images
- 10,000 test images

The dataset will be automatically downloaded on first run.

## Quick Start

### Option 1: Using uv

```bash
# Install dependencies
uv sync

# Train with default config
python src/train.py

# Train with specific config
python src/train.py model=resnet data.batch_size=64 trainer.max_epochs=20

# Run experiment
python src/train.py experiment=baseline
```

### Option 2: Using pixi

```bash
# Install dependencies
pixi install

# Train
pixi run train

# Train with config overrides
pixi run train model=resnet data.batch_size=64
```

### Option 3: Manual installation

```bash
pip install torch torchvision pytorch-lightning hydra-core tensorboard

python src/train.py
```

## Project Structure

```
image-classification/
├── configs/
│   ├── config.yaml              # Main config with defaults
│   ├── model/
│   │   ├── simple_cnn.yaml      # Simple CNN (default)
│   │   └── resnet.yaml          # ResNet-18 variant
│   ├── data/
│   │   └── cifar10.yaml         # CIFAR-10 dataset config
│   ├── trainer/
│   │   ├── default.yaml         # Training configuration
│   │   └── gpu.yaml             # GPU-specific settings
│   ├── logger/
│   │   ├── tensorboard.yaml
│   │   └── wandb.yaml
│   └── experiment/
│       ├── baseline.yaml        # Baseline experiment
│       └── augmented.yaml       # With data augmentation
├── src/
│   ├── train.py                 # Training script
│   ├── models/
│   │   ├── simple_cnn.py        # Simple CNN model
│   │   └── resnet.py            # ResNet variant
│   └── data/
│       └── cifar10_datamodule.py # CIFAR-10 DataModule
└── outputs/                     # Training outputs (created automatically)
```

## Configuration

### Model Selection

```bash
# Simple CNN (default)
python src/train.py model=simple_cnn

# ResNet-18 variant
python src/train.py model=resnet
```

### Data Augmentation

```bash
# Enable augmentation
python src/train.py data.augment_train=true

# Disable normalization
python src/train.py data.normalize=false
```

### Training Settings

```bash
# GPU training
python src/train.py trainer.accelerator=gpu trainer.devices=1

# Multi-GPU
python src/train.py trainer.accelerator=gpu trainer.devices=2 trainer.strategy=ddp

# CPU training
python src/train.py trainer.accelerator=cpu

# Quick test (1 batch per epoch)
python src/train.py trainer.fast_dev_run=true
```

### Experiment Tracking

```bash
# TensorBoard (default)
python src/train.py logger=tensorboard

# Weights & Biases
python src/train.py logger=wandb logger.wandb.project=cifar10-classification
```

## Expected Results

With default settings (simple CNN, 10 epochs):
- **Training accuracy**: ~70-75%
- **Validation accuracy**: ~65-70%
- **Training time**: ~5 minutes on GPU, ~30 minutes on CPU

With ResNet and augmentation (50 epochs):
- **Training accuracy**: ~95%+
- **Validation accuracy**: ~85-90%
- **Training time**: ~20 minutes on GPU

## Experiments

### Baseline

```bash
python src/train.py experiment=baseline
```

Simple CNN, no augmentation, 10 epochs. Good for quick testing.

### Augmented

```bash
python src/train.py experiment=augmented
```

ResNet-18 with data augmentation, 50 epochs. Better accuracy.

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006

### Weights & Biases

```bash
# Login first
wandb login

# Train with W&B
python src/train.py logger=wandb logger.wandb.project=my-project
```

## Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Advanced Usage

### Hyperparameter Sweeps

```bash
# Grid search
python src/train.py -m model=simple_cnn,resnet data.batch_size=32,64,128

# Multiple parameters
python src/train.py -m model.lr=0.001,0.0001 data.batch_size=32,64 trainer.max_epochs=10,20
```

### Resume Training

Lightning automatically saves checkpoints in `outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/`.

```bash
python src/train.py trainer.resume_from_checkpoint=outputs/.../checkpoints/last.ckpt
```

### Export Model

```bash
# After training, export to ONNX
python scripts/export_onnx.py --checkpoint outputs/.../checkpoints/best.ckpt
```

## Tips

1. **Start small**: Use `trainer.fast_dev_run=true` to test your setup
2. **Monitor overfitting**: Watch train vs validation accuracy
3. **Use GPU**: Training is 5-10x faster on GPU
4. **Experiment tracking**: Use W&B or TensorBoard to compare runs
5. **Data augmentation**: Helps prevent overfitting, especially with small datasets

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python src/train.py data.batch_size=16

# Reduce model size
python src/train.py model=simple_cnn model.hidden_channels=32

# Use gradient accumulation
python src/train.py trainer.accumulate_grad_batches=4
```

### Slow Data Loading

```bash
# Increase workers (be careful with memory)
python src/train.py data.num_workers=8

# Disable persistent workers if low memory
python src/train.py data.persistent_workers=false
```

### NaN Loss

- Check learning rate (try lower: `model.lr=0.0001`)
- Check for data normalization issues
- Try gradient clipping: `trainer.gradient_clip_val=1.0`

## References

- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Hydra Documentation](https://hydra.cc/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
