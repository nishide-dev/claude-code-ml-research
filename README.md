# ML Research Plugin for Claude Code

Comprehensive Claude Code plugin for machine learning research and experimentation. Provides project scaffolding, experiment management, training support, and debugging tools for PyTorch Lightning, Hydra, PyTorch Geometric, and Hugging Face Transformers.

## Features

### Commands

- `/ml:init` - Initialize new ML project with PyTorch Lightning + Hydra structure
- `/ml:config` - Generate and manage Hydra configuration files
- `/ml:train` - Execute training with monitoring and debugging
- `/ml:experiment` - Manage experiments, track results, and compare performance
- `/ml:debug` - Debug training issues (NaN loss, OOM, convergence problems)
- `/ml:data` - Create and manage data pipelines and preprocessing

### Agents

- **ml-architect**: Design ML system architectures and model designs
- **config-generator**: Generate and validate Hydra configurations
- **training-debugger**: Diagnose and fix training issues
- **pytorch-expert**: Write efficient PyTorch code and optimize performance
- **geometric-specialist**: Implement Graph Neural Networks with PyTorch Geometric

### Skills

- **lightning-basics**: PyTorch Lightning fundamentals and best practices
- **hydra-config**: Hydra configuration management
- **pytorch-geometric**: Graph Neural Network implementation
- **wandb-tracking**: Weights & Biases experiment tracking

### Supported Frameworks

- **PyTorch Lightning**: High-level training framework
- **PyTorch**: Core deep learning framework
- **PyTorch Geometric**: Graph Neural Networks
- **Hugging Face Transformers**: NLP and vision transformers
- **Hydra**: Configuration management
- **Weights & Biases**: Experiment tracking

### Supported Tools

- **pixi**: Conda-based package manager with PyPI integration (recommended for GPU/CUDA projects)
- **uv**: Fast Python package manager with pip compatibility (recommended for CPU or simple projects)
- **ruff**: Linting and formatting
- **ty**: Type checking

## Installation

### Via Claude Code Marketplace (Coming Soon)

```bash
claude plugin install ml-research
```

### Local Development

```bash
# Clone repository
git clone https://github.com/nishide-dev/claude-code-ml-research.git
cd claude-code-ml-research

# Test plugin locally
claude --plugin-dir . code
```

### Prerequisites

This plugin uses `copier` (via `uvx`) to generate projects from templates. Ensure you have `uv` installed:

```bash
# Check if uv is installed
uvx --version

# If not installed, install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

### Choosing a Package Manager

The template supports two package managers with different strengths:

#### Pixi (Recommended for GPU/ML Research)

**Best for:**
- Projects requiring CUDA/GPU support
- Complex ML dependencies (PyTorch, PyTorch Geometric)
- Reproducible environments across platforms
- Teams using conda ecosystems

**Architecture:**
- Python from conda (base environment)
- ML packages from PyPI via `[pypi-dependencies]` (faster, latest versions)
- PyTorch includes bundled CUDA (no separate CUDA installation needed)
- Automatic dependency resolution and locking

**Installation:**
```bash
curl -fsSL https://pixi.sh/install.sh | sh
pixi --version  # Should show v0.63+
```

**Commands:**
```bash
pixi install          # Install all dependencies
pixi run train        # Run training
pixi run test         # Run tests
pixi run format       # Format code
```

#### UV (Recommended for Simple/CPU Projects)

**Best for:**
- CPU-only projects
- Simple dependencies
- Fast iteration and prototyping
- Projects without complex CUDA requirements

**Architecture:**
- Pure Python packages from PyPI
- Fast dependency resolution
- pip-compatible workflows

**Commands:**
```bash
uv sync                                      # Install dependencies
uv run python src/my_project/train.py       # Run training
uv run pytest tests/                        # Run tests
```

### 1. Initialize New ML Project

```bash
/ml:init
```

This command uses `uvx copier copy` to create a project from the ML Research template.

**Interactive Configuration:**

The template will ask you to configure:
- **Package Manager:** uv (fast, pip-compatible) or pixi (conda-based, automatic CUDA)
- **Python Version:** 3.10, 3.11, 3.12, or 3.13
- **PyTorch + CUDA:** Choose from presets (PyTorch 2.4-2.9, CUDA 11.8-13.0)
- **ML Frameworks:** PyTorch Lightning, Hydra, PyTorch Geometric
- **Experiment Tracking:** TensorBoard, Weights & Biases, MLflow
- **Template Type:** Image classification, segmentation, GNN, text classification
- **Dataset:** MNIST, CIFAR-10, CIFAR-100, Fashion-MNIST (for vision tasks)

**What it creates:**
- `src/{{ package_name }}/` directory with your Python package
- Complete Hydra configuration structure
- PyTorch Lightning model and data module templates
- Test suite with pytest
- Development tools: ruff, ty, pytest configured
- Git repository initialized with proper .gitignore
- Virtual environment with all dependencies installed

**Manual Usage:**

```bash
# From the plugin directory
cd /path/to/claude-code-ml-research

# Create new project
uvx copier copy --trust templates/ml-research ~/projects/my-ml-project

# Follow the interactive prompts
# cd to your new project
cd ~/projects/my-ml-project

# Start training
uv run python src/my_ml_project/train.py  # if using uv
pixi run train  # if using pixi
```

### 2. Generate Configuration

```bash
/ml:config
```

Create new model, data, trainer, or experiment configurations with proper Hydra structure.

### 3. Train Model

```bash
/ml:train
```

Execute training with:
- Pre-training validation checks
- Real-time monitoring
- Automatic checkpointing
- Experiment tracking (W&B)
- Multi-GPU support

### 4. Debug Issues

```bash
/ml:debug
```

Diagnose and fix:
- NaN/Inf loss
- Memory errors (OOM)
- Slow training
- Convergence problems
- Overfitting/underfitting

## Example Project Structure

Projects generated from the template have this structure:

```
my-ml-project/
├── src/
│   └── my_ml_project/           # Package name auto-generated from project name
│       ├── __init__.py
│       ├── train.py             # Main training script with Hydra
│       ├── models/
│       │   ├── __init__.py
│       │   └── base_model.py    # LightningModule template
│       ├── data/
│       │   ├── __init__.py
│       │   └── datamodule.py    # LightningDataModule template
│       ├── utils/
│       │   └── __init__.py
│       └── py.typed             # PEP 561 type hint marker
├── configs/                     # Hydra configurations
│   ├── config.yaml              # Main config with defaults
│   ├── model/
│   │   └── default.yaml         # Model architecture config
│   ├── data/
│   │   └── default.yaml         # Dataset config
│   ├── trainer/
│   │   └── default.yaml         # Lightning Trainer config
│   ├── logger/
│   │   ├── tensorboard.yaml     # TensorBoard logger
│   │   ├── wandb.yaml           # W&B logger
│   │   └── mlflow.yaml          # MLflow logger
│   └── experiment/
│       └── baseline.yaml        # Experiment configuration
├── tests/
│   ├── __init__.py
│   └── test_my_ml_project.py    # Model and data tests
├── .venv/                       # Virtual environment (if using uv)
├── .pixi/                       # Pixi environment (if using pixi)
├── pyproject.toml               # Python project config (if using uv)
├── pixi.toml                    # Pixi config (if using pixi)
├── ruff.toml                    # Ruff linting config
├── .gitignore                   # Git ignore rules
├── .python-version              # Python version
├── README.md                    # Project documentation
└── uv.lock or pixi.lock         # Lock file
```

**Key Features:**
- `src/{{ package_name }}/` structure for proper Python packaging
- Complete Hydra configuration with composition support
- Lightning callbacks (checkpointing, early stopping, progress bar)
- Multiple logger options (TensorBoard/W&B/MLflow)
- Comprehensive tests with pytest
- Type hints throughout (validated by ty)
- Ruff-compliant code (no print statements, proper logging)

## Training Commands

Generated projects use the `src/{{ package_name }}/train.py` structure:

### Using Pixi (Recommended for GPU)

```bash
# Basic training
pixi run train

# Quick test (1 batch per epoch)
pixi run train-debug

# CPU-only training
pixi run train-cpu

# GPU training (explicit)
pixi run train-gpu

# Override Hydra parameters
pixi run train trainer.max_epochs=50 model.lr=0.001

# With specific experiment
pixi run train experiment=baseline

# Multi-GPU training
pixi run train trainer.devices=4 trainer.strategy=ddp

# Development tasks
pixi run test           # Run tests
pixi run test-cov       # Run tests with coverage
pixi run lint           # Check code style
pixi run format         # Format code
pixi run typecheck      # Type check with ty

# TensorBoard
pixi run tensorboard    # Launch TensorBoard (if configured)
```

### Using UV

```bash
# Basic training
uv run python src/my_ml_project/train.py

# With specific experiment
uv run python src/my_ml_project/train.py experiment=baseline

# Override parameters
uv run python src/my_ml_project/train.py model.lr=0.001 data.batch_size=128

# Multi-GPU training
uv run python src/my_ml_project/train.py trainer.devices=4 trainer.strategy=ddp

# Hyperparameter sweep
uv run python src/my_ml_project/train.py --multirun model.lr=0.001,0.01,0.1

# Resume from checkpoint
uv run python src/my_ml_project/train.py ckpt_path=checkpoints/best.ckpt

# Quick test (1 batch per epoch)
uv run python src/my_ml_project/train.py trainer.fast_dev_run=true

# Development tasks
uv run pytest tests/ -v              # Run tests
uv run ruff check src/ tests/        # Lint
uv run ruff format src/ tests/       # Format
```

## Configuration Examples

### Model Config

```yaml
# configs/model/resnet50.yaml
_target_: src.models.resnet.ResNetModel

arch: resnet50
num_classes: 1000
pretrained: true

optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 0.0001

scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: ${trainer.max_epochs}
```

### Experiment Config

```yaml
# configs/experiment/baseline.yaml
# @package _global_

defaults:
  - override /model: resnet50
  - override /data: imagenet
  - override /trainer: gpu_ddp

name: "resnet50_imagenet_baseline"
seed: 42

model:
  lr: 0.0001  # Fine-tuning LR

data:
  batch_size: 256

trainer:
  max_epochs: 50
  devices: 4
```

## PyTorch Geometric Support

```bash
# Initialize GNN project
/ml:init
# Select: PyTorch Geometric, Graph ML task

# Train node classification
python src/train.py model=gnn data=cora

# Train graph classification with batching
python src/train.py model=gnn data=proteins data.batch_size=32

# Large graph with sampling
python src/train.py model=gnn data=ogbn-products data.use_sampling=true
```

## Debugging Guide

### NaN Loss

```bash
# Run debug command
/ml:debug

# Common fixes:
# 1. Lower learning rate
python src/train.py model.optimizer.lr=0.0001

# 2. Enable gradient clipping
python src/train.py trainer.gradient_clip_val=1.0

# 3. Use full precision
python src/train.py trainer.precision=32
```

### Out of Memory

```bash
# 1. Reduce batch size
python src/train.py data.batch_size=32

# 2. Enable gradient accumulation
python src/train.py trainer.accumulate_grad_batches=4

# 3. Mixed precision
python src/train.py trainer.precision=16-mixed
```

## Experiment Tracking

### Weights & Biases

```bash
# Login to W&B
wandb login

# Train with W&B logging (automatic if logger configured)
python src/train.py

# View experiments
wandb workspace
```

### Compare Experiments

```bash
# List all experiments
python scripts/list_experiments.py

# Compare specific experiments
python scripts/compare_experiments.py exp_001 exp_002 exp_003

# Generate report
python scripts/generate_report.py --output report.md
```

## Best Practices

### General ML Practices

1. **Always use `self.save_hyperparameters()`** in LightningModule
2. **Use DataModule** for all data loading logic
3. **Enable mixed precision** for faster training
4. **Use W&B** for experiment tracking
5. **Set seeds** for reproducibility
6. **Use callbacks** for checkpointing and early stopping
7. **Test with `fast_dev_run`** before long training
8. **Monitor GPU utilization** (should be >80%)

### Package Manager Best Practices

#### For Pixi Projects

1. **Use `pixi.toml` tasks** instead of remembering long commands
   ```bash
   pixi run train-debug  # Instead of: pixi run python src/...
   ```

2. **Lock file is critical** - Always commit `pixi.lock`
   ```bash
   git add pixi.lock  # Ensures reproducible environments
   ```

3. **PyPI packages in `[pypi-dependencies]`** - ML packages install faster from PyPI
   - Template already optimized: PyTorch, Lightning from PyPI
   - CUDA is bundled with PyTorch (no separate installation)

4. **Platform-specific environments** - Pixi handles this automatically
   ```toml
   platforms = ["linux-64", "osx-arm64"]  # Intel macOS not supported by PyTorch 2.5+
   ```

5. **Feature environments** for optional dependencies
   ```bash
   pixi run --environment dev test  # Dev environment with extra tools
   ```

#### For UV Projects

1. **Use `uv.lock` for reproducibility** - Commit lock files
2. **Leverage uv's speed** - `uv sync` is much faster than pip
3. **Use extras for optional features**
   ```bash
   uv pip install -e ".[dev,wandb]"
   ```

## Development

### Adding New Commands

```bash
# Create command file
echo "# My Command\n\nCommand description..." > commands/my-command.md

# Plugin will automatically detect it
```

### Adding New Agents

```yaml
# agents/my-agent.md
---
name: my-agent
description: Agent description
tools: ["Read", "Write", "Edit"]
model: sonnet
---

You are an expert in...

## Your Role

- ...
```

### Adding New Skills

```bash
# Create skill directory
mkdir skills/my-skill

# Create SKILL.md
echo "# My Skill\n\nSkill content..." > skills/my-skill/SKILL.md
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Resources

### Documentation

- [Claude Code Documentation](https://code.claude.com/docs)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [Hydra](https://hydra.cc/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Weights & Biases](https://docs.wandb.ai/)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

### Package Managers

- [Pixi Documentation](https://pixi.sh/)
- [UV Documentation](https://docs.astral.sh/uv/)
- [Pixi PyPI Integration Guide](https://pixi.sh/latest/features/pypi_packages/)

### Templates

- [uv-torch-nix-template](https://github.com/nishide-dev/uv-torch-nix-template) - UV-based PyTorch template
- [uv-pyg-nix-template](https://github.com/nishide-dev/uv-pyg-nix-template) - UV-based PyTorch Geometric template

## Citation

If you use this plugin in your research, please cite:

```bibtex
@software{claude_code_ml_research,
  author = {ML Research Team},
  title = {ML Research Plugin for Claude Code},
  year = {2026},
  url = {https://github.com/nishide-dev/claude-code-ml-research}
}
```

## Support

- GitHub Issues: https://github.com/nishide-dev/claude-code-ml-research/issues
- Discussions: https://github.com/nishide-dev/claude-code-ml-research/discussions

---

Built with ❤️ for ML researchers using Claude Code
