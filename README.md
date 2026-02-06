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

- **pixi**: Package manager for ML dependencies (CUDA, PyTorch)
- **uv**: Fast Python package manager
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

## Quick Start

### 1. Initialize New ML Project

```bash
/ml:init
```

This will:
- Ask about project requirements (framework, task type, package manager)
- Create lightning-hydra-template style directory structure
- Generate pixi.toml or pyproject.toml with dependencies
- Create Hydra configuration files
- Setup ruff and ty configuration
- Generate starter code (LightningModule, DataModule)

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

```
my-ml-project/
├── configs/
│   ├── config.yaml              # Main config
│   ├── model/
│   │   ├── default.yaml
│   │   ├── resnet50.yaml
│   │   └── gnn.yaml
│   ├── data/
│   │   ├── default.yaml
│   │   └── imagenet.yaml
│   ├── trainer/
│   │   ├── gpu_single.yaml
│   │   └── gpu_ddp.yaml
│   ├── logger/
│   │   └── wandb.yaml
│   └── experiment/
│       ├── baseline.yaml
│       └── hp_sweep.yaml
├── src/
│   ├── models/
│   │   └── base_model.py
│   ├── data/
│   │   └── datamodule.py
│   ├── utils/
│   └── train.py
├── tests/
├── notebooks/
├── logs/
├── pixi.toml or pyproject.toml
├── ruff.toml
└── README.md
```

## Training Commands

```bash
# Basic training
python src/train.py

# With specific experiment
python src/train.py experiment=baseline

# Override parameters
python src/train.py model.lr=0.001 data.batch_size=128

# Multi-GPU training
python src/train.py trainer.devices=4 trainer.strategy=ddp

# Hyperparameter sweep
python src/train.py --multirun model.lr=0.001,0.01,0.1

# Resume from checkpoint
python src/train.py ckpt_path=checkpoints/best.ckpt
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

1. **Always use `self.save_hyperparameters()`** in LightningModule
2. **Use DataModule** for all data loading logic
3. **Enable mixed precision** for faster training
4. **Use W&B** for experiment tracking
5. **Set seeds** for reproducibility
6. **Use callbacks** for checkpointing and early stopping
7. **Test with `fast_dev_run`** before long training
8. **Monitor GPU utilization** (should be >80%)

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

- [Claude Code Documentation](https://code.claude.com/docs)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [Hydra](https://hydra.cc/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Weights & Biases](https://docs.wandb.ai/)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

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
