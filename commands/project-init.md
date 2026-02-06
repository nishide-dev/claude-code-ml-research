---
name: project-init
description: Initialize a new ML research project using the ML Research template with PyTorch Lightning, Hydra, and modern Python tooling
---

# ML Project Initialization

Initialize a new machine learning research project using the ML Research Copier template with PyTorch Lightning, Hydra configuration, and modern Python tooling (uv).

## Process

### 1. Explain the Template System

This command uses `uvx copier` to create projects from templates. The template supports:

- **PyTorch + CUDA**: Multiple version presets (2.4-2.9, CUDA 11.8-13.0)
- **PyTorch Lightning**: For training infrastructure
- **Hydra**: For configuration management
- **Experiment Tracking**: TensorBoard, W&B, MLflow
- **Modern Tooling**: uv (package manager), ruff (linter), ty (type checker)
- **Multiple Templates**: Image classification, segmentation, GNN, etc.

### 2. Check Prerequisites

Verify that `uvx` is available:

```bash
uvx --version
```

If not available, install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Run Copier Template

Execute the copier command pointing to the ML research template:

```bash
uvx copier copy <plugin-dir>/templates/ml-research <project-directory>
```

Where:

- `<plugin-dir>` is the absolute path to the claude-code-ml-research plugin directory
- `<project-directory>` is where the new project will be created

### 4. Interactive Configuration

Copier will ask the user to configure:

**Project Basics:**

- Project name (defaults to directory name)
- Package name (Python import name, auto-generated from project name)
- Description
- Author name and email
- Python version (3.10-3.13)

**Development Tools:**

- Use ruff? (default: yes)
- Use ty? (default: yes)
- Use pytest? (default: yes)
- Use GitHub Actions? (default: yes)
- Use Nix + direnv? (default: no)

**PyTorch/CUDA:**

- PyTorch + CUDA preset (interactive dropdown with compatible combinations)
  - PyTorch 2.8.0 + CUDA 12.6 (recommended)
  - PyTorch 2.9.0 + CUDA 12.6 (latest)
  - ... (many presets)
  - Custom (manual version entry)
- Include torchvision? (default: yes)
- Include torchaudio? (default: no)

**ML Frameworks:**

- Use PyTorch Lightning? (default: yes)
- Lightning version (if enabled)
- Use Hydra? (default: yes)
- Hydra version (if enabled)
- Use PyTorch Geometric? (default: no)

**Experiment Tracking:**

- Logger choice: TensorBoard / W&B / MLflow / Both / None
- W&B entity (if W&B selected)

**Template Type:**

- Image Classification (default)
- Segmentation
- Object Detection
- Text Classification
- GNN (Graph Neural Network)
- Minimal (custom template)

**Dataset:**

- MNIST / CIFAR-10 / CIFAR-100 / Fashion-MNIST / Custom (for image classification)

### 5. Post-Generation

Copier automatically executes post-generation tasks:

1. `git init -b main` - Initialize git repository
2. `uv venv` - Create virtual environment
3. `uv lock` - Lock dependencies
4. `uv sync` - Install dependencies
5. `ruff check . --fix` - Lint and auto-fix
6. `ruff format .` - Format code
7. `ty check` - Type check (if enabled)
8. `pytest` - Run tests (if enabled)

### 6. Project Structure

The generated project will have:

```text
<project-name>/
├── src/
│   └── <package-name>/
│       ├── __init__.py
│       ├── train.py          # Main training script with Hydra
│       ├── models/           # LightningModule definitions
│       ├── data/             # DataModule and datasets
│       └── utils/            # Utility functions
├── tests/
│   └── test_<package_name>.py
├── configs/                  # Hydra configuration
│   ├── config.yaml
│   ├── model/
│   ├── data/
│   ├── trainer/
│   ├── logger/
│   └── experiment/
├── pyproject.toml           # uv + project config
├── ruff.toml                # Ruff linting config
├── .gitignore
├── README.md
└── .venv/                   # Virtual environment (created)
```

### 7. Verify Installation

After project creation, verify the setup:

```bash
cd <project-name>

# Verify CUDA availability
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run a quick test
uv run pytest tests/ -v

# Check code quality
uv run ruff check .
```

### 8. Next Steps

Guide the user:

```bash
# Start training (with Hydra defaults)
uv run python src/<package-name>/train.py

# Override configuration
uv run python src/<package-name>/train.py trainer.max_epochs=20 data.batch_size=64

# Run specific experiment
uv run python src/<package-name>/train.py experiment=baseline

# Start TensorBoard (if using TensorBoard)
tensorboard --logdir logs/

# Login to W&B (if using W&B)
wandb login
```

## Alternative: Manual Template Selection

If the user wants a specific template variant without interactive prompts, they can use:

```bash
# Use defaults with --defaults flag
uvx copier copy --defaults <plugin-dir>/templates/ml-research <project-directory>

# Use data flags for non-interactive
uvx copier copy \
  --data project_name="my-project" \
  --data pytorch_cuda_preset="pytorch-2.8.0-cuda-12.6" \
  --data use_lightning=true \
  --data logger_choice="wandb" \
  <plugin-dir>/templates/ml-research <project-directory>
```

## Troubleshooting

### copier not found

```bash
# Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh

# Then uvx will be available
```

### CUDA version mismatch

Check available CUDA on system:

```bash
nvidia-smi
```

Select matching preset or custom CUDA version during copier prompts.

### Import errors after generation

Ensure virtual environment is activated or use `uv run`:

```bash
uv run python src/<package-name>/train.py
```

## Success Criteria

- [ ] Copier template executes without errors
- [ ] All dependencies installed successfully
- [ ] CUDA detection works (if GPU available)
- [ ] Tests pass
- [ ] Ruff checks pass
- [ ] README generated with correct instructions
- [ ] Git repository initialized

Project is ready for ML research!

## Example Usage

```bash
# Get plugin directory
PLUGIN_DIR=$(pwd)

# Create new project in ~/projects
cd ~/projects
uvx copier copy $PLUGIN_DIR/templates/ml-research my-cifar10-project

# Follow interactive prompts, then:
cd my-cifar10-project
uv run python src/my_cifar10_project/train.py
```
