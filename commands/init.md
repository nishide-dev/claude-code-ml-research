# ML Project Initialization

Initialize a new machine learning research project with PyTorch Lightning, Hydra configuration, and modern Python tooling.

## Process

### 1. Project Setup Questions

Ask the user about project configuration:
- Project name and description
- Framework preference (PyTorch Lightning, vanilla PyTorch, or Transformers)
- Package manager (pixi for ML dependencies with CUDA, uv for pure Python)
- Task type (classification, regression, generation, GNN, etc.)
- Experiment tracking (W&B, MLFlow, TensorBoard, local logs only)
- Include PyTorch Geometric for graph neural networks?

### 2. Directory Structure Creation

Create lightning-hydra-template inspired structure:

```
<project-name>/
├── configs/
│   ├── config.yaml          # Main config
│   ├── model/               # Model configurations
│   ├── data/                # Dataset configurations
│   ├── trainer/             # Trainer configurations
│   ├── logger/              # Logger configurations
│   └── experiment/          # Experiment configs
├── src/
│   ├── __init__.py
│   ├── models/              # LightningModule definitions
│   ├── data/                # DataModule and datasets
│   ├── utils/               # Utility functions
│   └── train.py             # Training entry point
├── tests/
│   ├── __init__.py
│   └── test_model.py
├── notebooks/               # Jupyter notebooks
├── logs/                    # Training logs
├── scripts/                 # Helper scripts
├── .gitignore
├── pyproject.toml or pixi.toml
├── README.md
└── ruff.toml
```

### 3. Package Manager Configuration

**If pixi selected (recommended for ML with GPU):**

Create `pixi.toml`:
```toml
[project]
name = "<project-name>"
version = "0.1.0"
channels = ["pytorch", "nvidia", "conda-forge"]
platforms = ["linux-64", "osx-arm64", "osx-64"]

[dependencies]
python = ">=3.10"
pytorch = ">=2.1"
pytorch-lightning = ">=2.1"
hydra-core = ">=1.3"
wandb = "*"          # If W&B selected
ruff = "*"
pytest = "*"

# Add PyTorch Geometric if requested
# pytorch-geometric = "*"
# torch-scatter = "*"
# torch-sparse = "*"

[tasks]
train = "python src/train.py"
test = "pytest tests/"
lint = "ruff check ."
format = "ruff format ."
```

**If uv selected:**

Create `pyproject.toml`:
```toml
[project]
name = "<project-name>"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "pytorch-lightning>=2.1",
    "hydra-core>=1.3",
    "wandb",  # If W&B selected
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "typer"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 4. Configuration Files

Generate Hydra configuration files in `configs/`:

**config.yaml:**
```yaml
defaults:
  - model: default
  - data: default
  - trainer: default
  - logger: wandb  # or tensorboard/mlflow
  - _self_

seed: 42
work_dir: ${hy dra:runtime.cwd}

hydra:
  run:
    dir: logs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

**configs/model/default.yaml** - based on task type
**configs/data/default.yaml** - dataset configuration
**configs/trainer/default.yaml** - Lightning trainer settings

### 5. Starter Code

Generate `src/train.py` with:
- Hydra decorator for configuration
- Lightning Trainer setup
- Proper logging initialization
- Seed setting for reproducibility
- W&B integration if selected

Generate `src/models/base_model.py` with:
- LightningModule template
- training_step, validation_step, test_step
- configure_optimizers
- Proper typing annotations

Generate `src/data/datamodule.py` with:
- LightningDataModule template
- setup(), train_dataloader(), val_dataloader(), test_dataloader()

### 6. Tooling Configuration

Create `ruff.toml`:
```toml
line-length = 100
target-version = "py310"

[lint]
select = [
    "E",   # pycodestyle errors
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "ANN", # flake8-annotations
    "B",   # flake8-bugbear
]
ignore = [
    "ANN101",  # Missing type annotation for self
    "ANN102",  # Missing type annotation for cls
]

[lint.per-file-ignores]
"__init__.py" = ["F401"]  # Unused imports
"tests/*" = ["ANN"]       # No annotations in tests
```

Create `.gitignore`:
```
# Python
__pycache__/
*.py[cod]
*.so
.Python
*.egg-info/
dist/
build/

# ML specific
logs/
wandb/
lightning_logs/
checkpoints/
*.ckpt
*.pth
*.pt

# Data
data/
*.csv
*.h5
*.hdf5

# IDEs
.vscode/
.idea/
*.swp

# Environment
.env
.pixi/
.venv/
```

### 7. Documentation

Generate `README.md` with:
- Project description
- Installation instructions (pixi install or uv sync)
- Usage examples (training commands)
- Project structure explanation
- Configuration guide

### 8. Initialize Tools

Run commands:
```bash
pixi install  # or uv sync
ruff check .
ruff format .
```

### 9. Git Initialization (if not already a repo)

```bash
git init
git add .
git commit -m "feat: initialize ML project with Lightning + Hydra

- Setup project structure
- Configure Hydra for experiments
- Add Lightning boilerplate
- Setup ruff linting and formatting"
```

## Success Criteria

- [ ] All directories created
- [ ] Package manager configured correctly
- [ ] Hydra configs generated
- [ ] Starter code compiles without errors
- [ ] Ruff checks pass
- [ ] README documentation complete
- [ ] Git repository initialized (if applicable)

Project is ready for ML research and experimentation!
