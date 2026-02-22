---
name: ml-validate
description: Comprehensive validation of ML project structure, configurations, code quality, and training readiness. Use when setting up a new project, before training runs, or debugging configuration issues. Validates config loading, data pipeline, model architecture, and dependencies.
disable-model-invocation: true
---

# ML Project Validation

Comprehensive validation of ML project structure, configurations, code quality, and training readiness.

## Quick Start

```bash
# Run full validation
python scripts/validate_project.py

# Quick config check
python src/train.py --cfg job

# Fast dev run (1 batch train/val/test)
python src/train.py trainer.fast_dev_run=true
```

## Validation Checks

### 1. Project Structure

**Required directories:**

- `src/` - Source code
- `src/models/` - Model implementations
- `src/data/` - DataModule implementations
- `configs/` - Hydra configuration files
- `tests/` - Unit tests (recommended)

**Required files:**

- `src/train.py` - Training script
- `configs/config.yaml` - Main config
- `pyproject.toml` or `pixi.toml` - Package manager

**Check manually:**

```bash
# Verify structure
test -d src && test -d configs && echo "✓ Basic structure OK"
test -f src/train.py && echo "✓ Training script found"
test -f configs/config.yaml && echo "✓ Main config found"
```

### 2. Configuration Validation

**YAML syntax:**

```bash
# Validate all YAML files
python -c "
import yaml
from pathlib import Path

for yaml_file in Path('configs').rglob('*.yaml'):
    try:
        yaml.safe_load(yaml_file.read_text())
        print(f'✓ {yaml_file}')
    except yaml.YAMLError as e:
        print(f'❌ {yaml_file}: {e}')
"
```

**Config composition:**

```bash
# Test Hydra config loads correctly
python src/train.py --cfg job
```

**_target_ validation:**

- All `_target_` paths must be importable
- Check model, data, trainer, logger targets
- Verify no typos in module paths

Use `scripts/validate_project.py` for automated checking.

### 3. Code Quality

**Linting:**

```bash
# Ruff checks
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/
```

**Type checking:**

```bash
# ty (type checker)
ty check src/

# mypy (alternative)
mypy src/ --ignore-missing-imports
```

**Import validation:**

```python
# Check all files have valid Python syntax
import ast
from pathlib import Path

for py_file in Path("src").rglob("*.py"):
    try:
        ast.parse(py_file.read_text())
        print(f"✓ {py_file}")
    except SyntaxError as e:
        print(f"❌ {py_file}: {e}")
```

### 4. Dependencies

**Required packages:**

- `torch` - PyTorch
- `pytorch_lightning` - Lightning framework
- `hydra-core` - Configuration management

**Optional but recommended:**

- `wandb` - Experiment tracking
- `tensorboard` - Visualization
- `torch_geometric` - For GNNs
- `transformers` - For NLP

**Check installation:**

```bash
python -c "
import torch
import pytorch_lightning
import hydra

print(f'PyTorch: {torch.__version__}')
print(f'Lightning: {pytorch_lightning.__version__}')
print(f'Hydra: {hydra.__version__}')
"
```

**GPU availability:**

```bash
python -c "
import torch

print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

### 5. Data Pipeline

**DataModule instantiation:**

```python
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path

# Load config
config_dir = Path.cwd() / "configs"
with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    cfg = compose(config_name="config")

# Instantiate DataModule
dm = instantiate(cfg.data)
print(f"✓ DataModule: {type(dm).__name__}")

# Test setup
dm.setup("fit")
print("✓ DataModule.setup() successful")

# Check dataloaders
train_loader = dm.train_dataloader()
print(f"✓ Train batches: {len(train_loader)}")
```

**Data directory:**

```bash
# Verify data path exists
python -c "
from omegaconf import OmegaConf
from pathlib import Path

cfg = OmegaConf.load('configs/config.yaml')
data_dir = Path(cfg.data.data_dir)

if data_dir.exists():
    print(f'✓ Data directory: {data_dir}')
    print(f'  Files: {len(list(data_dir.rglob(\"*\")))}')
else:
    print(f'⚠️  Data directory not found: {data_dir}')
"
```

### 6. Model Validation

**Model instantiation:**

```python
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path

# Load config
config_dir = Path.cwd() / "configs"
with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    cfg = compose(config_name="config")

# Instantiate model
model = instantiate(cfg.model)
print(f"✓ Model: {type(model).__name__}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")
```

**Forward pass test:**

```python
import torch

# Create dummy input (adjust for your model)
batch_size = 2
dummy_input = torch.randn(batch_size, 3, 224, 224)

# Test forward pass
model.eval()
with torch.no_grad():
    output = model(dummy_input)

print(f"✓ Forward pass OK")
print(f"  Input: {dummy_input.shape}")
print(f"  Output: {output.shape}")
```

### 7. Training Readiness

**Fast dev run:**

```bash
# Run 1 batch of train/val/test
python src/train.py trainer.fast_dev_run=true

# Expected output:
# - No errors
# - Completes in <1 minute
# - Shows train/val/test progress
```

**Logger check:**

```python
from hydra import compose, initialize_config_dir
from pathlib import Path
import os

config_dir = Path.cwd() / "configs"
with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    cfg = compose(config_name="config")

if "logger" in cfg:
    print(f"✓ Logger: {cfg.logger.get('_target_', 'unknown')}")

    # Check W&B credentials if using wandb
    if "wandb" in str(cfg.logger.get("_target_", "")):
        if "WANDB_API_KEY" in os.environ:
            print("✓ W&B API key set")
        else:
            print("⚠️  W&B not logged in (run: wandb login)")
```

## Validation Script

Use the automated validation script:

```bash
python scripts/validate_project.py
```

**What it checks:**

- ✓ Project structure (directories & files)
- ✓ Config YAML syntax
- ✓ Config composition
- ✓ _target_ paths are importable
- ✓ Code quality (ruff)
- ✓ Dependencies installed
- ✓ GPU availability
- ✓ Model instantiation
- ✓ DataModule instantiation
- ✓ Fast dev run

**Example output:**

```text
INFO: Starting ML project validation...
INFO: ✓ Project structure valid
INFO: ✓ All configs valid
INFO: ✓ Code quality OK
INFO: ✓ All dependencies installed
INFO: ✓ Model instantiated successfully
INFO: ✓ DataModule instantiated successfully
INFO: ✓ Fast dev run completed
INFO: ✓ All validation checks passed!
```

See `scripts/validate_project.py` for implementation.

## Quick Checks

### One-line Validation

```bash
# Config only
python src/train.py --cfg job && echo "✓ Config OK"

# Full validation
python scripts/validate_project.py && echo "✓ All OK"
```

### Pre-Training Checklist

```bash
# 1. Structure
test -d src -a -d configs -a -f src/train.py && echo "✓ Structure"

# 2. Config
python src/train.py --cfg job && echo "✓ Config"

# 3. Dependencies
python -c "import torch, pytorch_lightning, hydra" && echo "✓ Deps"

# 4. GPU
python -c "import torch; assert torch.cuda.is_available()" && echo "✓ GPU"

# 5. Fast dev run
python src/train.py trainer.fast_dev_run=true && echo "✓ Training"
```

## CI/CD Integration

Add to `.github/workflows/validate.yml`:

```yaml
name: Validate ML Project

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Validate project
        run: uv run python scripts/validate_project.py

      - name: Test config
        run: uv run python src/train.py --cfg job

      - name: Fast dev run
        run: uv run python src/train.py trainer.fast_dev_run=true
```

## Common Issues

### "Config composition failed"

**Cause:** Typo in defaults or invalid YAML.

**Fix:**

```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/config.yaml'))"

# Check defaults exist
ls configs/model/ configs/data/ configs/trainer/
```

### "_target_ not found"

**Cause:** Module path incorrect or not installed.

**Fix:**

```bash
# Check import works
python -c "from src.models.my_model import MyModel"

# Verify path in config matches file structure
```

### "DataModule setup failed"

**Cause:** Data directory missing or incorrect path.

**Fix:**

```bash
# Check data path in config
grep data_dir configs/data/*.yaml

# Create data directory
mkdir -p data/
```

### "Fast dev run failed"

**Cause:** Various issues in training loop.

**Fix:**

```bash
# Run with verbose logging
python src/train.py trainer.fast_dev_run=true --verbose

# Check logs for specific error
```

## Success Criteria

- [ ] Project structure valid
- [ ] All YAML files valid
- [ ] Config composes without errors
- [ ] All _target_ paths importable
- [ ] Code passes linting
- [ ] Required deps installed
- [ ] GPU available (if needed)
- [ ] Model instantiates
- [ ] DataModule instantiates
- [ ] Fast dev run succeeds
- [ ] Logger configured

✅ Project is ready for training!
