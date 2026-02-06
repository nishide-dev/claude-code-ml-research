---
name: validate
description: Comprehensive validation of ML project structure, configurations, code quality, and training readiness (config loading, data pipeline, model architecture)
---

# ML Project Validation

Comprehensive validation of ML project structure, configurations, code quality, and training readiness.

## Overview

This command performs multiple validation checks to ensure your ML project is properly configured and ready for training. It checks:

- Project structure and required files
- Configuration files (Hydra, PyTorch Lightning)
- Code quality (linting, type hints, imports)
- Dependencies and environment
- Data pipeline
- Model architecture
- Training readiness

## Validation Checks

### 1. Project Structure Validation

**Required Directories:**

```bash
# Check standard ML project structure
test -d src || echo "❌ Missing src/ directory"
test -d configs || echo "❌ Missing configs/ directory"
test -d tests || echo "⚠️  Missing tests/ directory (recommended)"

# Check source structure
test -d src/models || echo "❌ Missing src/models/"
test -d src/data || echo "❌ Missing src/data/"
```

**Required Files:**

```bash
# Core files
test -f src/train.py || echo "❌ Missing src/train.py"
test -f configs/config.yaml || echo "❌ Missing configs/config.yaml"
test -f README.md || echo "⚠️  Missing README.md"

# Package manager
test -f pyproject.toml -o -f pixi.toml || echo "❌ No package manager config found"
```

### 2. Configuration Validation

**Hydra Config Syntax:**

```bash
# Validate YAML syntax
python -c "
import yaml
from pathlib import Path

config_dir = Path('configs')
if config_dir.exists():
    for yaml_file in config_dir.rglob('*.yaml'):
        try:
            with open(yaml_file) as f:
                yaml.safe_load(f)
            print(f'✓ {yaml_file} is valid')
        except yaml.YAMLError as e:
            print(f'❌ {yaml_file} has syntax error: {e}')
"
```

**Check Config Composition:**

```python
# Validate Hydra config can be composed
from hydra import compose, initialize_config_dir
from pathlib import Path
import sys

try:
    config_dir = Path.cwd() / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config")
    print("✓ Config composition successful")

    # Check required fields
    required_fields = ["model", "data", "trainer"]
    for field in required_fields:
        if field not in cfg:
            print(f"❌ Missing required config field: {field}")
        else:
            print(f"✓ Config has {field}")

except Exception as e:
    print(f"❌ Config composition failed: {e}")
    sys.exit(1)
```

**Validate _target_ Paths:**

```python
# Check that all _target_ paths are importable
import importlib
from omegaconf import OmegaConf

def check_target(config, path=""):
    """Recursively check _target_ fields."""
    if isinstance(config, dict):
        if "_target_" in config:
            target = config["_target_"]
            try:
                module_path, class_name = target.rsplit(".", 1)
                module = importlib.import_module(module_path)
                getattr(module, class_name)
                print(f"✓ {path}._target_ is valid: {target}")
            except Exception as e:
                print(f"❌ {path}._target_ is invalid: {target} ({e})")

        for key, value in config.items():
            check_target(value, f"{path}.{key}" if path else key)
    elif isinstance(config, list):
        for i, item in enumerate(config):
            check_target(item, f"{path}[{i}]")

cfg = OmegaConf.load("configs/config.yaml")
check_target(cfg)
```

### 3. Code Quality Validation

**Ruff Linting:**

```bash
# Run ruff checks
if command -v ruff &> /dev/null; then
    echo "Running ruff linting..."
    ruff check src/ tests/ || echo "❌ Linting errors found"
else
    echo "⚠️  Ruff not installed, skipping linting"
fi
```

**Type Checking:**

```bash
# Run mypy
if command -v mypy &> /dev/null; then
    echo "Running type checking..."
    mypy src/ --ignore-missing-imports || echo "⚠️  Type errors found"
else
    echo "⚠️  Mypy not installed, skipping type checking"
fi
```

**Import Validation:**

```python
# Check that all source files can be imported
import sys
from pathlib import Path
import ast

src_dir = Path("src")
errors = []

for py_file in src_dir.rglob("*.py"):
    if py_file.name == "__init__.py":
        continue

    try:
        # Parse file to check for syntax errors
        with open(py_file) as f:
            ast.parse(f.read())
        print(f"✓ {py_file} has valid syntax")
    except SyntaxError as e:
        errors.append(f"❌ {py_file}: {e}")

if errors:
    for error in errors:
        print(error)
    sys.exit(1)
```

### 4. Dependency Validation

**Check Required Packages:**

```python
# Verify ML dependencies are installed
required_packages = [
    "torch",
    "pytorch_lightning",
    "hydra",
]

optional_packages = [
    "wandb",
    "tensorboard",
    "torch_geometric",
]

import importlib

print("\nChecking required packages:")
for package in required_packages:
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {package} {version}")
    except ImportError:
        print(f"❌ {package} is not installed")

print("\nChecking optional packages:")
for package in optional_packages:
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {package} {version}")
    except ImportError:
        print(f"⚠️  {package} is not installed (optional)")
```

**CUDA Availability (if applicable):**

```python
import torch

print("\nGPU Status:")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("⚠️  No GPU detected (CPU training will be slow)")
```

### 5. Data Pipeline Validation

**Check Data Directory:**

```bash
# Verify data paths in config
python -c "
from omegaconf import OmegaConf
from pathlib import Path

cfg = OmegaConf.load('configs/config.yaml')
if 'data' in cfg and 'data_dir' in cfg.data:
    data_dir = Path(cfg.data.data_dir)
    if data_dir.exists():
        print(f'✓ Data directory exists: {data_dir}')
        print(f'  Files: {len(list(data_dir.rglob(\"*\")))}')
    else:
        print(f'⚠️  Data directory not found: {data_dir}')
else:
    print('⚠️  No data_dir in config')
"
```

**Test DataModule:**

```python
# Try to instantiate DataModule
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path

try:
    config_dir = Path.cwd() / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config")

    if "data" in cfg and "_target_" in cfg.data:
        print("Testing DataModule instantiation...")
        dm = instantiate(cfg.data)
        print(f"✓ DataModule created: {type(dm).__name__}")

        # Try setup
        try:
            dm.setup("fit")
            print("✓ DataModule.setup() successful")
        except Exception as e:
            print(f"⚠️  DataModule.setup() failed: {e}")
    else:
        print("⚠️  No data config with _target_")

except Exception as e:
    print(f"❌ DataModule instantiation failed: {e}")
```

### 6. Model Validation

**Test Model Instantiation:**

```python
# Try to create model
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path

try:
    config_dir = Path.cwd() / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config")

    if "model" in cfg and "_target_" in cfg.model:
        print("Testing Model instantiation...")
        model = instantiate(cfg.model)
        print(f"✓ Model created: {type(model).__name__}")

        # Count parameters
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
    else:
        print("⚠️  No model config with _target_")

except Exception as e:
    print(f"❌ Model instantiation failed: {e}")
```

**Test Forward Pass:**

```python
# Try a forward pass with dummy data
import torch

try:
    # Create dummy batch based on config
    batch_size = 2
    # Adjust based on your model's expected input
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

except Exception as e:
    print(f"❌ Forward pass failed: {e}")
```

### 7. Training Readiness

**Fast Dev Run:**

```bash
# Try a quick training run
echo "Running fast_dev_run..."
python src/train.py trainer.fast_dev_run=true trainer.max_epochs=1 2>&1 | tail -20

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Fast dev run completed successfully"
else
    echo "❌ Fast dev run failed"
fi
```

**Check Logging:**

```python
# Verify logger configuration
from hydra import compose, initialize_config_dir
from pathlib import Path

config_dir = Path.cwd() / "configs"
with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    cfg = compose(config_name="config")

if "logger" in cfg:
    print(f"✓ Logger configured: {cfg.logger.get('_target_', 'unknown')}")

    # Check W&B login if using wandb
    if "wandb" in str(cfg.logger.get("_target_", "")):
        import os
        if "WANDB_API_KEY" in os.environ or Path.home() / ".netrc" exists():
            print("✓ W&B credentials found")
        else:
            print("⚠️  W&B not logged in (run: wandb login)")
else:
    print("⚠️  No logger configured")
```

## Validation Script

Create a comprehensive validation script:

```python
#!/usr/bin/env python3
"""Comprehensive ML project validation."""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ProjectValidator:
    """Validate ML project structure and configuration."""

    def __init__(self, project_dir: Path = Path.cwd()):
        self.project_dir = project_dir
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> bool:
        """Run all validation checks."""
        logger.info("Starting ML project validation...")

        self.check_structure()
        self.check_configs()
        self.check_code_quality()
        self.check_dependencies()

        # Report results
        if self.warnings:
            logger.warning(f"Found {len(self.warnings)} warning(s)")
            for warning in self.warnings:
                logger.warning(f"  {warning}")

        if self.errors:
            logger.error(f"Found {len(self.errors)} error(s)")
            for error in self.errors:
                logger.error(f"  {error}")
            return False

        logger.info("✓ All validation checks passed!")
        return True

    def check_structure(self):
        """Check project structure."""
        required_dirs = ["src", "configs"]
        for dir_name in required_dirs:
            if not (self.project_dir / dir_name).exists():
                self.errors.append(f"Missing directory: {dir_name}/")

        required_files = ["src/train.py", "configs/config.yaml"]
        for file_path in required_files:
            if not (self.project_dir / file_path).exists():
                self.errors.append(f"Missing file: {file_path}")

    def check_configs(self):
        """Validate configuration files."""
        import yaml

        configs_dir = self.project_dir / "configs"
        if not configs_dir.exists():
            return

        for yaml_file in configs_dir.rglob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                self.errors.append(f"Invalid YAML in {yaml_file}: {e}")

    def check_code_quality(self):
        """Check code quality with ruff."""
        import subprocess

        try:
            result = subprocess.run(
                ["ruff", "check", "src/", "tests/"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                self.warnings.append("Ruff found code quality issues")
        except FileNotFoundError:
            self.warnings.append("Ruff not installed")

    def check_dependencies(self):
        """Check required dependencies."""
        required = ["torch", "pytorch_lightning", "hydra"]

        for package in required:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                self.errors.append(f"Required package not installed: {package}")


def main() -> int:
    """Run validation."""
    validator = ProjectValidator()
    success = validator.validate_all()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
```

## Usage

```bash
# Run validation
/ml:validate

# Or run the script directly
python scripts/validate_project.py

# Save validation report
/ml:validate > validation_report.txt
```

## Quick Validation (One-liner)

```bash
# Quick check
python src/train.py --cfg job && echo "✓ Config is valid" || echo "❌ Config is invalid"
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
          python-version: "3.10"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Validate project
        run: python scripts/validate_project.py

      - name: Test config composition
        run: python src/train.py --cfg job

      - name: Fast dev run
        run: python src/train.py trainer.fast_dev_run=true
```

## Success Criteria

- [ ] Project structure is valid
- [ ] All config files have valid YAML syntax
- [ ] Configs can be composed without errors
- [ ] All _target_ paths are importable
- [ ] Code passes linting checks
- [ ] Required dependencies are installed
- [ ] Model can be instantiated
- [ ] DataModule can be instantiated
- [ ] Fast dev run completes successfully
- [ ] Logger is configured

Your project is ready for training!
