---
description: Setup development environment with modern Python tooling (uv/pixi), install dependencies, and configure development tools (ruff, ty, pytest)
---

# ML Environment Setup

Setup development environment with modern Python tooling (uv/pixi), install dependencies, and configure development tools.

## Process

### 1. Detect Project State

First, check what's already configured:

```bash
# Check for existing package managers
ls pyproject.toml 2>/dev/null && echo "Found pyproject.toml (uv/pip)"
ls pixi.toml 2>/dev/null && echo "Found pixi.toml (pixi)"
ls requirements.txt 2>/dev/null && echo "Found requirements.txt (pip)"
```

### 2. Ask User Preferences

If no package manager is configured, ask the user:

**Package Manager Choice:**

- **uv** (recommended for pure Python projects)
  - Fast dependency resolution
  - Compatible with pip/PyPI
  - Good for projects without CUDA requirements

- **pixi** (recommended for ML projects with GPU)
  - Conda-based, handles CUDA/cuDNN automatically
  - Better for complex ML dependencies (PyTorch, TensorFlow)
  - Cross-platform reproducibility

- **pip** (traditional, not recommended for new projects)
  - Slower than uv
  - Manual CUDA setup required

### 3. Install Package Manager (if needed)

**For uv:**

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

**For pixi:**

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | sh

# Verify installation
pixi --version
```

### 4. Initialize Project Dependencies

**With uv:**

```bash
# Initialize pyproject.toml if not exists
uv init --name ml-project

# Add ML dependencies
uv add torch pytorch-lightning hydra-core
uv add --dev pytest pytest-cov ruff mypy

# Create virtual environment and install
uv sync
```

**With pixi:**

```bash
# Initialize pixi.toml if not exists
pixi init

# Add ML dependencies with CUDA
pixi add pytorch pytorch-cuda=12.1 pytorch-lightning hydra-core
pixi add --feature dev pytest ruff mypy

# Install environment
pixi install
```

### 5. Configure Development Tools

**Setup ruff (linting and formatting):**

```bash
# Create ruff.toml if not exists
cat > ruff.toml << 'EOF'
line-length = 100
target-version = "py310"

[lint]
select = ["E", "F", "I", "N", "UP", "ANN", "B", "LOG", "G"]
ignore = ["ANN101", "ANN102"]

[lint.per-file-ignores]
"__init__.py" = ["F401"]
"tests/*" = ["ANN", "S101"]
EOF
```

**Setup mypy (type checking):**

```bash
# Add mypy config to pyproject.toml
cat >> pyproject.toml << 'EOF'

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true
EOF
```

**Setup pytest:**

```bash
# Add pytest config to pyproject.toml
cat >> pyproject.toml << 'EOF'

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = ["--cov=src", "--cov-report=html"]
EOF
```

### 6. Setup Pre-commit Hooks (Optional but Recommended)

```bash
# Install pre-commit
uv add --dev pre-commit  # or: pixi add --feature dev pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
EOF

# Install hooks
pre-commit install
```

### 7. Create Project Structure (if new project)

```bash
# Create standard ML project directories
mkdir -p src/{models,data,utils}
mkdir -p tests
mkdir -p configs/{model,data,trainer,logger,experiment}
mkdir -p notebooks
mkdir -p scripts

# Create __init__.py files
touch src/__init__.py
touch src/models/__init__.py
touch src/data/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
```

### 8. Validation

Run validation checks to ensure everything is setup correctly:

```bash
# Check package manager
if command -v uv &> /dev/null; then
    echo "✓ uv is installed"
    uv --version
elif command -v pixi &> /dev/null; then
    echo "✓ pixi is installed"
    pixi --version
fi

# Check ruff
uv run ruff check . || pixi run ruff check .
echo "✓ Ruff is configured"

# Check pytest
uv run pytest --collect-only || pixi run pytest --collect-only
echo "✓ Pytest is configured"

# Check Python version
python --version
echo "✓ Python environment is active"
```

### 9. Generate Documentation

Create README.md with setup instructions:

```markdown
# ML Project

## Setup

### Prerequisites
- Python 3.10+
- [uv](https://astral.sh/uv) or [pixi](https://pixi.sh)

### Installation

**With uv:**
\`\`\`bash
uv sync
\`\`\`

**With pixi:**
\`\`\`bash
pixi install
\`\`\`

### Development

\`\`\`bash
# Run tests
uv run pytest  # or: pixi run pytest

# Lint code
uv run ruff check .  # or: pixi run ruff check .

# Format code
uv run ruff format .  # or: pixi run ruff format .

# Type check
uv run mypy src/  # or: pixi run mypy src/
\`\`\`

### Training

\`\`\`bash
# Run training
uv run python src/train.py  # or: pixi run python src/train.py
\`\`\`
```

## Environment-Specific Notes

### CUDA/GPU Setup

**With pixi (automatic):**

```bash
pixi add pytorch pytorch-cuda=12.1
# CUDA toolkit and drivers are handled automatically
```

**With uv (manual):**

```bash
# Install PyTorch with CUDA
uv add torch --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### macOS (Apple Silicon)

```bash
# With uv
uv add torch  # MPS support included by default

# With pixi
pixi add pytorch
```

### Windows

```bash
# With uv (use PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
uv add torch --index-url https://download.pytorch.org/whl/cu121

# With pixi
iwr -useb https://pixi.sh/install.ps1 | iex
pixi add pytorch pytorch-cuda=12.1
```

## Troubleshooting

### Issue: "command not found: uv"

**Solution:** Add uv to PATH

```bash
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

### Issue: "CUDA not available"

**Solution:** Verify CUDA installation

```bash
nvidia-smi  # Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

If using uv, install correct CUDA version:

```bash
uv add torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "Permission denied" during installation

**Solution:** Run without sudo (install to user directory)

```bash
# Don't use sudo with uv/pixi installers
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Success Criteria

- [ ] Package manager installed (uv or pixi)
- [ ] Development dependencies installed
- [ ] Ruff, mypy, pytest configured
- [ ] Pre-commit hooks setup (optional)
- [ ] Project structure created
- [ ] All validation checks pass
- [ ] README.md generated

Your ML development environment is ready!
