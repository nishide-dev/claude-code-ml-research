---
name: tool-pixi
description: Comprehensive guide for Pixi package manager - Python environment management, CUDA/GPU support, PyPI integration, Docker/Pixi-Pack deployment, and best practices for ML research
---

# Pixi Package Manager for ML Research

## Overview

Pixi is a next-generation package manager designed to solve the "dependency hell" problem in machine learning projects. Built in Rust by the prefix.dev team, Pixi combines the best aspects of conda's binary package ecosystem with modern development workflows and strict reproducibility guarantees.

**Key Benefits for ML Research:**

- **Automatic CUDA management**: System requirements ensure GPU compatibility before installation
- **Fast dependency resolution**: Built on rattler (Rust-based conda resolver) - 10-100x faster than conda
- **Reproducible environments**: `pixi.lock` files guarantee exact environment replication
- **PyPI + conda integration**: Use conda for system libs, PyPI for Python packages via uv
- **Multi-environment support**: Dev, test, prod, CPU/GPU in one manifest
- **Built-in task runner**: No need for make/invoke
- **Cross-platform**: Works on Linux, macOS (including Apple Silicon), Windows

**When to use Pixi:**

- ✅ GPU/CUDA projects (PyTorch, JAX, TensorFlow)
- ✅ Projects needing both conda and PyPI packages
- ✅ Team projects requiring exact reproducibility
- ✅ Complex ML pipelines with multiple environments (CPU/GPU, dev/prod)
- ✅ HPC and air-gapped environments (via pixi-pack)

**Essential Resources:**

- Official docs: <https://pixi.prefix.dev/latest/>
- GitHub: <https://github.com/prefix-dev/pixi>
- PyTorch & CUDA guide: <https://pixi.prefix.dev/latest/python/pytorch/>
- System requirements: <https://pixi.prefix.dev/latest/features/system_requirements/>

---

## Installation

### Quick Install

```bash
# Linux/macOS
curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex

# Verify installation
pixi --version  # Should show v0.40.0+
```

### Shell Integration

Add to your shell config for autocompletion:

```bash
# Bash (~/.bashrc)
eval "$(pixi completion --shell bash)"

# Zsh (~/.zshrc)
eval "$(pixi completion --shell zsh)"

# Fish (~/.config/fish/config.fish)
pixi completion --shell fish | source
```

---

## Quick Start

### Initialize New Project

```bash
# Create new pixi project
pixi init my-ml-project
cd my-ml-project

# Or initialize in existing directory
pixi init
```

**Generated `pixi.toml`:**

```toml
[project]
name = "my-ml-project"
version = "0.1.0"
description = "Add a short description here"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64", "osx-64", "win-64"]

[tasks]

[dependencies]
```

### Add Dependencies

```bash
# Add conda package
pixi add python=3.11

# Add PyPI package
pixi add --pypi torch torchvision lightning

# Add with version constraint
pixi add "python>=3.10,<3.13"
pixi add --pypi "numpy>=1.24"
```

### Install and Run

```bash
# Install all dependencies (creates .pixi/ directory and pixi.lock)
pixi install

# Run commands in pixi environment
pixi run python --version

# Activate shell (like conda activate)
pixi shell
```

---

## Core Concepts

### 1. Project Manifest (pixi.toml)

Like Cargo.toml or pyproject.toml:

```toml
[project]
name = "deep-learning-project"
channels = ["conda-forge", "nvidia", "pytorch"]
platforms = ["linux-64"]

[dependencies]
python = "3.11.*"

[pypi-dependencies]
torch = ">=2.4"

[system-requirements]
cuda = "12.4"
```

### 2. Lock Files (pixi.lock)

Deterministic lock file contains:

- Exact package versions (including transitive dependencies)
- Build strings (e.g., `py311h06a4308_0`)
- SHA256 hashes for verification
- Platform-specific locks

Committing `pixi.lock` to Git ensures bit-identical environments across all machines.

### 3. Dependencies: Conda vs PyPI

**Key principle:** Use conda for **system libraries**, PyPI for **Python packages**.

**[dependencies] - Conda packages:**

- System libraries (CUDA, OpenMP, MKL)
- C/C++ libraries (OpenCV, FFmpeg)
- Python interpreter itself
- Packages with complex C dependencies

**[pypi-dependencies] - PyPI packages (resolved via uv):**

- Pure Python packages
- ML frameworks (PyTorch, TensorFlow - faster from PyPI)
- Latest package versions (PyPI updates faster than conda)
- Packages not on conda-forge

---

## PyTorch & CUDA Management

### Virtual Packages: Hardware Abstraction

Conda ecosystem has "virtual packages" representing system capabilities:

- `__cuda`: Maximum CUDA version supported by installed NVIDIA driver
- `__linux`: Linux kernel version
- `__glibc`: GNU C Library version
- `__osx`: macOS version

Pixi detects these at install time and validates compatibility before installing GPU libraries.

### System Requirements Definition

Define minimum system requirements in `pixi.toml`:

```toml
[system-requirements]
cuda = "12.4"  # Minimum CUDA version
libc = { family = "glibc", version = "2.28" }  # Minimum glibc version
linux = "5.10"  # Minimum kernel version
```

**Behavior:**

- Pixi selects packages compatible with these requirements
- If host machine doesn't meet requirements, `pixi run` fails with clear error
- System requirements specify **minimum** versions

### Strategy A: Conda Official Channels (Recommended)

Most stable approach:

```toml
[project]
channels = ["nvidia", "pytorch", "conda-forge"]  # Order matters: left has priority

[dependencies]
python = "3.11.*"

# GPU version: pytorch-cuda ensures GPU support
pytorch = { version = "2.4.1", channel = "pytorch" }
torchvision = { version = "*", channel = "pytorch" }
pytorch-cuda = { version = "12.4", channel = "pytorch" }

[system-requirements]
cuda = "12.4"
```

**Key points:**

- **`pytorch-cuda` package**: Explicitly specifies GPU version
- CUDA runtime libraries installed **inside** conda environment (no host contamination)
- Pixi automatically validates CUDA compatibility via virtual packages

### Strategy B: PyPI Packages

Use when you need specific nightly builds:

```toml
[pypi-dependencies]
torch = { version = "2.4.1+cu124", source = "url", url = "https://download.pytorch.org/whl/cu124/torch-2.4.1%2Bcu124-cp311-cp311-linux_x86_64.whl" }
```

### Platform-Specific CUDA Configuration

```toml
[project]
platforms = ["linux-64", "osx-arm64"]

# Linux: CUDA GPU
[target.linux-64.dependencies]
pytorch = { version = "2.4.1", channel = "pytorch" }
pytorch-cuda = { version = "12.4", channel = "pytorch" }

[target.linux-64.system-requirements]
cuda = "12.4"

# macOS: CPU/MPS
[target.osx-arm64.pypi-dependencies]
torch = ">=2.4"  # CPU/MPS support (no CUDA)
```

### Verify CUDA Setup

```bash
pixi run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
pixi run python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

---

## Multi-Environment & Features

Pixi's multi-environment feature (v0.13+) enables modular configurations for dev/test/prod and CPU/GPU switching.

### Complete Multi-Environment Configuration

```toml
[project]
name = "ml-research"
channels = ["nvidia", "pytorch", "conda-forge"]
platforms = ["linux-64", "osx-arm64"]

# Default feature (always included)
[dependencies]
python = "3.11.*"
git = ">=2.40"

[pypi-dependencies]
numpy = ">=1.24"
pandas = ">=2.0"
hydra-core = ">=1.3"

# Test feature
[feature.test.pypi-dependencies]
pytest = ">=7.4"
pytest-cov = ">=4.1"
ruff = ">=0.1"

[feature.test.tasks]
test = "pytest tests/ -v"
lint = "ruff check src/ tests/"

# CUDA feature (GPU training)
[feature.cuda.dependencies]
pytorch = { version = "2.4.1", channel = "pytorch" }
pytorch-cuda = { version = "12.4", channel = "pytorch" }

[feature.cuda.pypi-dependencies]
lightning = ">=2.0"

[feature.cuda.system-requirements]
cuda = "12.4"

[feature.cuda.tasks]
train-gpu = "python src/train.py trainer.accelerator=gpu"

# CPU feature (local development)
[feature.cpu.dependencies]
pytorch-cpu = { version = "2.4.1", channel = "pytorch" }

[feature.cpu.pypi-dependencies]
lightning = ">=2.0"

[feature.cpu.tasks]
train-cpu = "python src/train.py trainer.accelerator=cpu"

# Define environments (feature combinations)
[environments]
default = { features = ["cpu", "test"], solve-group = "default" }
prod-gpu = { features = ["cuda"], solve-group = "prod" }
prod-cpu = { features = ["cpu"], solve-group = "prod" }
ci = { features = ["test"], solve-group = "default" }
```

### Using Environments

```bash
# Install specific environment
pixi install --environment prod-gpu

# Run command in specific environment
pixi run --environment prod-gpu train-gpu

# Shell into environment
pixi shell --environment prod-gpu

# List all environments
pixi info
```

### Solve-Groups: Version Consistency

Environments in the same `solve-group` will use **identical versions** for shared dependencies (NumPy, Pandas, etc.), preventing GPU/CPU behavior divergence.

---

## Task Automation

### Basic Tasks

```toml
[tasks]
# Simple command
hello = "echo 'Hello from Pixi!'"

# Python script
train = "python src/train.py"

# With environment variables
train-gpu = { cmd = "python src/train.py", env = { CUDA_VISIBLE_DEVICES = "0" } }
```

**Run tasks:**

```bash
pixi run train
pixi run train-gpu
```

### Task Dependencies

```toml
[tasks]
lint = "ruff check src/"
test = "pytest tests/"
typecheck = "ty check src/"

# Run lint, test, typecheck in sequence
ci = { depends-on = ["lint", "test", "typecheck"] }
```

```bash
pixi run ci  # Runs all three tasks
```

### Complete Task Setup Example

```toml
[tasks]
# Training
train = "python src/train.py"
train-debug = "python src/train.py trainer.fast_dev_run=true"

# Testing
test = "pytest tests/ -v"
test-cov = "pytest tests/ --cov=src --cov-report=html"

# Code quality
lint = "ruff check src/ tests/"
format = "ruff format src/ tests/"
typecheck = "ty check src/"

# CI pipeline
ci = { depends-on = ["format", "lint", "typecheck", "test"] }

# Utilities
clean = "rm -rf .pixi __pycache__ .pytest_cache .ruff_cache"
```

---

## Production-Ready Configuration

### Complete ML Project pixi.toml

```toml
[project]
name = "ml-research"
version = "0.1.0"
description = "Production ML research environment"
channels = ["nvidia", "pytorch", "conda-forge"]
platforms = ["linux-64", "osx-arm64"]

# Base dependencies (all environments)
[dependencies]
python = "3.11.*"
git = ">=2.40"

[pypi-dependencies]
numpy = ">=1.24"
pandas = ">=2.0"
scikit-learn = ">=1.3"

# System requirements for production
[system-requirements]
libc = { family = "glibc", version = "2.28" }

# Development feature
[feature.dev.pypi-dependencies]
pytest = ">=7.4"
pytest-cov = ">=4.1"
ruff = ">=0.1"
ty = ">=0.2"
ipython = "*"
jupyter = "*"

[feature.dev.tasks]
test = "pytest tests/ -v"
test-cov = "pytest tests/ --cov=src --cov-report=html"
lint = "ruff check src/ tests/"
format = "ruff format src/ tests/"
typecheck = "ty check src/"
notebook = "jupyter lab"

# CUDA feature (GPU training)
[feature.cuda.dependencies]
pytorch = { version = "2.4.1", channel = "pytorch" }
pytorch-cuda = { version = "12.4", channel = "pytorch" }

[feature.cuda.pypi-dependencies]
lightning = ">=2.0"
torchvision = ">=0.19"
wandb = ">=0.15"
hydra-core = ">=1.3"

[feature.cuda.system-requirements]
cuda = "12.4"

[feature.cuda.tasks]
train = "python src/train.py"
train-debug = "python src/train.py trainer.fast_dev_run=true"

# CPU feature (local dev)
[feature.cpu.dependencies]
pytorch-cpu = { version = "2.4.1", channel = "pytorch" }

[feature.cpu.pypi-dependencies]
lightning = ">=2.0"
torchvision = ">=0.19"
wandb = ">=0.15"
hydra-core = ">=1.3"

[feature.cpu.tasks]
train = "python src/train.py trainer.accelerator=cpu"

# Environments
[environments]
default = { features = ["cpu", "dev"], solve-group = "dev" }
cuda-dev = { features = ["cuda", "dev"], solve-group = "dev" }
prod-gpu = { features = ["cuda"], solve-group = "prod" }
prod-cpu = { features = ["cpu"], solve-group = "prod" }
ci = { features = ["dev"], solve-group = "dev" }
```

---

## Common Workflows

### New ML Project Setup

```bash
# 1. Initialize
pixi init ml-project
cd ml-project

# 2. Add Python
pixi add python=3.11

# 3. Add ML frameworks
pixi add --pypi torch lightning wandb hydra-core

# 4. Add dev tools to feature
pixi add --pypi --feature dev pytest ruff ty

# 5. Edit pixi.toml to add [environments] and [system-requirements]

# 6. Install
pixi install

# 7. Commit
git init
git add pixi.toml pixi.lock .gitignore
git commit -m "Initial pixi setup"
```

### Cloning Existing Project

```bash
# Clone repository
git clone <repo-url>
cd <repo>

# Install from lock file (reproducible)
pixi install

# Run training
pixi run train
```

### Deploying to GPU Server

```bash
# On development machine
git push origin main

# On GPU server
git clone <repo-url>
cd <repo>
pixi install --environment prod-gpu
pixi run --environment prod-gpu train
```

---

## Troubleshooting

### Issue 1: CUDA Not Available

**Symptom:**

```python
import torch
torch.cuda.is_available()  # False
```

**Solution:**

```toml
# Ensure pytorch-cuda is specified
[dependencies]
pytorch = { version = "2.4.1", channel = "pytorch" }
pytorch-cuda = { version = "12.4", channel = "pytorch" }

[system-requirements]
cuda = "12.4"
```

### Issue 2: System Requirements Not Met

**Symptom:**

```text
Error: System requirements not satisfied
  cuda: required 12.4, found 11.8
```

**Solution:**

```toml
# Lower CUDA requirement
[system-requirements]
cuda = "11.8"

# And use matching pytorch-cuda
[dependencies]
pytorch-cuda = { version = "11.8", channel = "pytorch" }
```

### Issue 3: Cross-Platform Lock Generation Fails

**Solution (macOS → Linux):**

```bash
# Set override environment variable
export CONDA_OVERRIDE_CUDA="12.4"
pixi install
```

### Issue 4: Dependency Conflicts

**Solution:**

```toml
# Add explicit constraint
[pypi-dependencies]
numpy = ">=1.24,<2.0"  # Satisfy both requirements
```

### Issue 5: Glibc Incompatibility

**Symptom:**

```text
error: version 'GLIBC_2.35' not found
```

**Solution:**

```toml
# Lower glibc requirement for compatibility
[system-requirements]
libc = { family = "glibc", version = "2.28" }
```

---

## Best Practices

### 1. Lock File Discipline

```bash
# After adding dependencies
pixi install  # Regenerates pixi.lock
git add pixi.toml pixi.lock
git commit -m "Add PyTorch dependencies"

# Update dependencies periodically
pixi update
git diff pixi.lock  # Review changes
git add pixi.lock
git commit -m "Update dependencies"
```

### 2. Environment Strategy

```toml
[environments]
# Minimal for CI (no GPU)
ci = ["dev"]

# Local development (CPU)
default = ["cpu", "dev"]

# GPU development
cuda-dev = ["cuda", "dev"]

# Production GPU (no dev tools)
prod-gpu = ["cuda"]

# Production CPU (inference)
prod-cpu = ["cpu"]
```

### 3. Dependency Organization

```toml
[dependencies]
# Minimal: Only Python and system tools
python = ">=3.10,<3.13"
git = ">=2.40"

[pypi-dependencies]
# Core ML (group by purpose)
# ML frameworks
torch = ">=2.4"
lightning = ">=2.0"

# Data processing
numpy = ">=1.24"
pandas = ">=2.0"

# Experiment tracking
wandb = ">=0.15"

# Configuration
hydra-core = ">=1.3"

[feature.dev.pypi-dependencies]
# Development tools (separate feature)
pytest = "*"
ruff = "*"
ty = "*"
```

---

## Advanced Topics

For detailed guides on advanced features, see:

- **Docker Deployment**: [reference/advanced-topics.md](reference/advanced-topics.md#docker-deployment)
- **Pixi-Pack for HPC**: [reference/advanced-topics.md](reference/advanced-topics.md#pixi-pack-for-hpc--air-gapped-deployment)
- **System Requirements & Glibc**: [reference/advanced-topics.md](reference/advanced-topics.md#system-requirements--glibc-compatibility)
- **Migration Guides**: [reference/advanced-topics.md](reference/advanced-topics.md#migration-guides)

---

## Summary

Pixi represents a paradigm shift in ML environment management:

**Core Strengths:**

- **Deterministic reproducibility**: `pixi.lock` guarantees bit-identical environments
- **10-100x faster**: Rust-based solver eliminates conda's slowness
- **Unified ecosystem**: Conda + PyPI in one tool
- **Hardware-aware**: Virtual packages and system requirements prevent runtime failures
- **Flexible deployment**: Docker optimization + Pixi-Pack for HPC/air-gapped environments

**For ML Research:**

- Automatic CUDA/GPU management via system requirements
- Multi-environment CPU/GPU switching without separate projects
- Cross-platform development (macOS) → production (Linux GPU)
- Built-in task runner eliminates Makefile complexity
- Lock files solve "works on my machine" permanently

Pixi is not just a tool - it's a new standard for ML infrastructure that liberates engineers from "dependency hell" to focus on model development and experimentation.
