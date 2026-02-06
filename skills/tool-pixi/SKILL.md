---
name: tool-pixi
description: Comprehensive guide for Pixi package manager - Python environment management, CUDA/GPU support, PyPI integration, Docker/Pixi-Pack deployment, and best practices for ML research
---

# Pixi Package Manager for ML Research

## Overview & Philosophy

Pixi is a next-generation package manager designed to solve the "dependency hell" problem in machine learning projects. Built in Rust by the prefix.dev team, Pixi combines the best aspects of conda's binary package ecosystem with modern development workflows and strict reproducibility guarantees.

**Core Philosophy:**
- **Language-agnostic**: Manage Python, C++, CUDA, and system libraries in one tool
- **Complete reproducibility**: Deterministic lock files guarantee bit-identical environments
- **High-speed resolution**: Rust-based `rattler` solver is 10-100x faster than conda
- **Hardware-aware**: Virtual packages represent system capabilities (CUDA, glibc, kernel)

**Key Benefits for ML Research:**
- **Automatic CUDA management**: System requirements ensure GPU compatibility before installation
- **Fast dependency resolution**: Built on rattler (Rust-based conda resolver)
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

**Pixi vs Alternatives:**

| Feature | Pixi | Conda/Mamba | UV | Poetry |
|---------|------|-------------|-----|--------|
| Lock files | ✅ Built-in | ❌ No | ✅ Yes | ✅ Yes |
| PyPI integration | ✅ Via uv | ❌ Limited | ✅ Native | ✅ Native |
| Conda packages | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| Task runner | ✅ Built-in | ❌ No | ❌ No | ❌ No |
| Speed | ⚡ Fast | 🐌 Slow | ⚡⚡ Fastest | ⚡ Fast |
| CUDA support | ✅ System-req | ✅ Manual | ❌ Via PyPI | ❌ Via PyPI |
| Multi-env | ✅ Yes | ✅ Yes | ❌ No | ❌ No |

**Essential Resources:**
- Official docs: https://pixi.prefix.dev/latest/
- GitHub: https://github.com/prefix-dev/pixi
- PyTorch & CUDA guide: https://pixi.prefix.dev/latest/python/pytorch/
- System requirements: https://pixi.prefix.dev/latest/features/system_requirements/

---

## Architecture & Core Concepts

### 1. Conda Ecosystem Integration

Pixi uses the conda ecosystem (primarily conda-forge channels) as its package source. This allows unified management of:
- Python libraries (NumPy, Pandas)
- C/C++ libraries (OpenCV, Boost, FFmpeg)
- CUDA toolkit and cuDNN
- System tools (Git, CMake, compilers)
- Other languages (R, Rust, Julia)

Unlike traditional `conda`, Pixi is built in Rust from scratch, with optimized SAT solver algorithms and parallel network I/O, achieving 10-100x faster dependency resolution.

### 2. Project Manifest and Lock Files

**pixi.toml** - Project manifest (like Cargo.toml or pyproject.toml):
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

**pixi.lock** - Deterministic lock file:
- Exact package versions (including transitive dependencies)
- Build strings (e.g., `py311h06a4308_0`)
- SHA256 hashes for verification
- Platform-specific locks

Committing `pixi.lock` to Git ensures bit-identical environments across all machines, solving "works on my machine" problems.

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

**Resolution order:**
1. Conda dependencies are locked first
2. PyPI dependencies are resolved with conda packages as constraints
3. Both are written to `pixi.lock`

### 4. Cross-Platform Resolution

Pixi can generate lock files for multiple platforms simultaneously, regardless of the host OS. For example, a macOS developer can generate Linux lock files locally, then push them to Git for deployment on Linux servers.

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

## Python Version Management

### Specifying Python Versions

**Single version:**
```toml
[dependencies]
python = "3.11.*"  # Any 3.11.x
```

**Version range:**
```toml
[dependencies]
python = ">=3.10,<3.13"  # 3.10, 3.11, or 3.12
```

**Platform-specific Python:**
```toml
[target.osx-arm64.dependencies]
python = "3.11.*"  # Force 3.11 on Apple Silicon

[target.linux-64.dependencies]
python = ">=3.10,<3.13"  # Flexible on Linux
```

### Python Version Check

```bash
# Check resolved Python version
pixi run python --version

# List all packages
pixi list
```

---

## PyTorch & CUDA Management

This is the most complex aspect of ML environment setup. Pixi provides systematic solutions through virtual packages and system requirements.

### Virtual Packages: Hardware Abstraction

Conda ecosystem has "virtual packages" - packages that represent system capabilities rather than installed files:

- `__cuda`: Maximum CUDA version supported by installed NVIDIA driver
- `__linux`: Linux kernel version
- `__glibc`: GNU C Library version
- `__osx`: macOS version

Pixi detects these virtual packages at install time (equivalent to `nvidia-smi` check) and validates compatibility before installing GPU libraries.

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
- If the host machine doesn't meet requirements, `pixi run` will fail with a clear error message
- System requirements specify **minimum** versions (CUDA 12.1, 12.6 would both satisfy `cuda = "12"`)

### Strategy A: Conda Official Channels (Recommended)

The most stable approach that fully leverages Pixi's capabilities.

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
- CUDA runtime libraries are installed **inside** the conda environment (no host contamination)
- Pixi automatically validates CUDA compatibility via virtual packages

**Advantages:**
- Self-contained CUDA runtime (no host dependency)
- Automatic compatibility validation
- Works well with other CUDA libraries (RAPIDS, JAX)

### Strategy B: PyPI Packages

Use when you need specific nightly builds or packages not on conda-forge.

```toml
[pypi-dependencies]
torch = { version = "2.4.1+cu124", source = "url", url = "https://download.pytorch.org/whl/cu124/torch-2.4.1%2Bcu124-cp311-cp311-linux_x86_64.whl" }
```

**Trade-offs:**
- PyPI wheels bundle CUDA runtime (larger package size)
- Harder for Pixi to resolve CUDA version conflicts with other libraries
- Use this only when conda channels don't have what you need

### Cross-Platform CUDA Resolution Challenge

**Problem:** On macOS (no `__cuda` virtual package), Pixi cannot resolve Linux CUDA dependencies by default.

**Solution 1: Environment Variable Override**
```bash
export CONDA_OVERRIDE_CUDA="12.4"
pixi install
```

This tells Pixi to pretend CUDA 12.4 is available, forcing the solver to proceed.

**Solution 2: CI/CD Lock Generation (Recommended)**
Generate `pixi.lock` on Linux environment or in Docker/GitHub Actions:

```yaml
# .github/workflows/lock.yml
name: Update Lock File
on: [push]
jobs:
  lock:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:12.4.0-base-ubuntu22.04
    steps:
      - uses: actions/checkout@v3
      - name: Install Pixi
        run: curl -fsSL https://pixi.sh/install.sh | bash
      - name: Generate Lock
        run: pixi install
      - name: Commit Lock
        run: |
          git add pixi.lock
          git commit -m "Update pixi.lock" || true
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

Pixi's multi-environment feature (v0.13+) enables modular configurations for dev/test/prod and CPU/GPU switching in a single manifest.

### Features and Environments Concept

**Features** group dependencies and tasks. **Environments** combine features.

**Example Structure:**
- **Default feature**: Common libraries (Python, NumPy, Pandas)
- **Test feature**: Testing tools (Pytest, Ruff)
- **CUDA feature**: GPU libraries (PyTorch-CUDA, CuPy)
- **CPU feature**: CPU libraries (PyTorch-CPU)

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

### Solve-Groups: Version Consistency Across Environments

**Problem:** Different environments might resolve common dependencies (e.g., NumPy) to different versions, causing inconsistent behavior.

**Solution:** Solve-groups force consistent versions across environments.

```toml
[environments]
prod-gpu = { features = ["cuda"], solve-group = "prod" }
prod-cpu = { features = ["cpu"], solve-group = "prod" }
```

Environments in the same `solve-group` will use **identical versions** for shared dependencies (NumPy, Pandas, etc.), preventing GPU/CPU behavior divergence.

### Practical Workflow

1. **Local Development (macOS)**: Use `default` environment with CPU PyTorch
2. **CI Pipeline**: Use `ci` environment for testing and linting
3. **Remote Training (Linux GPU server)**:
   ```bash
   git clone <repo>
   cd <repo>
   pixi run --environment prod-gpu train-gpu
   ```
   Pixi automatically downloads CUDA libraries and starts training.

---

## Task Automation

Tasks replace Makefiles and shell scripts with a unified task runner.

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

### Environment-Specific Tasks

```toml
[feature.cuda.tasks]
train = "python src/train.py trainer.accelerator=gpu"
profile = "nsys profile python src/train.py"

[feature.cpu.tasks]
train = "python src/train.py trainer.accelerator=cpu"
```

```bash
pixi run --environment prod-gpu train  # Uses GPU version
pixi run --environment prod-cpu train  # Uses CPU version
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

## Platform-Specific Configuration

### Platform Selectors

**Available platforms:**
- `linux-64` - Linux x86_64
- `linux-aarch64` - Linux ARM64
- `osx-64` - macOS Intel
- `osx-arm64` - macOS Apple Silicon
- `win-64` - Windows x86_64

### Platform-Specific Dependencies

```toml
[project]
platforms = ["linux-64", "osx-arm64"]

# Linux-specific (NVIDIA GPU)
[target.linux-64.dependencies]
pytorch = { version = "2.4.1", channel = "pytorch" }
pytorch-cuda = { version = "12.4", channel = "pytorch" }

# macOS Apple Silicon (MPS)
[target.osx-arm64.dependencies]
pytorch = { version = "2.4.1", channel = "pytorch" }  # CPU/MPS
```

### Platform-Specific Tasks

```toml
# GPU tasks only on Linux
[target.linux-64.tasks]
train-gpu = "python src/train.py trainer.accelerator=gpu"
profile = "nsys profile python src/train.py"

# MPS tasks on Apple Silicon
[target.osx-arm64.tasks]
train-mps = "python src/train.py trainer.accelerator=mps"
```

---

## Lock Files & Reproducibility

### pixi.lock

Pixi automatically generates `pixi.lock` when you run `pixi install`.

**What it contains:**
- Exact package versions (including transitive dependencies)
- Build strings (e.g., `py311h06a4308_0`)
- SHA256 hashes for verification
- Source URLs
- Platform-specific locks

**Best practices:**
```bash
# Always commit pixi.lock
git add pixi.lock
git commit -m "Lock dependencies"

# Update dependencies
pixi update  # Updates pixi.lock

# Install from lock (reproducible)
pixi install  # Uses pixi.lock if present
```

**Lock file guarantees:**
- ✅ Exact same packages on all machines
- ✅ Exact same versions
- ✅ Cryptographic verification via hashes
- ✅ No "works on my machine" issues

---

## PyPI Integration Deep Dive

### PyPI Dependency Syntax

**Basic:**
```toml
[pypi-dependencies]
package-name = "*"  # Latest version
package-name = "==1.0.0"  # Exact version
package-name = ">=1.0,<2.0"  # Version range
```

**With extras:**
```toml
[pypi-dependencies]
fastapi = { version = "*", extras = ["all"] }
pandas = { version = ">=2.0", extras = ["sql", "excel"] }
```

**From Git:**
```toml
[pypi-dependencies]
my-package = { git = "https://github.com/user/repo.git", branch = "main" }
another = { git = "https://github.com/user/repo.git", tag = "v1.0.0" }
experimental = { git = "https://github.com/user/repo.git", rev = "abc123" }
```

**From local path (editable install):**
```toml
[pypi-dependencies]
my-lib = { path = "../my-lib", editable = true }
```

### When to Use PyPI vs Conda

**Use PyPI for:**
- ✅ Pure Python packages
- ✅ Latest versions (PyPI updates faster)
- ✅ ML frameworks (PyTorch, TensorFlow, JAX)
- ✅ Python development tools (ruff, pytest, mypy)
- ✅ Packages not on conda-forge

**Use Conda for:**
- ✅ Python interpreter itself
- ✅ System libraries (CUDA, MKL, OpenMP)
- ✅ C/C++ dependencies (OpenCV, FFmpeg)
- ✅ Packages with complex binary dependencies
- ✅ Cross-platform consistency

---

## Deployment Strategy I: Docker

Pixi optimizes Docker image construction for ML workloads, reducing build time and image size.

### Pixi-Based Docker Advantages

- **Fast builds**: Parallel downloads and fast solving reduce `docker build` time
- **Strict reproducibility**: `pixi.lock` ensures container matches development environment
- **Easy multi-stage builds**: Extract only necessary environments

### Multi-Stage Build for Size Optimization

QuantCo case study achieved 691MB → 209MB reduction with proper optimization.

**Optimized Dockerfile:**
```dockerfile
# Stage 1: Build Environment
FROM ghcr.io/prefix-dev/pixi:latest AS build
WORKDIR /app

# Copy manifest and lock file
COPY pixi.toml pixi.lock ./

# Install production environment only and remove cache
RUN pixi install --locked --environment prod && \
    rm -rf /root/.cache/rattler

# Stage 2: Runtime Environment
FROM ubuntu:22.04 AS production
WORKDIR /app

# Copy installed environment from build stage
# IMPORTANT: Keep absolute path identical (/app/.pixi/envs/prod)
COPY --from=build /app/.pixi/envs/prod /app/.pixi/envs/prod

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Set PATH to use pixi environment
ENV PATH="/app/.pixi/envs/prod/bin:$PATH"
ENV PYTHONPATH="/app"

# Run application
CMD ["python", "src/app.py"]
```

**Key points:**
- Conda environments are **not relocatable** - keep absolute paths identical between stages
- Remove cache (`rm -rf /root/.cache/rattler`) to reduce layer size
- Use `--locked` to ensure reproducibility

### Cache Management

```dockerfile
# Bad: Cache included in final image
RUN pixi install --environment prod

# Good: Cache removed in same layer
RUN pixi install --environment prod && \
    rm -rf /root/.cache/rattler
```

### Distroless Images (Advanced)

For ultimate minimization, use Google's distroless images (no shell, minimal OS):

```dockerfile
FROM gcr.io/distroless/python3-debian11
COPY --from=build /app/.pixi/envs/prod /app/.pixi/envs/prod
COPY src/ ./src/
ENV PATH="/app/.pixi/envs/prod/bin:$PATH"
CMD ["/app/.pixi/envs/prod/bin/python", "src/app.py"]
```

---

## Deployment Strategy II: Pixi-Pack

Pixi-Pack is a revolutionary distribution mechanism for environments where Docker is unavailable (HPC, air-gapped, Windows servers).

### Pixi-Pack Concept

**Docker** distributes entire filesystem (including OS). **Pixi-Pack** distributes package repository + construction recipe, then reconstructs environment on target machine.

**Archive contents:**
1. Binary packages (`.conda` or `.tar.bz2` format)
2. Metadata (`repodata.json`, `environment.yml`)
3. Pixi-pack configuration

**Key difference:** Docker requires Docker Engine (root access). Pixi-Pack runs with user permissions and doesn't need Pixi/Conda on target machine.

### Build & Deliver Process

#### 1. Pack (Build Archive)

```bash
# Install pixi-pack
pixi global install pixi-pack

# Create archive for specific platform
pixi-pack pack --manifest-file pixi.toml \
  --environment prod \
  --platform linux-64

# Output: environment.tar (self-contained, offline-capable)
```

#### 2. Transfer

```bash
# Copy to target machine (USB, scp, etc.)
scp environment.tar user@target-server:/path/to/
```

#### 3. Unpack (Deploy)

```bash
# On target machine (no Pixi/Conda needed!)
pixi-pack unpack environment.tar

# Creates ./env/ directory with activate scripts
```

#### 4. Execute

```bash
# Activate environment
source env/activate.sh

# Run application
python inference.py
```

### Inject Application Code

Pixi-Pack can bundle your application code as a conda package:

```bash
# Build app as conda package with rattler-build
rattler-build build --recipe recipe.yaml

# Inject into pack
pixi-pack pack --inject my-model-pkg-1.0.0.conda --environment prod

# After unpack, app is in PATH
source env/activate.sh
my-model-cli --input data.csv  # Application ready to use
```

### HPC & Air-Gapped Advantages

**Why Pixi-Pack excels in these environments:**
- **User permissions**: No root required (unlike Docker)
- **Fully offline**: Archive contains everything
- **No container overhead**: Direct execution on host
- **Singularity-compatible**: Can be base for HPC container formats

**Typical HPC workflow:**
```bash
# On laptop (with internet)
pixi-pack pack --environment gpu --platform linux-64

# Transfer to HPC login node
scp environment.tar hpc-cluster:/scratch/user/

# On HPC compute node (no internet)
cd /scratch/user/
pixi-pack unpack environment.tar
source env/activate.sh
sbatch train.slurm  # Submit GPU job
```

---

## System Requirements & Glibc Compatibility

The most overlooked yet critical issue in ML deployment is `glibc` (GNU C Library) version incompatibility.

### Binary Compatibility Problem

Linux programs (Python, PyTorch) dynamically link to system `glibc`. If you build/resolve on Ubuntu 22.04 (glibc 2.35) and run on CentOS 7 (glibc 2.17), you'll get:

```
error: version 'GLIBC_2.xx' not found
```

Docker avoids this by bundling the entire OS. Pixi-Pack requires explicit compatibility management.

### System Requirements for Compatibility

Define minimum system requirements to control binary selection:

```toml
[system-requirements]
linux = "4.18"
libc = { family = "glibc", version = "2.28" }
cuda = "12.1"
```

**Behavior:**
- Pixi solver selects binaries guaranteed to run on glibc 2.28+
- If target machine has glibc 2.17, `pixi run`/`unpack` fails with clear error (preventing mysterious runtime crashes)

### Supporting Legacy Systems

For old environments (legacy corporate clusters):

```toml
[system-requirements]
libc = { family = "glibc", version = "2.17" }  # Old but compatible
```

Pixi will find packages that work on ancient systems while maintaining modern features.

### Virtual Package Inspection

```bash
# Check system's virtual packages
conda info --all  # Shows __cuda, __glibc, etc.

# Or use Python
pixi run python -c "import platform; print(platform.libc_ver())"
```

---

## Complete ML Project Configuration

### Production-Ready pixi.toml

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
train-sweep = "wandb sweep configs/sweep.yaml"

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

## Best Practices for ML Projects

### 1. Dependency Organization

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

### 2. Lock File Discipline

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

### 3. Environment Strategy

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

### 4. Task Organization

```toml
[tasks]
# Training tasks
train = "python src/train.py"
train-debug = "python src/train.py trainer.fast_dev_run=true"

# Testing tasks
test = "pytest tests/ -v"
test-cov = "pytest tests/ --cov=src"

# Code quality tasks
lint = "ruff check src/"
format = "ruff format src/"
typecheck = "ty check src/"

# CI pipeline
ci = { depends-on = ["format", "lint", "typecheck", "test"] }
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

### Updating Dependencies

```bash
# Update specific package
pixi update torch

# Update all packages
pixi update

# Review changes
git diff pixi.lock

# Commit
git add pixi.lock
git commit -m "Update dependencies"
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

### Creating Offline Package for HPC

```bash
# On machine with internet
pixi global install pixi-pack
pixi-pack pack --environment prod-gpu --platform linux-64

# Transfer environment.tar to HPC
scp environment.tar hpc:/scratch/user/

# On HPC (no internet)
cd /scratch/user/
pixi-pack unpack environment.tar
source env/activate.sh
python train.py
```

---

## Troubleshooting

### Issue 1: CUDA Not Available

**Symptom:**
```python
import torch
torch.cuda.is_available()  # False
```

**Diagnosis:**
```bash
# Check PyTorch version
pixi run python -c "import torch; print(torch.__version__)"

# Check if CUDA version is in build string
# Should see: 2.4.1+cu124
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
```
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

**Symptom:**
```
Error: Could not resolve dependencies for linux-64
```

**Solution (macOS → Linux):**
```bash
# Set override environment variable
export CONDA_OVERRIDE_CUDA="12.4"
pixi install

# Or generate lock in CI (preferred)
```

### Issue 4: Dependency Conflicts

**Symptom:**
```
Error: Cannot resolve dependencies
  torch requires numpy<2.0
  package-x requires numpy>=2.0
```

**Solution:**
```toml
# Add explicit constraint
[pypi-dependencies]
numpy = ">=1.24,<2.0"  # Satisfy both
```

### Issue 5: Slow Installation

**Diagnosis:**
```bash
# Check if conda-forge is source
pixi list  # Look at package channels
```

**Solution:**
```toml
# Prioritize faster channels
[project]
channels = ["pytorch", "nvidia", "conda-forge"]  # pytorch first
```

### Issue 6: Glibc Incompatibility

**Symptom:**
```
error: version 'GLIBC_2.35' not found
```

**Solution:**
```toml
# Lower glibc requirement for compatibility
[system-requirements]
libc = { family = "glibc", version = "2.28" }
```

---

## Migration Guides

### From Conda/Mamba

**conda environment.yml:**
```yaml
name: ml-env
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.11
  - pytorch::pytorch
  - numpy
```

**Equivalent pixi.toml:**
```toml
[project]
name = "ml-env"
channels = ["pytorch", "conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = "3.11.*"
pytorch = { channel = "pytorch" }

[pypi-dependencies]
numpy = "*"
```

**Migration steps:**
```bash
# 1. Create pixi.toml
pixi init

# 2. Add channels
# Edit pixi.toml [project.channels]

# 3. Add dependencies
pixi add python=3.11
pixi add pytorch --channel pytorch
pixi add --pypi numpy

# 4. Install
pixi install

# 5. Test
pixi run python -c "import torch; print(torch.__version__)"
```

### From UV/Poetry

**pyproject.toml (Poetry):**
```toml
[tool.poetry.dependencies]
python = "^3.11"
torch = "^2.4"
numpy = "^1.24"
```

**Equivalent pixi.toml:**
```toml
[dependencies]
python = ">=3.11,<3.12"

[pypi-dependencies]
torch = ">=2.4,<3.0"
numpy = ">=1.24,<2.0"
```

**Key differences:**
- Poetry `^` (caret) → Pixi `>=x.y,<x+1.0`
- Poetry only manages Python packages → Pixi manages system dependencies too

---

## Advanced Topics

### Custom Conda Channels

```toml
[project]
channels = [
  "https://my-company.com/conda",  # Custom channel
  "pytorch",
  "conda-forge"
]
```

### Environment Variables

```toml
[tasks.train]
cmd = "python src/train.py"
env = {
  CUDA_VISIBLE_DEVICES = "0,1",
  OMP_NUM_THREADS = "8",
  WANDB_PROJECT = "my-project"
}
```

### Activation Scripts

```toml
[activation]
scripts = ["scripts/setup_env.sh"]
```

```bash
# scripts/setup_env.sh
export TORCH_HOME=/scratch/torch_cache
export HF_HOME=/scratch/huggingface_cache
```

---

## Essential Resources

### Official Documentation
- **Pixi Documentation**: https://pixi.prefix.dev/latest/
- **PyTorch & CUDA Setup**: https://pixi.prefix.dev/latest/python/pytorch/
- **System Requirements**: https://pixi.prefix.dev/latest/features/system_requirements/
- **Pixi-Pack Documentation**: https://pixi.prefix.dev/latest/deployment/pixi_pack/
- **Manifest Reference**: https://pixi.prefix.dev/latest/reference/pixi_manifest/

### Case Studies & Blog Posts
- **QuantCo: Pixi in Production**: https://tech.quantco.com/blog/pixi-production
  (Docker optimization, Pixi-Pack details)
- **Prefix.dev: Multi-Environment Support**: https://prefix.dev/blog/introducing_multi_env_pixi
  (CPU/GPU hybrid environments)

### Templates & Examples
- **Simple PyTorch Project**: https://hpc.nmsu.edu/discovery/software/pixi/python-example/
- **Marimo & Pixi Template**: https://github.com/marimo-team/marimo-pixi-starter-template
  (Modern notebook environment)

### Community
- **GitHub**: https://github.com/prefix-dev/pixi
- **Discord**: https://discord.gg/kKV8ZxyzY4
- **Discussions**: https://github.com/prefix-dev/pixi/discussions

### Related Tools
- **Rattler**: https://github.com/mamba-org/rattler (Rust conda implementation)
- **UV**: https://github.com/astral-sh/uv (Fast Python package installer)
- **Prefix.dev**: https://prefix.dev/ (Commercial conda hosting)

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

**When to Choose Pixi:**
- ✅ Projects requiring GPU/CUDA (PyTorch, JAX, TensorFlow)
- ✅ Teams needing strict reproducibility
- ✅ Mixed conda + PyPI dependencies
- ✅ HPC and air-gapped deployments
- ✅ Multi-environment workflows (dev/prod, CPU/GPU)

Pixi is not just a tool - it's a new standard for ML infrastructure that liberates engineers from "dependency hell" to focus on model development and experimentation.
