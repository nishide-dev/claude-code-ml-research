# Pixi Advanced Topics

Detailed guides for advanced Pixi features and deployment strategies.

## Docker Deployment

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

### Distroless Images

For ultimate minimization, use Google's distroless images (no shell, minimal OS):

```dockerfile
FROM gcr.io/distroless/python3-debian11
COPY --from=build /app/.pixi/envs/prod /app/.pixi/envs/prod
COPY src/ ./src/
ENV PATH="/app/.pixi/envs/prod/bin:$PATH"
CMD ["/app/.pixi/envs/prod/bin/python", "src/app.py"]
```

---

## Pixi-Pack for HPC & Air-Gapped Deployment

Pixi-Pack is a revolutionary distribution mechanism for environments where Docker is unavailable (HPC, air-gapped, Windows servers).

### Concept

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

```text
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
- If target machine doesn't meet requirements, `pixi run`/`unpack` fails with clear error (preventing mysterious runtime crashes)

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

## Custom Conda Channels

```toml
[project]
channels = [
  "https://my-company.com/conda",  # Custom channel
  "pytorch",
  "conda-forge"
]
```

---

## Environment Variables in Tasks

```toml
[tasks.train]
cmd = "python src/train.py"
env = {
  CUDA_VISIBLE_DEVICES = "0,1",
  OMP_NUM_THREADS = "8",
  WANDB_PROJECT = "my-project"
}
```

---

## Activation Scripts

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

## Essential Resources

### Official Documentation

- **Pixi Documentation**: <https://pixi.prefix.dev/latest/>
- **Pixi-Pack Documentation**: <https://pixi.prefix.dev/latest/deployment/pixi_pack/>
- **Manifest Reference**: <https://pixi.prefix.dev/latest/reference/pixi_manifest/>

### Case Studies & Blog Posts

- **QuantCo: Pixi in Production**: <https://tech.quantco.com/blog/pixi-production>
  (Docker optimization, Pixi-Pack details)
- **Prefix.dev: Multi-Environment Support**: <https://prefix.dev/blog/introducing_multi_env_pixi>
  (CPU/GPU hybrid environments)

### Community

- **GitHub**: <https://github.com/prefix-dev/pixi>
- **Discord**: <https://discord.gg/kKV8ZxyzY4>
- **Discussions**: <https://github.com/prefix-dev/pixi/discussions>
