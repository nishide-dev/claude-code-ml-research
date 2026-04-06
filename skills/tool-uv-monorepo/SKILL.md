---
name: tool-uv-monorepo
description: Comprehensive guide for building Python monorepos with uv workspaces - unified dependency resolution, shared lock files, editable installs, testing strategies, Docker optimization, and CI/CD patterns for managing multiple packages in a single repository
---

# UV Monorepo Development with Workspaces

Complete guide for building and managing Python monorepos using uv's workspace functionality.

## Overview

uv's workspace feature enables true monorepo architecture for Python projects, solving the historical challenge of managing multiple packages in a single repository. Inspired by Rust's Cargo, workspaces provide:

- **Unified dependency resolution**: Single `uv.lock` file for entire repository
- **Version consistency**: Eliminates version drift across packages
- **Automatic editable installs**: Code changes propagate instantly
- **Fast resolution**: Rust-powered solver 10-100x faster than pip
- **Standard compliance**: Built on PEP 621, PEP 735

**When to use uv workspaces:**

- ✅ Microservices sharing common libraries
- ✅ Multi-package applications (CLI + core + services)
- ✅ Internal library ecosystems
- ✅ Projects requiring strict dependency consistency

**Key resources:**

- Official docs: <https://docs.astral.sh/uv/concepts/projects/workspaces/>
- Projects guide: <https://docs.astral.sh/uv/>

---

## Core Concepts

### 1. Unified Lock File

The most powerful feature: a single `uv.lock` at repository root that:

- Resolves all workspace members' dependencies into one conflict-free graph
- Eliminates "version drift" where independent projects use different versions
- Forces all components to use identical versions of shared dependencies (e.g., `pydantic`, `fastapi`)

When you run `uv lock`, the system evaluates the entire workspace and creates a mathematically consistent dependency solution.

### 2. Python Version Constraints

Workspaces enforce a **single** `requires-python` for the entire repository, calculated as the **intersection** of all members' requirements:

| Package | requires-python | Workspace Result |
|---------|-----------------|------------------|
| root | >=3.10 | - |
| service-a | >=3.11 | >=3.11 |
| service-b | >=3.12 | >=3.12 (strictest wins) |

This ensures all members can coexist in the shared virtual environment (`.venv`).

### 3. Workspace vs Path Dependencies

Two approaches for managing related packages:

| Feature | Workspace | Path Dependencies |
|---------|-----------|-------------------|
| Lock file | Single `uv.lock` for all | Separate per project |
| Python version | Unified (intersection) | Independent per project |
| Virtual env | Single shared `.venv` | Individual `.venv` per project |
| Consistency | Enforced | Flexible |
| Best for | Tightly coupled services | Highly independent projects |

---

## Directory Structure

### Basic Layout

```text
my-monorepo/
├── pyproject.toml       # Workspace root config
├── uv.lock              # Unified lock file
├── .venv/               # Shared virtual environment
└── packages/
    ├── core/
    │   ├── pyproject.toml
    │   └── src/core/
    ├── api/
    │   ├── pyproject.toml
    │   └── src/api/
    └── cli/
        ├── pyproject.toml
        └── src/cli/
```

### Root Configuration

**pyproject.toml** (workspace root):

```toml
[project]
name = "my-monorepo-workspace"  # Must be unique from members!
version = "0.1.0"
requires-python = ">=3.10"

[tool.uv.workspace]
members = ["packages/*"]        # Glob patterns
# exclude = ["packages/legacy/*"]  # Optional exclusions

[tool.uv]
package = false                 # Virtual root (not installable)

[dependency-groups]
dev = [
    "pytest>=7.4",
    "ruff>=0.1",
    "mypy>=1.0",
]
```

#### Critical: Name Collision

The workspace root name must be **unique** from all member names. If both root and a member use `my-app`, `uv sync` fails with:

```text
Error: Duplicate workspace member: my-app
```

Use descriptive names: `my-app-workspace` for root, `my-app` for actual package.

---

## Package Dependencies

### Declaring Internal Dependencies

To make `api` depend on `core`, use two-step declaration:

**packages/api/pyproject.toml**:

```toml
[project]
name = "api"
dependencies = [
    "core",  # Standard PEP 621 declaration
]

[tool.uv.sources]
core = { workspace = true }  # uv-specific: resolve from workspace
```

**Why two declarations?**

- `[project.dependencies]`: Standard metadata, readable by all tools
- `[tool.uv.sources]`: Routing table for uv-specific resolution

If `[tool.uv.sources]` is missing, uv provides helpful error:

```text
Error: Package 'core' is a workspace member but missing sources entry
```

### Automatic Editable Install

With `workspace = true`, uv automatically installs members in **editable mode**. Code changes in `core` are instantly reflected in `api` without reinstallation.

---

## Development Dependencies

### PEP 735 Dependency Groups

For dev tools (pytest, ruff, mypy), use dependency groups in root:

```toml
[dependency-groups]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.1",
    "mypy>=1.0",
]
```

**Benefits:**

- Not included in build artifacts (wheels, sdists)
- Shared across all packages: `uv sync --group dev`
- Prevents version inconsistencies (e.g., different linters per service)

**Install:**

```bash
# Install all dependencies + dev group
uv sync --group dev

# Install only production dependencies
uv sync
```

---

## Testing with Pytest

### Import File Mismatch Problem

In monorepos with multiple `tests/` directories, pytest defaults cause errors:

```text
import file mismatch:
  imported module 'test_helpers' has this __file__: .../packages/cli/tests/test_helpers.py
  which is not the same as: .../packages/core/tests/test_helpers.py
```

**Cause:** Pytest's `prepend` mode inserts test directories into `sys.path`, causing name collisions.

### Solution: Importlib Mode

**Root pyproject.toml**:

```toml
[tool.pytest.ini_options]
addopts = ["--import-mode=importlib"]
```

This uses Python's `importlib` to import tests directly without modifying `sys.path`, eliminating name collisions.

**Important:** Do NOT add `__init__.py` to test directories when using `importlib` mode. This causes silent test skipping (tests are cached under one path but collected from another).

---

## Docker Optimization

### Problem: Monolithic Dependencies

A single `uv.lock` contains dependencies for ALL services. Naive Docker builds include unnecessary dependencies, inflating image size and attack surface.

### Solution: Multi-Stage Build with Export

#### Step 1: Extract Service-Specific Dependencies

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-alpine AS builder

WORKDIR /app
COPY uv.lock pyproject.toml ./
COPY packages/ ./packages/

# Extract only dependencies for 'api' service
RUN uv export --frozen --directory packages/api -o requirements.txt && \
    # Remove workspace member references (e.g., -e ./packages/core)
    sed -i '/^-e/d' requirements.txt
```

#### Step 2: Install External Dependencies (Cached Layer)

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip sync requirements.txt --no-cache --compile-bytecode
```

#### Step 3: Copy Internal Libraries and Application Code

```dockerfile
# Copy shared libraries first (changes less frequently)
COPY packages/core/ ./packages/core/

# Copy application code last (changes most frequently)
COPY packages/api/ ./packages/api/

# Install as editable
RUN uv pip install -e ./packages/api/
```

**Benefits:**

- External dependencies cached in Docker layer
- Application code changes don't invalidate dependency layer
- `--compile-bytecode` pre-generates `.pyc` files for faster startup

### Layer Caching Strategy

| Stage | Command | Cache Behavior |
|-------|---------|----------------|
| Base | Copy uv binary | Rarely changes |
| Dependencies | `uv export` + `uv pip sync` | Only invalidated when `uv.lock` changes |
| Shared libs | Copy `packages/core/` | Invalidated when core changes |
| Application | Copy `packages/api/` | Invalidated on every app code change |

---

## CI/CD Best Practices

### Global Cache Strategy (GitHub Actions)

**Problem:** Per-PR caches waste storage and slow down CI.

**Solution:** Single cache from `main` branch, read-only for PRs.

#### Workflow 1: Main Branch (Cache Write)

```yaml
name: Build Cache
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly refresh

jobs:
  cache:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: false  # Manual cache control

      - name: Sync dependencies
        run: uv sync --all-groups

      - name: Save cache
        uses: actions/cache/save@v4
        with:
          path: |
            ~/.cache/uv
            .venv
          key: uv-${{ runner.os }}-${{ hashFiles('uv.lock') }}
```

#### Workflow 2: PR Branches (Cache Read-Only)

```yaml
name: Test
on: pull_request

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Restore cache
        uses: actions/cache/restore@v4
        with:
          path: |
            ~/.cache/uv
            .venv
          key: uv-${{ runner.os }}-${{ hashFiles('uv.lock') }}
          restore-keys: |
            uv-${{ runner.os }}-

      - uses: astral-sh/setup-uv@v5
      - run: uv sync  # Uses cache, downloads only diff
      - run: uv run pytest
```

**Benefits:**

- PRs never pollute cache
- Always clean state from main
- Minimal storage usage

---

## Common Commands

### Workspace Management

```bash
# Initialize new workspace
uv init --package my-monorepo
cd my-monorepo

# Add workspace member
uv init --lib packages/core
uv init --lib packages/api

# Configure workspace in root pyproject.toml
# Add: [tool.uv.workspace] members = ["packages/*"]

# Sync all dependencies
uv sync

# Sync with dev tools
uv sync --group dev

# Run command in workspace context
uv run python -c "import core; import api"

# Run from specific package directory
cd packages/api
uv run uvicorn main:app
```

### Building & Publishing

```bash
# Build specific package
uv build --package api

# Build all packages
uv build --all-packages

# Check version
uv version --package core

# Publish to PyPI
uv publish --package core
```

---

## Migration from Poly-repo

### Pre-Migration: Version Alignment

**Critical:** Before creating workspace, align all dependency versions across services.

1. **Update all services to latest compatible versions:**

```bash
# In each service
uv sync --upgrade
uv run pytest  # Verify no breakage
```

1. **Resolve version conflicts:**

If Service A uses `Django 4.2` and Service B uses `Django 5.1`, choose one version and update all services.

1. **Create workspace structure:**

```bash
mkdir -p packages
mv service-a packages/
mv service-b packages/
```

1. **Configure root workspace:**

Create root `pyproject.toml` with workspace configuration.

1. **Convert path dependencies:**

Replace relative path dependencies with `workspace = true`.

**Expected outcome:** `uv lock` will surface any hidden version conflicts that were masked by isolated environments.

---

## Troubleshooting

### Error: Duplicate workspace member

**Cause:** Root and member have same name.

**Solution:** Rename root to `project-workspace`.

### Error: Missing sources entry

**Cause:** Package in `[project.dependencies]` but not in `[tool.uv.sources]`.

**Solution:** Add `{ workspace = true }` to sources.

### Pytest import file mismatch

**Cause:** Default `prepend` mode + duplicate test file names.

**Solution:** Add `--import-mode=importlib` to pytest config.

### CUDA/GPU dependencies in monorepo

**Solution:** Use platform-specific members or path dependencies for GPU-specific code.

---

## Best Practices

1. **Name virtual roots descriptively:** `my-app-workspace` not `my-app`
2. **Use dependency groups:** Keep dev tools in `[dependency-groups]`
3. **Importlib mode always:** Set `--import-mode=importlib` in pytest config
4. **Docker multi-stage:** Use `uv export` to extract service-specific deps
5. **Cache from main:** GitHub Actions cache strategy from main branch only
6. **Align before migration:** Update all versions before creating workspace

---

## Summary

uv workspaces provide:

**Core Benefits:**

- **10-100x faster** dependency resolution (Rust-powered)
- **Zero version drift** through unified lock file
- **Instant code propagation** with automatic editable installs
- **Standard compliance** (PEP 621, PEP 735)
- **Simplified CI/CD** through single environment

**Key Features:**

- Single `uv.lock` for entire repository
- Unified Python version constraint (intersection)
- Shared virtual environment (`.venv`)
- Automatic workspace member discovery
- Built-in Docker optimization patterns

**Resources:**

- **Official docs**: <https://docs.astral.sh/uv/concepts/projects/workspaces/>
- **Practical guide**: <https://dev.to/aws/3-things-i-wish-i-knew-before-setting-up-a-uv-workspace-30j6>
- **Advanced CI/CD**: <https://www.reddit.com/r/Python/comments/1iy4h5k/cracking_the_python_monorepo_build_pipelines_with/>

uv workspaces eliminate Python's historical monorepo pain points, bringing Cargo-like simplicity and performance to multi-package projects.
