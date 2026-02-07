# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Claude Code plugin** for machine learning research. It is NOT a Python package to be installed via pip. Instead, it extends Claude Code by providing:

- **Commands** (`commands/*.md`): Slash commands like `/train`, `/debug`, `/ml-config` that users invoke
- **Agents** (`agents/*.md`): Specialized sub-agents for ML architecture, debugging, configuration generation
- **Skills** (`skills/*/SKILL.md`): Knowledge bases (Lightning, Hydra, PyTorch Geometric, W&B, Pixi) that provide context
- **Rules** (`rules/*.md`): Coding standards and workflow constraints enforced automatically
- **Hooks** (`hooks/hooks.json`): Event-driven automation (auto-format Python files, validate YAML configs)

**Plugin manifest**: `.claude-plugin/plugin.json` defines the plugin structure. The `name` field is just an identifier—commands are invoked directly (e.g., `/train`, not `/ml-research:train`).

## Architecture: Four Component Types

### 1. Commands (commands/*.md)

Each command file must have YAML frontmatter:

```markdown
---
name: train
description: Execute training runs with PyTorch Lightning
---

# ML Training Execution

Content here...
```

Commands are referenced in `plugin.json` via `"skills": ["./commands/"]` (note: commands are treated as "skills" in the manifest).

**Note on `/project-init`**: This command uses the [ML Research Template](https://github.com/nishide-dev/ml-research-template), maintained as a separate repository. The template is referenced via GitHub URL (`gh:nishide-dev/ml-research-template`) for independent versioning and broader reusability.

### 2. Agents (agents/*.md)

Agent files have YAML frontmatter specifying tools, model, role, and optional color:

```markdown
---
name: ml-architect
description: Design ML system architectures
tools: ["Read", "Write", "Glob", "Grep"]
model: opus
color: "#4A90E2"
---

You are an expert ML architect...
```

**Model aliases**: Use simple aliases (`opus`, `sonnet`, `haiku`) instead of full version names. These automatically point to the latest versions (Opus 4.6, Sonnet 4.5, Haiku 4.5).

**Color field** (optional): Hex color code for visual distinction in Claude Code UI. Each agent has a semantic color:

- ml-architect: `#4A90E2` (blue - strategic)
- training-debugger: `#E74C3C` (red - debugging)
- config-generator: `#9B59B6` (purple - configuration)
- pytorch-expert: `#F39C12` (orange - implementation)
- geometric-specialist: `#16A085` (teal - graph domain)
- transformers-specialist: `#8B5CF6` (violet - NLP/LLM domain)

Agents are explicitly listed in `plugin.json` under `"agents"`.

### 3. Skills (skills/*/SKILL.md)

Skills are knowledge bases stored in subdirectories. Each skill has a `SKILL.md` file providing comprehensive guides on specific topics:

- `ml-lightning-basics/SKILL.md`: PyTorch Lightning patterns
- `ml-hydra-config/SKILL.md`: Hydra configuration management
- `ml-pytorch-geometric/SKILL.md`: Graph Neural Network implementations
- `ml-wandb-tracking/SKILL.md`: Experiment tracking with W&B
- `ml-cli-tools/SKILL.md`: Building CLIs with Typer and Rich
- `ml-transformers/SKILL.md`: Hugging Face Transformers + PyTorch Lightning integration
- `tool-pixi/SKILL.md`: Pixi package manager for ML projects

Skills are NOT invoked as commands—they provide context when referenced in conversation.

### 4. Rules (rules/ml/*.md)

Rules enforce coding standards and workflow constraints automatically. They are organized by category and referenced in `plugin.json` under `"rules"`:

- `rules/ml/coding-standards.md`: ML coding best practices (deterministic operations, type hints, tensor shape documentation)
- `rules/ml/security-practices.md`: Security guidelines (never log API keys, use environment variables, sanitize file paths)
- `rules/ml/workflow-constraints.md`: Workflow enforcement (validate configs before training, use checkpointing, tag experiments)

**Directory structure**: Rules are organized in subdirectories (`ml/`) to allow coexistence with rules from other plugins when installing to `~/.claude/rules/`.

Rules are automatically applied to guide Claude's behavior when writing code or suggesting workflows.

## Development Commands

### Setup

```bash
# Install dependencies
uv sync --all-extras --dev

# Install pre-commit hooks
uv run pre-commit install
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Fix linting issues automatically
uv run ruff check --fix .

# Type check
uv run ty check scripts/

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

### Testing

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=scripts --cov-report=term

# Run specific test
uv run pytest tests/test_plugin_structure.py::TestCommandFiles::test_command_files_have_title -v

# Validate plugin structure
uv run python scripts/validate_plugin.py
```

### CI

GitHub Actions runs:

- **Lint job**: Ruff format/check, ty type checking, markdownlint
- **Test job**: pytest, plugin validation, coverage upload (Python 3.12, 3.13)

All use `uv` for dependency management with `uv.lock` caching.

## Key Files for Plugin Structure

- `.claude-plugin/plugin.json`: Plugin manifest (defines agents, skills paths)
- `pyproject.toml`: Python dependencies for development (not for plugin users)
- `uv.lock`: Locked dependencies (must be committed for reproducible CI)
- `.pre-commit-config.yaml`: Pre-commit hooks (Ruff, ty, YAML validation)
- `scripts/validate_plugin.py`: Validates plugin structure (used in CI)
- `tests/test_plugin_structure.py`: Tests for plugin file structure

## Plugin File Naming Conventions

- **Commands**: Use kebab-case (e.g., `model-export.md`, `ml-config.md`)
- **Agents**: Use kebab-case (e.g., `training-debugger.md`)
- **Skills**: Directory name is kebab-case with prefix `ml-*` or `tool-*` (e.g., `ml-hydra-config/`, `tool-pixi/`)

Avoid names that conflict with Claude Code built-ins: `/init`, `/config`, `/export`, `/clear`, `/help`, etc.

## Command Files: YAML Frontmatter Required

Since Phase 2, ALL command files MUST have YAML frontmatter. Tests check for this:

```markdown
---
name: command-name
description: Brief description here
---

# Command Title

Content...
```

The test `test_command_files_have_title` skips frontmatter when checking for title (`# heading`).

## Scripts Directory

Contains Python utilities (not part of the plugin itself, only for development):

- `scripts/setup_pixi.py`: Initialize Pixi-based ML projects (CLI tool)
- `scripts/setup_uv.py`: Initialize UV-based ML projects (CLI tool)
- `scripts/validate_plugin.py`: Validate plugin structure (CI check)

These scripts use Typer for CLI and are tested with pytest. Coverage target: `scripts/` directory.

## Markdown Linting

Configuration: `.markdownlint.json`

Disabled rules:

- `MD013`: Line length (no limit)
- `MD033`: HTML allowed
- `MD041`: First line doesn't need to be top-level heading
- `MD060`: Table column style (too strict for compact tables)

Run: `npx markdownlint-cli2 "**/*.md" --config .markdownlint.json`

## Type Checking with ty

`ty` is used instead of `mypy`. Configuration in `pyproject.toml`:

```toml
# ty configuration (ty doesn't require config file, uses sensible defaults)
# Run with: ty check [path]
```

Note: `ty` version must be `>=0.0.15` (not `>=0.2.0`, which doesn't exist).

## Pre-commit Hooks

All hooks use `language: system` to avoid virtualenv creation issues on shared filesystems. This requires dependencies to be installed via `uv sync --all-extras --dev` before running hooks.

Hooks run on commit:

- Ruff linter (auto-fix)
- Ruff formatter
- ty type check
- Markdownlint (auto-fix)
- YAML validation
- Python AST check
- Trailing whitespace removal
- Merge conflict detection
- Plugin JSON validation

Manual-only (slow): `pytest` with `--hook-stage manual`

## Common Pitfalls

1. **Command naming**: Don't prefix all commands with `ml-`—only rename those that conflict with built-ins
2. **Plugin name vs commands**: The plugin `name` in `plugin.json` is NOT a namespace. Commands are invoked directly.
3. **YAML frontmatter**: All command files need it, but skills don't
4. **uv.lock tracking**: Must be committed to Git for CI caching to work
5. **ty version**: Use `>=0.0.15`, not `>=0.2.0`
6. **Coverage target**: pytest measures coverage for `scripts/`, not the plugin content itself
