# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Claude Code plugin** for machine learning research. It is NOT a Python package to be installed via pip. Instead, it extends Claude Code by providing:

- **Commands** (`commands/*.md`): Slash commands like `/train`, `/debug`, `/ml-config` that users invoke
- **Agents** (`agents/*.md`): Specialized sub-agents for ML architecture, debugging, configuration generation
- **Skills** (`skills/*/SKILL.md`): Knowledge bases (Lightning, Hydra, PyTorch Geometric, W&B, Pixi) that provide context
- **Rules** (`rules/*.md`): Coding standards and workflow constraints enforced automatically
- **Hooks** (`hooks/hooks.json`): Event-driven automation (auto-format Python files, validate YAML configs)

**Plugin manifest**: `.claude-plugin/plugin.json` defines the plugin structure. Commands in `./commands/` are auto-discovered and invoked directly (e.g., `/train`). Subdirectories create namespaces (e.g., `commands/ml/foo.md` → `/ml:foo`).

## Distribution

This plugin is distributed via marketplace.json, allowing users to install from GitHub:

**Marketplace Structure** (`.claude-plugin/marketplace.json`):

- Marketplace name: "ml-research" (same as plugin name)
- Single plugin entry with source: "." (repository root)
- Category: "machine-learning"
- Tags: deep-learning, pytorch-lightning, experiment-tracking, graph-neural-networks, nlp, computer-vision, training-automation, model-debugging, configuration-management, reproducible-research

**Installation methods**:

1. Marketplace: `/plugin marketplace add nishide-dev/claude-code-ml-research`
2. GitHub: `/plugin install gh:nishide-dev/claude-code-ml-research`
3. Local: `/plugin install ./claude-code-ml-research`

The marketplace.json enables Claude Code's plugin discovery and automatic updates.

## Architecture: Four Component Types

### 1. Skills and Commands

**IMPORTANT**: According to [official Claude Code documentation](https://code.claude.com/docs/ja/skills), **custom slash commands have been merged into skills**. Commands (`commands/*.md`) are now **legacy** — skills (`skills/*/SKILL.md`) are the recommended approach.

**Migration Status**: ✅ **Complete** - All 12 commands have been successfully migrated to skills with comprehensive supporting files (templates, examples, scripts).

#### Skills (skills/*/SKILL.md) - Recommended

Skills are directories containing `SKILL.md` plus optional supporting files:

```text
skills/ml-train/
├── SKILL.md (main instructions)
├── templates/
│   ├── basic-training.yaml
│   └── distributed-training.yaml
└── examples/
    └── image-classification.md
```

**Frontmatter fields** (in SKILL.md):

- `name` (optional): Skill identifier (defaults to directory name)
- `description` (recommended): What this skill does and when to use it
- `argument-hint` (optional): Document expected arguments like `[issue-number]`
- `disable-model-invocation` (optional): Set to `true` to prevent Claude from auto-invoking
- `user-invocable` (optional): Set to `false` to hide from `/` menu (background knowledge only)
- `allowed-tools` (optional): Restrict which tools this skill can use
- `model` (optional): Specify model (sonnet, opus, haiku)
- `context` (optional): Set to `fork` to run in subagent
- `agent` (optional): Subagent type when `context: fork` is set

Skills in `./skills/` are auto-discovered (no need to reference in `plugin.json`).

**Skill advantages over commands**:

1. **Supporting files**: Can include templates, examples, scripts in same directory
2. **Invocation control**: `disable-model-invocation` and `user-invocable` frontmatter fields
3. **Automatic loading**: Claude can load skills based on task context
4. **Better organization**: Related files grouped in one directory
5. **Agent Skills standard**: Part of open standard for cross-tool compatibility

**Note on `/ml-project-init`**: This skill uses the [ML Research Template](https://github.com/nishide-dev/ml-research-template), maintained as a separate repository. The template is referenced via GitHub URL (`gh:nishide-dev/ml-research-template`) for independent versioning and broader reusability.

### 2. Agents (agents/*.md)

Agent files have YAML frontmatter specifying name, description, tools, model, and color:

```markdown
---
name: ml-architect
description: Design ML system architectures
tools: ["Read", "Write", "Glob", "Grep"]
model: opus
color: blue
---

You are an expert ML architect...
```

**Frontmatter fields**:

- `name` (required): Agent identifier (kebab-case)
- `description` (required): When to trigger this agent (should include examples)
- `model` (required): `inherit`, `sonnet`, `opus`, or `haiku`
- `color` (required): Keyword color for UI (`blue`, `cyan`, `green`, `yellow`, `magenta`, `red`)
- `tools` (optional): Array of allowed tools

**Model aliases**: Use simple aliases (`opus`, `sonnet`, `haiku`) instead of full version names. These automatically point to the latest versions (Opus 4.6, Sonnet 4.5, Haiku 4.5).

**Color field**: Use keyword colors for consistency. Each agent has a semantic color:

- ml-architect: `blue` (strategic)
- training-debugger: `red` (debugging)
- config-generator: `magenta` (configuration)
- pytorch-expert: `yellow` (implementation)
- geometric-specialist: `cyan` (graph domain)
- transformers-specialist: `magenta` (NLP/LLM domain)

Agents are explicitly listed in `plugin.json` under `"agents"`.

### 2. Rules (rules/ml/*.md)

Rules enforce coding standards and workflow constraints automatically. They are organized by category:

- `rules/ml/coding-standards.md`: ML coding best practices (deterministic operations, type hints, tensor shape documentation)
- `rules/ml/security-practices.md`: Security guidelines (never log API keys, use environment variables, sanitize file paths)
- `rules/ml/workflow-constraints.md`: Workflow enforcement (validate configs before training, use checkpointing, tag experiments)

**Directory structure**: Rules are organized in subdirectories (`ml/`) to allow coexistence with rules from other plugins when installing to `~/.claude/rules/`.

**Installation**: Rules must be manually installed to `~/.claude/rules/` as plugins cannot distribute rules automatically:

```bash
cp -r rules/ml/* ~/.claude/rules/
```

Once installed, rules are automatically applied to guide Claude's behavior when writing code or suggesting workflows.

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

- `.claude-plugin/plugin.json`: Plugin manifest
  - Required fields: `name`, plus optional metadata (`version`, `description`, `author`, etc.)
  - `agents` field: Explicitly list agent files (not auto-discovered)
  - `skills`, `hooks` fields: Auto-discovered from standard directories, don't need to be specified
  - Note: LSP servers removed from manifest (can be configured per-project)
- `.claude-plugin/marketplace.json`: Marketplace configuration for distribution
- `hooks/hooks.json`: Event-driven automation (loaded automatically from standard location)
- `pyproject.toml`: Python dependencies for development (not for plugin users)
- `uv.lock`: Locked dependencies (must be committed for reproducible CI)
- `.pre-commit-config.yaml`: Pre-commit hooks (Ruff, ty, YAML validation)
- `scripts/validate_plugin.py`: Validates plugin structure (used in CI)
- `tests/test_plugin_structure.py`: Tests for plugin file structure

## Plugin File Naming Conventions

- **Skills**: Directory name is kebab-case with prefix `ml-*` or `tool-*` (e.g., `ml-train/`, `ml-hydra-config/`, `tool-pixi/`)
- **Agents**: Use kebab-case (e.g., `training-debugger.md`, `ml-architect.md`)

Avoid names that conflict with Claude Code built-ins: `/init`, `/config`, `/export`, `/clear`, `/help`, etc.

## Component File Frontmatter Requirements

### Skills

- `name` (optional): Skill identifier (defaults to directory name)
- `description` (recommended): What this skill does and when to use it (helps Claude decide when to load)
- `argument-hint` (optional): Document expected arguments
- `disable-model-invocation` (optional): Set to `true` to prevent Claude from auto-invoking (user-only invocation)
- `user-invocable` (optional): Set to `false` to hide from `/` menu (background knowledge only)
- `allowed-tools` (optional): Restrict which tools this skill can use
- `model` (optional): Specify model (sonnet, opus, haiku)
- `context` (optional): Set to `fork` to run in subagent
- `agent` (optional): Subagent type when `context: fork` is set

### Agents

- `name` (required): Agent identifier in kebab-case
- `description` (required): When to trigger (include examples)
- `model` (required): `inherit`, `sonnet`, `opus`, or `haiku`
- `color` (required): Keyword color (`blue`, `cyan`, `green`, `yellow`, `magenta`, `red`)
- `tools` (optional): Array of allowed tools

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

1. **Use skills for new development**: This plugin uses skills (`skills/*/SKILL.md`) exclusively for slash commands. Skills support supporting files (templates, examples, scripts), invocation control, and automatic loading.
2. **Skill naming**: Use kebab-case with `ml-` prefix for ML workflow skills (e.g., `ml-train`, `ml-debug`)
3. **Auto-discovery**: `./skills/` directory is auto-discovered—don't add skills to plugin.json
4. **Skill frontmatter**: Use `description` (recommended) and `disable-model-invocation` (optional), NOT `name` or `arguments`
5. **Agent frontmatter**: Use keyword colors (`blue`, `red`, etc.), NOT hex codes (`#4A90E2`)
6. **Supporting files organization**: Use `templates/`, `examples/`, `scripts/` subdirectories within skill directories
7. **SKILL.md size**: Keep SKILL.md concise (<500 lines recommended), split detailed content into support files
8. **uv.lock tracking**: Must be committed to Git for CI caching to work
9. **ty version**: Use `>=0.0.15`, not `>=0.2.0`
10. **Coverage target**: pytest measures coverage for `scripts/`, not the plugin content itself
