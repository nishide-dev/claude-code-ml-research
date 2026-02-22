# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Claude Code plugin** for machine learning research. It is NOT a Python package to be installed via pip. Instead, it extends Claude Code by providing:

- **Skills** (`skills/*/SKILL.md`): User-invocable commands and background knowledge (Lightning, Hydra, PyTorch Geometric, W&B, Pixi)
- **Agents** (`agents/*.md`): Specialized sub-agents for ML architecture, debugging, configuration generation
- **Rules** (`rules/ml/*.md`): Coding standards and workflow constraints enforced automatically
- **Hooks** (`hooks/hooks.json`): Event-driven automation (auto-format Python files, validate YAML configs)

**Plugin manifest**: `.claude-plugin/plugin.json` defines the plugin structure.

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

## Architecture: Plugin Components

This plugin follows the [Agent Skills](https://agentskills.io) open standard and Claude Code plugin architecture. Reference documentation:

- **Skills**: <https://code.claude.com/docs/ja/skills>
- **Sub-agents**: <https://code.claude.com/docs/ja/sub-agents>
- **Hooks**: <https://code.claude.com/docs/ja/hooks>
- **Plugins**: <https://code.claude.com/docs/ja/plugins>

### 1. Skills (`skills/*/SKILL.md`)

**Status**: ✅ All commands migrated to skills (12 workflow skills + 8 knowledge skills)

Skills are the primary way to extend Claude Code functionality. Each skill is a directory containing `SKILL.md` plus optional supporting files.

#### Skill Types

**Workflow Skills** (user-invocable commands with `disable-model-invocation: true`):

- `/ml-train` - Execute training runs
- `/ml-config-manager` - Generate Hydra configs
- `/ml-debug` - Debug training issues
- `/ml-experiment` - Manage experiments
- `/ml-validate` - Validate project structure
- `/ml-profile` - Profile performance
- `/ml-data-pipeline` - Manage data pipelines
- `/ml-setup` - Setup environment
- `/ml-project-init` - Initialize projects
- `/ml-lint` - Code quality checks
- `/ml-format` - Format code
- `/ml-model-export` - Export models

**Knowledge Skills** (background knowledge, auto-loaded by Claude):

- `ml-lightning-basics` - PyTorch Lightning patterns
- `ml-hydra-config` - Hydra configuration
- `ml-pytorch-geometric` - Graph Neural Networks
- `ml-wandb-tracking` - Experiment tracking
- `ml-transformers` - Hugging Face Transformers
- `ml-cli-tools` - Building CLIs with Typer/Rich
- `tool-pixi` - Pixi package manager
- `tool-marimo` - Marimo reactive notebooks

#### Skill Directory Structure

```text
skills/ml-train/
├── SKILL.md              # Main instructions (required)
├── templates/            # Config templates
│   ├── basic-training.yaml
│   └── distributed-training.yaml
├── examples/             # Usage examples
│   └── image-classification.md
└── scripts/              # Utility scripts
    └── validate.py
```

**Best practices**:

- Keep `SKILL.md` under 500 lines
- Move detailed content to supporting files
- Reference supporting files from SKILL.md so Claude knows when to load them

#### Skill Frontmatter Fields

Based on [official documentation](https://code.claude.com/docs/ja/skills#frontmatter-reference):

```yaml
---
name: my-skill                     # Optional: defaults to directory name
description: What this skill does  # Recommended: helps Claude decide when to load
argument-hint: [optional-args]     # Optional: shows expected arguments
disable-model-invocation: true     # Optional: prevent Claude auto-invoke (user-only)
user-invocable: false              # Optional: hide from `/` menu (background knowledge)
allowed-tools: Read, Grep          # Optional: restrict tool access
context: fork                      # Optional: run in subagent
agent: Explore                     # Optional: subagent type when context=fork
hooks:                             # Optional: lifecycle hooks for this skill
  PreToolUse: ...
---

Your skill instructions here...
```

**Important notes**:

- ⚠️ **`model` field is NOT documented** in official skills documentation, though it appears in the field list
- ✅ Use `context: fork` + `agent` to run skills in subagents with specific models
- Skills in `./skills/` are **auto-discovered** (don't add to plugin.json)
- Skills use **string replacements**: `$ARGUMENTS`, `${CLAUDE_SESSION_ID}`

#### Invocation Control

| Frontmatter                          | User invokes | Claude invokes | Context loading                        |
| :----------------------------------- | :----------- | :------------- | :------------------------------------- |
| (default)                            | Yes          | Yes            | Description always in context          |
| `disable-model-invocation: true`     | Yes          | No             | Not in context, loaded when user calls |
| `user-invocable: false`              | No           | Yes            | Description always in context          |

#### Running Skills in Subagents

Use `context: fork` to run a skill in an isolated subagent context:

```yaml
---
name: deep-research
description: Research a topic thoroughly
context: fork        # Run in subagent
agent: Explore       # Use Explore agent (read-only, fast)
---

Research $ARGUMENTS thoroughly...
```

Available agent types: `Explore` (Haiku, read-only), `Plan` (read-only), `general-purpose` (all tools), or custom agents from `.claude/agents/`.

### 2. Agents (`agents/*.md`)

**Purpose**: Specialized sub-agents that handle specific types of tasks in isolated contexts.

Each agent runs in its own context window with custom system prompt, specific tool access, and independent permissions.

#### Agent Directory Structure

Agent files are single markdown files (not directories):

```text
agents/
├── ml-architect.md
├── training-debugger.md
├── config-generator.md
├── pytorch-expert.md
├── geometric-specialist.md
└── transformers-specialist.md
```

#### Agent Frontmatter Fields

Based on [official documentation](https://code.claude.com/docs/ja/sub-agents#supported-frontmatter-fields):

```yaml
---
name: code-reviewer                # Required: unique identifier (kebab-case)
description: Reviews code for...   # Required: when to delegate to this agent
tools: Read, Grep, Glob            # Optional: allowed tools (inherits all if omitted)
disallowedTools: Write, Edit       # Optional: denied tools
model: sonnet                      # Optional: sonnet, opus, haiku, inherit (default: inherit)
permissionMode: default            # Optional: default, acceptEdits, dontAsk, bypassPermissions, plan
skills:                            # Optional: preload skills into agent context
  - api-conventions
  - error-handling-patterns
hooks:                             # Optional: lifecycle hooks
  PreToolUse: ...
memory: user                       # Optional: user, project, local (persistent memory)
color: blue                        # Required: blue, cyan, green, yellow, magenta, red
---

You are a senior code reviewer...
```

**Key differences from Skills**:

- ✅ `model` field **works in agents** (confirmed in documentation)
- ✅ `color` field **required** for UI identification
- Agents are **explicitly listed** in plugin.json (not auto-discovered)
- Agent body is the **system prompt** (not instructions for Claude)

#### Model Selection for Agents

Use model aliases that automatically point to latest versions:

- `sonnet` → Claude Sonnet 4.5
- `opus` → Claude Opus 4.6
- `haiku` → Claude Haiku 4.5
- `inherit` → Use same model as main conversation (default)

**Model selection strategy**:

- **Haiku**: Fast, low-cost for read-only exploration (e.g., Explore agent)
- **Sonnet**: Balanced performance for most tasks (recommended default)
- **Opus**: Maximum reasoning for complex architectural decisions

#### Current Agents

| Agent                    | Model  | Color     | Purpose                          |
| :----------------------- | :----- | :-------- | :------------------------------- |
| ml-architect             | opus   | blue      | Design ML system architectures   |
| training-debugger        | sonnet | red       | Diagnose training issues         |
| config-generator         | sonnet | magenta   | Generate Hydra configurations    |
| pytorch-expert           | sonnet | yellow    | PyTorch optimization             |
| geometric-specialist     | sonnet | cyan      | Graph Neural Networks            |
| transformers-specialist  | sonnet | magenta   | LLM fine-tuning, PEFT            |

#### Persistent Memory for Agents

Enable agents to build knowledge across sessions:

```yaml
---
name: code-reviewer
memory: user  # Stores in ~/.claude/agent-memory/<name>/
---
```

Memory scopes:

- `user`: `~/.claude/agent-memory/<name>/` - Cross-project learning
- `project`: `.claude/agent-memory/<name>/` - Project-specific (version controlled)
- `local`: `.claude/agent-memory-local/<name>/` - Project-specific (local only)

When enabled:

- Agent system prompt includes memory directory instructions
- First 200 lines of `MEMORY.md` injected into context
- Read, Write, Edit tools auto-enabled for memory management

### 3. Rules (`rules/ml/*.md`)

Rules enforce best practices automatically. Organized in `ml/` subdirectory for coexistence with other plugins.

**Current rules**:

- `rules/ml/coding-standards.md` - ML coding best practices
- `rules/ml/security-practices.md` - Security guidelines
- `rules/ml/workflow-constraints.md` - Workflow enforcement

**Installation** (manual, plugins cannot distribute rules automatically):

```bash
cp -r rules/ml/* ~/.claude/rules/
```

### 4. Hooks (`hooks/hooks.json`)

Event-driven automation loaded automatically from standard location.

**Current hooks**:

- Auto-format Python files after Write/Edit
- Validate YAML configs after changes
- Remind about dependency installation
- Welcome message on session start

See [Hooks documentation](https://code.claude.com/docs/ja/hooks) for configuration details.

## Development Workflow

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

# Validate plugin structure
uv run python scripts/validate_plugin.py
```

### Creating New Components

#### New Skill

```bash
# Create skill directory
mkdir -p skills/my-skill/{templates,examples,scripts}

# Create SKILL.md
cat > skills/my-skill/SKILL.md << 'EOF'
---
name: my-skill
description: What this skill does and when to use it
disable-model-invocation: true  # If user-only
---

# My Skill

Instructions for Claude...

## Templates

See [templates/](templates/) for...
EOF
```

#### New Agent

```bash
# Create agent file
cat > agents/my-agent.md << 'EOF'
---
name: my-agent
description: When to delegate to this agent
model: sonnet
color: blue
tools: Read, Grep, Glob
---

You are an expert in...
EOF

# Add to plugin.json
# Edit .claude-plugin/plugin.json and add "./agents/my-agent.md" to agents array
```

**Important**: Agents are NOT auto-discovered, must be listed in plugin.json.

## Key Files

- `.claude-plugin/plugin.json` - Plugin manifest
  - Required: `name`, `version`, `description`
  - `agents` field: **Explicitly list agent files** (not auto-discovered)
  - `skills`, `hooks`: Auto-discovered, don't specify
- `.claude-plugin/marketplace.json` - Marketplace configuration
- `hooks/hooks.json` - Event-driven automation (auto-loaded)
- `pyproject.toml` - Development dependencies (not for plugin users)
- `uv.lock` - Locked dependencies (must commit for CI)
- `ruff.toml` - Linting configuration with per-file ignores
- `.pre-commit-config.yaml` - Pre-commit hooks

## File Naming Conventions

- **Skills**: Directory name in kebab-case with `ml-*` or `tool-*` prefix
  - `ml-train/`, `ml-debug/`, `tool-pixi/`
- **Agents**: File name in kebab-case with descriptive name
  - `training-debugger.md`, `ml-architect.md`

Avoid names conflicting with Claude Code built-ins: `/init`, `/config`, `/export`, `/clear`, `/help`, etc.

## Frontmatter Reference Quick Guide

### Skills

```yaml
name: optional-name              # Defaults to directory name
description: recommended         # Helps Claude decide when to load
argument-hint: [args]            # Optional UI hint
disable-model-invocation: true   # Prevent auto-invoke
user-invocable: false            # Hide from menu
allowed-tools: Read, Grep        # Restrict tools
context: fork                    # Run in subagent
agent: Explore                   # Subagent type
hooks: {...}                     # Lifecycle hooks
```

### Agents

```yaml
name: required-name              # Unique identifier
description: required            # When to delegate
tools: [Read, Grep]              # Allowed tools
disallowedTools: [Write]         # Denied tools
model: sonnet                    # sonnet, opus, haiku, inherit
permissionMode: default          # Permission handling
skills: [skill1, skill2]         # Preload skills
hooks: {...}                     # Lifecycle hooks
memory: user                     # Persistent memory
color: blue                      # UI color (required)
```

## Common Patterns

### Skill with Supporting Files

```text
skills/ml-train/
├── SKILL.md (keep <500 lines, reference other files)
├── templates/
│   ├── basic-training.yaml
│   └── distributed-training.yaml
├── examples/
│   └── image-classification.md
└── reference.md (detailed docs loaded when needed)
```

### Skill Running in Subagent

```yaml
---
name: deep-research
context: fork
agent: Explore
---

Research $ARGUMENTS using read-only tools...
```

### Agent with Persistent Memory

```yaml
---
name: code-reviewer
memory: user
model: sonnet
---

Review code and update your memory with patterns you discover...
```

### Agent with Tool Restrictions

```yaml
---
name: safe-explorer
tools: Read, Grep, Glob
disallowedTools: Write, Edit, Bash
---

Explore codebase safely without making changes...
```

## Common Pitfalls

1. **Skills vs Agents for model control**: Use agents (not skills) when you need to specify a model
2. **Agent discovery**: Agents must be listed in plugin.json, skills are auto-discovered
3. **Skill size**: Keep SKILL.md under 500 lines, use supporting files for details
4. **Color for agents**: Use keyword colors (`blue`, `red`), not hex codes
5. **Auto-discovery**: Only skills and hooks auto-discover, agents don't
6. **Supporting files**: Reference them in SKILL.md so Claude knows when to load
7. **Memory scope**: Choose `user` for cross-project, `project` for codebase-specific
8. **Model field**: Only works in agents, not reliably in skills
9. **Invocation control**: Use `disable-model-invocation` for user-only skills
10. **Subagent execution**: Use `context: fork` in skills, or create dedicated agents

## Testing Checklist

Before committing:

- [ ] Run `uv run ruff check --fix .`
- [ ] Run `uv run ruff format .`
- [ ] Run `uv run ty check scripts/`
- [ ] Run `uv run python scripts/validate_plugin.py`
- [ ] Run `npx markdownlint-cli2 "**/*.md"`
- [ ] All pre-commit hooks pass
- [ ] Skills invokable via `/skill-name`
- [ ] Agents listed in plugin.json

## Documentation References

Always refer to official Claude Code documentation for authoritative information:

- **Skills**: <https://code.claude.com/docs/ja/skills>
- **Sub-agents**: <https://code.claude.com/docs/ja/sub-agents>
- **Hooks**: <https://code.claude.com/docs/ja/hooks>
- **Plugins**: <https://code.claude.com/docs/ja/plugins>
- **Agent Skills Standard**: <https://agentskills.io>
