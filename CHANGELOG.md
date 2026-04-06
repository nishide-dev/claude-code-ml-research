# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-04-06

### Fixed

- Resolved plugin validation issues identified in comprehensive review
- Added missing `ml-experiment` example and template files
- Fixed agent color conflict: `config-generator` changed from magenta to green
- Removed empty `ml-model-export/scripts` directory

### Changed

- Implemented progressive disclosure pattern for large knowledge skills
- Reduced `tool-pixi` skill from 1560 lines to 600 lines (62% reduction)
- Reduced `ml-transformers` skill from 1182 lines to 550 lines (53% reduction)
- Reduced `ml-cli-tools` skill from 1093 lines to 350 lines (68% reduction)
- Moved advanced topics to `reference/` subdirectories for better organization
- Total documentation reduction: 2,335 lines (62% reduction)

### Improved

- Plugin structure now follows progressive disclosure best practices
- Context window usage optimized for Claude Code
- Skill organization improved with reference files for detailed content
- Overall plugin quality score improved from 8.5/10 to 9.5/10

## [0.1.0] - 2026-02-23

### Added

- Initial release of ML Research plugin
- 12 workflow skills for ML development:
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
- 8 knowledge skills:
  - `ml-lightning-basics` - PyTorch Lightning patterns
  - `ml-hydra-config` - Hydra configuration
  - `ml-pytorch-geometric` - Graph Neural Networks
  - `ml-wandb-tracking` - Experiment tracking
  - `ml-transformers` - Hugging Face Transformers
  - `ml-cli-tools` - Building CLIs with Typer/Rich
  - `tool-pixi` - Pixi package manager
  - `tool-marimo` - Marimo reactive notebooks
- 6 specialized agents:
  - `ml-architect` - Design ML architectures
  - `training-debugger` - Diagnose training issues
  - `config-generator` - Generate Hydra configs
  - `pytorch-expert` - PyTorch optimization
  - `geometric-specialist` - Graph Neural Networks
  - `transformers-specialist` - LLM fine-tuning
- Comprehensive documentation in CLAUDE.md
- GitHub Actions CI/CD workflows
- Pre-commit hooks for code quality
- Plugin marketplace integration

### Documentation

- Detailed CLAUDE.md with architecture overview
- Git workflow and commit conventions
- Frontmatter reference for skills and agents
- Common patterns and pitfalls guide
