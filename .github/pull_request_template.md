# Pull Request

## Description

<!-- Provide a clear and concise description of your changes -->

## Type of Change

<!-- Check all that apply -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Performance improvement
- [ ] Dependency update

## Component Affected

<!-- Check all that apply -->

- [ ] Command (`/train`, `/debug`, `/experiment`, etc.)
- [ ] Agent (ml-architect, training-debugger, transformers-specialist, etc.)
- [ ] Skill (ml-transformers, ml-lightning-basics, ml-cli-tools, etc.)
- [ ] Rule (coding-standards, security-practices, workflow-constraints)
- [ ] Hook (auto-formatting, validation, etc.)
- [ ] Project Template (copier template, scaffolding)
- [ ] Infrastructure (CI, pre-commit, linters)
- [ ] Documentation (README, CLAUDE.md, skill guides)

## Motivation and Context

<!-- Why is this change required? What problem does it solve? -->

Fixes #(issue)

## Changes Made

<!-- Provide a detailed list of changes -->

-
-
-

## Testing Performed

<!-- Describe the tests you ran to verify your changes -->

- [ ] Validated plugin structure: `uv run python scripts/validate_plugin.py`
- [ ] Ran pytest tests: `uv run pytest -v`
- [ ] Ran pre-commit hooks: `uv run pre-commit run --all-files`
- [ ] Tested command invocation in Claude Code
- [ ] Tested agent behavior
- [ ] Verified skill content accuracy
- [ ] Checked markdown linting: `npx markdownlint-cli2 "**/*.md" --config .markdownlint.json`
- [ ] Type checking: `uv run ty check scripts/`
- [ ] Linting: `uv run ruff check .`
- [ ] Formatting: `uv run ruff format --check .`

## ML Framework Compatibility

<!-- If applicable, check frameworks tested with -->

- [ ] PyTorch Lightning
- [ ] Hugging Face Transformers
- [ ] PyTorch Geometric
- [ ] Hydra
- [ ] Weights & Biases
- [ ] Not framework-specific

## Documentation Updated

<!-- Check all that apply -->

- [ ] Updated command frontmatter (name, description)
- [ ] Updated agent frontmatter (name, description, tools, model, color)
- [ ] Updated skill frontmatter (name, description)
- [ ] Updated README.md
- [ ] Updated CLAUDE.md
- [ ] Updated plugin.json
- [ ] Added/updated code examples
- [ ] Added/updated docstrings

## Breaking Changes

<!-- If this PR introduces breaking changes, describe them here -->

- [ ] No breaking changes
- [ ] Changes command interface or arguments
- [ ] Changes agent behavior or tools
- [ ] Changes skill structure
- [ ] Changes rule enforcement
- [ ] Changes hook behavior
- [ ] Requires plugin.json update
- [ ] Requires user action (migration guide below)

### Migration Guide

<!-- If breaking changes, provide migration instructions -->

```markdown
<!-- Example:
Old usage: /train --config config.yaml
New usage: /train config.yaml --epochs 50
-->
```

## Checklist

- [ ] My code follows the coding standards in `rules/ml/coding-standards.md`
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Additional Context

<!-- Add any other context, screenshots, or references about the pull request here -->

## Related Issues and PRs

<!-- Link related issues and pull requests -->

- Related to #
- Depends on #
- Blocks #
