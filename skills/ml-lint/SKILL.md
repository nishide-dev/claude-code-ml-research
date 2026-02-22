---
name: ml-lint
description: Run comprehensive code quality checks with ruff (format, lint) and ty (type checking). Use when checking code quality, fixing linting errors, or ensuring code follows best practices before commits or PRs.
disable-model-invocation: true
---

# Code Quality Checks

Run comprehensive code quality checks including formatting, linting, and type checking.

## Process

### 1. Determine Scope

First, ask the user what to check:

- **All files**: Check entire project
- **Specific directory**: Check only specified directory (e.g., `src/`, `scripts/`)
- **Specific files**: Check only specified files
- **Changed files only**: Check only files modified in git working directory

If not specified, default to checking all Python files in the project.

### 2. Run Ruff Format Check

Check code formatting with ruff:

```bash
# Check all files
uv run ruff format --check .

# Check specific directory
uv run ruff format --check src/

# Check specific files
uv run ruff format --check src/train.py src/models/model.py
```

**Output interpretation:**

- If output shows "would reformat" → Files need formatting
- If no output → All files are properly formatted

### 3. Run Ruff Linter

Check code quality with ruff linter:

```bash
# Check all files
uv run ruff check .

# Check with auto-fix (if user requests)
uv run ruff check --fix .

# Check specific directory
uv run ruff check src/

# Check with specific rules
uv run ruff check --select F,E,W .  # pyflakes, pycodestyle errors/warnings
```

**Output interpretation:**

- Lists violations with file path, line number, rule code, and message
- Exit code 0 → No issues
- Exit code 1 → Issues found

**Common rule categories:**

- `F` - Pyflakes (undefined names, unused imports)
- `E` - Pycodestyle errors (syntax issues, indentation)
- `W` - Pycodestyle warnings (whitespace, line length)
- `I` - Import order (isort compatibility)
- `N` - PEP8 naming conventions
- `UP` - Pyupgrade (modernize Python syntax)

### 4. Run Type Checking (if applicable)

Check type hints with ty:

```bash
# Check all files
uv run ty check .

# Check specific directory
uv run ty check src/

# Check specific files
uv run ty check src/train.py
```

**Output interpretation:**

- Shows type errors with file path, line number, and description
- Exit code 0 → No type errors
- Exit code 1 → Type errors found

**Note**: Only run type checking if:

- Project uses type hints
- `ty` is installed in dependencies
- User requests it

### 5. Summary Report

After running all checks, provide a summary:

```text
Code Quality Report:
✓ Formatting: All files properly formatted (or X files need formatting)
✓ Linting: No issues found (or X issues found)
✓ Type checking: No type errors (or X errors found)

To fix formatting issues: uv run ruff format .
To fix auto-fixable lint issues: uv run ruff check --fix .
```

## Options

### Auto-fix Mode

If user requests auto-fix:

```bash
# Fix formatting
uv run ruff format .

# Fix auto-fixable lint issues
uv run ruff check --fix .
```

After auto-fixing, re-run checks to verify all issues are resolved.

### Strict Mode

For stricter checking:

```bash
# Format check with diff
uv run ruff format --check --diff .

# Lint with all rules
uv run ruff check --select ALL .

# Type check with strict mode (if ty supports it)
uv run ty check --strict .
```

### CI/Pre-commit Mode

For CI or pre-commit checks:

```bash
# Exit on first error, no auto-fix
uv run ruff format --check .
uv run ruff check .
uv run ty check .
```

All commands should fail fast (exit on first error) in CI mode.

## Error Handling

If commands fail:

1. **ruff not found**: Suggest `uv add --dev ruff` or `pixi add ruff`
2. **ty not found**: Suggest `uv add --dev ty` or `pixi add ty`
3. **uv not found**: Suggest installing uv or using alternative package manager

## Best Practices

1. **Run checks before committing**: Catch issues early
2. **Fix formatting first**: Reduces noise in lint output
3. **Address type errors**: Type hints improve code quality
4. **Use auto-fix judiciously**: Review changes before committing
5. **Integrate with pre-commit**: Automate checks on commit

## Related Commands

- `ml-validate`: Comprehensive project validation including code quality
- `ml-setup`: Setup development environment with linting tools
