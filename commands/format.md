---
description: Format Python code with ruff and optionally fix linting issues
model: haiku
---

# Code Formatting

Format Python code with ruff formatter and optionally fix auto-fixable linting issues.

## Process

### 1. Determine Scope

Ask the user what to format:

- **All files**: Format entire project (default)
- **Specific directory**: Format only specified directory (e.g., `src/`, `tests/`)
- **Specific files**: Format only specified files
- **Changed files only**: Format only files modified in git working directory

If not specified, format all Python files in the project.

### 2. Run Ruff Format

Format code with ruff:

```bash
# Format all files
uv run ruff format .

# Format specific directory
uv run ruff format src/

# Format specific files
uv run ruff format src/train.py src/models/model.py

# Preview changes without applying (dry-run)
uv run ruff format --check --diff .
```

**Output interpretation:**

- Shows files being formatted
- "X files reformatted, Y files left unchanged"
- Exit code 0 → Success

### 3. Run Ruff Check with Auto-fix (Optional)

After formatting, optionally fix auto-fixable linting issues:

```bash
# Fix all auto-fixable issues
uv run ruff check --fix .

# Fix specific directory
uv run ruff check --fix src/

# Fix specific rules only
uv run ruff check --fix --select F,I .  # Fix imports and undefined names
```

**Output interpretation:**

- Shows fixed violations
- Lists remaining violations that need manual fixes
- "Fixed X violations, Y remaining"

### 4. Verify Results

After formatting and fixing, verify the results:

```bash
# Check formatting is correct
uv run ruff format --check .

# Check remaining lint issues
uv run ruff check .
```

Report any remaining issues that need manual attention.

### 5. Git Status Check

If working in a git repository, show what changed:

```bash
# Show modified files
git status --short

# Show diff of changes (optional, if user wants to review)
git diff
```

Inform user about changes so they can review before committing.

## Options

### Check-only Mode

To preview changes without applying:

```bash
# Show what would be formatted
uv run ruff format --check .

# Show detailed diff
uv run ruff format --check --diff .
```

Use this when user wants to see changes before applying.

### Selective Fixing

To fix only specific types of issues:

```bash
# Fix only import order
uv run ruff check --fix --select I .

# Fix only unused imports
uv run ruff check --fix --select F401 .

# Fix only whitespace issues
uv run ruff check --fix --select W .
```

### Aggressive Formatting

For more aggressive formatting (use with caution):

```bash
# Format with maximum line length
uv run ruff format --line-length 120 .

# Note: This overrides project config, confirm with user first
```

## Git Integration

### Format Changed Files Only

```bash
# Get list of changed Python files
changed_files=$(git diff --name-only --diff-filter=ACMR "*.py")

if [ -n "$changed_files" ]; then
    # Format only changed files
    echo "$changed_files" | xargs uv run ruff format
    echo "$changed_files" | xargs uv run ruff check --fix
else
    echo "No Python files changed"
fi
```

### Pre-commit Integration

Suggest setting up pre-commit hooks:

```bash
# Install pre-commit (if using)
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files
```

## Error Handling

If commands fail:

1. **ruff not found**:
   - Suggest: `uv add --dev ruff`
   - Or: `pixi add ruff`

2. **Syntax errors in code**:
   - ruff format may fail on invalid Python syntax
   - Report the syntax error location
   - Ask user to fix syntax errors first

3. **Permission errors**:
   - Check file permissions
   - Suggest running with appropriate permissions

4. **uv not found**:
   - Suggest installing uv
   - Or use alternative: `python -m ruff format .`

## Best Practices

1. **Format before fixing**: Run formatter first, then linter
2. **Review changes**: Check git diff before committing
3. **Use in CI**: Add formatting checks to CI pipeline
4. **Consistent style**: Let ruff handle all formatting decisions
5. **Incremental adoption**: Format one directory at a time for large projects

## Output Example

```text
Formatting Python files with ruff...
✓ Formatted 15 files, 3 files left unchanged

Fixing auto-fixable lint issues...
✓ Fixed 8 violations
⚠ 2 violations remaining (need manual fix):
  - src/train.py:45:1 F841 Local variable 'x' is assigned but never used
  - src/model.py:12:5 E501 Line too long (95 > 88 characters)

Summary:
✓ Code formatted successfully
✓ 8 issues auto-fixed
⚠ 2 issues need manual attention

To review changes: git diff
To commit changes: git add -A && git commit -m "Format code with ruff"
```

## Related Commands

- `/lint`: Run code quality checks without modifying files
- `/validate`: Comprehensive project validation including code quality
