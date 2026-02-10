---
name: tool-marimo
description: Comprehensive guide for marimo - reactive Python notebooks as pure .py files, uv integration, AI-friendly architecture, reproducible data science workflows, and serverless deployment with WASM
---

# marimo for Reproducible Data Science

## Overview

marimo is a next-generation reactive notebook for Python that solves Jupyter's fundamental problems: hidden state, poor Git integration, and lack of reproducibility. It stores notebooks as pure `.py` files with reactive execution based on a DAG (Directed Acyclic Graph), making it ideal for AI-assisted development and production-ready data science workflows.

**Key Benefits:**

- **Reproducibility**: Reactive execution eliminates hidden state problems
- **Git-friendly**: Pure Python files (.py), not JSON (.ipynb)
- **AI-native**: Optimal format for LLMs (Claude, Copilot, Cursor)
- **Fast environment management**: Deep integration with `uv` package manager
- **Production-ready**: Notebooks are runnable Python scripts
- **Interactive apps**: Deploy as web apps or serverless WASM

**Resources:**

- Official docs: <https://docs.marimo.io>
- GitHub: <https://github.com/marimo-team/marimo>
- Sandboxed notebooks: <https://marimo.io/blog/sandboxed-notebooks>
- WASM deployment: <https://marimo.io/blog/wasm-github-pages>

---

## Core Concepts

### 1. Reactive Execution Model

Unlike Jupyter's imperative execution (run cells in any order), marimo uses **reactive programming**:

**How it works:**

1. **Static analysis**: marimo analyzes each cell to build a dependency graph (DAG)
2. **Automatic re-execution**: When you change a cell, all dependent cells automatically re-run
3. **No hidden state**: Cell execution order always matches code dependencies

**Example:**

```python
# Cell 1: Load data
import polars as pl
df = pl.read_csv("data.csv")

# Cell 2: Process (depends on df)
df_clean = df.drop_nulls()

# Cell 3: Visualize (depends on df_clean)
import altair as alt
chart = alt.Chart(df_clean).mark_line().encode(x='date', y='sales')
```

If you modify Cell 1, marimo automatically re-runs Cells 2 and 3 in the correct order. **No manual re-running needed**.

### 2. Pure Python Files (.py)

marimo notebooks are **pure Python scripts**, not JSON:

```python
import marimo

__generated_with = "0.9.0"
app = marimo.App()

@app.cell
def __(mo):
    mo.md("# Data Analysis")
    return

@app.cell
def __():
    import polars as pl
    df = pl.read_csv("data.csv")
    return df, pl

@app.cell
def __(df, alt):
    chart = alt.Chart(df).mark_bar().encode(x='category', y='sales')
    return chart,
```

**Benefits:**

- **Git-friendly**: Clean diffs, easy code review
- **Tool compatibility**: Works with ruff, mypy, pytest, etc.
- **Reusable**: Import cells as functions from other notebooks
- **AI-friendly**: LLMs understand Python better than JSON

### 3. Strict Variable Scoping

marimo enforces **clean variable management**:

- **No variable redefinition**: Each variable can only be defined in one cell
- **No circular dependencies**: DAG structure prevents cycles
- **Automatic cleanup**: Deleting a cell removes its variables from memory

This eliminates Jupyter's "zombie variables" problem where deleted cells leave variables in memory.

---

## Installation and Setup

### Install marimo

```bash
# With uv (recommended)
uv tool install marimo

# With pip
pip install marimo

# With pipx
pipx install marimo
```

### Create New Notebook

```bash
# Create and edit new notebook
marimo new notebook.py

# Edit existing notebook
marimo edit notebook.py

# Run as app (hides code)
marimo run notebook.py
```

### Convert from Jupyter

```bash
# Convert .ipynb to marimo .py
marimo convert notebook.ipynb -o notebook.py

# Then fix any dependency issues
marimo edit notebook.py
```

---

## Integration with uv

marimo has **deep integration** with the `uv` package manager for sandboxed, reproducible environments.

### PEP 723: Inline Script Metadata

marimo uses **PEP 723** to embed dependencies in the notebook file:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "polars==1.7.1",
#     "altair==5.4.1",
#     "duckdb==1.1.3",
# ]
# ///

import marimo
# ... rest of notebook
```

**Benefits:**

- **Self-contained**: Notebook includes its own dependency specification
- **Shareable**: Send a single `.py` file, receiver gets correct environment
- **Versioned**: Dependencies are locked to specific versions

### Sandboxed Execution

Run notebooks in **isolated environments** with automatic dependency management:

```bash
# Create sandbox environment and run
marimo edit --sandbox notebook.py
```

**What happens:**

1. marimo reads PEP 723 metadata from the file
2. `uv` creates a temporary virtual environment (fast!)
3. Dependencies are installed (10-100x faster than pip)
4. Notebook runs in isolated environment

**Auto-import detection:**

When you write `import pandas`, marimo:

1. Detects the import
2. Adds `pandas` to PEP 723 metadata
3. Installs it in the background with `uv`
4. No manual `pip install` needed!

### Lock File for Reproducibility

Generate lock file for exact reproducibility:

```bash
# Generate uv.lock from PEP 723 metadata
uv pip compile --script notebook.py -o requirements.lock

# Install from lock file
uv pip sync requirements.lock
```

---

## AI-Assisted Development

marimo is **optimized for AI coding** (Claude Code, Cursor, Copilot).

### Why marimo is AI-Friendly

**1. Pure Python format:**

- LLMs are trained on Python, not JSON
- No metadata noise → better token efficiency
- Execution order matches code order → less hallucination

**2. Structured code:**

- Each cell is a function with `@app.cell` decorator
- AI can easily identify and modify specific cells
- Clear dependency graph helps AI understand data flow

**3. File watching:**

- marimo hot-reloads when `.py` file changes
- AI edits file → marimo updates instantly → see results

### Workflow with Claude Code

**Recommended setup:**

```bash
# Terminal 1: Start marimo with file watching
marimo edit --watch notebook.py

# Terminal 2: Use Claude Code to edit
claude
```

**Process:**

1. Ask Claude: "Load data.csv, clean missing values, plot sales trend"
2. Claude edits `notebook.py` (pure Python, easy for LLM)
3. marimo detects changes and hot-reloads
4. View results in browser instantly
5. Iterate with natural language feedback

This is called **"Vibe Coding"** - watch results while AI codes.

### Workflow with Cursor

Cursor excels at refactoring marimo notebooks:

- **Extract function**: "Move data loading to separate cell"
- **Add visualization**: "Create bar chart of top 10 categories"
- **Optimize**: "Use DuckDB instead of pandas for aggregation"

Cursor understands `@app.cell` structure and makes precise edits.

### Built-in AI Assistant

marimo editor has **data-aware AI**:

```python
# In marimo editor, AI knows df's schema!
df  # Press Ctrl+Shift+I to ask AI

# AI sees: df has columns ['date', 'sales', 'category']
# You ask: "Plot monthly sales trend"
# AI generates: alt.Chart(df).encode(x='month(date):T', y='sum(sales):Q')
```

The AI has **access to runtime state** (DataFrame schemas, variable types), so it generates immediately executable code.

---

## Interactive UI Components

marimo provides **reactive UI elements** for building interactive notebooks:

### Basic Inputs

```python
import marimo as mo

# Slider
slider = mo.ui.slider(start=0, stop=100, value=50, label="Threshold")

# Dropdown
dropdown = mo.ui.dropdown(
    options=["A", "B", "C"],
    value="A",
    label="Category"
)

# Text input
text = mo.ui.text(placeholder="Enter query...")

# Date picker
date = mo.ui.date(label="Start Date")
```

### Reactive Binding

UI elements automatically trigger re-execution:

```python
# Cell 1: Create slider
import marimo as mo
threshold = mo.ui.slider(0, 100, value=50)
threshold

# Cell 2: Use slider value (automatically re-runs when slider changes)
import polars as pl
df_filtered = df.filter(pl.col("sales") > threshold.value)
df_filtered
```

**Key point**: `threshold.value` creates a dependency. When slider moves, this cell auto-updates.

### Data Tables

```python
# Interactive table with search, sort, filter
mo.ui.table(df, selection="multi")

# With custom formatters
mo.ui.table(
    df,
    formatters={
        "price": lambda x: f"${x:.2f}",
        "date": lambda x: x.strftime("%Y-%m-%d")
    }
)
```

### Forms

```python
# Group multiple inputs
form = mo.ui.form({
    "model": mo.ui.dropdown(["linear", "tree", "neural"]),
    "epochs": mo.ui.slider(1, 100, value=10),
    "lr": mo.ui.number(start=0.001, stop=0.1, step=0.001, value=0.01)
})

# Access submitted values
if form.value:
    model_type = form.value["model"]
    epochs = form.value["epochs"]
    learning_rate = form.value["lr"]
```

---

## Use Cases

### 1. Exploratory Data Analysis (EDA)

**Motivation**: Jupyter requires manual re-running when changing parameters.

**marimo solution**: Bind query parameters to UI elements.

```python
# Cell 1: UI controls
import marimo as mo
date_range = mo.ui.date_range()
category = mo.ui.dropdown(["Electronics", "Clothing", "Food"])

# Cell 2: Query (auto-updates when UI changes)
import duckdb
query = f"""
SELECT date, SUM(sales) as total
FROM sales
WHERE date BETWEEN '{date_range.value[0]}' AND '{date_range.value[1]}'
  AND category = '{category.value}'
GROUP BY date
"""
df = duckdb.sql(query).pl()

# Cell 3: Visualization (auto-updates)
import altair as alt
chart = alt.Chart(df).mark_line().encode(x='date', y='total')
```

Adjust date range or category → **entire pipeline re-runs instantly**.

### 2. ML Pipeline Development

**Motivation**: Convert experiment notebook to production pipeline without refactoring.

**marimo solution**: Notebooks are parameterized Python scripts.

```python
# notebook.py can be run as CLI:
# python notebook.py -- --epochs 50 --model-type transformer

import marimo as mo
import sys

app = mo.App()

@app.cell
def __():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model-type", default="linear")
    args = parser.parse_args()
    return args,

@app.cell
def __(args):
    # Train model with args.epochs and args.model_type
    model = train_model(epochs=args.epochs, model_type=args.model_type)
    return model,
```

**No rewrite needed** - same file works for:

- Interactive exploration: `marimo edit notebook.py`
- Batch training: `python notebook.py -- --epochs 100`
- Airflow/Dagster: Import and run as module

### 3. Interactive Dashboards

**Motivation**: Streamlit re-runs entire app on interaction (slow for large data).

**marimo solution**: Only re-run changed cells (reactive).

```python
# Build dashboard
import marimo as mo

# Cell 1: Load data once (doesn't re-run on interaction)
import polars as pl
df = pl.read_parquet("large_data.parquet")  # 10GB file

# Cell 2: UI filters (cheap to re-run)
filters = mo.ui.form({
    "region": mo.ui.dropdown(df["region"].unique().to_list()),
    "date": mo.ui.date_range()
})

# Cell 3: Filter data (only re-runs when filters change)
df_filtered = df.filter(
    (pl.col("region") == filters.value["region"]) &
    (pl.col("date").is_between(*filters.value["date"]))
)

# Cell 4: Viz (only re-runs when df_filtered changes)
chart = create_chart(df_filtered)
```

**Deploy as app:**

```bash
marimo run dashboard.py  # Code hidden, UI visible
```

### 4. Serverless WASM Deployment

**Motivation**: Share interactive analysis without server costs.

**marimo solution**: Export to WASM (runs in browser).

```bash
# Export to self-contained HTML with Python runtime
marimo export html-wasm notebook.py -o analysis.html
```

Upload `analysis.html` to GitHub Pages or S3:

- No server required
- Viewers can run Python code in browser (Pyodide)
- Interactive widgets work
- Perfect for: tutorials, demos, research supplements

---

## Advanced Patterns

### Lazy Evaluation

Defer expensive computations until needed:

```python
# Cell 1: Define lazy query
import duckdb
lazy_query = duckdb.sql("SELECT * FROM large_table WHERE ...")

# Cell 2: Only execute when needed
if user_clicked_button:
    result = lazy_query.pl()  # Materialize now
```

### Caching

Cache expensive computations:

```python
@app.cell
def __():
    import functools

    @functools.lru_cache
    def expensive_computation(param):
        # Heavy processing
        return result

    return expensive_computation,
```

### Modular Notebooks

Import cells from other notebooks:

```python
# utils.py (marimo notebook)
@app.cell
def load_data():
    import polars as pl
    return pl.read_csv("data.csv")

# analysis.py (another marimo notebook)
@app.cell
def __():
    from utils import load_data
    df = load_data()
    return df,
```

---

## Integration with ML Tools

### PyTorch Lightning

```python
# Cell 1: Define model
import lightning as L

class Model(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = ...
        self.lr = lr

# Cell 2: Interactive LR tuning
import marimo as mo
lr_slider = mo.ui.slider(0.0001, 0.01, step=0.0001, value=0.001)

# Cell 3: Train (re-runs when LR changes)
model = Model(lr=lr_slider.value)
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, train_loader)
```

### Weights & Biases

```python
# Cell 1: Initialize W&B
import wandb
run = wandb.init(project="marimo-demo")

# Cell 2: Log metrics reactively
wandb.log({"accuracy": model_accuracy, "loss": model_loss})
```

### Hydra Configs

```python
# Cell 1: Load Hydra config
from hydra import compose, initialize
with initialize(config_path="configs"):
    cfg = compose(config_name="config")

# Cell 2: Interactive config override
import marimo as mo
batch_size = mo.ui.slider(16, 128, value=cfg.batch_size)

# Cell 3: Use config
dataloader = DataLoader(dataset, batch_size=batch_size.value)
```

---

## Best Practices

### ✅ DO

1. **One variable = one cell**: Keep variable definitions isolated
2. **Use reactive UI**: Bind parameters to sliders/dropdowns for exploration
3. **Leverage uv sandboxing**: `--sandbox` flag for reproducibility
4. **Git workflows**: Use `.py` format for clean PRs and code review
5. **Modularize**: Extract reusable cells to separate notebooks
6. **AI assistance**: Use Claude/Cursor to edit `.py` files with `--watch` mode
7. **Document**: Use `mo.md()` for explanatory text

### ❌ DON'T

1. **Don't use global mutations**: Avoid modifying global state
2. **Don't use `globals()`**: Breaks static analysis
3. **Don't create circular dependencies**: marimo will error
4. **Don't redefine variables**: One name = one definition
5. **Don't mix notebook + script logic**: Keep notebooks focused

---

## Common Issues and Solutions

### Issue 1: "MultipleDefinitionError"

**Problem**: Same variable defined in multiple cells.

```python
# Cell 1
x = 1

# Cell 2
x = 2  # Error!
```

**Solution**: Use unique names or consolidate logic.

```python
# Cell 1
x_initial = 1

# Cell 2
x_processed = x_initial * 2
```

### Issue 2: Slow re-execution

**Problem**: Large cells re-run too often.

**Solution 1**: Split into smaller cells (only changed parts re-run).

**Solution 2**: Use lazy evaluation or caching.

```python
@functools.lru_cache
def load_large_data():
    return pd.read_parquet("huge.parquet")
```

### Issue 3: UI not updating

**Problem**: Cell doesn't depend on UI element.

**Solution**: Reference `ui_element.value` to create dependency.

```python
# Wrong (no dependency)
slider = mo.ui.slider(0, 100)
result = compute(50)  # Hardcoded value

# Correct (reactive)
slider = mo.ui.slider(0, 100)
result = compute(slider.value)  # Creates dependency
```

---

## Migration from Jupyter

### Step 1: Convert

```bash
marimo convert notebook.ipynb -o notebook.py
```

### Step 2: Fix Dependencies

Open in marimo and resolve errors:

- **Multiple definitions**: Consolidate or rename variables
- **Hidden state**: Explicitly define all variables
- **Out-of-order execution**: marimo enforces DAG order

### Step 3: Add Metadata

Add PEP 723 dependencies:

```python
# /// script
# dependencies = ["pandas", "matplotlib"]
# ///
```

### Step 4: Test

```bash
# Test in sandbox
marimo edit --sandbox notebook.py

# Verify reproducibility
rm -rf .venv
marimo edit --sandbox notebook.py  # Should work fresh
```

---

## Comparison Summary

| Feature | Jupyter | marimo | Advantage |
|---------|---------|--------|-----------|
| Execution | Manual, any order | Automatic, DAG-based | marimo (reproducibility) |
| File format | JSON (.ipynb) | Python (.py) | marimo (Git, AI-friendly) |
| State management | Hidden state | Always synced | marimo (no bugs) |
| Package management | Manual pip/conda | uv integration | marimo (speed, reproducibility) |
| AI coding | Poor (JSON noise) | Excellent (pure Python) | marimo |
| Ecosystem | Massive, mature | Growing | Jupyter |
| Learning curve | Low | Medium | Jupyter |

**Recommendation:**

- **New projects**: Start with marimo for better engineering practices
- **AI-heavy workflows**: marimo's `.py` format is optimal for LLMs
- **Team collaboration**: marimo enables proper code review
- **Production**: marimo notebooks are production-ready scripts
- **Legacy projects**: Jupyter is fine if already working

---

## Resources

### Official Documentation

- **marimo Docs**: <https://docs.marimo.io>
- **API Reference**: <https://docs.marimo.io/api/index.html>
- **GitHub**: <https://github.com/marimo-team/marimo>

### Key Guides

- **Sandboxed Notebooks**: <https://marimo.io/blog/sandboxed-notebooks>
- **WASM Deployment**: <https://marimo.io/blog/wasm-github-pages>
- **Coming from Jupyter**: <https://docs.marimo.io/guides/coming_from/jupyter/>
- **Coming from Streamlit**: <https://docs.marimo.io/guides/coming_from/streamlit/>

### Japanese Resources

- **AI時代のJupyter代替**: <https://zenn.dev/mkj/articles/7c6f38e1b70594>

---

## Quick Reference

**Create notebook:**

```bash
marimo new notebook.py
```

**Edit notebook:**

```bash
marimo edit notebook.py
marimo edit --sandbox notebook.py  # Isolated environment
marimo edit --watch notebook.py    # Hot reload for AI editing
```

**Run as app:**

```bash
marimo run notebook.py  # Hide code, show outputs
```

**Convert from Jupyter:**

```bash
marimo convert notebook.ipynb -o notebook.py
```

**Export:**

```bash
marimo export html notebook.py -o output.html
marimo export html-wasm notebook.py -o app.html  # Serverless
marimo export script notebook.py -o script.py     # Pure Python
```

**Minimal notebook:**

```python
import marimo

app = marimo.App()

@app.cell
def __():
    import marimo as mo
    mo.md("# Hello, marimo!")
    return mo,

@app.cell
def __(mo):
    slider = mo.ui.slider(0, 100, value=50)
    slider
    return slider,

@app.cell
def __(slider):
    f"Value: {slider.value}"
    return
```

---

## Summary

marimo is the future of Python notebooks:

- **Reproducible**: No hidden state, DAG-based execution
- **Production-ready**: Pure Python files, parameterizable scripts
- **AI-native**: Optimal format for LLMs, hot reload for AI editing
- **Fast**: uv integration, reactive execution
- **Shareable**: WASM deployment, Git-friendly

For modern data science workflows that prioritize engineering quality, reproducibility, and AI collaboration, marimo is the superior choice over Jupyter.
