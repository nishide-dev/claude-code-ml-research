---
name: ml-cli-tools
description: Building professional CLIs with Typer and Rich - type-safe argument parsing, progress bars, model visualization, Hydra integration, RichHandler logging, and multi-process handling for ML workflows
---

# ML CLI Tools with Typer and Rich

Building professional command-line interfaces for machine learning workflows using Typer and Rich.

## Overview

This skill covers modern CLI development for ML projects using:

- **Typer**: Type-safe CLI framework leveraging Python type hints
- **Rich**: Terminal UI library for beautiful progress bars, tables, and formatting
- **Integration**: Combining with Hydra, PyTorch Lightning, and logging systems

## Why Typer + Rich for ML?

Traditional `argparse` becomes unmaintainable for complex ML workflows with numerous hyperparameters. Modern alternatives provide:

### Typer Benefits

- **Type Safety**: Automatic validation based on type hints
- **Less Boilerplate**: No manual parser configuration
- **Auto-completion**: Shell completion for Bash, Zsh, Fish, PowerShell
- **Self-documenting**: Help text generated from docstrings and type hints
- **Subcommands**: Easy organization of complex workflows

### Rich Benefits

- **Progress Visualization**: Real-time training progress with custom metrics
- **Structured Output**: Tables, trees, panels for model architectures
- **Logging Integration**: Prevents progress bar corruption from log messages
- **User Experience**: Professional terminal output improves usability

---

## Installation

```bash
# Using UV
uv add "typer[all]" rich hydra-core pydantic

# Using Pixi
pixi add --pypi typer[all] rich hydra-core pydantic
```

---

## Typer Fundamentals

### Basic CLI with Type Hints

```python
import typer
from typing_extensions import Annotated
from pathlib import Path

app = typer.Typer()

@app.command()
def train(
    data_path: Annotated[Path, typer.Argument(
        exists=True,
        dir_okay=True,
        help="Path to training data directory"
    )],
    epochs: Annotated[int, typer.Option(
        min=1,
        max=1000,
        help="Number of training epochs"
    )] = 50,
    lr: Annotated[float, typer.Option(
        "--learning-rate",
        "-lr",
        help="Learning rate"
    )] = 1e-3,
):
    """
    Train the ML model with specified parameters.

    Example:
        python main.py train ./data --epochs 100 --learning-rate 0.001
    """
    typer.echo(f"Training with {epochs} epochs at LR {lr}")

if __name__ == "__main__":
    app()
```

**What Typer provides automatically**:

- Type validation: `epochs` must be int between 1-1000
- Path validation: `data_path` must exist and be a directory
- Help generation: `--help` flag shows formatted documentation
- Shell completion: Tab completion in terminal

### Using Enum for Choices

Restrict inputs to valid options using `Enum`:

```python
from enum import Enum

class ModelArchitecture(str, Enum):
    """Supported model architectures."""
    resnet50 = "resnet50"
    vit_b_16 = "vit_b_16"
    efficientnet_b0 = "efficientnet_b0"

@app.command()
def train(
    model: Annotated[ModelArchitecture, typer.Option(
        case_sensitive=False,
        help="Model architecture to train"
    )] = ModelArchitecture.resnet50,
):
    """Train with specified model."""
    typer.echo(f"Training {model.value}")
```

**Benefits**:

- IDE autocomplete for valid values
- Type-safe: No typos like "renet50"
- Help text shows all valid options

### Subcommands for Complex Workflows

```python
# main.py
import typer
from ml_cli.commands import data, train, evaluate

app = typer.Typer(name="ml-cli", help="ML Research CLI Tool")

app.add_typer(data.app, name="data", help="Data preprocessing")
app.add_typer(train.app, name="train", help="Model training")
app.add_typer(evaluate.app, name="eval", help="Model evaluation")

if __name__ == "__main__":
    app()
```

**Usage**:

```bash
python main.py data preprocess
python main.py train start --epochs 100
python main.py eval metrics --checkpoint best.ckpt
```

---

## Rich: Beautiful Terminal Output

### Progress Bars for Training

```python
from rich.progress import Progress

with Progress() as progress:
    task = progress.add_task("[cyan]Training...", total=100)

    for epoch in range(100):
        # Training logic
        progress.update(task, advance=1)
```

### Custom Progress with Training Metrics

```python
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

class MetricsColumn(TextColumn):
    """Custom column to display training metrics."""

    def render(self, task):
        loss = task.fields.get("loss", 0.0)
        acc = task.fields.get("acc", 0.0)
        return f"Loss: {loss:.4f} | Acc: {acc:.2f}%"

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
    MetricsColumn(),
) as progress:
    task = progress.add_task("[cyan]Training...", total=num_epochs)

    for epoch in range(num_epochs):
        loss, acc = train_epoch(model, dataloader)

        progress.update(
            task,
            advance=1,
            loss=loss,
            acc=acc,
        )
```

### Tables for Results

```python
from rich.table import Table
from rich.console import Console

console = Console()

def show_results(results: list[dict]):
    """Display experiment results in a table."""
    table = Table(title="Experiment Results")

    table.add_column("Experiment", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("Loss", justify="right")

    for result in results:
        table.add_row(
            result["name"],
            f"{result['accuracy']:.2f}%",
            f"{result['loss']:.4f}",
        )

    console.print(table)
```

### Model Architecture Tree

```python
from rich.tree import Tree
import torch.nn as nn

def build_model_tree(module: nn.Module, tree: Tree | None = None) -> Tree:
    """Recursively build a Rich tree from PyTorch module."""
    if tree is None:
        tree = Tree("[bold blue]Model[/]")

    for name, child in module.named_children():
        class_name = child.__class__.__name__
        branch = tree.add(f"[green]{name}[/]: [yellow]{class_name}[/]")
        build_model_tree(child, branch)

    return tree

# Usage
model = torchvision.models.resnet18()
tree = build_model_tree(model)
console.print(tree)
```

---

## Logging Integration

### RichHandler for Clean Logs

Use Rich's logging handler to prevent progress bar corruption:

```python
import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(console=console)],
)

logger = logging.getLogger("ml_cli")

# Logs work cleanly with progress bars
with Progress(console=console) as progress:
    task = progress.add_task("Training", total=100)

    for i in range(100):
        logger.info(f"Epoch {i+1} completed")  # Won't corrupt progress
        progress.update(task, advance=1)
```

---

## Project Structure

Recommended structure for ML CLIs:

```text
ml-cli-tool/
├── pyproject.toml       # Dependencies and metadata
├── src/
│   └── ml_cli/
│       ├── __init__.py
│       ├── main.py      # CLI entry point
│       ├── commands/    # Subcommands
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   └── data.py
│       ├── core/        # Core logic
│       └── utils/       # Rich utilities
├── configs/             # Hydra configs
└── tests/
```

---

## Complete Example

```python
import typer
from typing_extensions import Annotated
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, BarColumn
from rich.console import Console
import logging
from rich.logging import RichHandler

app = typer.Typer()
console = Console()

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger(__name__)

@app.command()
def train(
    data_path: Annotated[Path, typer.Argument(exists=True)],
    epochs: Annotated[int, typer.Option(min=1)] = 50,
    lr: Annotated[float, typer.Option("--learning-rate")] = 1e-3,
):
    """
    Train ML model.

    Example:
        python main.py train ./data --epochs 100 --learning-rate 0.001
    """
    logger.info(f"Starting training with LR={lr}")

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Training...", total=epochs)

        for epoch in range(epochs):
            # Training logic
            loss = train_epoch(data_path, lr)
            logger.info(f"Epoch {epoch+1}: Loss={loss:.4f}")
            progress.update(task, advance=1)

    console.print("[bold green]Training completed![/]")

if __name__ == "__main__":
    app()
```

---

## Advanced Topics

For detailed guides on advanced features, see [reference/advanced-examples.md](reference/advanced-examples.md):

- **Hydra Integration**: Combining Typer with Hydra Compose API
- **Multi-Process Logging**: QueueHandler pattern for DataLoader workers
- **Complete Examples**: Training scripts, data preprocessing, model evaluation
- **Best Practices**: CLI design, error handling, testing

---

## Summary

**Key Takeaways**:

- Typer + Rich = professional ML CLIs with minimal boilerplate
- Type hints drive automatic validation and help generation
- Rich provides beautiful progress bars and structured output
- RichHandler prevents logging from corrupting progress bars

**When to Use**:

- ✅ Training scripts with many hyperparameters
- ✅ Data preprocessing pipelines
- ✅ Model evaluation and comparison tools
- ✅ Experiment management CLIs
- ✅ Any ML workflow that benefits from rich feedback

**Resources**:

- **Typer**: <https://typer.tiangolo.com/>
- **Rich**: <https://rich.readthedocs.io/>
- **Hydra**: <https://hydra.cc/>
