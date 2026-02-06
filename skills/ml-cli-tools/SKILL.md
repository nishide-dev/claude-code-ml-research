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

## Project Structure

Use the `src` layout for proper Python packaging:

```text
ml-cli-tool/
├── pyproject.toml       # UV: Dependencies and project metadata
├── pixi.toml            # Pixi: Alternative with conda + PyPI
├── uv.lock / pixi.lock  # Locked dependencies
├── src/
│   └── ml_cli/
│       ├── __init__.py
│       ├── main.py      # CLI entry point
│       ├── commands/    # Subcommands (train, eval, data)
│       │   ├── __init__.py
│       │   ├── train.py
│       │   ├── predict.py
│       │   └── data.py
│       ├── core/        # Core logic
│       │   ├── config.py
│       │   └── logging.py
│       └── utils/       # Rich utilities
│           └── display.py
├── configs/             # Hydra configurations
│   ├── config.yaml
│   └── model/
└── tests/
```

### Package Manager Setup

**Using UV (recommended for CPU projects)**:

```bash
uv init --package ml-cli-tool
cd ml-cli-tool
uv add "typer[all]" rich hydra-core pydantic torch
```

**Using Pixi (recommended for GPU projects)**:

```bash
pixi init ml-cli-tool
cd ml-cli-tool
# Add Python from conda
pixi add python=3.11
# Add ML packages from PyPI (faster, latest versions)
pixi add --pypi typer[all] rich hydra-core pydantic torch
```

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
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
):
    """
    Train the ML model with specified parameters.

    Example:
        python main.py train ./data --epochs 100 --learning-rate 0.001
    """
    typer.echo(f"Training with {epochs} epochs at LR {lr}")
    # Training logic here

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

class Optimizer(str, Enum):
    """Supported optimizers."""
    adam = "adam"
    adamw = "adamw"
    sgd = "sgd"

@app.command()
def train(
    model: Annotated[ModelArchitecture, typer.Option(
        case_sensitive=False,
        help="Model architecture to train"
    )] = ModelArchitecture.resnet50,
    optimizer: Annotated[Optimizer, typer.Option(
        case_sensitive=False
    )] = Optimizer.adamw,
):
    """Train with specified model and optimizer."""
    typer.echo(f"Training {model.value} with {optimizer.value}")
    # Access enum value: model.value, optimizer.value
```

**Benefits**:

- IDE autocomplete for valid values
- Type-safe: No typos like "renet50" or "aadm"
- Help text shows all valid options
- Case-insensitive matching

### Subcommands for Complex Workflows

Organize related commands into groups:

```python
# src/ml_cli/main.py
import typer
from ml_cli.commands import data, train, evaluate

app = typer.Typer(
    name="ml-cli",
    help="ML Research CLI Tool",
    add_completion=True,
)

# Add subcommand groups
app.add_typer(data.app, name="data", help="Data preprocessing and management")
app.add_typer(train.app, name="train", help="Model training commands")
app.add_typer(evaluate.app, name="eval", help="Model evaluation and testing")

if __name__ == "__main__":
    app()
```

```python
# src/ml_cli/commands/data.py
import typer

app = typer.Typer()

@app.command("preprocess")
def preprocess_data():
    """Preprocess raw data for training."""
    typer.echo("Preprocessing data...")

@app.command("augment")
def augment_data():
    """Apply data augmentation."""
    typer.echo("Augmenting data...")
```

**Usage**:

```bash
python main.py data preprocess
python main.py train start --epochs 100
python main.py eval metrics --checkpoint best.ckpt
```

### Optional and Required Arguments

```python
@app.command()
def train(
    # Required positional argument
    config_name: Annotated[str, typer.Argument(help="Hydra config name")],

    # Optional with default
    epochs: Annotated[int, typer.Option()] = 50,

    # Optional without default (can be None)
    checkpoint: Annotated[Path | None, typer.Option(
        help="Resume from checkpoint"
    )] = None,

    # Flag (boolean)
    debug: Annotated[bool, typer.Option("--debug")] = False,

    # Multiple values
    overrides: Annotated[list[str], typer.Option(
        "--override",
        "-o",
        help="Hydra config overrides"
    )] = None,
):
    """Train with flexible options."""
    if overrides is None:
        overrides = []

    if checkpoint:
        typer.echo(f"Resuming from {checkpoint}")

    if debug:
        typer.echo("Debug mode enabled")
```

## Rich: Beautiful Terminal Output

### Console and Error Output

Create shared console objects for consistent formatting:

```python
from rich.console import Console

# Standard output
console = Console()

# Error output (stderr) with styling
err_console = Console(stderr=True, style="bold red")

# Usage
console.print("[bold blue]Training started[/]")
err_console.print("[bold red]Error:[/] Invalid configuration")
```

**Why separate consoles**:

- Progress bars and logs don't mix when redirecting output
- Errors always visible even when stdout redirected to file
- Different styling for different message types

### Progress Bars for Training

Basic progress bar:

```python
from rich.progress import Progress

with Progress() as progress:
    task = progress.add_task("[cyan]Training...", total=100)

    for epoch in range(100):
        # Training logic
        progress.update(task, advance=1)
```

### Custom Progress Columns with Metrics

Display live training metrics alongside progress:

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
        lr = task.fields.get("lr", 0.0)
        return f"Loss: {loss:.4f} | Acc: {acc:.2f}% | LR: {lr:.2e}"

# Create progress with custom columns
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
    MetricsColumn(),
    console=console,
) as progress:
    task = progress.add_task(
        "[cyan]Training...",
        total=num_epochs,
        loss=0.0,
        acc=0.0,
        lr=learning_rate,
    )

    for epoch in range(num_epochs):
        # Training step
        loss, acc = train_epoch(model, dataloader)

        # Update progress with metrics
        progress.update(
            task,
            advance=1,
            loss=loss,
            acc=acc,
            lr=optimizer.param_groups[0]["lr"],
        )
```

### Model Architecture Visualization

Display PyTorch model structure as a tree:

```python
from rich.tree import Tree
import torch.nn as nn

def build_model_tree(module: nn.Module, tree: Tree | None = None, name: str = "Model") -> Tree:
    """Recursively build a Rich tree from PyTorch module."""
    if tree is None:
        tree = Tree(f"[bold blue]{name}[/]")

    for child_name, child_module in module.named_children():
        # Format module info
        class_name = child_module.__class__.__name__
        info = f"[green]{child_name}[/]: [yellow]{class_name}[/]"

        # Add parameter count for layers
        if isinstance(child_module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            num_params = sum(p.numel() for p in child_module.parameters())
            info += f" [cyan]({num_params:,} params)[/]"

        # Add branch and recurse
        branch = tree.add(info)
        build_model_tree(child_module, branch, child_name)

    return tree

# Usage
import torchvision
model = torchvision.models.resnet18()
tree = build_model_tree(model, name="ResNet18")
console.print(tree)
```

**Output**:

```text
ResNet18
├── conv1: Conv2d (9,408 params)
├── bn1: BatchNorm2d (128 params)
├── relu: ReLU
├── maxpool: MaxPool2d
├── layer1: Sequential
│   ├── 0: BasicBlock
│   │   ├── conv1: Conv2d (36,864 params)
│   │   ├── bn1: BatchNorm2d (128 params)
│   │   └── ...
```

### Tables for Results

Display experiment results in formatted tables:

```python
from rich.table import Table

def show_experiment_results(results: list[dict]):
    """Display experiment results in a table."""
    table = Table(title="Experiment Results", show_header=True, header_style="bold magenta")

    table.add_column("Experiment", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("Epochs", justify="right")
    table.add_column("LR", justify="right")

    for result in results:
        table.add_row(
            result["name"],
            f"{result['accuracy']:.2f}%",
            f"{result['loss']:.4f}",
            str(result["epochs"]),
            f"{result['lr']:.2e}",
        )

    console.print(table)

# Usage
results = [
    {"name": "baseline", "accuracy": 85.3, "loss": 0.412, "epochs": 50, "lr": 1e-3},
    {"name": "augmented", "accuracy": 87.1, "loss": 0.389, "epochs": 50, "lr": 1e-3},
    {"name": "tuned", "accuracy": 89.5, "loss": 0.341, "epochs": 75, "lr": 5e-4},
]
show_experiment_results(results)
```

### Live Dashboard

Create a real-time dashboard with multiple panels:

```python
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel

def create_training_dashboard():
    """Create a live training dashboard."""
    layout = Layout()

    # Split into header and body
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
    )

    # Split body into left and right
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )

    return layout

# Usage with Live
layout = create_training_dashboard()

with Live(layout, refresh_per_second=4, console=console):
    for epoch in range(num_epochs):
        # Update header
        layout["header"].update(
            Panel(f"[bold blue]Training Epoch {epoch+1}/{num_epochs}[/]")
        )

        # Update left panel with metrics
        metrics_table = create_metrics_table(current_metrics)
        layout["left"].update(Panel(metrics_table, title="Metrics"))

        # Update right panel with logs
        logs = get_recent_logs()
        layout["right"].update(Panel(logs, title="Logs"))

        # Training step
        train_epoch(model, dataloader)
```

## Hydra Integration with Typer

### Problem: Hydra vs Typer Conflict

Both Hydra (`@hydra.main`) and Typer control the application entry point. Using both decorators causes conflicts.

### Solution: Hydra Compose API

Use Hydra's `compose()` function inside Typer commands:

```python
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import typer

app = typer.Typer()

def load_hydra_config(config_name: str, overrides: list[str]) -> DictConfig:
    """Load and compose Hydra configuration."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg

@app.command()
def train(
    config: Annotated[str, typer.Option(help="Hydra config name")] = "config",
    overrides: Annotated[list[str], typer.Option(
        "--override", "-o",
        help="Hydra config overrides (e.g., model.lr=0.01)"
    )] = None,
    show_config: Annotated[bool, typer.Option(
        "--show-config",
        help="Print config and exit"
    )] = False,
):
    """
    Train model with Hydra configuration.

    Example:
        python main.py train --config baseline -o model.lr=0.01 -o data.batch_size=64
    """
    if overrides is None:
        overrides = []

    # Load Hydra config
    cfg = load_hydra_config(config, overrides)

    if show_config:
        console.print(OmegaConf.to_yaml(cfg))
        raise typer.Exit()

    # Run training
    run_training(cfg)
```

**Benefits**:

- Typer handles CLI interface (help, validation, completion)
- Hydra handles configuration management (composition, overrides)
- Full flexibility: change configs from command line
- Type-safe CLI arguments + powerful config system

### Combining with Pydantic Settings

For secrets and environment variables, combine Pydantic Settings with Hydra:

```python
# src/ml_cli/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    """Application settings from environment variables."""

    # API Keys (from .env)
    wandb_api_key: str
    hf_token: str | None = None

    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

# Global settings instance
settings = Settings()
```

```python
# In your training command
from ml_cli.core.config import settings

@app.command()
def train(config: str = "config"):
    # Load Hydra config for experiment settings
    cfg = load_hydra_config(config, [])

    # Use Pydantic settings for secrets
    import wandb
    wandb.init(
        project=cfg.experiment.project,
        api_key=settings.wandb_api_key,  # From .env
    )

    # Training logic
    run_training(cfg, output_dir=settings.output_dir)
```

## Logging Integration

### Problem: Logs Corrupt Progress Bars

Standard `print()` and `logging` statements can break Rich progress bar rendering.

### Solution: RichHandler

Use Rich's logging handler for proper integration:

```python
import logging
from rich.logging import RichHandler

# Shared console for progress and logs
console = Console()

# Configure logging with RichHandler
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)

logger = logging.getLogger("ml_cli")

# Now logs and progress bars work together
with Progress(console=console) as progress:
    task = progress.add_task("Training", total=100)

    for i in range(100):
        logger.info(f"Epoch {i+1} completed")  # Won't corrupt progress bar
        progress.update(task, advance=1)
```

### Multi-Process Logging (DataLoader Workers)

PyTorch `DataLoader` with `num_workers > 0` creates child processes. Logs from workers can break Rich layouts.

#### Solution: QueueHandler Pattern

```python
import logging
from logging.handlers import QueueHandler, QueueListener
import multiprocessing as mp
from rich.logging import RichHandler

def setup_logging(queue: mp.Queue):
    """Setup logging for worker processes."""
    handler = QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

def worker_init_fn(queue: mp.Queue):
    """Initialize logging in DataLoader workers."""
    setup_logging(queue)

# Main process
if __name__ == "__main__":
    # Create queue for log records
    log_queue = mp.Queue()

    # Create listener in main process
    rich_handler = RichHandler(console=console)
    listener = QueueListener(log_queue, rich_handler)
    listener.start()

    # Create DataLoader with worker logging
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        worker_init_fn=lambda worker_id: worker_init_fn(log_queue),
    )

    # Training loop
    with Progress(console=console) as progress:
        task = progress.add_task("Training", total=len(dataloader))

        for batch in dataloader:
            # Worker logs are properly handled
            process_batch(batch)
            progress.update(task, advance=1)

    # Cleanup
    listener.stop()
```

## Real-World ML CLI Examples

### Complete Training Script

```python
# src/ml_cli/commands/train.py
import typer
from typing_extensions import Annotated
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from hydra import compose, initialize
import torch
import logging
from rich.logging import RichHandler

app = typer.Typer()
console = Console()

# Setup logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger(__name__)

def load_config(config_name: str, overrides: list[str]):
    """Load Hydra configuration."""
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg

@app.command()
def start(
    config: Annotated[str, typer.Option(help="Config name")] = "config",
    overrides: Annotated[list[str], typer.Option("--override", "-o")] = None,
    checkpoint: Annotated[Path | None, typer.Option(help="Resume from checkpoint")] = None,
    debug: Annotated[bool, typer.Option(help="Fast dev run")] = False,
):
    """
    Start model training.

    Example:
        python main.py train start --config baseline -o model.lr=0.001
    """
    if overrides is None:
        overrides = []

    # Load configuration
    logger.info("Loading configuration...")
    cfg = load_config(config, overrides)

    if debug:
        cfg.trainer.fast_dev_run = True
        logger.warning("Debug mode: fast_dev_run enabled")

    # Setup model and data
    logger.info("Initializing model and data...")
    model = create_model(cfg.model)
    datamodule = create_datamodule(cfg.data)

    # Training loop with progress bar
    num_epochs = 1 if debug else cfg.trainer.max_epochs

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Training...", total=num_epochs)

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")

            # Training step
            train_loss = train_epoch(model, datamodule.train_dataloader())
            val_loss = validate_epoch(model, datamodule.val_dataloader())

            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save checkpoint
            if epoch % cfg.checkpoint_interval == 0:
                checkpoint_path = Path(f"checkpoints/epoch_{epoch}.ckpt")
                save_checkpoint(model, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")

            progress.update(task, advance=1)

    console.print("[bold green]Training completed![/]")

if __name__ == "__main__":
    app()
```

### Data Preprocessing Pipeline

```python
# src/ml_cli/commands/data.py
import typer
from typing_extensions import Annotated
from pathlib import Path
from rich.progress import Progress, track
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()

@app.command()
def preprocess(
    input_dir: Annotated[Path, typer.Argument(exists=True, dir_okay=True)],
    output_dir: Annotated[Path, typer.Argument()],
    num_workers: Annotated[int, typer.Option(min=1, max=32)] = 4,
):
    """
    Preprocess raw data for training.

    Processes all files in input_dir and saves results to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all files
    files = list(input_dir.glob("**/*.jpg"))
    console.print(f"Found {len(files)} files to process")

    # Process with progress bar
    processed = []
    for file in track(files, description="Processing images..."):
        result = process_image(file, output_dir)
        processed.append(result)

    # Show summary table
    table = Table(title="Preprocessing Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Files", str(len(files)))
    table.add_row("Processed", str(len(processed)))
    table.add_row("Output Dir", str(output_dir))

    console.print(table)

@app.command()
def split(
    data_dir: Annotated[Path, typer.Argument(exists=True)],
    train_ratio: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.8,
    val_ratio: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.1,
    seed: Annotated[int, typer.Option()] = 42,
):
    """
    Split dataset into train/val/test sets.

    Ratios must sum to <= 1.0. Remaining data goes to test set.
    """
    test_ratio = 1.0 - train_ratio - val_ratio

    if test_ratio < 0:
        console.print("[red]Error: train_ratio + val_ratio must be <= 1.0[/]")
        raise typer.Exit(1)

    console.print(f"Split ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")

    # Perform split
    splits = split_dataset(data_dir, train_ratio, val_ratio, seed)

    # Show results
    table = Table(title="Dataset Split")
    table.add_column("Split", style="cyan")
    table.add_column("Samples", justify="right")
    table.add_column("Percentage", justify="right")

    for split_name, samples in splits.items():
        pct = len(samples) / sum(len(s) for s in splits.values())
        table.add_row(split_name, str(len(samples)), f"{pct:.1%}")

    console.print(table)
```

### Model Evaluation and Comparison

```python
# src/ml_cli/commands/evaluate.py
import typer
from typing_extensions import Annotated
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
import torch

app = typer.Typer()
console = Console()

@app.command()
def metrics(
    checkpoint: Annotated[Path, typer.Argument(exists=True)],
    data_path: Annotated[Path, typer.Argument(exists=True)],
    device: Annotated[str, typer.Option()] = "cuda",
):
    """
    Evaluate model checkpoint on test data.

    Loads checkpoint and computes accuracy, F1, precision, recall.
    """
    console.print(f"[cyan]Loading checkpoint:[/] {checkpoint}")

    # Load model
    model = load_checkpoint(checkpoint)
    model = model.to(device)
    model.eval()

    # Evaluate
    with console.status("[bold green]Evaluating..."):
        results = evaluate_model(model, data_path, device)

    # Display results
    table = Table(title=f"Evaluation Results: {checkpoint.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    for metric_name, value in results.items():
        if isinstance(value, float):
            table.add_row(metric_name, f"{value:.4f}")
        else:
            table.add_row(metric_name, str(value))

    console.print(table)

@app.command()
def compare(
    checkpoints: Annotated[list[Path], typer.Argument()],
    data_path: Annotated[Path, typer.Option(exists=True)] = Path("data/test"),
):
    """
    Compare multiple checkpoints.

    Evaluates each checkpoint and displays comparison table.
    """
    results = {}

    for ckpt in checkpoints:
        console.print(f"[cyan]Evaluating:[/] {ckpt.name}")
        model = load_checkpoint(ckpt)
        metrics = evaluate_model(model, data_path)
        results[ckpt.name] = metrics

    # Comparison table
    table = Table(title="Model Comparison")
    table.add_column("Checkpoint", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1 Score", justify="right")
    table.add_column("Params", justify="right")

    for name, metrics in results.items():
        table.add_row(
            name,
            f"{metrics['accuracy']:.2%}",
            f"{metrics['f1']:.4f}",
            f"{metrics['num_params']:,}",
        )

    console.print(table)

@app.command()
def inspect(
    checkpoint: Annotated[Path, typer.Argument(exists=True)],
):
    """
    Inspect checkpoint structure and metadata.

    Shows model architecture, hyperparameters, and training info.
    """
    ckpt = torch.load(checkpoint, map_location="cpu")

    # Show checkpoint info
    console.print("[bold]Checkpoint Information[/]\n")

    if "epoch" in ckpt:
        console.print(f"Epoch: {ckpt['epoch']}")
    if "global_step" in ckpt:
        console.print(f"Global Step: {ckpt['global_step']}")

    # Model architecture tree
    if "state_dict" in ckpt:
        console.print("\n[bold]Model Architecture[/]\n")
        model = reconstruct_model(ckpt)
        tree = build_model_tree(model)
        console.print(tree)

    # Hyperparameters
    if "hyper_parameters" in ckpt:
        console.print("\n[bold]Hyperparameters[/]\n")
        for key, value in ckpt["hyper_parameters"].items():
            console.print(f"  {key}: {value}")
```

## Best Practices

### CLI Design Principles

1. **Use type hints everywhere**: Enables validation and IDE support
2. **Provide good defaults**: Make simple cases simple
3. **Document with docstrings**: Generates help text automatically
4. **Use Enum for choices**: Prevents typos and improves UX
5. **Organize with subcommands**: Group related functionality
6. **Add examples in help**: Show common usage patterns

### Rich Integration Patterns

1. **Shared console**: Create once, use everywhere
2. **Separate stdout/stderr**: Use different consoles for errors
3. **Progress for long operations**: Any loop > 10 seconds
4. **Tables for results**: Better than printing dicts
5. **Trees for hierarchies**: Models, file structures
6. **RichHandler for logging**: Prevents progress bar corruption

### Hydra + Typer Integration

1. **Typer for CLI, Hydra for configs**: Clear separation of concerns
2. **Compose API inside commands**: Flexible config loading
3. **Pass overrides from CLI**: `--override model.lr=0.01`
4. **Show config option**: Debug configuration issues
5. **Combine with Pydantic**: Use Settings for secrets

### Error Handling

```python
@app.command()
def train(config: str):
    try:
        cfg = load_hydra_config(config, [])
    except Exception as e:
        err_console.print(f"[red]Error loading config:[/] {e}")
        raise typer.Exit(1)

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/]")
        raise typer.Exit(0)
    except Exception as e:
        err_console.print(f"[red]Training failed:[/] {e}")
        raise typer.Exit(1)
```

### Testing CLIs

```python
from typer.testing import CliRunner

runner = CliRunner()

def test_train_command():
    """Test training command with valid inputs."""
    result = runner.invoke(app, ["train", "baseline", "--epochs", "1"])
    assert result.exit_code == 0
    assert "Training completed" in result.output

def test_invalid_config():
    """Test error handling for invalid config."""
    result = runner.invoke(app, ["train", "nonexistent"])
    assert result.exit_code == 1
    assert "Error loading config" in result.output
```

## Summary

**Key Takeaways**:

- Typer + Rich = professional ML CLIs with minimal boilerplate
- Type hints drive automatic validation and help generation
- Rich provides beautiful progress bars and structured output
- Hydra Compose API integrates cleanly with Typer commands
- RichHandler prevents logging from corrupting progress bars
- Proper console separation enables stdout/stderr handling

**When to Use**:

- ✅ Training scripts with many hyperparameters
- ✅ Data preprocessing pipelines
- ✅ Model evaluation and comparison tools
- ✅ Experiment management CLIs
- ✅ Any ML workflow that benefits from rich feedback

**Resources**:

- Typer: <https://typer.tiangolo.com/>
- Rich: <https://rich.readthedocs.io/>
- Hydra: <https://hydra.cc/>
- PyTorch Lightning: <https://lightning.ai/docs/pytorch/>
