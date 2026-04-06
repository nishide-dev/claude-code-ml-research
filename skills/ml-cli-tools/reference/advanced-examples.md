# ML CLI Advanced Examples and Patterns

Detailed implementation examples for real-world ML CLIs with Typer, Rich, and Hydra integration.

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

---

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

---

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

---

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

---

## Resources

- **Typer**: <https://typer.tiangolo.com/>
- **Rich**: <https://rich.readthedocs.io/>
- **Hydra**: <https://hydra.cc/>
- **Pydantic Settings**: <https://docs.pydantic.dev/latest/concepts/pydantic_settings/>
