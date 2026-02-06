# ML Coding Standards

These coding standards must be followed when writing machine learning code in this project.

## Deterministic Operations

- **Always set random seeds** for reproducibility:
  - Python random: `random.seed(seed)`
  - NumPy: `np.random.seed(seed)`
  - PyTorch: `torch.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)`
  - PyTorch Lightning: Use `seed_everything()` in `LightningModule`

- **Disable non-deterministic CUDA operations** when reproducibility is required:

  ```python
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```

- **Document when non-determinism is acceptable** (e.g., for production inference speed)

## Type Hints

- **Use type hints for all function signatures**:

  ```python
  def train_model(
      model: nn.Module,
      dataloader: DataLoader,
      optimizer: torch.optim.Optimizer,
      epochs: int
  ) -> dict[str, float]:
      ...
  ```

- **Use modern type syntax** (Python 3.10+):
  - `list[str]` instead of `List[str]`
  - `dict[str, int]` instead of `Dict[str, int]`
  - `tuple[int, ...]` instead of `Tuple[int, ...]`

## Tensor Shape Documentation

- **Document tensor shapes in docstrings** using Einstein notation or explicit descriptions:

  ```python
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      """
      Forward pass through the network.

      Args:
          x: Input tensor of shape (batch_size, channels, height, width)

      Returns:
          Output tensor of shape (batch_size, num_classes)
      """
      ...
  ```

- **Use inline comments for complex transformations**:

  ```python
  # (B, C, H, W) -> (B, C*4, H/2, W/2)
  x = self.downsample(x)
  ```

## PyTorch Lightning Integration

- **Use PyTorch Lightning's built-in logging methods**:

  ```python
  # Good
  self.log("train_loss", loss, on_step=True, on_epoch=True)

  # Bad - direct logging to wandb/tensorboard
  wandb.log({"train_loss": loss})
  ```

- **Log metrics in the correct Lightning hooks**:
  - `training_step`: Log training metrics
  - `validation_step`: Log validation metrics
  - `test_step`: Log test metrics
  - `on_train_epoch_end`: Log epoch-level aggregations

## Gradient Operations

- **Avoid in-place operations that break autograd**:

  ```python
  # Good
  x = x + 1
  x = torch.relu(x)

  # Bad - in-place operations
  x += 1
  x.relu_()
  ```

- **Use `.detach()` when you don't need gradients**:

  ```python
  # Good - detach when logging or computing metrics
  loss_val = loss.detach().item()

  # Bad - keeping computation graph unnecessarily
  loss_val = loss.item()  # Still attached if loss requires grad
  ```

## Code Organization

- **One model per file** in `src/models/`
- **One dataset per file** in `src/data/`
- **One training script** in `src/train.py` or `scripts/train.py`
- **Keep configs in separate files** under `configs/`

## Error Handling

- **Validate tensor shapes early**:

  ```python
  assert x.shape[-1] == expected_dim, f"Expected dim {expected_dim}, got {x.shape[-1]}"
  ```

- **Check for NaN/Inf during training**:

  ```python
  if torch.isnan(loss) or torch.isinf(loss):
      raise ValueError(f"Invalid loss value: {loss.item()}")
  ```

## Performance Optimization

- **Use `torch.no_grad()` for inference**:

  ```python
  @torch.no_grad()
  def evaluate(model, dataloader):
      ...
  ```

- **Move data to device efficiently**:

  ```python
  # Good - single device transfer
  batch = {k: v.to(device) for k, v in batch.items()}

  # Bad - multiple transfers
  inputs = inputs.to(device)
  labels = labels.to(device)
  masks = masks.to(device)
  ```

- **Use mixed precision training** for faster training on modern GPUs:

  ```python
  # In Lightning
  trainer = Trainer(precision="16-mixed")
  ```

## CLI Tool Creation

When building command-line interfaces for ML workflows (training scripts, data preprocessing, model evaluation), use **Typer + Rich** for type-safe, user-friendly CLIs.

### Use Typer for CLI Definition

- **Leverage type hints with `Annotated`** for automatic argument parsing and validation:

  ```python
  import typer
  from typing_extensions import Annotated
  from pathlib import Path

  app = typer.Typer()

  @app.command()
  def train(
      data_path: Annotated[Path, typer.Argument(exists=True, dir_okay=True)],
      epochs: Annotated[int, typer.Option(min=1, max=1000)] = 50,
      lr: Annotated[float, typer.Option("--learning-rate")] = 1e-3,
  ):
      """Train the model with specified parameters."""
      typer.echo(f"Training with {epochs} epochs...")
  ```

- **Use `Enum` for restricted choices**:

  ```python
  from enum import Enum

  class ModelArch(str, Enum):
      resnet50 = "resnet50"
      vit_b_16 = "vit_b_16"
      efficientnet_b0 = "efficientnet_b0"

  @app.command()
  def train(
      model: Annotated[ModelArch, typer.Option()] = ModelArch.resnet50
  ):
      print(f"Training {model.value}")
  ```

- **Organize with subcommands** for complex workflows:

  ```python
  # main.py
  from ml_project.commands import data, train, evaluate

  app = typer.Typer()
  app.add_typer(data.app, name="data")
  app.add_typer(train.app, name="train")
  app.add_typer(evaluate.app, name="eval")
  ```

### Use Rich for Output Formatting

- **Create a shared console** for consistent output:

  ```python
  from rich.console import Console

  console = Console()
  err_console = Console(stderr=True, style="bold red")
  ```

- **Use Progress bars for training loops**:

  ```python
  from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn

  with Progress(
      SpinnerColumn(),
      BarColumn(),
      TaskProgressColumn(),
      console=console
  ) as progress:
      task = progress.add_task("[cyan]Training...", total=epochs)
      for epoch in range(epochs):
          # training logic
          progress.update(task, advance=1)
  ```

- **Display model architectures with Tree**:

  ```python
  from rich.tree import Tree

  def show_model_tree(module: nn.Module) -> Tree:
      tree = Tree(f"[bold blue]{module.__class__.__name__}[/]")
      for name, child in module.named_children():
          branch = tree.add(f"[green]{name}[/]: [yellow]{child.__class__.__name__}[/]")
      return tree

  console.print(show_model_tree(model))
  ```

- **Use RichHandler for logging** to prevent progress bar flicker:

  ```python
  import logging
  from rich.logging import RichHandler

  logging.basicConfig(
      level="INFO",
      format="%(message)s",
      handlers=[RichHandler(console=console)]
  )
  ```

### Integrate with Hydra for Configuration

- **Use Hydra Compose API inside Typer commands** for flexible config management:

  ```python
  from hydra import compose, initialize
  from omegaconf import DictConfig

  def load_config(config_name: str, overrides: list[str]) -> DictConfig:
      with initialize(version_base=None, config_path="../configs"):
          cfg = compose(config_name=config_name, overrides=overrides)
      return cfg

  @app.command()
  def train(
      config: str = "config",
      overrides: list[str] = None,
  ):
      cfg = load_config(config, overrides or [])
      # Use cfg for training
  ```

**See the `ml-cli-tools` skill** for comprehensive patterns and best practices.

## Hugging Face Transformers Integration

When integrating Hugging Face Transformers with PyTorch Lightning, follow these patterns for correctness and reproducibility.

### Always Use save_hyperparameters()

- **Call in `__init__`** to save all hyperparameters to checkpoints:

  ```python
  class TransformerClassifier(pl.LightningModule):
      def __init__(self, model_name_or_path: str, learning_rate: float = 2e-5):
          super().__init__()
          self.save_hyperparameters()  # Critical for reproducibility
          self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
  ```

- **Why**: Enables loading from checkpoint without manually specifying arguments
- **Benefits**: Experiment reproducibility, automatic W&B logging

### Delegate Loss Calculation to HF Models

- **Let HF models compute loss** by passing `labels` argument:

  ```python
  # Good - HF handles loss internally
  def training_step(self, batch, batch_idx):
      outputs = self.model(**batch)  # batch contains labels
      loss = outputs.loss
      return loss

  # Bad - manual loss calculation (prone to errors)
  def training_step(self, batch, batch_idx):
      logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
      loss = F.cross_entropy(logits, batch["labels"])  # Risk of double softmax, wrong ignore_index
      return loss
  ```

- **Prevents**: Double softmax, incorrect ignore_index, task-specific loss errors

### Weight Decay Exclusion Pattern

- **Exclude bias and LayerNorm** from weight decay (standard for transformers):

  ```python
  def configure_optimizers(self):
      no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
      optimizer_grouped_parameters = [
          {
              "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
              "weight_decay": 0.01,
          },
          {
              "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
              "weight_decay": 0.0,
          },
      ]
      optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
      return optimizer
  ```

### Use estimated_stepping_batches for Schedulers

- **Never manually calculate** total training steps:

  ```python
  from transformers import get_linear_schedule_with_warmup

  def configure_optimizers(self):
      optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

      # Good - auto-calculated total steps
      scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=500,
          num_training_steps=self.trainer.estimated_stepping_batches
      )

      return {
          "optimizer": optimizer,
          "lr_scheduler": {
              "scheduler": scheduler,
              "interval": "step",
              "frequency": 1,
          },
      }
  ```

- **Why**: Accounts for gradient accumulation, multi-GPU, and number of epochs

### Enable Gradient Checkpointing for Large Models

- **Use for models >1B parameters** to reduce memory:

  ```python
  def __init__(self, model_name_or_path: str):
      super().__init__()
      self.save_hyperparameters()
      self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

      # Enable gradient checkpointing
      self.model.gradient_checkpointing_enable()
  ```

- **Trade-off**: 30% slower, but 2-3x less memory usage

### Use Dynamic Padding with DataCollatorWithPadding

- **Always use** for variable-length sequences:

  ```python
  from transformers import DataCollatorWithPadding

  def train_dataloader(self):
      return DataLoader(
          self.train_dataset,
          batch_size=32,
          collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer),
      )
  ```

- **Benefits**: 2x speedup on variable-length datasets

**See the `ml-transformers` skill** for comprehensive integration patterns, distributed training strategies, and PEFT (LoRA) implementations.

## Documentation

- **Every model must have a docstring** explaining:
  - Architecture overview
  - Input/output shapes
  - Key hyperparameters
  - Example usage

- **Complex functions need docstrings** with:
  - Args section with types and descriptions
  - Returns section with type and description
  - Example usage if not obvious

## Testing

- **Test model shapes**:

  ```python
  def test_model_output_shape():
      model = MyModel(input_dim=10, output_dim=5)
      x = torch.randn(32, 10)
      out = model(x)
      assert out.shape == (32, 5)
  ```

- **Test training step runs**:

  ```python
  def test_training_step():
      model = MyLightningModule()
      batch = create_dummy_batch()
      loss = model.training_step(batch, 0)
      assert isinstance(loss, torch.Tensor)
      assert not torch.isnan(loss)
  ```
