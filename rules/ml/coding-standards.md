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
