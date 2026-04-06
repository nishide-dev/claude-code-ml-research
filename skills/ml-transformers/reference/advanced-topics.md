# Transformers Advanced Topics

Detailed guides for distributed training, PEFT, evaluation, and configuration management.

## Distributed Training Strategies

### DDP (Distributed Data Parallel)

**When to use**: Model fits in single GPU memory

```python
# Single GPU
trainer = pl.Trainer(accelerator="gpu", devices=1)

# Multi-GPU DDP
trainer = pl.Trainer(
    accelerator="gpu",
    devices=4,  # 4 GPUs
    strategy="ddp",  # Data parallel
)
```

**How DDP works**:

- Model replicated on each GPU
- Data split across GPUs
- Gradients synchronized after backward pass
- Efficient for models < 10B parameters

### FSDP (Fully Sharded Data Parallel)

**When to use**: Model too large for single GPU

```python
from lightning.pytorch.strategies import FSDPStrategy

# FSDP with automatic settings
trainer = pl.Trainer(
    accelerator="gpu",
    devices=4,
    strategy="fsdp",
    precision="bf16-mixed",  # BF16 recommended for Ampere+ GPUs
)

# FSDP with custom settings
fsdp_strategy = FSDPStrategy(
    cpu_offload=True,  # Offload params to CPU (slower, more memory)
    # ... other options
)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=4,
    strategy=fsdp_strategy,
)
```

**How FSDP works**:

- **Shards** parameters, gradients, optimizer states across GPUs
- Each GPU only stores 1/N of model (N = num GPUs)
- Gathers parameters during forward/backward, then discards
- Can train models larger than single GPU VRAM

**Memory savings**: ~75% reduction with 4 GPUs

### DeepSpeed Integration

**When to use**: Need extreme optimization or already have DeepSpeed configs

```python
from lightning.pytorch.strategies import DeepSpeedStrategy

# DeepSpeed ZeRO Stage 3 (equivalent to FSDP)
trainer = pl.Trainer(
    accelerator="gpu",
    devices=4,
    strategy="deepspeed_stage_3",
    precision="bf16-mixed",
)

# Custom DeepSpeed config
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
    },
    "bf16": {"enabled": True},
}

trainer = pl.Trainer(
    strategy=DeepSpeedStrategy(config=deepspeed_config),
)
```

**ZeRO Stages**:

- **Stage 1**: Optimizer state sharding (minimal savings)
- **Stage 2**: + Gradient sharding (moderate savings)
- **Stage 3**: + Parameter sharding (maximum savings, equivalent to FSDP)
- **Stage 3 + Offload**: + CPU/NVMe offload (can run 100B+ models on single GPU)

#### FSDP vs DeepSpeed Decision Matrix

| Scenario | Recommendation |
|----------|----------------|
| PyTorch-native solution preferred | FSDP |
| Simplest setup | FSDP |
| Existing DeepSpeed configs | DeepSpeed |
| Need NVMe offload | DeepSpeed Stage 3 Offload |
| Maximum performance tuning | DeepSpeed (custom kernels) |
| Latest PyTorch features | FSDP (better compatibility) |

### Memory Optimization Techniques

#### 1. Gradient Checkpointing

**Recompute activations** during backward instead of storing:

```python
def __init__(self, model_name_or_path: str, ...):
    super().__init__()
    self.save_hyperparameters()

    # Load model
    self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # Enable gradient checkpointing
    self.model.gradient_checkpointing_enable()
```

**Trade-off**:

- ✅ Massive memory savings (can fit 2-3x larger models)
- ❌ ~30% slower training (recomputation cost)

**When to use**: Training large models (>1B parameters) where memory is bottleneck

#### 2. Mixed Precision Training

**BF16** (recommended for Ampere+ GPUs: A100, RTX 30/40 series):

```python
trainer = pl.Trainer(
    precision="bf16-mixed",  # BF16 mixed precision
)
```

**FP16** (older GPUs: V100, RTX 20 series):

```python
trainer = pl.Trainer(
    precision="16-mixed",  # FP16 mixed precision
)
```

**Why BF16 > FP16**:

- Same range as FP32 (no loss scaling needed)
- More stable training
- Standard for LLM training

**Memory savings**: 50% (16-bit vs 32-bit)

---

## PEFT: Parameter-Efficient Fine-Tuning

### LoRA (Low-Rank Adaptation)

**Problem**: Full fine-tuning of LLMs (70B parameters) requires enormous GPU resources.

**Solution**: Freeze base model, train small adapter layers.

#### Mathematical Foundation

LoRA approximates weight updates as low-rank decomposition:

```text
W' = W₀ + ΔW = W₀ + BA
```

Where:

- W₀: Frozen pretrained weights
- A ∈ ℝ^(r×d), B ∈ ℝ^(d×r): Trainable adapters
- r << d: Rank (typically 8, 16, 32)

**Result**: <1% trainable parameters, no inference overhead after merging

#### Implementation

```python
from peft import LoraConfig, get_peft_model, TaskType
import lightning.pytorch as pl
from transformers import AutoModelForCausalLM

class LitLoRAModel(pl.LightningModule):
    """LLM with LoRA adapters."""

    def __init__(
        self,
        base_model_name: str,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load base model (frozen)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,  # Rank
            lora_alpha=lora_alpha,  # Scaling factor
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],  # Which layers to adapt
        )

        # Wrap model with LoRA
        self.model = get_peft_model(base_model, peft_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()
        # Output: trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.062

    def training_step(self, batch, batch_idx):
        """Training step with LoRA."""
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        """Optimizer for LoRA parameters only."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),  # Only LoRA params are trainable
            lr=self.hparams.learning_rate
        )
        return optimizer
```

#### Saving LoRA Adapters

Save **only adapters** (not full model) for efficiency:

```python
# In training script
from lightning.pytorch.callbacks import ModelCheckpoint

# Custom callback to save only adapters
class LoRACheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Save only LoRA weights (a few MB instead of GB)
        pl_module.model.save_pretrained(self.dirpath)

trainer = pl.Trainer(
    callbacks=[LoRACheckpoint(dirpath="lora_adapters")]
)
```

#### Loading for Inference

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "lora_adapters/")

# Merge adapters into base weights (optional, for faster inference)
model = model.merge_and_unload()
```

### QLoRA (Quantized LoRA)

**Further optimization**: Quantize base model to 4-bit, train LoRA on top.

```python
from transformers import BitsAndBytesConfig

class LitQLoRAModel(pl.LightningModule):
    def __init__(self, base_model_name: str, ...):
        super().__init__()
        self.save_hyperparameters()

        # Quantization config (4-bit)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load model in 4-bit
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        # Add LoRA (same as before)
        peft_config = LoraConfig(...)
        self.model = get_peft_model(base_model, peft_config)
```

**Memory savings**: 4x reduction (4-bit vs 16-bit base model)

**Result**: Fine-tune 70B models on single 24GB GPU

---

## Evaluation and Metrics

### TorchMetrics for Distributed Evaluation

**Problem**: Naive metric calculation is incorrect in multi-GPU training.

**Solution**: Use TorchMetrics (automatically syncs across GPUs).

```python
import torchmetrics

class TransformerClassifier(pl.LightningModule):
    def __init__(self, num_classes: int, ...):
        super().__init__()
        # ... model setup ...

        # Initialize metrics
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="macro"
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)

        # Update metric state (accumulate)
        self.train_acc(preds, batch["labels"])

        self.log("train/loss", loss)
        return loss

    def on_train_epoch_end(self):
        """Compute and log metrics at epoch end."""
        # Compute aggregated metric (synced across GPUs)
        acc = self.train_acc.compute()
        self.log("train/acc_epoch", acc)

        # Reset for next epoch
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)

        # Update metrics
        self.val_acc(preds, batch["labels"])
        self.val_f1(preds, batch["labels"])

        self.log("val/loss", loss)

    def on_validation_epoch_end(self):
        """Log validation metrics."""
        self.log("val/acc", self.val_acc.compute())
        self.log("val/f1", self.val_f1.compute())

        # Reset
        self.val_acc.reset()
        self.val_f1.reset()
```

**Why TorchMetrics**:

- Handles distributed sync automatically
- State accumulation across batches
- Correct averaging in multi-GPU scenarios

### Weights & Biases Integration

```python
from lightning.pytorch.loggers import WandbLogger

# Create logger
wandb_logger = WandbLogger(
    project="transformers-lightning",
    name="bert-text-classification",
    log_model=True,  # Log checkpoints to W&B
)

# Pass to trainer
trainer = pl.Trainer(
    logger=wandb_logger,
    log_every_n_steps=50,
)

# Train
trainer.fit(model, datamodule)
```

**Log custom data**:

```python
def on_validation_epoch_end(self):
    """Log custom visualizations."""
    if self.global_rank == 0:  # Only on main process
        # Log confusion matrix
        import wandb
        self.logger.experiment.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=self.val_labels,
                preds=self.val_preds,
                class_names=self.class_names,
            )
        })
```

---

## Configuration Management with Hydra

### Lightning-Hydra-Template Structure

Recommended project structure:

```text
project/
├── configs/
│   ├── config.yaml                 # Main config
│   ├── model/
│   │   ├── bert_base.yaml
│   │   ├── roberta_large.yaml
│   │   └── t5_small.yaml
│   ├── data/
│   │   ├── glue_sst2.yaml
│   │   └── squad.yaml
│   ├── trainer/
│   │   ├── default.yaml
│   │   ├── ddp.yaml
│   │   └── fsdp.yaml
│   └── experiment/
│       ├── bert_sst2.yaml          # Complete experiment config
│       └── t5_squad.yaml
├── src/
│   ├── models/
│   │   └── transformer_module.py
│   ├── data/
│   │   └── transformer_datamodule.py
│   └── train.py
└── logs/
```

### Example Configs

**configs/config.yaml** (main):

```yaml
defaults:
  - model: bert_base
  - data: glue_sst2
  - trainer: default
  - _self_

seed: 42
```

**configs/model/bert_base.yaml**:

```yaml
_target_: src.models.transformer_module.TransformerClassifier

model_name_or_path: bert-base-uncased
num_labels: 2
learning_rate: 2e-5
warmup_steps: 500
weight_decay: 0.01
```

**configs/experiment/bert_sst2_large.yaml** (override):

```yaml
# @package _global_

defaults:
  - override /model: bert_base
  - override /data: glue_sst2
  - override /trainer: ddp

# Overrides
model:
  model_name_or_path: bert-large-uncased
  learning_rate: 1e-5

data:
  batch_size: 16  # Smaller for large model

trainer:
  max_epochs: 5
  devices: 4
```

### Training with Hydra

```python
# src/train.py
import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Instantiate model from config
    model = hydra.utils.instantiate(cfg.model)

    # Instantiate datamodule
    datamodule = hydra.utils.instantiate(cfg.data)

    # Instantiate trainer
    trainer = pl.Trainer(**cfg.trainer)

    # Train
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()
```

**Run experiments**:

```bash
# Default config
python src/train.py

# With experiment
python src/train.py experiment=bert_sst2_large

# With overrides
python src/train.py model.learning_rate=1e-4 data.batch_size=64

# Multirun (sweep)
python src/train.py -m model.learning_rate=1e-5,2e-5,3e-5
```

---

## Troubleshooting and Best Practices

### Common Pitfalls

#### 1. Forgetting `save_hyperparameters()`

**Problem**: Cannot load checkpoint without manually specifying all arguments.

**Solution**: Always call in `__init__`:

```python
def __init__(self, model_name: str, lr: float = 2e-5):
    super().__init__()
    self.save_hyperparameters()  # ← Critical
```

#### 2. Manual Loss Calculation

**Problem**: Double softmax, wrong ignore_index, incompatible with model.

**Solution**: Let HF model compute loss:

```python
# Good
outputs = self.model(input_ids=ids, attention_mask=mask, labels=labels)
loss = outputs.loss

# Bad
logits = self.model(input_ids=ids, attention_mask=mask).logits
loss = F.cross_entropy(logits, labels)  # Prone to errors
```

#### 3. Incorrect Scheduler Steps

**Problem**: Scheduler configured for wrong number of steps.

**Solution**: Use `self.trainer.estimated_stepping_batches`:

```python
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=self.hparams.warmup_steps,
    num_training_steps=self.trainer.estimated_stepping_batches  # Auto-calculated
)
```

#### 4. Not Using TorchMetrics in Distributed Training

**Problem**: Metrics computed incorrectly across GPUs.

**Solution**: Use TorchMetrics for automatic sync:

```python
self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
```

### Performance Optimization

#### Gradient Accumulation

Simulate larger batch sizes:

```python
trainer = pl.Trainer(
    accumulate_grad_batches=4,  # Accumulate gradients over 4 batches
)
```

#### Mixed Precision

Always use BF16/FP16:

```python
trainer = pl.Trainer(
    precision="bf16-mixed",  # Or "16-mixed" for older GPUs
)
```

#### Compiled Models (PyTorch 2.0+)

```python
def configure_model(self):
    if not self.model.training:
        return

    # Compile model for faster training
    self.model = torch.compile(self.model)
```

### Memory Management

#### Gradient Checkpointing

For large models:

```python
self.model.gradient_checkpointing_enable()
```

#### CPU Offloading

For FSDP/DeepSpeed:

```python
fsdp_strategy = FSDPStrategy(
    cpu_offload=True,  # Offload to CPU
)
```

---

## Resources

### Official Documentation

- **Hugging Face Transformers**: <https://huggingface.co/docs/transformers>
- **PyTorch Lightning**: <https://lightning.ai/docs/pytorch>
- **PEFT Library**: <https://huggingface.co/docs/peft>
- **Weights & Biases**: <https://docs.wandb.ai/guides/integrations/lightning>

### Templates & Examples

- **Lightning-Hydra-Template**: <https://github.com/ashleve/lightning-hydra-template>
- **HF Transformers Examples**: <https://github.com/huggingface/transformers/tree/main/examples/pytorch>

### Papers

- **LoRA**: <https://arxiv.org/abs/2106.09685>
- **QLoRA**: <https://arxiv.org/abs/2305.14314>
- **FSDP**: <https://arxiv.org/abs/2304.11277>
