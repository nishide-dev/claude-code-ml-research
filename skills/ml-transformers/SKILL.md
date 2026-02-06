---
name: ml-transformers
description: Hugging Face Transformers with PyTorch Lightning - LightningModule integration, distributed training (FSDP/DeepSpeed), PEFT (LoRA/QLoRA), data pipelines with HF Datasets, evaluation metrics, and common NLP tasks
---

# Hugging Face Transformers with PyTorch Lightning

Comprehensive guide for integrating Hugging Face Transformers with PyTorch Lightning for scalable NLP and LLM development.

## Overview

This skill covers the integration of two industry-standard libraries:

- **Hugging Face Transformers**: Pretrained models and tokenizers for NLP/LLM tasks
- **PyTorch Lightning**: High-level training framework with distributed strategies

**Why This Integration**:

- **HF provides**: Model architectures, pretrained weights, tokenizers, datasets
- **PL provides**: Training loops, distributed strategies (FSDP, DeepSpeed), experiment tracking
- **Together**: Clean separation of model definition and training logic

## Prerequisites

Install required packages:

```bash
# Using UV
uv add transformers datasets lightning torchmetrics wandb

# Using Pixi
pixi add --pypi transformers datasets lightning torchmetrics wandb

# For PEFT (LoRA)
uv add peft  # or: pixi add --pypi peft

# For DeepSpeed (optional)
uv add deepspeed  # or: pixi add --pypi deepspeed
```

## Integration Pattern: LightningModule + Transformers

### Basic Architecture

The fundamental pattern is **encapsulating** HF's `PreTrainedModel` inside PL's `LightningModule`:

```python
import lightning.pytorch as pl
from transformers import AutoModelForSequenceClassification, AutoConfig
import torch

class TransformerClassifier(pl.LightningModule):
    """Text classification with pretrained transformers."""

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
    ):
        super().__init__()

        # Save all hyperparameters (critical for reproducibility)
        self.save_hyperparameters()

        # Load configuration
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels
        )

        # Load pretrained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=self.config
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass for inference and training."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        """Training step - called for each batch."""
        # Unpack batch (dictionary from HF datasets)
        outputs = self(**batch)

        # HF models compute loss automatically when labels provided
        loss = outputs.loss

        # Log metrics (sync across GPUs automatically)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self(**batch)
        loss = outputs.loss

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Will be covered in detail below
        pass
```

### Critical Implementation Details

#### 1. `save_hyperparameters()` - Reproducibility

**Always** call `self.save_hyperparameters()` in `__init__`:

```python
def __init__(self, model_name_or_path: str, learning_rate: float = 2e-5):
    super().__init__()
    self.save_hyperparameters()  # Saves all __init__ args to self.hparams
```

**Benefits**:

- Hyperparameters saved in checkpoint automatically
- Can load model without remembering arguments: `Model.load_from_checkpoint(path)`
- Essential for experiment reproducibility

#### 2. HF's Built-in Loss Computation

HF models have **built-in loss calculation** when `labels` are provided:

```python
# Good - let HF compute loss
outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss  # HF handles loss internally

# Bad - manual loss (prone to errors like double softmax)
logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
loss = F.cross_entropy(logits, labels)  # Risk of mistakes
```

**Why delegate to HF**:

- Prevents common errors (double softmax, wrong ignore_index)
- Handles different task types correctly (classification, regression, generation)
- Automatically compatible with model architecture

#### 3. `forward()` vs `training_step()` Separation

- **`forward()`**: For inference, should return model outputs
- **`training_step()`**: For training, should return loss

```python
def forward(self, input_ids, attention_mask):
    """Inference mode - no labels."""
    return self.model(input_ids=input_ids, attention_mask=attention_mask)

def training_step(self, batch, batch_idx):
    """Training mode - with labels."""
    outputs = self(**batch)  # batch contains labels
    return outputs.loss
```

## Optimizer Configuration

### Weight Decay Exclusion Pattern

**Standard practice**: Exclude bias and LayerNorm parameters from weight decay:

```python
def configure_optimizers(self):
    """Configure AdamW with weight decay exclusion."""

    # Parameters that should NOT have weight decay
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    # Group parameters
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in self.model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in self.model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=self.hparams.learning_rate
    )

    return optimizer
```

### Learning Rate Schedulers with Warmup

Transformers typically use **warmup + linear decay**:

```python
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

def configure_optimizers(self):
    # ... optimizer setup ...

    # Linear decay with warmup (BERT-style)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=self.hparams.warmup_steps,
        num_training_steps=self.trainer.estimated_stepping_batches
    )

    # Or cosine annealing with warmup (GPT-style)
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=self.hparams.warmup_steps,
    #     num_training_steps=self.trainer.estimated_stepping_batches
    # )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",  # Update every step, not epoch
            "frequency": 1,
        },
    }
```

**Critical**: Use `self.trainer.estimated_stepping_batches` instead of manual calculation. This accounts for:

- Gradient accumulation
- Number of GPUs
- Number of epochs
- Dataset size

## Data Pipeline with LightningDataModule

### Structure

```python
import lightning.pytorch as pl
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

class TransformerDataModule(pl.LightningDataModule):
    """Data module for HF datasets."""

    def __init__(
        self,
        model_name_or_path: str,
        dataset_name: str,
        max_length: int = 128,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def prepare_data(self):
        """Download data (run once, on rank 0 only)."""
        # Download dataset (cached automatically)
        load_dataset(self.hparams.dataset_name)

    def setup(self, stage: str = None):
        """Load and preprocess data (run on all GPUs)."""

        # Load dataset
        dataset = load_dataset(self.hparams.dataset_name)

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.hparams.max_length,
            )

        # Apply tokenization (cached automatically)
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        # Set format for PyTorch
        tokenized_datasets.set_format("torch")

        # Split datasets
        if stage == "fit" or stage is None:
            self.train_dataset = tokenized_datasets["train"]
            self.val_dataset = tokenized_datasets["validation"]

        if stage == "test" or stage is None:
            self.test_dataset = tokenized_datasets["test"]

    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=DataCollatorWithPadding(self.tokenizer),  # Dynamic padding
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            pin_memory=True,
        )
```

### Dynamic Padding

**Always use `DataCollatorWithPadding`** for efficiency:

```python
from transformers import DataCollatorWithPadding

collate_fn = DataCollatorWithPadding(tokenizer=self.tokenizer)
```

**Why**:

- Pads to **max length in batch**, not max length in dataset
- Can achieve 2x speedup on datasets with variable sequence lengths
- Reduces wasted computation on padding tokens

### Streaming Large Datasets

For datasets too large to download:

```python
def setup(self, stage: str = None):
    """Load dataset in streaming mode."""

    # Stream instead of download
    dataset = load_dataset(
        self.hparams.dataset_name,
        streaming=True  # No download, read on-the-fly
    )

    # Tokenize (on-the-fly, no caching)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # IterableDataset - no shuffling needed
    self.train_dataset = tokenized_datasets["train"]
```

**Note**: Streaming returns `IterableDataset`, which PL handles automatically.

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

## Common Tasks and Recipes

### Text Classification (BERT, RoBERTa)

```python
# Model
model = TransformerClassifier(
    model_name_or_path="bert-base-uncased",
    num_labels=2,  # Binary classification
    learning_rate=2e-5,
)

# Data
datamodule = TransformerDataModule(
    model_name_or_path="bert-base-uncased",
    dataset_name="glue",
    dataset_config_name="sst2",
    batch_size=32,
)

# Trainer
trainer = pl.Trainer(
    max_epochs=3,
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
)

# Train
trainer.fit(model, datamodule)
```

### Sequence-to-Sequence (T5, BART)

```python
from transformers import AutoModelForSeq2SeqLM

class Seq2SeqModel(pl.LightningModule):
    def __init__(self, model_name: str = "t5-small"):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def training_step(self, batch, batch_idx):
        # T5 expects input_ids, attention_mask, labels
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],  # Target sequence
        )
        loss = outputs.loss
        self.log("train/loss", loss)
        return loss

    def generate(self, input_ids, attention_mask, **kwargs):
        """Generate text."""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
```

### Causal Language Modeling (GPT, Llama)

```python
from transformers import AutoModelForCausalLM

class CausalLM(pl.LightningModule):
    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def training_step(self, batch, batch_idx):
        # For causal LM, labels = input_ids (shifted internally by HF)
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],  # Self-supervised
        )
        loss = outputs.loss
        self.log("train/loss", loss)
        return loss
```

### Token Classification (NER)

```python
from transformers import AutoModelForTokenClassification

class NERModel(pl.LightningModule):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],  # Token-level labels
        )
        loss = outputs.loss
        self.log("train/loss", loss)
        return loss
```

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
    num_warmup_steps=warmup_steps,
    num_training_steps=self.trainer.estimated_stepping_batches  # ← Auto-computed
)
```

#### 4. Missing Weight Decay Exclusion

**Problem**: Applying weight decay to bias and LayerNorm degrades performance.

**Solution**: Exclude specific parameters:

```python
no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
# ... group parameters ...
```

### Memory Profiling

```python
# Enable memory profiling
trainer = pl.Trainer(
    profiler="simple",  # or "advanced" for detailed profiling
)

# Check memory usage in training
def training_step(self, batch, batch_idx):
    if batch_idx % 100 == 0:
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    outputs = self(**batch)
    return outputs.loss
```

### Performance Optimization Checklist

- ✅ Use BF16 mixed precision (`precision="bf16-mixed"`)
- ✅ Enable gradient checkpointing for large models
- ✅ Use dynamic padding (`DataCollatorWithPadding`)
- ✅ Set `num_workers > 0` in DataLoader (typically 4-8)
- ✅ Use `pin_memory=True` in DataLoader
- ✅ Increase batch size (use gradient accumulation if OOM)
- ✅ Profile with `profiler="advanced"` to find bottlenecks
- ✅ Use FSDP/DeepSpeed for models >10B parameters
- ✅ Consider LoRA for fine-tuning large models

## Resources

### Official Documentation

- **PyTorch Lightning**: <https://lightning.ai/docs/pytorch/stable/>
- **Hugging Face Transformers**: <https://huggingface.co/docs/transformers/>
- **HF Datasets**: <https://huggingface.co/docs/datasets/>
- **PEFT (LoRA)**: <https://huggingface.co/docs/peft/>
- **TorchMetrics**: <https://lightning.ai/docs/torchmetrics/>

### Templates and Examples

- **Lightning-Hydra-Template**: <https://github.com/ashleve/lightning-hydra-template>
- **LitGPT**: <https://github.com/Lightning-AI/litgpt>
- **Transformers Examples**: <https://github.com/huggingface/transformers/tree/main/examples/pytorch>

### Key Papers

- **LoRA**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- **QLoRA**: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- **FSDP**: "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" (Meta AI, 2023)

## Summary

**Key Takeaways**:

1. **Encapsulation**: Wrap HF models in LightningModule for clean separation
2. **save_hyperparameters()**: Essential for reproducibility
3. **Delegate loss**: Let HF models compute loss (avoid manual calculation)
4. **Weight decay exclusion**: Standard pattern for transformers
5. **Dynamic padding**: Use DataCollatorWithPadding for efficiency
6. **FSDP/DeepSpeed**: Required for large models (>10B parameters)
7. **LoRA**: Fine-tune with <1% parameters
8. **TorchMetrics**: Correct distributed evaluation
9. **Hydra**: Professional config management

**When to Use This Stack**:

- ✅ NLP tasks (classification, generation, QA, NER)
- ✅ LLM fine-tuning (with LoRA/QLoRA)
- ✅ Multi-GPU training
- ✅ Experiment tracking and reproducibility
- ✅ Production-grade ML pipelines

This integration represents the industry standard for transformer-based model development in 2025.
