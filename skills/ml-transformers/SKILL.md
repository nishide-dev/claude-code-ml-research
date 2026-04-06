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

---

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
        # See below for implementation
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

---

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
from transformers import get_linear_schedule_with_warmup

def configure_optimizers(self):
    # ... optimizer setup ...

    # Linear decay with warmup (BERT-style)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=self.hparams.warmup_steps,
        num_training_steps=self.trainer.estimated_stepping_batches
    )

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

---

## Data Pipeline with LightningDataModule

### Complete DataModule

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

---

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

---

## Complete Training Example

```python
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Model
model = TransformerClassifier(
    model_name_or_path="bert-base-uncased",
    num_labels=2,
    learning_rate=2e-5,
    warmup_steps=500,
)

# Data
datamodule = TransformerDataModule(
    model_name_or_path="bert-base-uncased",
    dataset_name="glue",
    dataset_config_name="sst2",
    batch_size=32,
)

# Logger
wandb_logger = WandbLogger(
    project="transformers-lightning",
    name="bert-classification",
)

# Callbacks
callbacks = [
    ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=3,
    ),
    EarlyStopping(
        monitor="val/loss",
        patience=3,
        mode="min",
    ),
]

# Trainer
trainer = pl.Trainer(
    max_epochs=3,
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
    logger=wandb_logger,
    callbacks=callbacks,
    log_every_n_steps=50,
)

# Train
trainer.fit(model, datamodule)

# Test
trainer.test(model, datamodule)
```

---

## Advanced Topics

For detailed guides on advanced features, see [reference/advanced-topics.md](reference/advanced-topics.md):

- **Distributed Training**: DDP, FSDP, DeepSpeed strategies
- **PEFT**: LoRA and QLoRA for parameter-efficient fine-tuning
- **Evaluation**: TorchMetrics for distributed evaluation
- **Configuration**: Hydra integration for experiment management
- **Troubleshooting**: Common pitfalls and best practices

---

## Resources

### Official Documentation

- **Hugging Face Transformers**: <https://huggingface.co/docs/transformers>
- **PyTorch Lightning**: <https://lightning.ai/docs/pytorch>
- **HF Datasets**: <https://huggingface.co/docs/datasets>
- **PEFT Library**: <https://huggingface.co/docs/peft>

### Templates & Examples

- **Lightning-Hydra-Template**: <https://github.com/ashleve/lightning-hydra-template>
- **HF Transformers Examples**: <https://github.com/huggingface/transformers/tree/main/examples/pytorch>

---

## Summary

This integration pattern provides:

**Core Strengths**:

- **Clean separation**: Model definition (HF) vs training logic (PL)
- **Automatic optimizations**: Mixed precision, gradient accumulation, distributed training
- **Built-in safety**: HF loss computation prevents common errors
- **Easy scaling**: DDP → FSDP → DeepSpeed with minimal code changes
- **Reproducibility**: save_hyperparameters() + checkpoint system

**Best Practices**:

1. Always call `save_hyperparameters()` in `__init__`
2. Let HF models compute loss (don't do it manually)
3. Use `DataCollatorWithPadding` for dynamic padding
4. Use `self.trainer.estimated_stepping_batches` for schedulers
5. Start with DDP, scale to FSDP/DeepSpeed as needed

This pattern is production-ready and scales from single GPU to hundreds of GPUs with minimal code changes.
