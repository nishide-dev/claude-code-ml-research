# Text Classification Training Example

Complete example for training a text classification model using Hugging Face Transformers + PyTorch Lightning.

## Setup

```bash
# Install dependencies
uv add torch pytorch-lightning transformers datasets hydra-core

# Login to Hugging Face (optional, for private models)
huggingface-cli login
```

## Project Structure

```text
project/
├── configs/
│   ├── config.yaml
│   ├── model/
│   │   └── bert.yaml
│   ├── data/
│   │   └── imdb.yaml
│   └── experiment/
│       └── imdb_bert.yaml
├── src/
│   ├── models/
│   │   └── text_classifier.py
│   ├── data/
│   │   └── text_datamodule.py
│   └── train.py
└── data/
```

## Model Implementation

**`src/models/text_classifier.py`:**

```python
import pytorch_lightning as pl
import torch
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torchmetrics import Accuracy, F1Score


class TextClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_labels, average="macro")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        logits = outputs.logits

        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, batch["labels"])

        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        logits = outputs.logits

        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, batch["labels"])
        self.val_f1(preds, batch["labels"])

        # Logging
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # Separate learning rates for transformer and classifier
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
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
            optimizer_grouped_parameters, lr=self.hparams.learning_rate
        )

        # Learning rate scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
```

## DataModule Implementation

**`src/data/text_datamodule.py`:**

```python
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "imdb",
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        # Download dataset
        load_dataset(self.dataset_name)

    def setup(self, stage=None):
        # Load dataset
        dataset = load_dataset(self.dataset_name)

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        if stage == "fit" or stage is None:
            self.train_dataset = dataset["train"].map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
            )
            self.train_dataset = self.train_dataset.rename_column("label", "labels")
            self.train_dataset.set_format("torch")

            # Use test set as validation (IMDB doesn't have a val split)
            self.val_dataset = dataset["test"].map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
            )
            self.val_dataset = self.val_dataset.rename_column("label", "labels")
            self.val_dataset.set_format("torch")

        if stage == "test" or stage is None:
            self.test_dataset = dataset["test"].map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
            )
            self.test_dataset = self.test_dataset.rename_column("label", "labels")
            self.test_dataset.set_format("torch")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
```

## Configuration

**`configs/experiment/imdb_bert.yaml`:**

```yaml
# @package _global_

defaults:
  - override /model: bert
  - override /data: imdb
  - override /trainer: default
  - override /logger: wandb

# Experiment name
experiment_name: imdb_bert_classification

# Model settings
model:
  model_name: bert-base-uncased
  num_labels: 2  # Binary classification (positive/negative)
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 500

# Data settings
data:
  dataset_name: imdb
  model_name: ${model.model_name}  # Use same tokenizer as model
  max_length: 512
  batch_size: 16
  num_workers: 4

# Training settings
trainer:
  max_epochs: 3  # Transformers typically need fewer epochs
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  accumulate_grad_batches: 2  # Effective batch size = 32

  # Gradient clipping
  gradient_clip_val: 1.0

# Callbacks
callbacks:
  early_stopping:
    patience: 3  # Stop if no improvement for 3 epochs

# Logging
logger:
  wandb:
    project: text-classification
    tags: ["imdb", "bert-base-uncased"]
    log_model: true
```

## Training Commands

**Basic training:**

```bash
python src/train.py experiment=imdb_bert
```

**With different model:**

```bash
python src/train.py experiment=imdb_bert \
  model.model_name=roberta-base \
  data.model_name=roberta-base
```

**Multi-GPU training:**

```bash
python src/train.py experiment=imdb_bert \
  trainer.devices=4 \
  trainer.strategy=ddp \
  data.batch_size=8
```

**Hyperparameter search:**

```bash
python src/train.py experiment=imdb_bert --multirun \
  model.learning_rate=1e-5,2e-5,5e-5 \
  data.batch_size=8,16,32
```

## Expected Results

**IMDB with BERT-base:**

- Validation accuracy: ~93-95%
- Training time: ~2 hours on single GPU (V100)
- Convergence: Usually within 3 epochs

**Metrics to monitor:**

- Loss should decrease quickly in first epoch
- Accuracy should improve significantly early on
- F1 score should be close to accuracy for balanced datasets

## Common Issues

**Out of memory:**

- Reduce batch_size (try 8 or 4)
- Reduce max_length (try 256 or 128)
- Use gradient accumulation
- Use 16-mixed precision

**Slow training:**

- Increase batch_size if memory allows
- Use more num_workers
- Enable pin_memory=True
- Use persistent_workers=True

**Overfitting:**

- Add dropout in classifier head
- Reduce number of epochs
- Use weight decay
- Try data augmentation (backtranslation, synonym replacement)

**Poor performance:**

- Try different pretrained models (RoBERTa, DeBERTa)
- Adjust learning rate (try 1e-5 to 5e-5)
- Increase max_length for longer texts
- Check dataset quality and class balance

## Advanced Techniques

**LoRA fine-tuning (parameter-efficient):**

```python
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)
```

**Gradient checkpointing (save memory):**

```python
model.gradient_checkpointing_enable()
```

**Mixed batch sizes for different lengths:**

```python
# Use dynamic padding in dataloader
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
