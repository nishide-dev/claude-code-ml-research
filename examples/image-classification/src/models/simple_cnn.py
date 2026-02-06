"""Simple CNN model for CIFAR-10."""

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(pl.LightningModule):
    """Simple CNN for image classification.

    Architecture:
        Conv(3->64) -> ReLU -> MaxPool -> Conv(64->128) -> ReLU -> MaxPool ->
        Flatten -> Linear(128*8*8->512) -> ReLU -> Dropout -> Linear(512->num_classes)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_classes: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        optimizer: dict[str, Any] | None = None,
        scheduler: dict[str, Any] | None = None,
    ) -> None:
        """Initialize model.

        Args:
            in_channels: Number of input channels (3 for RGB)
            hidden_channels: Number of channels in first conv layer
            num_classes: Number of output classes
            lr: Learning rate
            weight_decay: Weight decay
            optimizer: Optimizer config (Hydra instantiation dict)
            scheduler: Scheduler config (Hydra instantiation dict)
        """
        super().__init__()
        self.save_hyperparameters()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels * 2)

        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        # After 2 pooling layers: 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(hidden_channels * 2 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 32x32 -> 16x16

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 16x16 -> 8x8

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Input batch (images, labels)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Input batch (images, labels)
            batch_idx: Batch index
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step.

        Args:
            batch: Input batch (images, labels)
            batch_idx: Batch index
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log("test/loss", loss)
        self.log("test/acc", acc)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dictionary with optimizer and optional scheduler
        """
        from hydra.utils import instantiate

        # Instantiate optimizer
        if self.hparams.optimizer is not None:
            optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )

        # No scheduler
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        # Instantiate scheduler
        scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
