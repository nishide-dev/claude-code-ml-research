"""ResNet model adapted for CIFAR-10."""

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize basic block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first convolution
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(pl.LightningModule):
    """ResNet model adapted for CIFAR-10 (32x32 images).

    Modified from standard ResNet to work with smaller CIFAR-10 images.
    Uses smaller initial kernel and no initial pooling.
    """

    def __init__(
        self,
        num_classes: int = 10,
        blocks: list[int] | None = None,
        lr: float = 0.0001,
        weight_decay: float = 0.0005,
        optimizer: dict[str, Any] | None = None,
        scheduler: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ResNet model.

        Args:
            num_classes: Number of output classes
            blocks: Number of blocks in each layer (default: [2,2,2,2] for ResNet-18)
            lr: Learning rate
            weight_decay: Weight decay
            optimizer: Optimizer config (Hydra instantiation dict)
            scheduler: Scheduler config (Hydra instantiation dict)
        """
        super().__init__()
        self.save_hyperparameters()

        if blocks is None:
            blocks = [2, 2, 2, 2]  # ResNet-18

        self.in_channels = 64

        # Initial convolution - smaller kernel for CIFAR-10
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = self._make_layer(64, blocks[0], stride=1)
        self.layer2 = self._make_layer(128, blocks[1], stride=2)
        self.layer3 = self._make_layer(256, blocks[2], stride=2)
        self.layer4 = self._make_layer(512, blocks[3], stride=2)

        # Average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a residual layer.

        Args:
            out_channels: Number of output channels
            num_blocks: Number of blocks in this layer
            stride: Stride for first block

        Returns:
            Sequential container with residual blocks
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride_val in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride_val))
            self.in_channels = out_channels * BasicBlock.expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)))

        # Residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Global average pooling
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        # Classifier
        out = self.fc(out)

        return out

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
