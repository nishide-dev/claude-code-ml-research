# src/data/vision_datamodule.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class VisionDataModule(pl.LightningDataModule):
    """DataModule for computer vision tasks."""

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 128,
        num_workers: int = 4,
        image_size: int = 224,
        train_val_test_split: tuple[float, float, float] = (0.8, 0.1, 0.1),
        augment: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_val_test_split = train_val_test_split
        self.augment = augment

    def setup(self, stage: str | None = None):
        """Setup datasets for each stage."""
        # Training transforms with augmentation
        train_transforms = (
            transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            if self.augment
            else self._base_transforms()
        )

        # Validation/test transforms (no augmentation)
        eval_transforms = self._base_transforms()

        if stage == "fit" or stage is None:
            full_dataset = datasets.ImageFolder(
                root=f"{self.data_dir}/train", transform=train_transforms
            )

            # Split into train/val
            train_size = int(self.train_val_test_split[0] * len(full_dataset))
            val_size = len(full_dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )

            # Apply eval transforms to val
            self.val_dataset.dataset.transform = eval_transforms

        if stage == "test" or stage is None:
            self.test_dataset = datasets.ImageFolder(
                root=f"{self.data_dir}/test", transform=eval_transforms
            )

    def _base_transforms(self):
        """Base transforms without augmentation."""
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
