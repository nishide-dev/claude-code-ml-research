# src/data/graph_datamodule.py

import pytorch_lightning as pl
from torch.utils.data import random_split
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures


class GraphDataModule(pl.LightningDataModule):
    """DataModule for graph neural networks."""

    def __init__(
        self,
        dataset_name: str = "Cora",
        data_dir: str = "data/graphs/",
        batch_size: int = 32,
        num_workers: int = 4,
        use_sampling: bool = False,
        num_neighbors: list[int] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_sampling = use_sampling
        self.num_neighbors = num_neighbors or [15, 10, 5]

    def setup(self, stage: str | None = None):
        """Load graph dataset."""
        # Node classification datasets (e.g., Cora, PubMed)
        if self.dataset_name in ["Cora", "CiteSeer", "PubMed"]:
            self.dataset = Planetoid(
                root=self.data_dir,
                name=self.dataset_name,
                transform=NormalizeFeatures(),
            )
            self.data = self.dataset[0]  # Single graph
            self.task_type = "node_classification"

        # Graph classification datasets (e.g., PROTEINS, ENZYMES)
        elif self.dataset_name in ["PROTEINS", "ENZYMES", "MUTAG"]:
            self.dataset = TUDataset(
                root=self.data_dir,
                name=self.dataset_name,
                transform=NormalizeFeatures(),
            )
            self.task_type = "graph_classification"

            # Split dataset
            train_size = int(0.8 * len(self.dataset))
            val_size = int(0.1 * len(self.dataset))
            test_size = len(self.dataset) - train_size - val_size

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset, [train_size, val_size, test_size]
            )

    def train_dataloader(self):
        if self.task_type == "node_classification":
            # For node classification, typically use full graph
            return DataLoader([self.data], batch_size=1)
        # For graph classification, batch multiple graphs
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if self.task_type == "node_classification":
            return DataLoader([self.data], batch_size=1)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if self.task_type == "node_classification":
            return DataLoader([self.data], batch_size=1)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
