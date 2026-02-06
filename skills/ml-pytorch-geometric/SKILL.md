---
name: ml-pytorch-geometric
description: Complete guide for PyTorch Geometric (PyG) - graph neural networks, message passing, large-scale distributed graph learning, Lightning integration, and heterogeneous graphs
---

# PyTorch Geometric for Graph Neural Networks

## Overview

PyTorch Geometric (PyG) is the standard library for geometric deep learning with PyTorch. It provides optimized implementations of Graph Neural Networks (GNNs), efficient data structures for graph data, and scalable solutions for large-scale graph learning.

**Key Capabilities:**

- Efficient sparse tensor operations for graphs
- 100+ pre-implemented GNN layers (GCN, GAT, GraphSAGE, etc.)
- Scalable data loaders with neighbor sampling
- Large-scale distributed graph learning
- Heterogeneous and temporal graph support
- Integration with PyTorch Lightning

**Resources:**

- Official docs: <https://pytorch-geometric.readthedocs.io/>
- Stanford CS224W: <http://web.stanford.edu/class/cs224w/>
- GitHub: <https://github.com/pyg-team/pytorch_geometric>

---

## Core Concepts

### 1. Graph Data Representation

PyG uses a tensor-centric approach. Each graph is represented by a `Data` object containing node features, edge indices, and optional attributes.

**Basic Data object:**

```python
import torch
from torch_geometric.data import Data

# Create a simple graph: 0 -> 1 -> 2
#                        ^    |
#                        |____|
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 2, 0, 1]], dtype=torch.long)
x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float)  # Node features
y = torch.tensor([0], dtype=torch.long)  # Graph label

data = Data(x=x, edge_index=edge_index, y=y)

print(data)
# Data(x=[3, 1], edge_index=[2, 4], y=[1])

# Access attributes
print(f"Nodes: {data.num_nodes}")        # 3
print(f"Edges: {data.num_edges}")        # 4
print(f"Features: {data.num_node_features}")  # 1
print(f"Directed: {data.is_directed()}")  # True
```

**Common Data attributes:**

| Attribute | Shape | Type | Description |
|-----------|-------|------|-------------|
| `data.x` | `[num_nodes, num_node_features]` | float | Node feature matrix |
| `data.edge_index` | `[2, num_edges]` | long | Graph connectivity in COO format |
| `data.edge_attr` | `[num_edges, num_edge_features]` | float | Edge feature matrix |
| `data.y` | Varies | Any | Target labels (node/graph level) |
| `data.pos` | `[num_nodes, num_dimensions]` | float | Node positions (for point clouds) |
| `data.batch` | `[num_nodes]` | long | Batch assignment vector |

### 2. Message Passing Framework

GNNs work through message passing: nodes aggregate information from neighbors to update their representations.

**MessagePassing base class:**

```python
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "add", "mean", "max"
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Add self-loops to adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linear transformation
        x = self.lin(x)

        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Start propagating messages
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # Normalize node features
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # Update node embeddings
        return aggr_out
```

**Message passing steps:**

1. **message()**: Constructs messages from source nodes to target nodes
2. **aggregate()**: Aggregates messages (sum, mean, max)
3. **update()**: Updates node embeddings based on aggregated messages

### 3. Common GNN Layers

PyG provides 100+ pre-implemented layers. Here are the most important:

**GCN (Graph Convolutional Network):**

```python
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

**GAT (Graph Attention Networks):**

```python
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

**GraphSAGE (inductive learning):**

```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

---

## Graph-Level Tasks

### Graph Classification

Aggregate node representations to classify entire graphs (molecules, social networks, etc.).

**Complete example:**

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

# Define model
class GraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.lin = torch.nn.Linear(64, num_classes)

    def forward(self, x, edge_index, batch):
        # Node embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        # Graph-level readout (pooling)
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]

        # Classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

# Load dataset
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

# Split dataset
train_dataset = dataset[:540]
test_dataset = dataset[540:]

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train model
model = GraphClassifier(dataset.num_node_features, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(200):
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
```

**Graph pooling layers:**

- `global_mean_pool`: Average node features
- `global_max_pool`: Max pooling over nodes
- `global_add_pool`: Sum node features
- `global_sort_pool`: Sort pooling (SortPool)
- `SAGPooling`: Self-attention graph pooling

---

## Node-Level Tasks

### Node Classification

Classify nodes within a single large graph.

**Node classification example:**

```python
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Load Cora dataset (citation network)
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]  # Single graph

# Model
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Training
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    # Only compute loss on training nodes
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
```

### Neighbor Sampling for Large Graphs

For massive graphs, load only k-hop neighborhoods instead of the entire graph.

**NeighborLoader:**

```python
from torch_geometric.loader import NeighborLoader

# Sample 2-hop neighborhoods
loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],  # Sample 25 neighbors in 1st hop, 10 in 2nd hop
    batch_size=128,
    input_nodes=data.train_mask,  # Only create batches from training nodes
    num_workers=4,
    shuffle=True,
)

# Training loop with mini-batches
model.train()
for batch in loader:
    optimizer.zero_grad()
    out = model(batch.x, batch.edge_index)[:batch.batch_size]  # Only predict for center nodes
    loss = F.nll_loss(out, batch.y[:batch.batch_size])
    loss.backward()
    optimizer.step()
```

---

## Integration with PyTorch Lightning

PyG provides Lightning-compatible wrappers for seamless integration.

### LightningDataset (Graph Classification)

```python
import lightning as L
from torch_geometric.datasets import TUDataset
from torch_geometric.data import LightningDataset

# Create Lightning wrapper
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
datamodule = LightningDataset(
    train_dataset=dataset[:540],
    val_dataset=dataset[540:600],
    test_dataset=dataset[600:],
    batch_size=32,
    num_workers=4,
)

# LightningModule
class LitGNN(L.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x

    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        acc = (out.argmax(dim=1) == batch.y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

# Train
model = LitGNN(dataset.num_node_features, 64, dataset.num_classes)
trainer = L.Trainer(max_epochs=100, accelerator='gpu', devices=1)
trainer.fit(model, datamodule)
```

### LightningNodeData (Node Classification)

```python
from torch_geometric.data import LightningNodeData
from torch_geometric.datasets import Planetoid

# Load dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Create Lightning wrapper with neighbor sampling
datamodule = LightningNodeData(
    data,
    input_train_nodes=data.train_mask,
    input_val_nodes=data.val_mask,
    input_test_nodes=data.test_mask,
    loader='neighbor',  # Use NeighborLoader
    num_neighbors=[25, 10],  # 2-hop sampling
    batch_size=128,
    num_workers=4,
)

# Training works the same as before
model = LitGNN(dataset.num_node_features, 64, dataset.num_classes)
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, datamodule)
```

---

## Large-Scale Graph Learning

### Distributed Training

PyG 2.5+ supports distributed training for billion-scale graphs using DDP + RPC.

**Architecture:**

1. **Graph Partitioning**: Split graph using METIS to minimize edge cuts
2. **DDP**: Replicate model across GPUs with gradient synchronization
3. **RPC**: Fetch features/structure from remote partitions

**Partition graph:**

```bash
# Use METIS to partition graph
python -m torch_geometric.distributed.partition \
    --dataset ogbn-products \
    --num_parts 4 \
    --output_dir ./partitions
```

**Distributed training script:**

```python
import torch.distributed as dist
from torch_geometric.distributed import DistNeighborLoader
from torch_geometric.nn import GraphSAGE

def run_distributed(rank, world_size):
    # Initialize process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # Load partition
    data = torch.load(f'./partitions/part_{rank}.pt')

    # Create distributed loader
    loader = DistNeighborLoader(
        data,
        num_neighbors=[25, 10],
        batch_size=1024,
        shuffle=True,
    )

    # Model
    model = GraphSAGE(in_channels, hidden_channels, out_channels)
    model = torch.nn.parallel.DistributedDataParallel(model)

    # Training loop
    for epoch in range(100):
        for batch in loader:
            # Training logic
            pass

if __name__ == '__main__':
    world_size = 4
    torch.multiprocessing.spawn(run_distributed, args=(world_size,), nprocs=world_size)
```

### Remote Backend (Out-of-Core Learning)

Use `FeatureStore` and `GraphStore` for data that doesn't fit in memory.

**Custom FeatureStore:**

```python
from torch_geometric.data import FeatureStore, GraphStore
import rocksdb

class RocksDBFeatureStore(FeatureStore):
    def __init__(self, path):
        self.db = rocksdb.DB(path, rocksdb.Options(create_if_missing=True))

    def get_tensor(self, key):
        # Fetch features from disk
        data = self.db.get(key.encode())
        return torch.frombuffer(data, dtype=torch.float32)

    def set_tensor(self, key, tensor):
        # Store features on disk
        self.db.put(key.encode(), tensor.numpy().tobytes())

# Use with NeighborLoader
feature_store = RocksDBFeatureStore('./features_db')
loader = NeighborLoader(
    data=(feature_store, graph_store),
    num_neighbors=[15, 10],
    batch_size=128,
)
```

---

## Heterogeneous Graphs

Handle graphs with multiple node and edge types (e.g., knowledge graphs).

**HeteroData:**

```python
from torch_geometric.data import HeteroData

data = HeteroData()

# Add node types
data['paper'].x = torch.randn(1000, 128)  # 1000 papers with 128 features
data['author'].x = torch.randn(500, 64)   # 500 authors with 64 features

# Add edge types
data['author', 'writes', 'paper'].edge_index = torch.randint(0, 500, (2, 2000))
data['paper', 'cites', 'paper'].edge_index = torch.randint(0, 1000, (2, 5000))

print(data)
# HeteroData(
#   paper={ x=[1000, 128] },
#   author={ x=[500, 64] },
#   (author, writes, paper)={ edge_index=[2, 2000] },
#   (paper, cites, paper)={ edge_index=[2, 5000] }
# )
```

**Heterogeneous GNN:**

```python
from torch_geometric.nn import HeteroConv, GCNConv, Linear

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = HeteroConv({
            ('author', 'writes', 'paper'): GCNConv(-1, hidden_channels),
            ('paper', 'cites', 'paper'): GCNConv(-1, hidden_channels),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('author', 'writes', 'paper'): GCNConv(-1, hidden_channels),
            ('paper', 'cites', 'paper'): GCNConv(-1, hidden_channels),
        }, aggr='sum')

        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return self.lin(x_dict['paper'])
```

---

## Advanced Techniques

### 1. Graph Augmentation

```python
from torch_geometric.transforms import RandomNodeSplit, AddSelfLoops, NormalizeFeatures

# Compose transforms
transform = T.Compose([
    AddSelfLoops(),
    NormalizeFeatures(),
    RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1),
])

dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
```

### 2. Custom Datasets

```python
from torch_geometric.data import Dataset, download_url

class MyDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['data.csv']

    @property
    def processed_file_names(self):
        return ['data_0.pt', 'data_1.pt', ...]

    def download(self):
        # Download raw data
        download_url('https://example.com/data.csv', self.raw_dir)

    def process(self):
        # Process raw data into Data objects
        for idx, raw_data in enumerate(raw_data_list):
            data = Data(...)
            torch.save(data, f'{self.processed_dir}/data_{idx}.pt')

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(f'{self.processed_dir}/data_{idx}.pt')
        return data
```

### 3. Explainability

```python
from torch_geometric.explain import Explainer, GNNExplainer

model = GCN(...)
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
)

# Explain prediction for node 0
explanation = explainer(data.x, data.edge_index, index=0)
print(f'Node mask: {explanation.node_mask}')
print(f'Edge mask: {explanation.edge_mask}')
```

---

## Best Practices

### ✅ DO

1. **Use neighbor sampling for large graphs**: Don't load entire graph into memory
2. **Leverage pre-implemented layers**: PyG has 100+ optimized GNN layers
3. **Normalize features**: Use `NormalizeFeatures()` transform for stable training
4. **Add self-loops**: Many GNN layers require self-loops (use `add_self_loops()`)
5. **Use dropout**: GNNs overfit easily; add dropout between layers
6. **Monitor gradient flow**: Check for vanishing/exploding gradients in deep GNNs
7. **Profile memory usage**: Graph data can be memory-intensive
8. **Use Lightning for distributed training**: Simplifies multi-GPU setup

### ❌ DON'T

1. **Don't ignore edge direction**: Undirected graphs need bidirectional edges in `edge_index`
2. **Don't use too many layers**: Deep GNNs suffer from over-smoothing (3-5 layers typical)
3. **Don't forget to set model.eval()**: Dropout/BatchNorm behavior differs in eval mode
4. **Don't use dense adjacency matrices**: Always use sparse COO format
5. **Don't mix node/graph-level tasks**: Be clear about prediction granularity
6. **Don't skip data validation**: Check for isolated nodes, NaN features, etc.

---

## Essential Resources

### Official Documentation

- **PyG Docs**: <https://pytorch-geometric.readthedocs.io/>
- **API Reference**: <https://pytorch-geometric.readthedocs.io/en/latest/modules/root.html>
- **Examples**: <https://github.com/pyg-team/pytorch_geometric/tree/master/examples>

### Learning Resources

- **Stanford CS224W**: <http://web.stanford.edu/class/cs224w/> (Machine Learning with Graphs)
- **UvA Deep Learning**: <https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html>
- **Distill.pub GNN Article**: <https://distill.pub/2021/gnn-intro/>

### Advanced Topics

- **Distributed Training**: <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/distributed_pyg.html>
- **Remote Backends**: <https://pytorch-geometric.readthedocs.io/en/latest/advanced/remote.html>
- **Heterogeneous Graphs**: <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html>

---

## Summary

PyTorch Geometric provides:

- **Efficient sparse operations**: Optimized for graph data structures
- **Rich model library**: 100+ GNN layers, datasets, and transforms
- **Scalability**: Neighbor sampling, distributed training, out-of-core learning
- **Flexibility**: Custom message passing, heterogeneous graphs, temporal data
- **Integration**: Seamless PyTorch Lightning integration for production

Combined with Lightning, PyG enables research-to-production graph deep learning at any scale.
