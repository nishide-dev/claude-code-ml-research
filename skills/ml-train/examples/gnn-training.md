# Graph Neural Network Training with PyTorch Geometric

Complete guide for training GNNs with PyTorch Lightning and PyTorch Geometric.

## PyTorch Geometric Specific Training

### Node Classification

Train a GNN for node classification on citation networks:

```bash
# Cora dataset (small graph, 2708 nodes)
python src/train.py \
  model=gnn \
  data=graph \
  data.dataset_name=Cora

# PubMed dataset (larger graph, 19717 nodes)
python src/train.py \
  model=gnn \
  data=graph \
  data.dataset_name=PubMed \
  model.hidden_channels=128 \
  trainer.max_epochs=200
```

### Graph Classification with Batching

Train on graph-level tasks with proper batching:

```bash
# PROTEINS dataset (biological graphs)
python src/train.py \
  model=gnn \
  data=graph \
  data.dataset_name=PROTEINS \
  data.batch_size=32 \
  data.use_batching=true

# MUTAG dataset (molecular graphs)
python src/train.py \
  model=gnn \
  data=graph \
  data.dataset_name=MUTAG \
  data.batch_size=64
```

### Large Graph Sampling

For very large graphs, use neighbor sampling:

```bash
# Reddit dataset (large social network)
python src/train.py \
  model=gnn \
  data=graph \
  data.dataset_name=Reddit \
  data.use_sampling=true \
  data.num_neighbors=[15,10,5] \
  data.batch_size=1024

# OGB-Products (large e-commerce graph)
python src/train.py \
  model=gnn \
  data=graph \
  data.dataset_name=ogbn-products \
  data.use_sampling=true \
  data.num_neighbors=[25,20,15] \
  data.batch_size=2048
```

### Sampling Strategies

**Neighbor Sampling:**

- `num_neighbors=[15,10,5]` - Sample 15 neighbors at layer 0, 10 at layer 1, 5 at layer 2
- Best for: Very large graphs (millions of nodes)
- Trade-off: Faster training, approximate gradients

**GraphSAINT Sampling:**

```bash
python src/train.py \
  model=gnn \
  data=graph \
  data.sampler=graphsaint \
  data.walk_length=3 \
  data.num_steps=30
```

**ClusterGCN Sampling:**

```bash
python src/train.py \
  model=gnn \
  data=graph \
  data.sampler=cluster \
  data.num_clusters=1500
```

## GNN-Specific Monitoring Metrics

Track these metrics specific to graph learning:

**Node-level tasks:**

- Node classification accuracy
- Per-class F1 scores
- Homophily ratio (same-class neighbor ratio)

**Graph-level tasks:**

- Graph classification accuracy
- ROC-AUC for binary classification
- Mean average precision for multi-label

**Training health metrics:**

- **Over-smoothing detection**: Node representations becoming too similar

  ```python
  # Log pairwise cosine similarity of node embeddings
  from torch.nn.functional import cosine_similarity

  def validation_step(self, batch, batch_idx):
      # ... existing code ...

      # Check over-smoothing
      h = self.get_node_embeddings(batch)
      pairwise_sim = cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=2)
      avg_sim = pairwise_sim.mean()

      self.log("val/avg_node_similarity", avg_sim)
      # Warning: If avg_sim > 0.9, likely over-smoothing
  ```

- **Graph connectivity statistics**: Number of connected components, average degree
- **Layer-wise gradient norms**: Check if gradients vanish in deep GNNs

## GNN Model Architectures

### Message Passing Layers

**GCN (Graph Convolutional Network):**

```python
from torch_geometric.nn import GCNConv

class GCN(LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
```

**GAT (Graph Attention Network):**

```python
from torch_geometric.nn import GATConv

class GAT(LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

**GraphSAGE:**

```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(LightningModule):
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

## Common GNN Training Issues

**Over-smoothing:**

- **Problem**: Node representations become indistinguishable in deep GNNs
- **Solution**:
  - Limit number of layers (2-4 typically)
  - Use skip connections (JumpingKnowledge)
  - Add layer normalization
  - Use adaptive depth (learned stop condition)

**Exploding/vanishing gradients:**

- **Problem**: Gradients become too large or too small in deep GNNs
- **Solution**:
  - Use gradient clipping
  - Add batch/layer normalization
  - Use residual connections
  - Careful initialization

**Memory issues with large graphs:**

- **Problem**: Full graph doesn't fit in GPU memory
- **Solution**:
  - Use sampling (neighbor/cluster/random walk)
  - Increase num_workers for faster sampling
  - Use CPU-GPU offloading
  - Reduce batch size

**Class imbalance in node classification:**

- **Problem**: Skewed class distribution
- **Solution**:
  - Use weighted cross-entropy loss
  - Oversample minority classes
  - Use focal loss
  - Modify evaluation metrics (F1 instead of accuracy)

## Example Configuration

**GNN training config:**

```yaml
# configs/experiment/gnn_node_classification.yaml
defaults:
  - override /model: gcn
  - override /data: cora
  - override /trainer: default

model:
  in_channels: 1433  # Cora feature dimension
  hidden_channels: 128
  out_channels: 7  # Number of classes
  num_layers: 3
  dropout: 0.5
  learning_rate: 0.01
  weight_decay: 5e-4

data:
  dataset_name: Cora
  split_type: public  # Use standard split
  use_sampling: false  # Full-batch training

trainer:
  max_epochs: 200
  accelerator: gpu
  devices: 1
  check_val_every_n_epoch: 1

callbacks:
  early_stopping:
    monitor: val/acc
    patience: 50
    mode: max
```

## Performance Tips

**For small graphs (< 10K nodes):**

- Use full-batch training (no sampling)
- Can use deeper networks (4-6 layers)
- Fast convergence (50-200 epochs)

**For medium graphs (10K - 1M nodes):**

- Use neighbor sampling
- 2-3 layers typically sufficient
- Larger batch sizes (512-2048)

**For very large graphs (> 1M nodes):**

- Essential to use sampling
- Consider ClusterGCN or GraphSAINT
- May need distributed training
- Focus on sampling efficiency

**Multi-GPU training for GNNs:**

```bash
# Partition graph across GPUs
python src/train.py \
  model=gnn \
  data=graph \
  trainer.devices=4 \
  trainer.strategy=ddp \
  data.use_sampling=true
```

## Heterogeneous Graphs

For graphs with multiple node/edge types:

```python
from torch_geometric.nn import HeteroConv, SAGEConv

class HeteroGNN(LightningModule):
    def __init__(self, metadata, hidden_channels):
        super().__init__()

        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        })

        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        })
```

Training heterogeneous graphs:

```bash
python src/train.py \
  model=hetero_gnn \
  data=hetero_graph \
  data.dataset_name=DBLP
```

## Resources

- **PyTorch Geometric examples**: [GitHub](https://github.com/pyg-team/pytorch_geometric/tree/master/examples)
- **OGB benchmarks**: [Open Graph Benchmark](https://ogb.stanford.edu/)
- **GNN papers**: [Distill GNN guide](https://distill.pub/2021/gnn-intro/)
