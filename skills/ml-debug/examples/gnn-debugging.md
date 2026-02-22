# GNN Debugging Guide (PyTorch Geometric)

Common issues and solutions for Graph Neural Networks with PyTorch Geometric.

## 1. Over-smoothing

**Symptoms:**

- All node representations become similar
- Performance degrades with more layers
- Node embeddings lose distinctiveness

**Diagnosis:**

```python
# Check embedding similarity
import torch.nn.functional as F

def check_oversmoothing(embeddings):
    """Measure embedding similarity (higher = more over-smooth)."""
    # Normalize embeddings
    norm_emb = F.normalize(embeddings, p=2, dim=1)

    # Compute pairwise cosine similarity
    similarity = torch.mm(norm_emb, norm_emb.t())

    # Average similarity (excluding diagonal)
    mask = ~torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    avg_sim = similarity[mask].mean().item()

    print(f"Average embedding similarity: {avg_sim:.4f}")
    if avg_sim > 0.9:
        print("⚠️  HIGH OVER-SMOOTHING DETECTED")

    return avg_sim
```

**Solutions:**

### Solution 1: Reduce Number of Layers

```yaml
model:
  num_layers: 2  # Start with 2-3 layers for most graphs
```

Most graphs don't need deep GNNs:

- Citation networks (Cora, CiteSeer): 2-3 layers
- Social networks: 2-4 layers
- Molecular graphs: 3-5 layers

### Solution 2: Add Skip Connections

```python
class GNNWithSkip(nn.Module):
    def forward(self, x, edge_index):
        x_orig = x

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)

            # Skip connection
            if i > 0:  # Skip first layer
                x = x + x_orig  # Or: x = 0.5 * x + 0.5 * x_orig

            x_orig = x  # Update for next skip

        return x
```

### Solution 3: Use Jumping Knowledge

```yaml
model:
  jk_mode: "cat"  # Concatenate representations from all layers
```

```python
from torch_geometric.nn import JumpingKnowledge

class GNNWithJK(nn.Module):
    def __init__(self, jk_mode='cat'):
        super().__init__()
        self.convs = nn.ModuleList([...])
        self.jk = JumpingKnowledge(mode=jk_mode)  # 'cat', 'max', 'lstm'

    def forward(self, x, edge_index):
        xs = []

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            xs.append(x)

        # Aggregate representations from all layers
        x = self.jk(xs)
        return x
```

### Solution 4: Use Initial Residual Connections

```python
class GNNWithInitialResidual(nn.Module):
    def forward(self, x, edge_index):
        x_init = x  # Save initial features

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # Add initial features
        x = x + self.skip_proj(x_init)  # Project if dimensions differ
        return x
```

## 2. Large Graph OOM

**Symptoms:**

- CUDA out of memory
- Graph doesn't fit in GPU memory
- Training on full graph crashes

**Solutions:**

### Solution 1: Mini-Batch Training with NeighborSampler

```yaml
data:
  use_sampling: true
  batch_size: 1024  # Number of nodes per batch
  num_neighbors: [15, 10, 5]  # Neighbors sampled per layer
```

```python
from torch_geometric.loader import NeighborLoader

# In DataModule
def train_dataloader(self):
    return NeighborLoader(
        self.data,
        num_neighbors=[15, 10, 5],  # For 3-layer GNN
        batch_size=1024,
        shuffle=True,
        num_workers=4,
    )
```

### Solution 2: Cluster-GCN

```python
from torch_geometric.loader import ClusterData, ClusterLoader

# Partition graph into clusters
cluster_data = ClusterData(data, num_parts=1500)

# Create loader
loader = ClusterLoader(cluster_data, batch_size=20, shuffle=True)
```

### Solution 3: GraphSAINT Sampling

```python
from torch_geometric.loader import GraphSAINTRandomWalkSampler

loader = GraphSAINTRandomWalkSampler(
    data,
    batch_size=6000,
    walk_length=2,
    num_steps=30,
    sample_coverage=100,
    num_workers=4,
)
```

### Solution 4: Mixed Precision + Gradient Checkpointing

```yaml
trainer:
  precision: "16-mixed"

model:
  use_checkpoint: true  # Enable gradient checkpointing
```

```python
# In model
from torch.utils.checkpoint import checkpoint

def forward(self, x, edge_index):
    for conv in self.convs:
        # Use checkpointing for memory efficiency
        x = checkpoint(conv, x, edge_index, use_reentrant=False)
    return x
```

## 3. Heterogeneous Graph Issues

**Symptoms:**

- Inconsistent edge types
- Node type mismatches
- Dimension mismatches between node types

**Solutions:**

### Use Heterogeneous GNN Layers

```python
from torch_geometric.nn import HeteroConv, GCNConv

class HeteroGNN(nn.Module):
    def __init__(self, metadata):
        super().__init__()

        # Define convolutions for each edge type
        self.conv1 = HeteroConv({
            ('user', 'rates', 'movie'): GCNConv(-1, 64),
            ('movie', 'rated_by', 'user'): GCNConv(-1, 64),
            ('user', 'friends', 'user'): GCNConv(-1, 64),
        }, aggr='mean')

    def forward(self, x_dict, edge_index_dict):
        # x_dict: {'user': tensor, 'movie': tensor}
        # edge_index_dict: {('user', 'rates', 'movie'): edge_index, ...}

        x_dict = self.conv1(x_dict, edge_index_dict)

        # Apply activation per node type
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        return x_dict
```

### Handle Different Node Feature Dimensions

```python
class HeteroGNNWithProjection(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels):
        super().__init__()

        # Project each node type to common dimension
        self.projections = nn.ModuleDict({
            node_type: nn.Linear(in_ch, hidden_channels)
            for node_type, in_ch in in_channels_dict.items()
        })

        # Heterogeneous convolution
        self.conv = HeteroConv({
            edge_type: GCNConv(hidden_channels, hidden_channels)
            for edge_type in metadata[1]  # Edge types
        })

    def forward(self, x_dict, edge_index_dict):
        # Project to common dimension
        x_dict = {
            node_type: self.projections[node_type](x)
            for node_type, x in x_dict.items()
        }

        # Apply GNN
        x_dict = self.conv(x_dict, edge_index_dict)
        return x_dict
```

## 4. Edge Feature Handling

**Symptoms:**

- Edge features not used
- Dimension mismatches with edge_attr
- Edge features cause OOM

**Solutions:**

### Use Edge-Conditioned Convolutions

```python
from torch_geometric.nn import GATConv, TransformerConv

# GAT with edge features
conv = GATConv(
    in_channels,
    out_channels,
    heads=8,
    edge_dim=edge_feat_dim,  # Edge feature dimension
)

# Transformer with edge features
conv = TransformerConv(
    in_channels,
    out_channels,
    heads=8,
    edge_dim=edge_feat_dim,
)
```

### Reduce Edge Feature Dimensions

```python
class GNNWithEdgeProjection(nn.Module):
    def __init__(self, edge_dim, reduced_dim=8):
        super().__init__()

        # Project high-dim edge features to lower dim
        self.edge_proj = nn.Linear(edge_dim, reduced_dim)

        self.conv = GATConv(in_channels, out_channels, edge_dim=reduced_dim)

    def forward(self, x, edge_index, edge_attr):
        # Reduce edge feature dimension
        edge_attr = self.edge_proj(edge_attr)

        # Apply convolution
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        return x
```

## 5. Global Pooling Issues

**Symptoms:**

- Poor graph-level predictions
- Loss not decreasing
- NaN in global pooling

**Solutions:**

### Try Different Global Pooling

```yaml
model:
  global_pool: "mean"  # Try: mean, max, add, attention, set2set
```

```python
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GlobalAttention, Set2Set

# Mean pooling (simple, robust)
x = global_mean_pool(x, batch)

# Max pooling (good for important features)
x = global_max_pool(x, batch)

# Attention pooling (learnable, more expressive)
gate_nn = nn.Linear(hidden_channels, 1)
pool = GlobalAttention(gate_nn)
x = pool(x, batch)

# Set2Set (most expressive, but slower)
pool = Set2Set(hidden_channels, processing_steps=3)
x = pool(x, batch)
```

### Hierarchical Pooling

```python
from torch_geometric.nn import TopKPooling

class HierarchicalGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.5)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.5)

    def forward(self, x, edge_index, batch):
        # Layer 1
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        # Layer 2
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        # Global pooling
        x = global_mean_pool(x, batch)
        return x
```

## 6. Directed vs Undirected Graphs

**Symptoms:**

- Poor performance on directed graphs
- Asymmetric message passing needed
- Undirected assumption fails

**Solutions:**

### Make Graph Undirected

```python
from torch_geometric.utils import to_undirected

# Convert directed edges to undirected
edge_index = to_undirected(edge_index)
```

### Use Directed GNN Layers

```python
# Separate convolutions for in/out edges
class DirectedGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = GCNConv(in_channels, out_channels)
        self.conv_out = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Split edge_index into incoming and outgoing
        row, col = edge_index

        # Incoming messages
        edge_index_in = torch.stack([col, row])  # Reverse
        x_in = self.conv_in(x, edge_index_in)

        # Outgoing messages
        x_out = self.conv_out(x, edge_index)

        # Combine
        x = (x_in + x_out) / 2
        return x
```

## 7. Temporal Graph Issues

**Symptoms:**

- Time information not used
- Data leakage across time
- Train/val/test splits by time needed

**Solutions:**

### Temporal Train/Val/Test Split

```python
def temporal_split(data, ratios=[0.7, 0.15, 0.15]):
    """Split graph data by time."""
    timestamps = data.t  # Assume data has timestamps

    # Sort by time
    sorted_indices = torch.argsort(timestamps)
    n = len(sorted_indices)

    # Split indices
    train_end = int(n * ratios[0])
    val_end = int(n * (ratios[0] + ratios[1]))

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[sorted_indices[:train_end]] = True
    val_mask[sorted_indices[train_end:val_end]] = True
    test_mask[sorted_indices[val_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data
```

### Use Temporal GNN

```python
from torch_geometric.nn import TGNMemory, TransformerConv

class TemporalGNN(nn.Module):
    def __init__(self, num_nodes, raw_msg_dim, memory_dim):
        super().__init__()

        # TGN memory module
        self.memory = TGNMemory(
            num_nodes,
            raw_msg_dim,
            memory_dim,
            time_dim=memory_dim,
        )

        # GNN layers
        self.conv = TransformerConv(memory_dim, memory_dim)

    def forward(self, n_id, edge_index, t, msg):
        # Update memory
        self.memory.update_state(n_id, msg, t)

        # Get memory
        z = self.memory(n_id)

        # Apply GNN
        z = self.conv(z, edge_index)
        return z
```

## Debugging Checklist for GNNs

**Graph Structure:**

- [ ] Edge index is [2, num_edges] shape
- [ ] Node indices in range [0, num_nodes)
- [ ] No self-loops (unless intended)
- [ ] Undirected if needed (edges in both directions)

**Node Features:**

- [ ] Shape is [num_nodes, num_features]
- [ ] No NaN or Inf values
- [ ] Properly normalized

**Edge Features:**

- [ ] Shape is [num_edges, edge_dim]
- [ ] Matches edge_index length
- [ ] Compatible with GNN layer (check edge_dim support)

**Batching:**

- [ ] Batch index is correct
- [ ] Global node indexing is correct
- [ ] Edge indices offset by batch

**Model:**

- [ ] Number of layers appropriate (2-5)
- [ ] Global pooling for graph-level tasks
- [ ] Aggregation function appropriate (mean, max, add)

**Training:**

- [ ] No data leakage (proper train/val/test split)
- [ ] Temporal split for temporal graphs
- [ ] Appropriate loss for task (node vs graph)

## Common GNN Errors

**"IndexError: index out of range":**

```python
# Check edge_index validity
assert edge_index.max() < num_nodes, "Edge index out of range"
assert edge_index.min() >= 0, "Negative indices in edge_index"
```

**"RuntimeError: Expected tensor for argument #1 'index' to have scalar type Long":**

```python
# Ensure edge_index is Long type
edge_index = edge_index.long()
```

**"RuntimeError: Sizes of tensors must match":**

- Check node features match num_nodes
- Verify batch index length
- Ensure edge features match num_edges

**"No batch dimension":**

```python
# Add batch for single graph
from torch_geometric.data import Batch

if not hasattr(data, 'batch'):
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
```
