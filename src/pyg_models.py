"""PyTorch Geometric models for the node-feature experiment.

Reproduces the architectures from ``Paper_NodeFeatures``:
    gcn       - 2-layer GCNConv
    gat       - 2-layer GATConv (multi-head)
    gcn_aug   - 3-layer GCN + BatchNorm + dropout + edge dropout
    appnp     - APPNP propagation over a 2-layer MLP
    gcn_edge  - 2-layer GCN with learned edge weights from edge features

Each entry in ``MODEL_SPECS`` carries the exact training hyperparameters used
in the notebook.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, APPNP

from metrics import report_metrics


def _dropout_edge(edge_index, p, training):
    """Edge dropout that works across PyG versions (dropout_edge or dropout_adj)."""
    try:
        from torch_geometric.utils import dropout_edge
        return dropout_edge(edge_index, p=p, training=training)[0]
    except Exception:
        from torch_geometric.utils import dropout_adj
        return dropout_adj(edge_index, p=p, training=training)[0]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, **_):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x = torch.relu(self.conv1(data.x, data.edge_index))
        return self.conv2(x, data.edge_index)


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=3, heads=16, **_):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, data):
        x = torch.relu(self.conv1(data.x, data.edge_index))
        return self.conv2(x, data.edge_index)


class GCNWithAugmentation(nn.Module):
    def __init__(self, input_dim, hidden_dim1=None, hidden_dim2=32, output_dim=3,
                 dropout=0.5, edge_dropout_p=0.2, **_):
        super().__init__()
        hidden_dim1 = hidden_dim1 or input_dim // 2
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.edge_dropout_p = edge_dropout_p

    def forward(self, data):
        edge_index = _dropout_edge(data.edge_index, self.edge_dropout_p, self.training)
        x = self.dropout(torch.relu(self.bn1(self.conv1(data.x, edge_index))))
        x = self.dropout(torch.relu(self.bn2(self.conv2(x, edge_index))))
        return self.conv3(x, edge_index)


class APPNPNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3,
                 dropout=0.5, K=10, alpha=0.1, edge_dropout_p=0.2, **_):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.prop = APPNP(K=K, alpha=alpha, cached=False)
        self.edge_dropout_p = edge_dropout_p

    def forward(self, data):
        edge_index = data.edge_index
        if self.training and self.edge_dropout_p > 0:
            edge_index = _dropout_edge(edge_index, self.edge_dropout_p, True)
        x = self.dropout(F.relu(self.lin1(data.x)))
        x = self.lin2(x)
        return self.prop(x, edge_index)


class EdgeWeightedGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, edge_feat_dim=4, **_):
        super().__init__()
        self.edge_mlp = nn.Linear(edge_feat_dim, 1)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        ew = torch.sigmoid(self.edge_mlp(data.edge_attr)).squeeze()
        x = torch.relu(self.conv1(data.x, data.edge_index, edge_weight=ew))
        return self.conv2(x, data.edge_index, edge_weight=ew)


# name -> (class, model kwargs, training kwargs)
MODEL_SPECS: Dict[str, tuple] = {
    "gcn": (GCN, dict(hidden_dim=128), dict(lr=0.0005, weight_decay=0.0, epochs=200)),
    "gat": (GAT, dict(hidden_dim=32, heads=16), dict(lr=0.0005, weight_decay=5e-4, epochs=20)),
    "gcn_aug": (GCNWithAugmentation, dict(hidden_dim2=32, dropout=0.5),
                dict(lr=0.01, weight_decay=1e-4, epochs=100, patience=10, label_smoothing=0.1)),
    "appnp": (APPNPNet, dict(hidden_dim=64, K=10, alpha=0.1, dropout=0.5),
              dict(lr=0.01, weight_decay=5e-4, epochs=200, patience=10, label_smoothing=0.1)),
    "gcn_edge": (EdgeWeightedGCN, dict(hidden_dim=128), dict(lr=0.0005, weight_decay=0.0, epochs=200)),
}


# --------------------------------------------------------------------------- #
# Data + training
# --------------------------------------------------------------------------- #
def build_pyg_data(node_x, edges, edge_attr, labels, cfg) -> Data:
    """Wrap numpy arrays into a PyG ``Data`` object with train/val/test masks."""
    from sklearn.model_selection import train_test_split

    data = Data(
        x=torch.tensor(node_x, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=torch.tensor(labels, dtype=torch.long),
    )
    idx = np.arange(data.num_nodes)
    train_idx, holdout = train_test_split(idx, test_size=cfg.test_size, random_state=cfg.seed)
    val_idx, test_idx = train_test_split(holdout, test_size=cfg.val_test_size, random_state=cfg.seed)
    for name, ids in (("train_mask", train_idx), ("val_mask", val_idx), ("test_mask", test_idx)):
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[ids] = True
        setattr(data, name, mask)
    return data


def train_eval(data: Data, model_name: str, cfg) -> dict:
    """Train ``model_name`` and report test metrics. Returns the metrics dict."""
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_SPECS)}")
    model_cls, model_kwargs, train_kwargs = MODEL_SPECS[model_name]

    device = torch.device("cuda" if (cfg.device != "cpu" and torch.cuda.is_available()) else "cpu")
    data = data.to(device)
    edge_feat_dim = data.edge_attr.shape[1] if data.edge_attr is not None else 4
    model = model_cls(
        input_dim=data.num_node_features, output_dim=cfg.num_classes,
        edge_feat_dim=edge_feat_dim, **model_kwargs,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_kwargs["lr"], weight_decay=train_kwargs["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=train_kwargs.get("label_smoothing", 0.0))

    patience = train_kwargs.get("patience")
    best_val, best_state, counter = float("inf"), None, 0
    for epoch in range(train_kwargs["epochs"]):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(data)[data.val_mask], data.y[data.val_mask]).item()
        print(f"Epoch {epoch + 1:03d} | Train {loss.item():.4f} | Val {val_loss:.4f}")

        if patience is not None:
            if val_loss < best_val:
                best_val, best_state, counter = val_loss, model.state_dict(), 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()
    return report_metrics(f"PyG {model_name} [test]", y_true, y_pred, show_report=True)
