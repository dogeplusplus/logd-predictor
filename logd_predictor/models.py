from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BatchedGraph:
    node_features: torch.Tensor  # (total_nodes, node_feat_dim)
    edge_features: torch.Tensor  # (total_edges, edge_feat_dim)
    edge_index: torch.Tensor  # (2, total_edges) - global node indices
    batch: torch.Tensor  # (total_nodes,)   - which graph each node belongs to


class MLPRegressor(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden: int = 200,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_features
        for _ in range(n_layers):
            layers += [nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            dim = hidden
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GCNLayer(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.norm = nn.LayerNorm(out_feats)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        # Sum-aggregate neighbour features (with implicit self-loop via residual)
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.size(1)), h[src])
        return self.drop(F.relu(self.norm(self.linear(agg + h))))


class GCNRegressor(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        hidden: int = 200,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_feat_dim, hidden)
        self.layers = nn.ModuleList(
            [_GCNLayer(hidden, hidden, dropout) for _ in range(n_layers)]
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, g: BatchedGraph) -> torch.Tensor:
        h = F.relu(self.input_proj(g.node_features))
        for layer in self.layers:
            h = layer(h, g.edge_index)
        # Global mean-pool per graph
        n_graphs = int(g.batch.max().item()) + 1
        pooled = torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        pooled.scatter_add_(0, g.batch.unsqueeze(1).expand_as(h), h)
        counts = (
            torch.bincount(g.batch, minlength=n_graphs)
            .to(h.dtype)
            .clamp(min=1)
            .unsqueeze(1)
        )
        return self.out(pooled / counts)


class _AttentiveFPLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.attn = nn.Linear(2 * node_dim + edge_dim, 1)
        self.msg = nn.Linear(node_dim + edge_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        attn_in = torch.cat([h[src], h[dst], edge_feat], dim=1)
        alpha = torch.sigmoid(self.attn(attn_in)).to(h.dtype)
        msg_in = torch.cat([h[src], edge_feat], dim=1)
        msgs = alpha * F.relu(self.msg(msg_in))
        agg = torch.zeros(h.size(0), msgs.size(1), device=h.device, dtype=msgs.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msgs), msgs)
        return F.relu(self.norm(agg))


class AttentiveFPRegressor(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden: int = 200,
        n_layers: int = 2,
        num_timesteps: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_feat_dim, hidden)
        self.edge_proj = nn.Linear(edge_feat_dim, hidden)
        self.mp_layers = nn.ModuleList(
            [_AttentiveFPLayer(hidden, hidden, hidden) for _ in range(n_layers)]
        )
        # Readout: num_timesteps rounds of graph-level attention
        self.readout_attn = nn.Linear(hidden, 1)
        self.readout_gru = nn.GRUCell(hidden, hidden)
        self.num_timesteps = num_timesteps
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, g: BatchedGraph) -> torch.Tensor:
        h = F.relu(self.node_proj(g.node_features))
        e = F.relu(self.edge_proj(g.edge_features))

        for layer in self.mp_layers:
            h = self.drop(layer(h, g.edge_index, e))

        # Attentive readout per graph
        n_graphs = int(g.batch.max().item()) + 1
        # Initial graph state: mean-pool
        graph_h = torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        graph_h.scatter_add_(0, g.batch.unsqueeze(1).expand_as(h), h)
        counts = (
            torch.bincount(g.batch, minlength=n_graphs)
            .to(h.dtype)
            .clamp(min=1)
            .unsqueeze(1)
        )
        graph_h = graph_h / counts

        for _ in range(self.num_timesteps):
            # Node-level attention from graph context
            ctx = graph_h[g.batch]  # broadcast graph state to nodes
            alpha = torch.sigmoid(self.readout_attn(h * ctx))
            # Weighted sum - cast to context dtype in case GRUCell promoted graph_h to float32
            context = torch.zeros_like(graph_h)
            weighted = (alpha * h).to(context.dtype)
            context.scatter_add_(0, g.batch.unsqueeze(1).expand_as(weighted), weighted)
            graph_h = self.drop(self.readout_gru(context, graph_h))

        return self.out(graph_h)


# Matches the RDKit-based featurizer in ml_pipeline.py.
# Node: 13+7+6+5+1+1+1 = 34  |  Edge: 4+1+1+6 = 12
_GRAPH_NODE_DIM = 34
_GRAPH_EDGE_DIM = 12


def build_net(
    model_type: str,
    featurizer_type: str,
    in_features: int,
    hidden: int = 200,
    n_layers: int = 2,
    num_timesteps: int = 2,
    dropout: float = 0.2,
    node_feat_dim: int = _GRAPH_NODE_DIM,
    edge_feat_dim: int = _GRAPH_EDGE_DIM,
) -> nn.Module:
    """Instantiate the appropriate nn.Module for a given model + featurizer combo."""
    if model_type == "RandomForest":
        raise ValueError("RandomForest is not a PyTorch model - use sklearn directly.")

    if featurizer_type == "MolGraphConv":
        if model_type == "GCN":
            return GCNRegressor(
                node_feat_dim=node_feat_dim,
                hidden=hidden,
                n_layers=n_layers,
                dropout=dropout,
            )
        if model_type == "AttentiveFP":
            return AttentiveFPRegressor(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden=hidden,
                n_layers=n_layers,
                num_timesteps=num_timesteps,
                dropout=dropout,
            )
        raise ValueError(f"Unknown graph model: {model_type}")

    # Fingerprint / descriptor models → MLP
    return MLPRegressor(
        in_features=in_features,
        hidden=hidden,
        n_layers=n_layers,
        dropout=dropout,
    )
