"""Tests for PyTorch model forward passes."""

import pytest
import torch

from logd_predictor.models import (
    AttentiveFPRegressor,
    BatchedGraph,
    GCNRegressor,
    MLPRegressor,
    build_net,
)

BATCH = 4
N_NODES = 20
N_EDGES = 40
NODE_DIM = 34
EDGE_DIM = 12
FP_DIM = 2048
HIDDEN = 64


def _make_batched_graph(batch_size: int = BATCH) -> BatchedGraph:
    """Synthetic batched graph with `batch_size` graphs of equal size."""
    nodes_per = N_NODES // batch_size
    edges_per = N_EDGES // batch_size
    total_nodes = nodes_per * batch_size
    total_edges = edges_per * batch_size

    node_feats = torch.randn(total_nodes, NODE_DIM)
    edge_feats = torch.randn(total_edges, EDGE_DIM)
    batch = torch.repeat_interleave(torch.arange(batch_size), nodes_per)

    # random valid edge_index (global indices within each graph's node range)
    src = torch.cat(
        [
            torch.randint(0, nodes_per, (edges_per,)) + i * nodes_per
            for i in range(batch_size)
        ]
    )
    dst = torch.cat(
        [
            torch.randint(0, nodes_per, (edges_per,)) + i * nodes_per
            for i in range(batch_size)
        ]
    )
    edge_index = torch.stack([src, dst])

    return BatchedGraph(
        node_features=node_feats,
        edge_features=edge_feats,
        edge_index=edge_index,
        batch=batch,
    )


class TestMLPRegressor:
    def test_output_shape(self):
        model = MLPRegressor(in_features=FP_DIM, hidden=HIDDEN, n_layers=2)
        x = torch.randn(BATCH, FP_DIM)
        out = model(x)
        assert out.shape == (BATCH, 1)

    def test_single_sample(self):
        model = MLPRegressor(in_features=FP_DIM, hidden=HIDDEN)
        out = model(torch.randn(1, FP_DIM))
        assert out.shape == (1, 1)

    def test_different_layer_counts(self):
        for n in [1, 3, 5]:
            model = MLPRegressor(in_features=FP_DIM, hidden=HIDDEN, n_layers=n)
            out = model(torch.randn(BATCH, FP_DIM))
            assert out.shape == (BATCH, 1)

    def test_dropout_in_eval_mode(self):
        model = MLPRegressor(in_features=FP_DIM, hidden=HIDDEN, dropout=0.9)
        model.eval()
        x = torch.randn(BATCH, FP_DIM)
        with torch.no_grad():
            out1, out2 = model(x), model(x)
        assert torch.allclose(out1, out2)


class TestGCNRegressor:
    def test_output_shape(self):
        model = GCNRegressor(node_feat_dim=NODE_DIM, hidden=HIDDEN, n_layers=2)
        g = _make_batched_graph()
        out = model(g)
        assert out.shape == (BATCH, 1)

    def test_single_graph(self):
        model = GCNRegressor(node_feat_dim=NODE_DIM, hidden=HIDDEN)
        g = _make_batched_graph(batch_size=1)
        out = model(g)
        assert out.shape == (1, 1)

    def test_no_nan(self):
        model = GCNRegressor(node_feat_dim=NODE_DIM, hidden=HIDDEN)
        g = _make_batched_graph()
        out = model(g)
        assert not torch.isnan(out).any()


class TestAttentiveFPRegressor:
    def test_output_shape(self):
        model = AttentiveFPRegressor(
            node_feat_dim=NODE_DIM, edge_feat_dim=EDGE_DIM, hidden=HIDDEN, n_layers=2
        )
        g = _make_batched_graph()
        out = model(g)
        assert out.shape == (BATCH, 1)

    def test_single_graph(self):
        model = AttentiveFPRegressor(
            node_feat_dim=NODE_DIM, edge_feat_dim=EDGE_DIM, hidden=HIDDEN
        )
        g = _make_batched_graph(batch_size=1)
        out = model(g)
        assert out.shape == (1, 1)

    def test_num_timesteps(self):
        for ts in [1, 3]:
            model = AttentiveFPRegressor(
                node_feat_dim=NODE_DIM,
                edge_feat_dim=EDGE_DIM,
                hidden=HIDDEN,
                num_timesteps=ts,
            )
            g = _make_batched_graph()
            out = model(g)
            assert out.shape == (BATCH, 1)

    def test_no_nan(self):
        model = AttentiveFPRegressor(
            node_feat_dim=NODE_DIM, edge_feat_dim=EDGE_DIM, hidden=HIDDEN
        )
        g = _make_batched_graph()
        out = model(g)
        assert not torch.isnan(out).any()


class TestBuildNet:
    def test_gcn(self):
        net = build_net("GCN", "MolGraphConv", in_features=FP_DIM, hidden=HIDDEN)
        assert isinstance(net, GCNRegressor)

    def test_attentivefp(self):
        net = build_net(
            "AttentiveFP", "MolGraphConv", in_features=FP_DIM, hidden=HIDDEN
        )
        assert isinstance(net, AttentiveFPRegressor)

    def test_mlp_circular(self):
        net = build_net("MLP", "CircularFingerprint", in_features=FP_DIM, hidden=HIDDEN)
        assert isinstance(net, MLPRegressor)

    def test_mlp_rdkit(self):
        net = build_net("MLP", "RDKitDescriptors", in_features=200, hidden=HIDDEN)
        assert isinstance(net, MLPRegressor)

    def test_random_forest_raises(self):
        with pytest.raises(ValueError, match="RandomForest"):
            build_net("RandomForest", "CircularFingerprint", in_features=FP_DIM)

    def test_unknown_graph_model_raises(self):
        with pytest.raises(ValueError, match="Unknown graph model"):
            build_net("UnknownModel", "MolGraphConv", in_features=FP_DIM)
