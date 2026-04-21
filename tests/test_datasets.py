"""Tests for dataset classes and collate functions."""

from pathlib import Path

import numpy as np
import pytest
import torch

from logd_predictor.datasets import FingerprintDataset, GraphDataset
from logd_predictor.featurize import GRAPH_EDGE_DIM, GRAPH_NODE_DIM
from logd_predictor.models import BatchedGraph

N_MOLS = 10
FP_DIM = 256


def _write_graph_npz(path: Path, n: int = N_MOLS) -> None:
    """Write a minimal graph dataset npz file."""
    nodes_per = 5
    edges_per = 4

    n_nodes = np.full(n, nodes_per, dtype=np.int64)
    n_edges = np.full(n, edges_per, dtype=np.int64)
    graph_offsets = np.concatenate([[0], np.cumsum(n_nodes[:-1])])
    edge_offsets = np.concatenate([[0], np.cumsum(n_edges[:-1])])

    total_nodes = nodes_per * n
    total_edges = edges_per * n

    # Local edge indices (0..nodes_per-1) repeated for each graph
    local_ei = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    edge_index = np.tile(local_ei, n)  # still local - matches GraphDataset expectation

    np.savez(
        path / "data.npz",
        node_feats=np.random.randn(total_nodes, GRAPH_NODE_DIM).astype(np.float32),
        edge_feats=np.random.randn(total_edges, GRAPH_EDGE_DIM).astype(np.float32),
        edge_index=edge_index,
        graph_offsets=graph_offsets,
        edge_offsets=edge_offsets,
        n_nodes=n_nodes,
        n_edges=n_edges,
        targets=np.random.randn(n).astype(np.float32),
    )


def _write_fp_npz(path: Path, n: int = N_MOLS) -> None:
    np.savez(
        path / "data.npz",
        X=np.random.randn(n, FP_DIM).astype(np.float32),
        targets=np.random.randn(n).astype(np.float32),
    )


@pytest.fixture()
def graph_dir(tmp_path):
    _write_graph_npz(tmp_path)
    return tmp_path


@pytest.fixture()
def fp_dir(tmp_path):
    _write_fp_npz(tmp_path)
    return tmp_path


class TestGraphDataset:
    def test_len(self, graph_dir):
        ds = GraphDataset(str(graph_dir))
        assert len(ds) == N_MOLS

    def test_max_samples(self, graph_dir):
        ds = GraphDataset(str(graph_dir), max_samples=3)
        assert len(ds) == 3

    def test_item_shapes(self, graph_dir):
        ds = GraphDataset(str(graph_dir))
        node_feats, edge_feats, edge_index, target = ds[0]
        assert node_feats.shape == (5, GRAPH_NODE_DIM)
        assert edge_feats.shape == (4, GRAPH_EDGE_DIM)
        assert edge_index.shape == (2, 4)
        assert target.shape == ()


class TestGraphDatasetCollate:
    def test_collate_returns_batched_graph(self, graph_dir):
        ds = GraphDataset(str(graph_dir))
        batch = [ds[i] for i in range(4)]
        g, targets = GraphDataset.collate(batch)
        assert isinstance(g, BatchedGraph)
        assert targets.shape == (4,)

    def test_total_nodes(self, graph_dir):
        ds = GraphDataset(str(graph_dir))
        batch_size = 4
        batch = [ds[i] for i in range(batch_size)]
        g, _ = GraphDataset.collate(batch)
        assert g.node_features.shape[0] == 5 * batch_size

    def test_total_edges(self, graph_dir):
        ds = GraphDataset(str(graph_dir))
        batch = [ds[i] for i in range(4)]
        g, _ = GraphDataset.collate(batch)
        assert g.edge_features.shape[0] == 4 * 4

    def test_batch_vector_values(self, graph_dir):
        ds = GraphDataset(str(graph_dir))
        batch = [ds[i] for i in range(3)]
        g, _ = GraphDataset.collate(batch)
        # Each graph has 5 nodes; batch should be [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2]
        expected = torch.repeat_interleave(torch.arange(3), 5)
        assert torch.equal(g.batch, expected)

    def test_edge_index_global_after_offset(self, graph_dir):
        ds = GraphDataset(str(graph_dir))
        batch = [ds[i] for i in range(2)]
        g, _ = GraphDataset.collate(batch)
        # First graph: local indices 0-4, second graph: global indices 5-9
        src, dst = g.edge_index
        assert src.max().item() < 10
        assert dst.max().item() < 10


class TestFingerprintDataset:
    def test_len(self, fp_dir):
        ds = FingerprintDataset(str(fp_dir))
        assert len(ds) == N_MOLS

    def test_max_samples(self, fp_dir):
        ds = FingerprintDataset(str(fp_dir), max_samples=5)
        assert len(ds) == 5

    def test_item_shape(self, fp_dir):
        ds = FingerprintDataset(str(fp_dir))
        x, y = ds[0]
        assert x.shape == (FP_DIM,)
        assert y.shape == ()

    def test_dtype(self, fp_dir):
        ds = FingerprintDataset(str(fp_dir))
        x, y = ds[0]
        assert x.dtype == torch.float32
