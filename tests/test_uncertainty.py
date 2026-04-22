"""Tests for uncertainty quantification utilities in predict_uncertainty.py."""

from logd_predictor.models import AttentiveFPRegressor, BatchedGraph
from logd_predictor.featurize import FeaturizerConfig, FeaturizerType
from predict_uncertainty import (
    _enable_dropout,
    _featurize_smiles,
    _graph_batches,
    _fp_batches,
)
import sys
from pathlib import Path

import numpy as np
import torch

# The script lives at project root; add it to path so we can import helpers.
sys.path.insert(0, str(Path(__file__).parent.parent))


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
ETHANOL = "CCO"
INVALID = "not_a_smiles"

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------


class TestConformaCalibration:
    def test_quantile_level_gives_coverage(self):
        """Calibration quantile should cover at least the requested fraction."""
        rng = np.random.default_rng(42)
        residuals = np.abs(rng.normal(0, 1, size=1000))
        coverage = 0.9
        n = len(residuals)
        q_level = min(np.ceil((n + 1) * coverage) / n, 1.0)
        q = float(np.quantile(residuals, q_level, method="higher"))
        actual_coverage = float(np.mean(residuals <= q))
        assert actual_coverage >= coverage

    def test_higher_coverage_gives_wider_interval(self):
        rng = np.random.default_rng(0)
        residuals = np.abs(rng.normal(0, 1, size=500))
        n = len(residuals)

        def q_at(cov: float) -> float:
            lvl = min(np.ceil((n + 1) * cov) / n, 1.0)
            return float(np.quantile(residuals, lvl, method="higher"))

        assert q_at(0.95) > q_at(0.90) >= q_at(0.80)

    def test_finite_sample_inflation(self):
        # With n=10 calibration points and 90% coverage, the quantile level
        # should be ceil(11 * 0.9) / 10 = ceil(9.9)/10 = 10/10 = 1.0
        residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        n = len(residuals)
        q_level = min(np.ceil((n + 1) * 0.9) / n, 1.0)
        assert q_level == 1.0  # must use the max residual


# ---------------------------------------------------------------------------
# MC Dropout activation
# ---------------------------------------------------------------------------


class TestEnableDropout:
    def test_dropout_enabled_during_eval(self):
        model = AttentiveFPRegressor(
            node_feat_dim=34,
            edge_feat_dim=12,
            hidden=32,
            n_layers=1,
            num_timesteps=1,
            dropout=0.5,
        )
        model.eval()
        _enable_dropout(model)
        import torch.nn as nn

        dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
        assert all(m.training for m in dropout_layers), (
            "Dropout layers should be in train mode"
        )

    def test_mc_dropout_produces_variance(self):
        """With dropout=0.5 and N passes, predictions should vary."""
        model = AttentiveFPRegressor(
            node_feat_dim=34,
            edge_feat_dim=12,
            hidden=32,
            n_layers=1,
            num_timesteps=1,
            dropout=0.5,
        )
        model.eval()
        _enable_dropout(model)

        g = _make_small_graph()
        preds = []
        with torch.no_grad():
            for _ in range(20):
                out = model(g).squeeze(-1).numpy()
                preds.append(out)

        stds = np.stack(preds).std(axis=0)
        assert stds.mean() > 0, "MC Dropout should produce non-zero variance"

    def test_full_eval_deterministic(self):
        """Without dropout enabled, repeated passes should be identical."""
        model = AttentiveFPRegressor(
            node_feat_dim=34,
            edge_feat_dim=12,
            hidden=32,
            n_layers=1,
            num_timesteps=1,
            dropout=0.5,
        )
        model.eval()  # dropout OFF

        g = _make_small_graph()
        preds = []
        with torch.no_grad():
            for _ in range(5):
                out = model(g).squeeze(-1).numpy()
                preds.append(out)

        for p in preds[1:]:
            np.testing.assert_array_equal(preds[0], p)


# ---------------------------------------------------------------------------
# In-memory featurization
# ---------------------------------------------------------------------------


class TestFeaturizeSmiles:
    def test_graph_valid_molecules(self):
        cfg = FeaturizerConfig(featurizer_type=FeaturizerType.MOL_GRAPH_CONV)
        valid_idx, graphs = _featurize_smiles([ASPIRIN, CAFFEINE], cfg)
        assert valid_idx == [0, 1]
        assert len(graphs) == 2
        assert all("node_feats" in g for g in graphs)

    def test_graph_skips_invalid(self):
        cfg = FeaturizerConfig(featurizer_type=FeaturizerType.MOL_GRAPH_CONV)
        valid_idx, graphs = _featurize_smiles([ASPIRIN, INVALID, CAFFEINE], cfg)
        assert valid_idx == [0, 2]
        assert len(graphs) == 2

    def test_fingerprint_valid(self):
        cfg = FeaturizerConfig(featurizer_type=FeaturizerType.CIRCULAR, fp_size=512)
        valid_idx, X = _featurize_smiles([ASPIRIN, CAFFEINE], cfg)
        assert valid_idx == [0, 1]
        assert X.shape == (2, 512)

    def test_fingerprint_skips_invalid(self):
        cfg = FeaturizerConfig(featurizer_type=FeaturizerType.CIRCULAR)
        valid_idx, X = _featurize_smiles([INVALID, ASPIRIN], cfg)
        assert valid_idx == [1]
        assert X.shape[0] == 1

    def test_rdkit_valid(self):
        cfg = FeaturizerConfig(featurizer_type=FeaturizerType.RDKIT)
        valid_idx, X = _featurize_smiles([ASPIRIN], cfg)
        assert valid_idx == [0]
        assert X.ndim == 2 and X.shape[0] == 1
        assert np.all(np.isfinite(X))

    def test_all_invalid_returns_empty(self):
        cfg = FeaturizerConfig(featurizer_type=FeaturizerType.MOL_GRAPH_CONV)
        valid_idx, graphs = _featurize_smiles([INVALID, "???"], cfg)
        assert valid_idx == []
        assert graphs == []


class TestGraphBatches:
    def test_batches_cover_all_graphs(self):
        cfg = FeaturizerConfig(featurizer_type=FeaturizerType.MOL_GRAPH_CONV)
        smiles = [ASPIRIN, CAFFEINE, ETHANOL, "c1ccccc1"]
        _, graphs = _featurize_smiles(smiles, cfg)
        batches = list(_graph_batches(graphs, batch_size=2, device=DEVICE))
        assert len(batches) == 2  # 4 graphs / batch_size 2

    def test_batch_is_batched_graph(self):
        cfg = FeaturizerConfig(featurizer_type=FeaturizerType.MOL_GRAPH_CONV)
        _, graphs = _featurize_smiles([ASPIRIN, CAFFEINE], cfg)
        batches = list(_graph_batches(graphs, batch_size=4, device=DEVICE))
        assert len(batches) == 1
        assert isinstance(batches[0], BatchedGraph)


class TestFpBatches:
    def test_batches_cover_all(self):
        X = np.random.randn(10, 512).astype(np.float32)
        batches = list(_fp_batches(X, batch_size=3, device=DEVICE))
        total = sum(b.shape[0] for b in batches)
        assert total == 10

    def test_last_batch_smaller(self):
        X = np.random.randn(10, 512).astype(np.float32)
        batches = list(_fp_batches(X, batch_size=3, device=DEVICE))
        assert batches[-1].shape[0] == 1  # 10 % 3 == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_graph(n_mols: int = 2) -> BatchedGraph:
    nodes_per = 8
    edges_per = 6
    total_nodes = nodes_per * n_mols
    total_edges = edges_per * n_mols
    batch = torch.repeat_interleave(torch.arange(n_mols), nodes_per)
    src = torch.cat(
        [torch.arange(edges_per) % nodes_per + i * nodes_per for i in range(n_mols)]
    )
    dst = torch.cat(
        [
            (torch.arange(edges_per) + 1) % nodes_per + i * nodes_per
            for i in range(n_mols)
        ]
    )
    return BatchedGraph(
        node_features=torch.randn(total_nodes, 34),
        edge_features=torch.randn(total_edges, 12),
        edge_index=torch.stack([src, dst]),
        batch=batch,
    )
