"""Tests for training utilities."""

import numpy as np
import pytest

from logd_predictor.featurize import FeaturizerConfig, FeaturizerType
from logd_predictor.configs import ModelConfig, ModelType
from logd_predictor.models import AttentiveFPRegressor, GCNRegressor, MLPRegressor
from logd_predictor.training import build_lit_model, compute_metrics, LitRegressor


PERFECT = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


class TestComputeMetrics:
    def test_perfect_predictions(self):
        metrics = compute_metrics(PERFECT, PERFECT, ["mae", "rmse", "r2"])
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)

    def test_mae(self):
        preds = np.array([1.0, 3.0])
        targets = np.array([2.0, 2.0])
        metrics = compute_metrics(preds, targets, ["mae"])
        assert metrics["mae"] == pytest.approx(1.0)

    def test_rmse(self):
        preds = np.array([0.0, 2.0])
        targets = np.array([1.0, 1.0])
        metrics = compute_metrics(preds, targets, ["rmse"])
        assert metrics["rmse"] == pytest.approx(1.0)

    def test_r2_zero_variance_target(self):
        # All targets equal - R² is 0 by convention
        preds = np.array([1.0, 1.0])
        targets = np.array([1.0, 1.0])
        metrics = compute_metrics(preds, targets, ["r2"])
        assert metrics["r2"] == pytest.approx(0.0)

    def test_subset_metrics(self):
        metrics = compute_metrics(PERFECT, PERFECT, ["mae"])
        assert "mae" in metrics
        assert "rmse" not in metrics
        assert "r2" not in metrics

    def test_empty_metrics_list(self):
        metrics = compute_metrics(PERFECT, PERFECT, [])
        assert metrics == {}


class TestBuildLitModel:
    def test_attentivefp_graph(self):
        model_cfg = ModelConfig(model_type=ModelType.ATTENTIVE_FP)
        feat_cfg = FeaturizerConfig(featurizer_type=FeaturizerType.MOL_GRAPH_CONV)
        lit = build_lit_model(model_cfg, feat_cfg)
        assert isinstance(lit, LitRegressor)
        assert isinstance(lit.net, AttentiveFPRegressor)

    def test_gcn_graph(self):
        model_cfg = ModelConfig(model_type=ModelType.GCN)
        feat_cfg = FeaturizerConfig(featurizer_type=FeaturizerType.MOL_GRAPH_CONV)
        lit = build_lit_model(model_cfg, feat_cfg)
        assert isinstance(lit.net, GCNRegressor)

    def test_mlp_circular(self):
        model_cfg = ModelConfig(model_type=ModelType.ATTENTIVE_FP)
        feat_cfg = FeaturizerConfig(
            featurizer_type=FeaturizerType.CIRCULAR, fp_size=1024
        )
        lit = build_lit_model(model_cfg, feat_cfg)
        assert isinstance(lit.net, MLPRegressor)

    def test_mlp_rdkit(self):
        model_cfg = ModelConfig(model_type=ModelType.ATTENTIVE_FP)
        feat_cfg = FeaturizerConfig(featurizer_type=FeaturizerType.RDKIT)
        lit = build_lit_model(model_cfg, feat_cfg)
        assert isinstance(lit.net, MLPRegressor)

    def test_learning_rate_propagated(self):
        model_cfg = ModelConfig(model_type=ModelType.GCN, learning_rate=5e-4)
        feat_cfg = FeaturizerConfig(featurizer_type=FeaturizerType.MOL_GRAPH_CONV)
        lit = build_lit_model(model_cfg, feat_cfg)
        assert lit.lr == pytest.approx(5e-4)
