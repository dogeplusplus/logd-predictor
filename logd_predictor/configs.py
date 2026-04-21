"""Pydantic config models for model training and evaluation."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    ATTENTIVE_FP = "AttentiveFP"
    GCN = "GCN"
    RANDOM_FOREST = "RandomForest"


class ModelConfig(BaseModel):
    model_type: ModelType = ModelType.ATTENTIVE_FP
    epochs: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 2048
    num_layers: int = 2
    num_timesteps: int = 2
    graph_feat_size: int = 200
    dropout: float = 0.2
    n_estimators: int = 100
    model_dir: Path = Path("artifacts/model")
    patience: int = 5

    model_config = {"arbitrary_types_allowed": True}


class EvalConfig(BaseModel):
    metrics: list[str] = Field(default_factory=lambda: ["mae", "rmse", "r2"])
    output_path: Path = Path("artifacts/ml_metrics.json")
    log_every_n_epochs: int = 10

    model_config = {"arbitrary_types_allowed": True}
