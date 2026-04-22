"""Lightning training: LitRegressor, model builder, train/eval helpers, MLflow trial runner."""

import json
import logging
from pathlib import Path

import joblib
import lightning as L
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor

from logd_predictor._io import console
from logd_predictor.configs import EvalConfig, ModelConfig, ModelType
from logd_predictor.datasets import FingerprintDataset, MoleculeDataModule
from logd_predictor.tui_callback import TUICallback
from logd_predictor.featurize import (
    GRAPH_EDGE_DIM,
    GRAPH_NODE_DIM,
    FeaturizerConfig,
    FeaturizerType,
    featurize_to_disk,
    _DESC_FNS,
)
from logd_predictor.models import (
    AttentiveFPRegressor,
    BatchedGraph,
    GCNRegressor,
    MLPRegressor,
)

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

# Keyed by (featurizer_type, radius, fp_size, use_edges, split_dir).
# Keeps dataset tensors and persistent DataLoader workers alive across trials
# so the 15 GB dataset is only loaded once per featurizer configuration.
_dm_cache: dict[tuple, "MoleculeDataModule"] = {}


class LitRegressor(L.LightningModule):
    def __init__(self, net: nn.Module, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.lr = lr
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_r2 = torchmetrics.R2Score()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_r2 = torchmetrics.R2Score()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        x, y = batch
        if isinstance(x, BatchedGraph):
            x = BatchedGraph(
                node_features=x.node_features.to(device, non_blocking=True),
                edge_features=x.edge_features.to(device, non_blocking=True),
                edge_index=x.edge_index.to(device, non_blocking=True),
                batch=x.batch.to(device, non_blocking=True),
            )
        else:
            x = x.to(device, non_blocking=True)
        return x, y.to(device, non_blocking=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self.net(x).squeeze(-1), y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=y.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.net(x).squeeze(-1)
        bs = y.shape[0]
        self.val_mae(pred, y)
        self.val_rmse(pred, y)
        self.val_r2(pred, y)
        self.log(
            "val_mae",
            self.val_mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=bs,
        )
        self.log("val_rmse", self.val_rmse, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_r2", self.val_r2, on_step=False, on_epoch=True, batch_size=bs)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.net(x).squeeze(-1)
        bs = y.shape[0]
        self.test_mae(pred, y)
        self.test_rmse(pred, y)
        self.test_r2(pred, y)
        self.log("test_mae", self.test_mae, on_step=False, on_epoch=True, batch_size=bs)
        self.log(
            "test_rmse", self.test_rmse, on_step=False, on_epoch=True, batch_size=bs
        )
        self.log("test_r2", self.test_r2, on_step=False, on_epoch=True, batch_size=bs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def _get_or_create_dm(
    dataset_dirs: dict[str, str],
    feat_cfg: FeaturizerConfig,
    batch_size: int,
    split_dir: str,
) -> "MoleculeDataModule":
    key = (
        feat_cfg.featurizer_type,
        feat_cfg.radius,
        feat_cfg.fp_size,
        feat_cfg.use_edges,
        split_dir,
    )
    if key in _dm_cache:
        logger.info("Reusing cached DataModule — skipping dataset reload")
        dm = _dm_cache[key]
        dm.batch_size = batch_size
        return dm
    dm = MoleculeDataModule(
        dataset_dirs=dataset_dirs,
        featurizer_type=feat_cfg.featurizer_type,
        batch_size=batch_size,
    )
    logger.info("Loading datasets into memory (train ~15 GB, may take ~60 s)…")
    dm.setup()
    logger.info("Datasets loaded — starting training")
    _dm_cache[key] = dm
    return dm


def build_lit_model(model_cfg: ModelConfig, feat_cfg: FeaturizerConfig) -> LitRegressor:
    if feat_cfg.featurizer_type == FeaturizerType.MOL_GRAPH_CONV:
        edge_dim = GRAPH_EDGE_DIM if feat_cfg.use_edges else 0
        net: nn.Module = (
            GCNRegressor(
                node_feat_dim=GRAPH_NODE_DIM,
                hidden=model_cfg.graph_feat_size,
                n_layers=model_cfg.num_layers,
                dropout=model_cfg.dropout,
            )
            if model_cfg.model_type == ModelType.GCN
            else AttentiveFPRegressor(
                node_feat_dim=GRAPH_NODE_DIM,
                edge_feat_dim=edge_dim,
                hidden=model_cfg.graph_feat_size,
                n_layers=model_cfg.num_layers,
                num_timesteps=model_cfg.num_timesteps,
                dropout=model_cfg.dropout,
            )
        )
    else:
        fp_dim = (
            feat_cfg.fp_size
            if feat_cfg.featurizer_type == FeaturizerType.CIRCULAR
            else len(_DESC_FNS)
        )
        net = MLPRegressor(
            in_features=fp_dim,
            hidden=model_cfg.graph_feat_size,
            n_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
        )
    return LitRegressor(net, lr=model_cfg.learning_rate)


def save_model_bundle(
    lit: LitRegressor,
    model_cfg: ModelConfig,
    feat_cfg: FeaturizerConfig,
    metrics: dict[str, float],
    bundle_dir: Path,
) -> None:
    """Save a self-contained inference bundle: weights + config + metadata.

    The bundle can be loaded for cold-start inference without the Hydra
    run-directory or any Lightning / training dependencies:

        config = json.load(open("bundle/config.json"))
        lit = build_lit_model(ModelConfig(**config["model"]),
                              FeaturizerConfig(**config["featurizer"]))
        lit.net.load_state_dict(torch.load("bundle/weights.pt"))
        lit.eval()
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap torch.compile wrapper if present so we save the raw state dict.
    net = lit.net
    if hasattr(net, "_orig_mod"):
        net = net._orig_mod
    torch.save(net.state_dict(), bundle_dir / "weights.pt")

    (bundle_dir / "config.json").write_text(
        json.dumps(
            {
                "model": json.loads(model_cfg.model_dump_json()),
                "featurizer": json.loads(feat_cfg.model_dump_json()),
            },
            indent=2,
        )
    )

    (bundle_dir / "metadata.json").write_text(
        json.dumps(
            {
                **metrics,
                "model_type": model_cfg.model_type.value,
                "featurizer_type": feat_cfg.featurizer_type.value,
            },
            indent=2,
        )
    )
    logger.info("Model bundle saved → %s", bundle_dir)


def compute_metrics(
    preds: np.ndarray, targets: np.ndarray, names: list[str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    if "mae" in names:
        out["mae"] = float(np.mean(np.abs(preds - targets)))
    if "rmse" in names:
        out["rmse"] = float(np.sqrt(np.mean((preds - targets) ** 2)))
    if "r2" in names:
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        out["r2"] = (
            float(1 - np.sum((targets - preds) ** 2) / ss_tot) if ss_tot > 0 else 0.0
        )
    return out


def _train_rf(
    dataset_dirs: dict[str, str], model_cfg: ModelConfig, max_samples: int | None = None
) -> str:
    ds = FingerprintDataset(dataset_dirs["train"], max_samples=max_samples)
    rf = RandomForestRegressor(
        n_estimators=model_cfg.n_estimators, random_state=42, n_jobs=-1
    )
    with console.status("[bold blue]Fitting RandomForest…[/]", spinner="dots"):
        rf.fit(ds.X.numpy(), ds.targets.numpy())
    path = str(model_cfg.model_dir / "rf_model.pkl")
    joblib.dump(rf, path)
    console.print("[bold green]✓[/] RandomForest fit complete\n")
    return path


def _eval_rf(
    model_path: str, dataset_dirs: dict[str, str], eval_cfg: EvalConfig
) -> dict[str, dict[str, float]]:
    rf = joblib.load(model_path)
    results: dict[str, dict[str, float]] = {}
    for split in ("validation", "test"):
        ds = FingerprintDataset(dataset_dirs[split])
        results[split] = compute_metrics(
            rf.predict(ds.X.numpy()), ds.targets.numpy(), eval_cfg.metrics
        )
        logger.info("%s: %s", split, {k: f"{v:.4f}" for k, v in results[split].items()})
    return results


class _MLflowCallback(L.Callback):
    """Logs train/val metrics to the active MLflow run after each epoch."""

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        step = trainer.current_epoch + 1
        for key in ("train_loss",):
            if (v := trainer.callback_metrics.get(key)) is not None:
                mlflow.log_metric(key, float(v), step=step)

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        step = trainer.current_epoch + 1
        for key in ("val_mae", "val_rmse", "val_r2"):
            if (v := trainer.callback_metrics.get(key)) is not None:
                mlflow.log_metric(key, float(v), step=step)


def _log_dataset_metadata(dataset_dirs: dict[str, str]) -> None:
    for split, path in dataset_dirs.items():
        meta_path = Path(path) / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            mlflow.log_param(f"n_{split}", meta.get("n_mols", "?"))


def run_trial(
    cfg: DictConfig,
    trial_number: int | None = None,
    output_dir: Path | None = None,
    tui_app=None,
    total_trials: int = 0,
) -> float:
    """Run one training trial inside an active MLflow run. Returns best val MAE."""
    feat_cfg = FeaturizerConfig(**OmegaConf.to_container(cfg.featurizer, resolve=True))
    model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    eval_cfg = EvalConfig(**OmegaConf.to_container(cfg.eval, resolve=True))

    if output_dir is not None:
        model_cfg = model_cfg.model_copy(update={"model_dir": output_dir / "model"})
        eval_cfg = eval_cfg.model_copy(
            update={"output_path": output_dir / "ml_metrics.json"}
        )

    mlflow.log_params(OmegaConf.to_container(cfg.model, resolve=True))
    mlflow.log_params(
        {
            f"feat_{k}": v
            for k, v in OmegaConf.to_container(cfg.featurizer, resolve=True).items()
        }
    )
    tags: dict[str, str] = {
        "model_type": model_cfg.model_type.value,
        "featurizer": feat_cfg.featurizer_type.value,
    }
    if trial_number is not None:
        tags["trial"] = str(trial_number)
    mlflow.set_tags(tags)

    dataset_dirs = featurize_to_disk(
        split_dir=cfg.split_dir,
        cfg=feat_cfg,
        dataset_dir="data/processed/datasets",
        smiles_col=cfg.smiles_col,
        target_col=cfg.target_col,
    )
    _log_dataset_metadata(dataset_dirs)

    results: dict[str, dict[str, float]] = {}
    best_val_mae = float("inf")

    if model_cfg.model_type == ModelType.RANDOM_FOREST:
        ckpt_path = _train_rf(dataset_dirs, model_cfg)
        results = _eval_rf(ckpt_path, dataset_dirs, eval_cfg)
        best_val_mae = float(results.get("validation", {}).get("mae", float("inf")))
        for split, scores in results.items():
            for metric, value in scores.items():
                mlflow.log_metric(f"{split}_{metric}", float(value))
        mlflow.log_metric("val_mae", best_val_mae)
    else:
        dm = _get_or_create_dm(
            dataset_dirs=dataset_dirs,
            feat_cfg=feat_cfg,
            batch_size=model_cfg.batch_size,
            split_dir=cfg.split_dir,
        )

        ckpt_cb = ModelCheckpoint(
            dirpath=str(model_cfg.model_dir),
            filename="best",
            monitor="val_mae",
            mode="min",
            save_top_k=1,
        )
        callbacks: list = [ckpt_cb, _MLflowCallback()]
        if model_cfg.patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_mae", patience=model_cfg.patience, mode="min"
                )
            )
        if tui_app is not None:
            callbacks.append(TUICallback(tui_app, trial_number or 1, total_trials))

        lit = build_lit_model(model_cfg, feat_cfg)
        # if torch.cuda.is_available():
        #     try:
        #         lit.net = torch.compile(
        #             lit.net, dynamic=True, mode="reduce-overhead"
        #         )
        #     except Exception as exc:
        #         logger.warning("torch.compile failed (%s) — running eager", exc)

        L.Trainer(
            max_epochs=model_cfg.epochs,
            callbacks=callbacks,
            check_val_every_n_epoch=eval_cfg.log_every_n_epochs,
            enable_progress_bar=tui_app is None,
            enable_model_summary=False,
            logger=False,
            accelerator="auto",
            devices=torch.cuda.device_count(),
            precision=model_cfg.precision,
            strategy="ddp",
        ).fit(lit, datamodule=dm)

        best_val_mae = float(ckpt_cb.best_model_score or float("inf"))
        mlflow.log_metric("best_val_mae", best_val_mae)

        ckpt = torch.load(
            ckpt_cb.best_model_path, map_location="cpu", weights_only=True
        )
        lit = build_lit_model(model_cfg, feat_cfg)
        lit.load_state_dict(ckpt["state_dict"])
        lit.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lit = lit.to(device)

        for split, loader in [
            ("validation", dm.val_dataloader()),
            ("test", dm.test_dataloader()),
        ]:
            preds_list, targets_list = [], []
            with torch.no_grad():
                for x, y in loader:
                    if isinstance(x, BatchedGraph):
                        x = BatchedGraph(
                            node_features=x.node_features.to(device),
                            edge_features=x.edge_features.to(device),
                            edge_index=x.edge_index.to(device),
                            batch=x.batch.to(device),
                        )
                    else:
                        x = x.to(device)
                    preds_list.append(lit.net(x).squeeze(-1).cpu().numpy())
                    targets_list.append(y.numpy())
            scores = compute_metrics(
                np.concatenate(preds_list),
                np.concatenate(targets_list),
                eval_cfg.metrics,
            )
            results[split] = scores
            for metric, value in scores.items():
                mlflow.log_metric(f"{split}_{metric}", float(value))
            logger.info("%s: %s", split, {k: f"{v:.4f}" for k, v in scores.items()})

    eval_cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_cfg.output_path.write_text(json.dumps(results, indent=2))
    mlflow.log_artifact(str(eval_cfg.output_path))
    if Path(str(model_cfg.model_dir)).exists():
        mlflow.log_artifacts(str(model_cfg.model_dir), artifact_path="model")

    # Save a self-contained bundle alongside the checkpoint so inference
    # works from a cold start without the Hydra run-directory structure.
    if model_cfg.model_type != ModelType.RANDOM_FOREST and "lit" in dir():
        bundle_dir = model_cfg.model_dir / "bundle"
        val_metrics = results.get("validation", {})
        save_model_bundle(lit, model_cfg, feat_cfg, val_metrics, bundle_dir)
        mlflow.log_artifacts(str(bundle_dir), artifact_path="model/bundle")

    return best_val_mae
