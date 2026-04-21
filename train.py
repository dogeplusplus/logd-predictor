import logging

import hydra
import mlflow
from omegaconf import DictConfig

from logd_predictor.training import run_trial

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """Run one trial. Returns best val MAE so Optuna can minimise it."""
    experiment_name = cfg.get("mlflow", {}).get("experiment_name", "logd-predictor")
    mlflow.set_experiment(experiment_name)

    model_type = cfg.model.get("model_type", "unknown")
    feat_type = cfg.featurizer.get("featurizer_type", "unknown")
    run_name = f"{model_type}/{feat_type}"

    with mlflow.start_run(run_name=run_name):
        best_val_mae = run_trial(cfg)

    logger.info("Trial complete - best val MAE: %.4f", best_val_mae)
    return best_val_mae


if __name__ == "__main__":
    main()
