import logging
from pathlib import Path

import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from logd_predictor.training import run_trial
from logd_predictor.tui import ensure_tui, wait_for_exit

logger = logging.getLogger(__name__)

_trial_counter = 0
_total_trials = 0


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """Run one trial. Returns best val MAE so Optuna can minimise it."""
    global _trial_counter, _total_trials

    if _trial_counter == 0:
        try:
            _total_trials = int(cfg.hydra.sweeper.get("n_trials", 0))
        except Exception:
            _total_trials = 0

    _trial_counter += 1
    trial_num = _trial_counter

    tui = ensure_tui()

    experiment_name = cfg.get("mlflow", {}).get("experiment_name", "logd-predictor")
    mlflow.set_experiment(experiment_name)

    model_type = cfg.model.get("model_type", "unknown")
    feat_type = cfg.featurizer.get("featurizer_type", "unknown")
    run_name = f"{model_type}/{feat_type}"

    output_dir = Path(HydraConfig.get().runtime.output_dir)

    with mlflow.start_run(run_name=run_name):
        try:
            best_val_mae = run_trial(
                cfg,
                trial_number=trial_num,
                output_dir=output_dir,
                tui_app=tui,
                total_trials=_total_trials,
            )
        except Exception:
            logger.exception("Trial %d failed", trial_num)
            raise

    logger.info("Trial %d complete — best val MAE: %.4f", trial_num, best_val_mae)
    return best_val_mae


if __name__ == "__main__":
    tui = None
    success = False
    try:
        main()
        success = True
    except (Exception, SystemExit):
        pass  # exception already logged to TUI via logger.exception above
    finally:
        if _trial_counter > 0:
            tui = ensure_tui()
            if success:
                tui.mark_done()
            else:
                tui.log_message(
                    "Training ended with errors — see log above", logging.ERROR
                )
            wait_for_exit()
