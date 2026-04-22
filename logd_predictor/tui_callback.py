"""Lightning callback that drives the Textual TUI during training."""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

import lightning as L

if TYPE_CHECKING:
    from logd_predictor.tui import TrainingApp

_STEP_UPDATE_INTERVAL = 5  # update step bar at most every N batches
_BPS_WINDOW = 50  # rolling window for batches/s average


class TUICallback(L.Callback):
    def __init__(self, app: TrainingApp, trial_num: int, total_trials: int) -> None:
        self._app = app
        self._trial_num = trial_num
        self._total_trials = total_trials
        self._batch_t0: float = 0.0
        self._batch_times: deque[float] = deque(maxlen=_BPS_WINDOW)
        self._epoch_t0: float = 0.0

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        feat = getattr(trainer.datamodule, "featurizer_type", "—")
        feat_str = feat.value if hasattr(feat, "value") else str(feat)
        self._app.set_trial_info(
            trial_num=self._trial_num,
            total_trials=self._total_trials,
            model_type=trainer.model.__class__.__name__,
            feat_type=feat_str,
            total_epochs=trainer.max_epochs or 0,
        )
        self._app.set_step_progress(0, trainer.num_training_batches)

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._epoch_t0 = time.perf_counter()

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch, batch_idx: int
    ) -> None:
        self._batch_t0 = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        self._batch_times.append(time.perf_counter() - self._batch_t0)

        if batch_idx % _STEP_UPDATE_INTERVAL != 0:
            return
        self._app.set_step_progress(batch_idx + 1, trainer.num_training_batches)

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        epoch = trainer.current_epoch + 1
        loss = float(
            trainer.callback_metrics.get("train_loss_epoch")
            or trainer.callback_metrics.get("train_loss")
            or 0.0
        )
        self._app.set_epoch_progress(epoch, loss)
        self._app.set_step_progress(0, trainer.num_training_batches)

        s_per_epoch = time.perf_counter() - self._epoch_t0
        remaining = max(0, (trainer.max_epochs or epoch) - epoch)
        bps = (
            1.0 / (sum(self._batch_times) / len(self._batch_times))
            if self._batch_times
            else 0.0
        )
        self._app.set_throughput(
            bps=bps,
            s_per_epoch=s_per_epoch,
            eta_seconds=int(s_per_epoch * remaining),
        )

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        self._app.set_val_metrics(
            float(m.get("val_mae", 0.0)),
            float(m.get("val_rmse", 0.0)),
            float(m.get("val_r2", 0.0)),
        )

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._app.complete_trial(
            self._trial_num,
            float(trainer.callback_metrics.get("val_mae", float("inf"))),
        )
