import io
import logging
import threading
import time
from dataclasses import dataclass, field

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    DataTable,
    Footer,
    Label,
    ProgressBar,
    RichLog,
    Sparkline,
    Static,
)

_HISTORY_LEN = 120  # max sparkline points retained per chart

_CSS = """
Screen { background: $surface; }

#header-bar {
    dock: top;
    height: 1;
    background: $primary-darken-2;
    layout: horizontal;
    padding: 0 1;
}
#hdr-info  { width: 1fr; }
#hdr-timer { width: 9; text-align: right; color: $text-muted; }

#main-content { height: 1fr; layout: horizontal; }

/* ── Trials ──────────────────────────────── */
#trials-panel {
    width: 40;
    border-right: tall $primary-darken-1;
    layout: vertical;
}
#trials-title {
    height: 1;
    background: $primary-darken-1;
    padding: 0 1;
    text-style: bold;
}
DataTable { height: 1fr; }

/* ── Right panel ─────────────────────────── */
#right-panel { width: 1fr; layout: vertical; }

#progress-panel {
    height: auto;
    border-bottom: tall $primary-darken-1;
    padding: 1 2 1 2;
}
#progress-title {
    height: 1;
    background: $primary-darken-1;
    margin: 0 -2 1 -2;
    padding: 0 1;
    text-style: bold;
}
.prow { height: 2; layout: horizontal; align: left middle; }
.plabel   { width: 7; color: $text-muted; }
.pcounter { width: 11; text-align: right; color: $text-muted; }
ProgressBar { width: 1fr; padding: 0 1; }
#throughput-row { height: 1; color: $text-muted; margin-top: 1; }

#metrics-panel {
    height: 1fr;
    padding: 1 2;
    border-bottom: tall $primary-darken-1;
}
#metrics-title {
    height: 1;
    background: $primary-darken-1;
    margin: 0 -2 1 -2;
    padding: 0 1;
    text-style: bold;
}
.mrow  { height: 1; layout: horizontal; }
.mname { width: 14; color: $text-muted; }
.mval  { width: 10; color: $success; }

/* ── Charts + Log (bottom row) ───────────── */
#bottom-row { height: 18; layout: horizontal; }

#loss-chart {
    width: 1fr;
    border-right: tall $primary-darken-1;
    padding: 0 2;
}
#mae-chart {
    width: 1fr;
    border-right: tall $primary-darken-1;
    padding: 0 2;
}
.chart-title {
    height: 1;
    background: $primary-darken-1;
    margin: 0 -2;
    padding: 0 1;
    text-style: bold;
}
Sparkline { height: 5; margin-top: 1; }
.chart-stats { height: 1; color: $text-muted; }

#gpu-chart {
    width: 1fr;
    border-right: tall $primary-darken-1;
    padding: 0 2;
}

#log-panel { width: 2fr; padding: 0 2; }
#log-title {
    height: 1;
    background: $primary-darken-1;
    margin: 0 -2;
    padding: 0 1;
    text-style: bold;
}
RichLog { height: 1fr; }
"""


@dataclass
class _State:
    trial_num: int = 0
    total_trials: int = 0
    model_type: str = ""
    feat_type: str = ""

    epoch: int = 0
    total_epochs: int = 0
    step: int = 0
    total_steps: int = 0

    train_loss: float | None = None
    val_mae: float | None = None
    val_rmse: float | None = None
    val_r2: float | None = None

    loss_history: list[float] = field(default_factory=list)
    mae_history: list[float] = field(default_factory=list)
    gpu_history: list[float] = field(default_factory=list)

    bps: float = 0.0
    s_per_epoch: float = 0.0
    eta_seconds: int = 0


class TrainingApp(App):
    CSS = _CSS
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("c", "clear_charts", "Clear charts"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._state = _State()
        self._start = time.monotonic()
        self._mount_event = threading.Event()
        self._added_rows: set[str] = set()
        self._col_mae_key = None

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with Container(id="header-bar"):
            yield Static("logd-predictor", id="hdr-info")
            yield Static("00:00:00", id="hdr-timer")

        with Horizontal(id="main-content"):
            with Vertical(id="trials-panel"):
                yield Static(" ALL TRIALS", id="trials-title")
                yield DataTable(id="trials-table", show_cursor=False)

            with Vertical(id="right-panel"):
                with Container(id="progress-panel"):
                    yield Static(" PROGRESS", id="progress-title")
                    with Horizontal(classes="prow"):
                        yield Label("Epoch", classes="plabel")
                        yield ProgressBar(total=1, show_eta=False, id="epoch-bar")
                        yield Static("—", id="epoch-counter", classes="pcounter")
                    with Horizontal(classes="prow"):
                        yield Label("Step", classes="plabel")
                        yield ProgressBar(total=1, show_eta=False, id="step-bar")
                        yield Static("—", id="step-counter", classes="pcounter")
                    yield Static("", id="throughput-row")

                with Container(id="metrics-panel"):
                    yield Static(" METRICS", id="metrics-title")
                    for widget_id, label in [
                        ("m-train-loss", "train_loss"),
                        ("m-val-mae", "val_mae"),
                        ("m-val-rmse", "val_rmse"),
                        ("m-val-r2", "val_r2"),
                    ]:
                        with Horizontal(classes="mrow"):
                            yield Label(label, classes="mname")
                            yield Static("—", id=widget_id, classes="mval")

        with Horizontal(id="bottom-row"):
            with Container(id="loss-chart"):
                yield Static(" TRAIN LOSS", classes="chart-title")
                yield Sparkline([], id="loss-spark")
                yield Static("", id="loss-stats", classes="chart-stats")
            with Container(id="mae-chart"):
                yield Static(" VAL MAE", classes="chart-title")
                yield Sparkline([], id="mae-spark")
                yield Static("", id="mae-stats", classes="chart-stats")
            with Container(id="gpu-chart"):
                yield Static(" GPU USAGE", classes="chart-title")
                yield Sparkline([], id="gpu-spark")
                yield Static("", id="gpu-stats", classes="chart-stats")
            with Container(id="log-panel"):
                yield Static(" OUTPUT", id="log-title")
                yield RichLog(id="tui-log", highlight=False, markup=False, wrap=True)

        yield Footer()

    def on_mount(self) -> None:
        tbl: DataTable = self.query_one("#trials-table")
        cols = tbl.add_columns(" #", "Model", "Feat", "Val MAE")
        self._col_mae_key = cols[3]
        self.set_interval(1.0, self._tick)
        self.set_interval(2.0, self._poll_gpu)
        self._mount_event.set()

    # ── Timer ─────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        s = int(time.monotonic() - self._start)
        self.query_one("#hdr-timer", Static).update(
            f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
        )

    # ── Thread-safe public API ────────────────────────────────────────────────

    def set_trial_info(
        self,
        trial_num: int,
        total_trials: int,
        model_type: str,
        feat_type: str,
        total_epochs: int,
    ) -> None:
        self.call_from_thread(
            self._do_trial_info,
            trial_num,
            total_trials,
            model_type,
            feat_type,
            total_epochs,
        )

    def set_step_progress(self, step: int, total_steps: int) -> None:
        self.call_from_thread(self._do_step, step, total_steps)

    def set_epoch_progress(self, epoch: int, train_loss: float) -> None:
        self.call_from_thread(self._do_epoch, epoch, train_loss)

    def set_val_metrics(self, val_mae: float, val_rmse: float, val_r2: float) -> None:
        self.call_from_thread(self._do_val, val_mae, val_rmse, val_r2)

    def set_throughput(self, bps: float, s_per_epoch: float, eta_seconds: int) -> None:
        self.call_from_thread(self._do_throughput, bps, s_per_epoch, eta_seconds)

    def complete_trial(self, trial_num: int, best_val_mae: float) -> None:
        self.call_from_thread(self._do_complete, trial_num, best_val_mae)

    def log_message(self, text: str, level: int = logging.INFO) -> None:
        self.call_from_thread(self._do_log, text, level)

    def mark_done(self) -> None:
        self.call_from_thread(
            self.query_one("#hdr-info", Static).update,
            Text("  All trials complete — press q to exit", style="bold green"),
        )

    def _do_trial_info(
        self,
        trial_num: int,
        total_trials: int,
        model_type: str,
        feat_type: str,
        total_epochs: int,
    ) -> None:
        s = self._state
        s.trial_num, s.total_trials = trial_num, total_trials
        s.model_type, s.feat_type = model_type, feat_type
        s.total_epochs = total_epochs
        s.epoch = s.step = 0
        s.train_loss = s.val_mae = s.val_rmse = s.val_r2 = None

        trial_label = (
            f"Trial {trial_num}/{total_trials}"
            if total_trials
            else f"Trial {trial_num}"
        )
        header = Text()
        header.append(f"  {model_type}", style="bold")
        header.append(" / ", style="dim")
        header.append(feat_type)
        header.append(f"   [{trial_label}]", style="dim")
        self.query_one("#hdr-info", Static).update(header)

        self.query_one("#epoch-bar", ProgressBar).update(progress=0, total=total_epochs)
        self.query_one("#epoch-counter", Static).update(f"0/{total_epochs}")
        self.query_one("#step-bar", ProgressBar).update(progress=0, total=1)
        self.query_one("#step-counter", Static).update("—")
        for wid in ("m-train-loss", "m-val-mae", "m-val-rmse", "m-val-r2"):
            self.query_one(f"#{wid}", Static).update("—")

        row_key = str(trial_num)
        if row_key not in self._added_rows:
            tbl: DataTable = self.query_one("#trials-table")
            tbl.add_row(str(trial_num), model_type[:5], feat_type[:9], "…", key=row_key)
            self._added_rows.add(row_key)
            tbl.move_cursor(row=tbl.row_count - 1)

    def _do_step(self, step: int, total_steps: int) -> None:
        self._state.step, self._state.total_steps = step, total_steps
        self.query_one("#step-bar", ProgressBar).update(
            progress=step, total=total_steps
        )
        self.query_one("#step-counter", Static).update(f"{step}/{total_steps}")

    def _do_epoch(self, epoch: int, train_loss: float) -> None:
        s = self._state
        s.epoch, s.train_loss = epoch, train_loss
        s.loss_history = (s.loss_history + [train_loss])[-_HISTORY_LEN:]

        self.query_one("#epoch-bar", ProgressBar).update(progress=epoch)
        self.query_one("#epoch-counter", Static).update(f"{epoch}/{s.total_epochs}")
        self.query_one("#m-train-loss", Static).update(f"{train_loss:.4f}")

        spark: Sparkline = self.query_one("#loss-spark")
        spark.data = list(s.loss_history)
        self.query_one("#loss-stats", Static).update(
            f"{s.loss_history[0]:.4f} → {train_loss:.4f}  (epoch {epoch})"
        )

    def _do_val(self, val_mae: float, val_rmse: float, val_r2: float) -> None:
        s = self._state
        s.val_mae, s.val_rmse, s.val_r2 = val_mae, val_rmse, val_r2
        s.mae_history = (s.mae_history + [val_mae])[-_HISTORY_LEN:]

        self.query_one("#m-val-mae", Static).update(f"{val_mae:.4f}")
        self.query_one("#m-val-rmse", Static).update(f"{val_rmse:.4f}")
        r2_col = "green" if val_r2 > 0.7 else "yellow" if val_r2 > 0 else "red"
        self.query_one("#m-val-r2", Static).update(Text(f"{val_r2:.4f}", style=r2_col))

        spark: Sparkline = self.query_one("#mae-spark")
        spark.data = list(s.mae_history)
        self.query_one("#mae-stats", Static).update(
            f"{s.mae_history[0]:.4f} → {val_mae:.4f}  best {min(s.mae_history):.4f}"
        )

    def _do_throughput(self, bps: float, s_per_epoch: float, eta_seconds: int) -> None:
        s = self._state
        s.bps, s.s_per_epoch, s.eta_seconds = bps, s_per_epoch, eta_seconds
        h, rem = divmod(eta_seconds, 3600)
        m, sec = divmod(rem, 60)
        eta_str = f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
        self.query_one("#throughput-row", Static).update(
            f"{bps:.1f} bat/s  ·  {s_per_epoch:.1f}s/ep  ·  ETA {eta_str}"
        )

    def _poll_gpu(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_gb = mem_info.used / 1024**3
            mem_total_gb = mem_info.total / 1024**3
        except Exception:
            return

        s = self._state
        s.gpu_history = (s.gpu_history + [float(util)])[-_HISTORY_LEN:]
        spark: Sparkline = self.query_one("#gpu-spark")
        spark.data = list(s.gpu_history)
        self.query_one("#gpu-stats", Static).update(
            f"{util}%  {mem_used_gb:.1f}/{mem_total_gb:.1f} GB"
        )

    def _do_complete(self, trial_num: int, best_val_mae: float) -> None:
        tbl: DataTable = self.query_one("#trials-table")
        row_key = str(trial_num)
        if row_key in self._added_rows and self._col_mae_key is not None:
            tbl.update_cell(row_key, self._col_mae_key, f"{best_val_mae:.4f}")

    def _do_log(self, text: str, level: int) -> None:
        log: RichLog = self.query_one("#tui-log")
        if level >= logging.ERROR:
            style = "bold red"
        elif level >= logging.WARNING:
            style = "yellow"
        elif level == 0:  # raw console write
            style = "dim"
        else:
            style = "default"
        log.write(Text(text.rstrip(), style=style))

    def action_clear_charts(self) -> None:
        self._state.loss_history.clear()
        self._state.mae_history.clear()
        self._state.gpu_history.clear()
        self.query_one("#loss-spark", Sparkline).data = []
        self.query_one("#mae-spark", Sparkline).data = []
        self.query_one("#gpu-spark", Sparkline).data = []
        self.query_one("#loss-stats", Static).update("")
        self.query_one("#mae-stats", Static).update("")
        self.query_one("#gpu-stats", Static).update("")


class _TUIWriter(io.TextIOBase):
    """File-like object that routes console writes into the TUI log panel.

    Buffers until a newline, then ships complete lines to avoid sending
    partial progress-bar cursor-movement sequences into the log.
    """

    def __init__(self, app: TrainingApp) -> None:
        self._app = app
        self._buf = ""

    def write(self, text: str) -> int:
        # Strip carriage returns used by in-place progress bar rewrites
        text = text.replace("\r", "")
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.strip()
            if stripped:
                self._app.log_message(stripped, level=0)
        return len(text)

    def flush(self) -> None:
        if self._buf.strip():
            self._app.log_message(self._buf.strip(), level=0)
            self._buf = ""

    def isatty(self) -> bool:
        return False


_OUR_LOGGERS = {"logd_predictor", "__main__", "train"}
_NOISY_LIBS = {
    "lightning",
    "torch",
    "mlflow",
    "urllib3",
    "filelock",
    "optuna",
    "httpx",
    "httpcore",
    "PIL",
    "matplotlib",
}


class _TUILogFilter(logging.Filter):
    """Pass through:
    - WARNING+ from any logger (important library errors)
    - INFO/DEBUG from our own loggers only
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return record.name.split(".")[0] in _OUR_LOGGERS


class _TUILogHandler(logging.Handler):
    """Logging handler that forwards filtered records to the TUI log panel."""

    def __init__(self, app: TrainingApp) -> None:
        super().__init__(level=logging.DEBUG)
        self.setFormatter(logging.Formatter("%(name)s  %(message)s"))
        self.addFilter(_TUILogFilter())
        self._app = app

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._app.log_message(self.format(record), level=record.levelno)
        except Exception:
            pass


_app: TrainingApp | None = None
_app_thread: threading.Thread | None = None


def ensure_tui() -> TrainingApp:
    """Start the TUI in a background daemon thread (idempotent).

    Also patches the shared Rich console and root logger so all training
    output is routed into the TUI's log panel rather than the raw terminal.
    """
    global _app, _app_thread
    if _app is not None:
        return _app

    # Textual's LinuxDriver installs a SIGTSTP handler (Ctrl+Z) via
    # signal.signal(), which raises ValueError from a non-main thread.
    import signal as _signal

    _orig_signal = _signal.signal

    def _thread_safe_signal(sig, handler):
        try:
            return _orig_signal(sig, handler)
        except ValueError:
            pass

    _signal.signal = _thread_safe_signal

    _app = TrainingApp()
    _app_thread = threading.Thread(target=_app.run, daemon=True)
    _app_thread.start()
    _app._mount_event.wait(timeout=10.0)

    writer = _TUIWriter(_app)

    # Route the shared Rich console into the TUI log.
    # Modules that did `from _io import console` hold a reference to the same
    # Console object, so patching its _file attribute affects all of them.
    from logd_predictor._io import console as _rich_console

    _rich_console._file = writer
    # Disable terminal detection so Rich emits plain text (no ANSI codes).
    _rich_console._force_terminal = False
    _rich_console._is_terminal = False

    # Capture stderr — this surfaces Python tracebacks and Hydra error output
    # that would otherwise be swallowed by Textual's raw terminal mode.
    import sys

    sys.stderr = writer

    # Route Python logging to the TUI.
    # Don't touch the root logger's level — lowering it wakes up verbose
    # library loggers (Lightning, MLflow, etc). The filter on the handler
    # is what controls what actually appears in the TUI log.
    _handler = _TUILogHandler(_app)
    logging.getLogger().addHandler(_handler)
    # Ensure our own package emits at INFO so status messages come through.
    logging.getLogger("logd_predictor").setLevel(logging.INFO)

    return _app


def wait_for_exit() -> None:
    """Block until the user dismisses the TUI (presses q)."""
    if _app_thread is not None:
        _app_thread.join()


def shutdown_tui() -> None:
    global _app, _app_thread
    if _app is not None:
        _app.call_from_thread(_app.exit)
        if _app_thread:
            _app_thread.join(timeout=3.0)
    _app = None
    _app_thread = None
