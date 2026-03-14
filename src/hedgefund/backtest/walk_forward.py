"""Walk-forward analysis with anchored and rolling windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog

from hedgefund.learning.base import TradingModel

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class WalkForwardWindow:
    """Record of a single train/test window in the walk-forward analysis."""

    step: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    predictions: np.ndarray | None = None


@dataclass(slots=True)
class WalkForwardConfig:
    """Configuration for the walk-forward analyser."""

    train_window: int = 252  # trading days
    test_window: int = 63
    step_size: int = 21  # how much the test window slides each step
    anchored: bool = True  # if True, training always starts at index 0
    min_train_samples: int = 100
    retrain_at_each_step: bool = True


class WalkForwardAnalyzer:
    """Anchored and rolling walk-forward analysis.

    At each step the analyser:
    1. Defines a train and test window.
    2. Retrains the model on the training data.
    3. Evaluates on the test data.
    4. Slides the window forward by ``step_size``.

    The result is a sequence of :class:`WalkForwardWindow` objects that
    can be aggregated for out-of-sample performance.
    """

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()
        self._windows: list[WalkForwardWindow] = []
        self._log = log.bind(component="walk_forward")

    @property
    def windows(self) -> list[WalkForwardWindow]:
        return list(self._windows)

    # ── Main entry point ──────────────────────────────────────────────

    def run(
        self,
        model_factory: type[TradingModel],
        X: np.ndarray,
        y: np.ndarray,
        *,
        model_kwargs: dict[str, Any] | None = None,
    ) -> list[WalkForwardWindow]:
        """Execute walk-forward analysis over the full dataset.

        A fresh model is constructed from *model_factory* at each step
        (when ``retrain_at_each_step`` is True).

        Returns:
            Ordered list of window results.
        """
        n = len(X)
        cfg = self.config
        kwargs = model_kwargs or {}
        windows: list[WalkForwardWindow] = []

        # Compute window start positions.
        test_starts = self._compute_test_starts(n)
        self._log.info(
            "walk_forward_started",
            n_samples=n,
            n_steps=len(test_starts),
            anchored=cfg.anchored,
        )

        model: TradingModel | None = None

        for step_idx, test_start in enumerate(test_starts):
            test_end = min(test_start + cfg.test_window, n)
            if cfg.anchored:
                train_start = 0
            else:
                train_start = max(test_start - cfg.train_window, 0)
            train_end = test_start

            if train_end - train_start < cfg.min_train_samples:
                self._log.debug(
                    "skipping_step_insufficient_train",
                    step=step_idx,
                    train_size=train_end - train_start,
                )
                continue

            # Train.
            if cfg.retrain_at_each_step or model is None:
                model = model_factory(**kwargs)
                train_metrics = model.train(
                    X[train_start:train_end],
                    y[train_start:train_end],
                )
            else:
                train_metrics = {}

            # Predict and evaluate on test window.
            y_pred = model.predict(X[test_start:test_end])
            y_true = y[test_start:test_end]

            # Align lengths (sequence models may drop initial rows).
            min_len = min(len(y_true), len(y_pred))
            y_true_aligned = np.asarray(y_true[-min_len:]).ravel()
            y_pred_aligned = np.asarray(y_pred[-min_len:]).ravel()

            test_metrics = self._compute_metrics(y_true_aligned, y_pred_aligned)

            window = WalkForwardWindow(
                step=step_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                predictions=y_pred_aligned,
            )
            windows.append(window)
            self._log.debug(
                "step_complete",
                step=step_idx,
                train=f"{train_start}:{train_end}",
                test=f"{test_start}:{test_end}",
                **test_metrics,
            )

        self._windows = windows
        self._log.info(
            "walk_forward_complete",
            n_windows=len(windows),
            aggregate=self.aggregate_metrics(),
        )
        return windows

    # ── Aggregation ───────────────────────────────────────────────────

    def aggregate_metrics(self) -> dict[str, float]:
        """Compute mean and std of test metrics across all windows."""
        if not self._windows:
            return {}
        keys = self._windows[0].test_metrics.keys()
        agg: dict[str, float] = {}
        for k in keys:
            vals = [w.test_metrics.get(k, 0) for w in self._windows]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        return agg

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with one row per walk-forward window."""
        rows = []
        for w in self._windows:
            row: dict[str, Any] = {
                "step": w.step,
                "train_start": w.train_start,
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
            }
            row.update({f"train_{k}": v for k, v in w.train_metrics.items()})
            row.update({f"test_{k}": v for k, v in w.test_metrics.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    # ── Internal ──────────────────────────────────────────────────────

    def _compute_test_starts(self, n: int) -> list[int]:
        cfg = self.config
        if cfg.anchored:
            first_test = cfg.train_window
        else:
            first_test = cfg.train_window
        starts: list[int] = []
        pos = first_test
        while pos < n:
            starts.append(pos)
            pos += cfg.step_size
        return starts

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        if len(y_true) == 0:
            return {"rmse": 0.0, "mae": 0.0, "directional_accuracy": 0.0}
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))

        dir_acc = 0.0
        if len(y_true) > 1:
            true_dir = np.sign(np.diff(y_true))
            pred_dir = np.sign(np.diff(y_pred))
            dir_acc = float(np.mean(true_dir == pred_dir))

        return {"rmse": rmse, "mae": mae, "directional_accuracy": dir_acc}
