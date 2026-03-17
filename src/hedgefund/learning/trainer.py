"""Orchestrates training for all trading models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from hedgefund.learning.base import TradingModel

log = structlog.get_logger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TrainerConfig:
    """Settings for the model training orchestrator."""

    output_dir: Path = field(default_factory=lambda: Path("models"))
    n_cv_splits: int = 5
    walk_forward_train_pct: float = 0.7
    walk_forward_step_pct: float = 0.1
    max_train_hours: float = 4.0
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-5
    log_hyperparameters: bool = True
    select_best_by: str = "val_loss"  # metric name for model selection
    minimize_metric: bool = True


# ── Cross-Validation Helpers ──────────────────────────────────────────────────


def _time_series_cv_splits(
    n_samples: int, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window train/validation index pairs.

    Each fold uses all data up to a cutoff for training and the next
    chunk for validation, ensuring no look-ahead.
    """
    fold_size = n_samples // (n_splits + 1)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, n_splits + 1):
        train_end = fold_size * (i + 1)
        val_end = min(train_end + fold_size, n_samples)
        if val_end <= train_end:
            break
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        splits.append((train_idx, val_idx))
    return splits


def _walk_forward_splits(
    n_samples: int,
    train_pct: float,
    step_pct: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Anchored walk-forward: training window grows, test window slides."""
    train_size = int(n_samples * train_pct)
    step_size = max(int(n_samples * step_pct), 1)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    test_start = train_size
    while test_start < n_samples:
        test_end = min(test_start + step_size, n_samples)
        train_idx = np.arange(0, test_start)
        val_idx = np.arange(test_start, test_end)
        splits.append((train_idx, val_idx))
        test_start = test_end
    return splits


# ── Training Run Record ──────────────────────────────────────────────────────


@dataclass(slots=True)
class TrainingRun:
    """Immutable record of a single training run."""

    model_name: str
    version: str
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float = 0.0
    fold: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    artifact_path: str = ""
    status: str = "running"  # running | completed | failed


# ── Trainer ───────────────────────────────────────────────────────────────────


class ModelTrainer:
    """Orchestrates training for arbitrary :class:`TradingModel` instances.

    Supports:
    - Walk-forward training schedule
    - Time-series cross-validation
    - Hyperparameter logging
    - Early stopping (delegated to the underlying model)
    - Best model selection across folds
    """

    def __init__(self, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        self._runs: list[TrainingRun] = []
        self._log = log.bind(component="model_trainer")

    @property
    def runs(self) -> list[TrainingRun]:
        return list(self._runs)

    # ── Single training run ───────────────────────────────────────────

    def train_model(
        self,
        model: TradingModel,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        *,
        fold: int = 0,
    ) -> TrainingRun:
        """Execute a single training run and persist the best artifact."""
        run = TrainingRun(
            model_name=model.metadata.model_name,
            version=model.metadata.version,
            started_at=datetime.now(timezone.utc),
            fold=fold,
            hyperparameters=model.metadata.hyperparameters,
        )
        if self.config.log_hyperparameters:
            self._log.info(
                "training_run_start",
                model=run.model_name,
                fold=fold,
                hyperparameters=run.hyperparameters,
            )

        t0 = time.monotonic()
        try:
            metrics = model.train(X, y, X_val, y_val)
            run.metrics = metrics
            run.status = "completed"

            # Persist artifact.
            artifact_dir = (
                self.config.output_dir
                / model.metadata.model_name
                / model.metadata.version
                / f"fold_{fold}"
            )
            model.save(artifact_dir)
            run.artifact_path = str(artifact_dir)

        except Exception:
            run.status = "failed"
            self._log.exception("training_run_failed", model=run.model_name, fold=fold)
            raise
        finally:
            elapsed = time.monotonic() - t0
            run.duration_seconds = elapsed
            run.finished_at = datetime.now(timezone.utc)
            self._runs.append(run)
            self._log.info(
                "training_run_end",
                model=run.model_name,
                fold=fold,
                status=run.status,
                duration=round(elapsed, 2),
                metrics=run.metrics,
            )

        return run

    # ── Cross-validation ──────────────────────────────────────────────

    def cross_validate(
        self,
        model_factory: type[TradingModel],
        X: np.ndarray,
        y: np.ndarray,
        *,
        model_kwargs: dict[str, Any] | None = None,
    ) -> list[TrainingRun]:
        """Time-series cross-validation using expanding windows.

        A fresh model is instantiated from *model_factory* for each fold.
        """
        splits = _time_series_cv_splits(len(X), self.config.n_cv_splits)
        self._log.info("cross_validation_start", n_splits=len(splits))

        runs: list[TrainingRun] = []
        kwargs = model_kwargs or {}
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            model = model_factory(**kwargs)
            run = self.train_model(
                model,
                X[train_idx],
                y[train_idx],
                X[val_idx],
                y[val_idx],
                fold=fold_idx,
            )
            runs.append(run)

        self._log.info(
            "cross_validation_complete",
            mean_metric={
                k: float(np.mean([r.metrics.get(k, 0) for r in runs]))
                for k in runs[0].metrics
            },
        )
        return runs

    # ── Walk-forward training ─────────────────────────────────────────

    def walk_forward_train(
        self,
        model_factory: type[TradingModel],
        X: np.ndarray,
        y: np.ndarray,
        *,
        model_kwargs: dict[str, Any] | None = None,
    ) -> list[TrainingRun]:
        """Anchored walk-forward: retrain at each step, evaluate on next window."""
        splits = _walk_forward_splits(
            len(X),
            self.config.walk_forward_train_pct,
            self.config.walk_forward_step_pct,
        )
        self._log.info("walk_forward_start", n_steps=len(splits))

        runs: list[TrainingRun] = []
        kwargs = model_kwargs or {}
        for step_idx, (train_idx, val_idx) in enumerate(splits):
            model = model_factory(**kwargs)
            run = self.train_model(
                model,
                X[train_idx],
                y[train_idx],
                X[val_idx],
                y[val_idx],
                fold=step_idx,
            )
            runs.append(run)

        self._log.info("walk_forward_complete", n_runs=len(runs))
        return runs

    # ── Best model selection ──────────────────────────────────────────

    def select_best_run(
        self, runs: list[TrainingRun] | None = None
    ) -> TrainingRun | None:
        """Return the run with the best value of the selection metric.

        Uses ``config.select_best_by`` and ``config.minimize_metric``.
        """
        candidates = [r for r in (runs or self._runs) if r.status == "completed"]
        if not candidates:
            return None

        metric_key = self.config.select_best_by

        def _score(r: TrainingRun) -> float:
            val = r.metrics.get(metric_key, float("inf") if self.config.minimize_metric else float("-inf"))
            return val if self.config.minimize_metric else -val

        best = min(candidates, key=_score)
        self._log.info(
            "best_run_selected",
            model=best.model_name,
            fold=best.fold,
            metric={metric_key: best.metrics.get(metric_key)},
        )
        return best
