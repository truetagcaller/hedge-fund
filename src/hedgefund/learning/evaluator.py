"""Model evaluation: walk-forward analysis, out-of-sample testing, metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import structlog

from hedgefund.learning.base import TradingModel

log = structlog.get_logger(__name__)


# ── Metric computation ────────────────────────────────────────────────────────


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of times the predicted direction matches the actual direction."""
    if len(y_true) < 2:
        return 0.0
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir))


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Accuracy, precision, recall, F1 for discrete labels."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    correct = y_true == y_pred
    accuracy = float(correct.mean()) if len(correct) > 0 else 0.0

    labels = np.unique(np.concatenate([y_true, y_pred]))
    precisions, recalls, f1s = [], [], []
    for lab in labels:
        tp = int(((y_pred == lab) & (y_true == lab)).sum())
        fp = int(((y_pred == lab) & (y_true != lab)).sum())
        fn = int(((y_pred != lab) & (y_true == lab)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


# ── Evaluation result ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class EvaluationResult:
    model_name: str
    dataset: str  # e.g. "validation", "test", "walk_forward_step_3"
    n_samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: np.ndarray | None = None
    residuals: np.ndarray | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "dataset": self.dataset,
            "n_samples": self.n_samples,
            **self.metrics,
        }


# ── Evaluator ─────────────────────────────────────────────────────────────────


class ModelEvaluator:
    """Evaluate trading models with walk-forward analysis and standard metrics.

    Works with any :class:`TradingModel` subclass.
    """

    def __init__(self) -> None:
        self._results: list[EvaluationResult] = []
        self._log = log.bind(component="model_evaluator")

    @property
    def results(self) -> list[EvaluationResult]:
        return list(self._results)

    # ── Out-of-sample evaluation ──────────────────────────────────────

    def evaluate(
        self,
        model: TradingModel,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        *,
        dataset_label: str = "test",
        task: str = "regression",
    ) -> EvaluationResult:
        """Evaluate a trained model on a held-out dataset.

        Args:
            model: A trained ``TradingModel``.
            X: Feature matrix.
            y: Ground-truth targets.
            dataset_label: Human-readable identifier for the dataset.
            task: ``"regression"`` or ``"classification"``.
        """
        y_true = np.asarray(y).ravel()
        y_pred = model.predict(X).ravel()

        # Align lengths (sequence models may drop initial rows).
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[-min_len:]
        y_pred = y_pred[-min_len:]

        if task == "classification":
            metrics = _classification_metrics(y_true, y_pred)
        else:
            metrics = regression_metrics(y_true, y_pred)

        result = EvaluationResult(
            model_name=model.metadata.model_name,
            dataset=dataset_label,
            n_samples=min_len,
            metrics=metrics,
            predictions=y_pred,
            residuals=y_true - y_pred,
        )
        self._results.append(result)
        self._log.info("evaluation_complete", **result.summary())
        return result

    # ── Walk-forward analysis ─────────────────────────────────────────

    def walk_forward_evaluate(
        self,
        model_factory: type[TradingModel],
        X: np.ndarray,
        y: np.ndarray,
        *,
        train_pct: float = 0.7,
        step_pct: float = 0.1,
        model_kwargs: dict[str, Any] | None = None,
        task: str = "regression",
    ) -> list[EvaluationResult]:
        """Anchored walk-forward: retrain, predict on next window, repeat.

        Returns one :class:`EvaluationResult` per step.
        """
        n = len(X)
        train_size = int(n * train_pct)
        step_size = max(int(n * step_pct), 1)
        kwargs = model_kwargs or {}
        results: list[EvaluationResult] = []

        self._log.info("walk_forward_start", n_samples=n, train_size=train_size, step=step_size)

        test_start = train_size
        step_idx = 0
        while test_start < n:
            test_end = min(test_start + step_size, n)
            model = model_factory(**kwargs)
            model.train(X[:test_start], y[:test_start])
            result = self.evaluate(
                model,
                X[test_start:test_end],
                y[test_start:test_end],
                dataset_label=f"walk_forward_step_{step_idx}",
                task=task,
            )
            results.append(result)
            test_start = test_end
            step_idx += 1

        self._log.info(
            "walk_forward_complete",
            n_steps=len(results),
            mean_metrics={
                k: float(np.mean([r.metrics.get(k, 0) for r in results]))
                for k in results[0].metrics
            }
            if results
            else {},
        )
        return results

    # ── Model comparison ──────────────────────────────────────────────

    def compare_models(
        self,
        models: list[TradingModel],
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        *,
        task: str = "regression",
    ) -> pd.DataFrame:
        """Evaluate multiple models on the same dataset and return a comparison table."""
        rows: list[dict[str, Any]] = []
        for model in models:
            result = self.evaluate(model, X, y, dataset_label="comparison", task=task)
            rows.append(result.summary())
        df = pd.DataFrame(rows)
        self._log.info("model_comparison", models=df["model"].tolist())
        return df

    # ── Aggregate statistics ──────────────────────────────────────────

    def summary_table(self) -> pd.DataFrame:
        """Return a DataFrame summarising all evaluation results collected so far."""
        return pd.DataFrame([r.summary() for r in self._results])
