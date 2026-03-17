"""ML-based signal generator using scikit-learn classifiers.

Supports Random Forest and Gradient Boosted Trees for signal classification.
Tracks feature importances and emits probability-based confidence scores.
"""

from __future__ import annotations

import asyncio
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from hedgefund.signals.base import SignalGenerator
from hedgefund.types import (
    MarketRegime,
    SentimentResult,
    SignalAction,
    SignalDirection,
    TradeSignal,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Label mapping: classifier outputs integer class labels
#   0 -> SHORT, 1 -> NEUTRAL, 2 -> LONG
# ---------------------------------------------------------------------------
_CLASS_TO_DIRECTION: dict[int, SignalDirection] = {
    0: SignalDirection.SHORT,
    1: SignalDirection.NEUTRAL,
    2: SignalDirection.LONG,
}

_DIRECTION_TO_ACTION: dict[SignalDirection, SignalAction] = {
    SignalDirection.LONG: SignalAction.BUY_CALL,
    SignalDirection.SHORT: SignalAction.BUY_PUT,
    SignalDirection.NEUTRAL: SignalAction.NO_TRADE,
}

# Default feature columns the model expects.
_DEFAULT_FEATURES: list[str] = [
    "close",
    "volume",
    "rsi",
    "ema_fast",
    "ema_slow",
    "macd",
    "macd_hist",
    "atr",
    "bb_upper",
    "bb_lower",
    "iv_rank",
    "iv_percentile",
    "delta",
    "gamma",
    "theta",
    "vega",
    "put_call_ratio",
    "sentiment_score",
]


class MLSignalGenerator(SignalGenerator):
    """Generate trade signals with a scikit-learn classification model.

    The model predicts one of three classes (SHORT / NEUTRAL / LONG) and
    exposes per-class probabilities that become the signal's confidence.

    Parameters:
        model_path: Path to a pickled scikit-learn classifier.  If *None* the
            generator will lazily train a default model on the first call (or
            return empty signals if no training data is available).
        feature_columns: Ordered list of column names the model expects.
        min_confidence: Minimum predicted probability to emit a signal.
        atr_stop_mult: ATR multiplier for stop-loss distance.
        min_rr: Minimum risk/reward ratio for emitted signals.
        strategy_name: Name attached to generated signals.
    """

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        feature_columns: list[str] | None = None,
        min_confidence: float = 0.55,
        atr_stop_mult: float = 1.5,
        min_rr: float = 2.0,
        strategy_name: str = "ml_classifier",
    ) -> None:
        self._model: Any | None = None
        self._model_path = Path(model_path) if model_path else None
        self._features = feature_columns or _DEFAULT_FEATURES
        self._min_conf = min_confidence
        self._atr_mult = atr_stop_mult
        self._min_rr = min_rr
        self._name = strategy_name
        self._feature_importances: dict[str, float] = {}

    # -- model lifecycle ----------------------------------------------------

    def load_model(self, path: str | Path | None = None) -> None:
        """Load a pickled model from disk."""
        p = Path(path) if path else self._model_path
        if p is None or not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        with open(p, "rb") as fh:
            self._model = pickle.load(fh)  # noqa: S301
        self._extract_importances()
        log.info("ml_model_loaded", path=str(p))

    def save_model(self, path: str | Path) -> None:
        """Persist the current model to disk."""
        if self._model is None:
            raise RuntimeError("No model to save")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(self._model, fh)
        log.info("ml_model_saved", path=str(p))

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        model_type: str = "gradient_boosting",
    ) -> dict[str, float]:
        """Train a new classifier on labelled data.

        Args:
            X: Feature matrix (columns matching ``self._features``).
            y: Integer labels (0=SHORT, 1=NEUTRAL, 2=LONG).
            model_type: ``"random_forest"`` or ``"gradient_boosting"``.

        Returns:
            Feature importance dict.
        """
        from sklearn.ensemble import (  # type: ignore[import-untyped]
            GradientBoostingClassifier,
            RandomForestClassifier,
        )

        available = [c for c in self._features if c in X.columns]
        if not available:
            raise ValueError("No matching feature columns in training data")

        X_clean = X[available].fillna(0)

        if model_type == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            )
        else:
            clf = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )

        clf.fit(X_clean, y)
        self._model = clf
        self._features = available
        self._extract_importances()
        log.info(
            "ml_model_trained",
            model_type=model_type,
            n_samples=len(X_clean),
            n_features=len(available),
        )
        return dict(self._feature_importances)

    @property
    def feature_importances(self) -> dict[str, float]:
        return dict(self._feature_importances)

    # -- signal generation --------------------------------------------------

    async def generate(
        self,
        features_df: pd.DataFrame,
        regime: MarketRegime,
        sentiment: SentimentResult,
    ) -> list[TradeSignal]:
        if self._model is None:
            if self._model_path and self._model_path.exists():
                self.load_model()
            else:
                log.warning("ml_no_model_available")
                return []

        if features_df.empty:
            return []

        available = [c for c in self._features if c in features_df.columns]
        if not available:
            log.warning("ml_no_matching_features")
            return []

        latest_row = features_df.iloc[[-1]][available].fillna(0)

        pred_class, probabilities = await asyncio.to_thread(
            self._predict, latest_row
        )

        direction = _CLASS_TO_DIRECTION.get(pred_class, SignalDirection.NEUTRAL)
        if direction == SignalDirection.NEUTRAL:
            return []

        confidence = float(probabilities[pred_class])
        if confidence < self._min_conf:
            return []

        close = float(features_df.iloc[-1]["close"])
        atr = float(features_df.iloc[-1].get("atr", 0))
        if atr <= 0:
            return []

        signal = self._build_signal(direction, close, atr, confidence, regime)
        if signal is None:
            return []

        signal.underlying = str(features_df.iloc[-1].get("symbol", ""))
        signal.metadata["feature_importances_top5"] = dict(
            sorted(self._feature_importances.items(), key=lambda x: -x[1])[:5]
        )
        signal.metadata["class_probabilities"] = {
            str(k): round(v, 4) for k, v in enumerate(probabilities)
        }

        return [signal]

    # -- internals ----------------------------------------------------------

    def _predict(self, row: pd.DataFrame) -> tuple[int, np.ndarray]:
        proba = self._model.predict_proba(row)[0]  # type: ignore[union-attr]
        pred = int(np.argmax(proba))
        return pred, proba

    def _extract_importances(self) -> None:
        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
            self._feature_importances = {
                name: float(imp)
                for name, imp in zip(self._features, importances)
            }

    def _build_signal(
        self,
        direction: SignalDirection,
        close: float,
        atr: float,
        confidence: float,
        regime: MarketRegime,
    ) -> TradeSignal | None:
        stop_dist = atr * self._atr_mult
        target_dist = stop_dist * self._min_rr

        if direction == SignalDirection.LONG:
            action = SignalAction.BUY_CALL
            stop = close - stop_dist
            target = close + target_dist
        else:
            action = SignalAction.BUY_PUT
            stop = close + stop_dist
            target = close - target_dist

        rr = target_dist / stop_dist if stop_dist > 0 else 0.0
        if rr < self._min_rr:
            return None

        return TradeSignal(
            signal_id=TradeSignal.generate_id(),
            timestamp=datetime.now(timezone.utc),
            underlying="",
            action=action,
            direction=direction,
            confidence=round(confidence, 4),
            strategy_name=self._name,
            entry_price=close,
            stop_loss=round(stop, 4),
            target_price=round(target, 4),
            risk_reward_ratio=round(rr, 2),
            reasoning=f"ML {direction.value} (p={confidence:.2%})",
            metadata={"regime": regime.value},
        )
