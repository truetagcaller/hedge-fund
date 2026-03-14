"""Rule-based signal generator implementing classic technical-analysis rules.

Supported rules:
  * EMA crossover (fast/slow)
  * RSI overbought / oversold
  * MACD histogram divergence
  * Breakout detection (N-bar high/low)
  * Support / resistance zone proximity

Each rule independently produces candidate signals with a confidence score.
All signals include ATR-based stop-loss and a target that satisfies a minimum
risk-reward ratio.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

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
# Default parameters
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "ema_fast": 9,
    "ema_slow": 21,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "breakout_period": 20,
    "atr_period": 14,
    "atr_stop_mult": 1.5,
    "min_rr": 2.0,
    "sr_lookback": 50,
    "sr_tolerance_pct": 0.005,
}


def _ensure_column(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


class RuleBasedSignalGenerator(SignalGenerator):
    """Technical-analysis rule engine.

    Parameters:
        params: Override any key from ``_DEFAULTS``.
        strategy_name: Name attached to generated signals.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        *,
        strategy_name: str = "rule_based",
    ) -> None:
        self.p = {**_DEFAULTS, **(params or {})}
        self._name = strategy_name

    async def generate(
        self,
        features_df: pd.DataFrame,
        regime: MarketRegime,
        sentiment: SentimentResult,
    ) -> list[TradeSignal]:
        if features_df.empty or len(features_df) < self.p["ema_slow"] + 1:
            return []

        df = features_df.copy()
        self._compute_indicators(df)

        signals: list[TradeSignal] = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        atr = latest.get("atr")
        if atr is None or atr <= 0:
            return []

        close = float(latest["close"])

        # -- EMA crossover -------------------------------------------------
        sig = self._ema_crossover(latest, prev, close, atr, regime)
        if sig is not None:
            signals.append(sig)

        # -- RSI oversold/overbought ---------------------------------------
        sig = self._rsi_signal(latest, close, atr, regime)
        if sig is not None:
            signals.append(sig)

        # -- MACD divergence -----------------------------------------------
        sig = self._macd_signal(latest, prev, close, atr, regime)
        if sig is not None:
            signals.append(sig)

        # -- Breakout detection --------------------------------------------
        sig = self._breakout_signal(df, latest, close, atr, regime)
        if sig is not None:
            signals.append(sig)

        # -- Support / resistance ------------------------------------------
        sig = self._sr_signal(df, latest, close, atr, regime)
        if sig is not None:
            signals.append(sig)

        return signals

    # ---- indicator computation -------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> None:
        """Add columns that may be missing from the feature DataFrame."""
        close = df["close"]

        if not _ensure_column(df, "ema_fast"):
            df["ema_fast"] = close.ewm(span=self.p["ema_fast"], adjust=False).mean()
        if not _ensure_column(df, "ema_slow"):
            df["ema_slow"] = close.ewm(span=self.p["ema_slow"], adjust=False).mean()

        if not _ensure_column(df, "rsi"):
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(self.p["rsi_period"]).mean()
            loss = (-delta.clip(upper=0)).rolling(self.p["rsi_period"]).mean()
            rs = gain / loss.replace(0, 1e-10)
            df["rsi"] = 100.0 - 100.0 / (1.0 + rs)

        if not _ensure_column(df, "macd"):
            ema_f = close.ewm(span=self.p["macd_fast"], adjust=False).mean()
            ema_s = close.ewm(span=self.p["macd_slow"], adjust=False).mean()
            df["macd"] = ema_f - ema_s
            df["macd_signal"] = df["macd"].ewm(span=self.p["macd_signal"], adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]

        if not _ensure_column(df, "atr"):
            high = df["high"]
            low = df["low"]
            prev_close = close.shift(1)
            tr = pd.concat(
                [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1,
            ).max(axis=1)
            df["atr"] = tr.rolling(self.p["atr_period"]).mean()

    # ---- individual rules ------------------------------------------------

    def _make_signal(
        self,
        direction: SignalDirection,
        close: float,
        atr: float,
        confidence: float,
        reasoning: str,
        regime: MarketRegime,
    ) -> TradeSignal | None:
        """Build a signal with ATR-based stop and min RR target."""
        stop_distance = atr * self.p["atr_stop_mult"]
        min_target_distance = stop_distance * self.p["min_rr"]

        if direction == SignalDirection.LONG:
            action = SignalAction.BUY_CALL
            stop_loss = close - stop_distance
            target = close + min_target_distance
        elif direction == SignalDirection.SHORT:
            action = SignalAction.BUY_PUT
            stop_loss = close + stop_distance
            target = close - min_target_distance
        else:
            return None

        rr = min_target_distance / stop_distance if stop_distance > 0 else 0.0
        if rr < self.p["min_rr"]:
            return None

        return TradeSignal(
            signal_id=TradeSignal.generate_id(),
            timestamp=datetime.utcnow(),
            underlying="",  # caller should fill from features
            action=action,
            direction=direction,
            confidence=max(0.0, min(1.0, confidence)),
            strategy_name=self._name,
            entry_price=close,
            stop_loss=round(stop_loss, 4),
            target_price=round(target, 4),
            risk_reward_ratio=round(rr, 2),
            reasoning=reasoning,
            metadata={"regime": regime.value},
        )

    def _ema_crossover(
        self,
        latest: pd.Series,
        prev: pd.Series,
        close: float,
        atr: float,
        regime: MarketRegime,
    ) -> TradeSignal | None:
        try:
            fast_now = float(latest["ema_fast"])
            slow_now = float(latest["ema_slow"])
            fast_prev = float(prev["ema_fast"])
            slow_prev = float(prev["ema_slow"])
        except (KeyError, TypeError):
            return None

        # Bullish crossover
        if fast_prev <= slow_prev and fast_now > slow_now:
            spread_pct = (fast_now - slow_now) / close
            confidence = min(0.5 + spread_pct * 20, 0.85)
            return self._make_signal(
                SignalDirection.LONG, close, atr, confidence,
                f"EMA {self.p['ema_fast']}/{self.p['ema_slow']} bullish crossover",
                regime,
            )
        # Bearish crossover
        if fast_prev >= slow_prev and fast_now < slow_now:
            spread_pct = (slow_now - fast_now) / close
            confidence = min(0.5 + spread_pct * 20, 0.85)
            return self._make_signal(
                SignalDirection.SHORT, close, atr, confidence,
                f"EMA {self.p['ema_fast']}/{self.p['ema_slow']} bearish crossover",
                regime,
            )
        return None

    def _rsi_signal(
        self,
        latest: pd.Series,
        close: float,
        atr: float,
        regime: MarketRegime,
    ) -> TradeSignal | None:
        try:
            rsi = float(latest["rsi"])
        except (KeyError, TypeError):
            return None

        if rsi <= self.p["rsi_oversold"]:
            distance = self.p["rsi_oversold"] - rsi
            confidence = min(0.5 + distance / 60, 0.9)
            return self._make_signal(
                SignalDirection.LONG, close, atr, confidence,
                f"RSI oversold ({rsi:.1f})", regime,
            )
        if rsi >= self.p["rsi_overbought"]:
            distance = rsi - self.p["rsi_overbought"]
            confidence = min(0.5 + distance / 60, 0.9)
            return self._make_signal(
                SignalDirection.SHORT, close, atr, confidence,
                f"RSI overbought ({rsi:.1f})", regime,
            )
        return None

    def _macd_signal(
        self,
        latest: pd.Series,
        prev: pd.Series,
        close: float,
        atr: float,
        regime: MarketRegime,
    ) -> TradeSignal | None:
        try:
            hist_now = float(latest["macd_hist"])
            hist_prev = float(prev["macd_hist"])
        except (KeyError, TypeError):
            return None

        # Histogram crosses above zero
        if hist_prev <= 0 and hist_now > 0:
            confidence = min(0.45 + abs(hist_now) / close * 100, 0.80)
            return self._make_signal(
                SignalDirection.LONG, close, atr, confidence,
                "MACD histogram bullish crossover", regime,
            )
        # Histogram crosses below zero
        if hist_prev >= 0 and hist_now < 0:
            confidence = min(0.45 + abs(hist_now) / close * 100, 0.80)
            return self._make_signal(
                SignalDirection.SHORT, close, atr, confidence,
                "MACD histogram bearish crossover", regime,
            )
        return None

    def _breakout_signal(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        close: float,
        atr: float,
        regime: MarketRegime,
    ) -> TradeSignal | None:
        period = self.p["breakout_period"]
        if len(df) < period + 1:
            return None

        lookback = df.iloc[-(period + 1) : -1]
        high_max = float(lookback["high"].max())
        low_min = float(lookback["low"].min())

        if close > high_max:
            pct_above = (close - high_max) / high_max
            confidence = min(0.55 + pct_above * 10, 0.85)
            return self._make_signal(
                SignalDirection.LONG, close, atr, confidence,
                f"Breakout above {period}-bar high ({high_max:.2f})", regime,
            )
        if close < low_min:
            pct_below = (low_min - close) / low_min
            confidence = min(0.55 + pct_below * 10, 0.85)
            return self._make_signal(
                SignalDirection.SHORT, close, atr, confidence,
                f"Breakdown below {period}-bar low ({low_min:.2f})", regime,
            )
        return None

    def _sr_signal(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        close: float,
        atr: float,
        regime: MarketRegime,
    ) -> TradeSignal | None:
        """Generate a signal when price is near a support/resistance zone."""
        lookback = self.p["sr_lookback"]
        tol = self.p["sr_tolerance_pct"]
        if len(df) < lookback:
            return None

        window = df.iloc[-lookback:]
        support = float(window["low"].min())
        resistance = float(window["high"].max())

        # Near support -> long
        if abs(close - support) / close <= tol and close > support:
            confidence = 0.50
            return self._make_signal(
                SignalDirection.LONG, close, atr, confidence,
                f"Bounce off support zone ({support:.2f})", regime,
            )
        # Near resistance -> short
        if abs(close - resistance) / close <= tol and close < resistance:
            confidence = 0.50
            return self._make_signal(
                SignalDirection.SHORT, close, atr, confidence,
                f"Rejection at resistance zone ({resistance:.2f})", regime,
            )
        return None
