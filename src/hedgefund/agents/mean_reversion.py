"""Mean Reversion AI Agent.

Identifies over-extended price moves using Bollinger Bands, RSI extremes,
and z-score analysis.  Expects price to revert to the mean.  Receives
highest weight in MEAN_REVERTING regimes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from hedgefund.agents.base import AgentSignal, TradingAgent
from hedgefund.streaming.event_bus import EventBus
from hedgefund.types import MarketRegime, SignalAction, SignalDirection

log = structlog.get_logger(__name__)

_REGIME_WEIGHTS: dict[MarketRegime, float] = {
    MarketRegime.MEAN_REVERTING: 0.40,
    MarketRegime.LOW_VOL_BULLISH: 0.20,
    MarketRegime.LOW_VOL_BEARISH: 0.20,
    MarketRegime.HIGH_VOL_BULLISH: 0.10,
    MarketRegime.HIGH_VOL_BEARISH: 0.10,
    MarketRegime.TRENDING: 0.05,
}


class MeanReversionAgent(TradingAgent):
    """Trades reversion to the mean after extended moves.

    Strategy logic
    --------------
    1. **Bollinger Band**: price at/beyond 2σ band → reversion expected.
    2. **RSI extremes**: RSI < 30 (oversold) or RSI > 70 (overbought).
    3. **Z-score**: price z-score vs 20-bar mean > 2.0 or < -2.0.
    4. **Entry**: at current price when conditions align.
    5. **Stop**: beyond the Bollinger band extreme + 0.5 × ATR.
    6. **Target**: mid-band (20-period SMA).
    """

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__("mean_reversion", event_bus)

    def get_weight(self, regime: MarketRegime) -> float:
        return _REGIME_WEIGHTS.get(regime, 0.15)

    async def analyze(
        self,
        symbol: str,
        market_data: dict[str, Any],
    ) -> AgentSignal | None:
        price = market_data.get("price", 0.0)
        if price <= 0:
            return None

        rsi = market_data.get("rsi", 50.0)
        bb_upper = market_data.get("bollinger_upper", 0.0)
        bb_lower = market_data.get("bollinger_lower", 0.0)
        bb_mid = market_data.get("bollinger_mid", 0.0)
        atr = market_data.get("atr", 0.0)

        if not all([bb_upper, bb_lower, atr]):
            return None

        # If no mid provided, estimate it
        if bb_mid <= 0:
            bb_mid = (bb_upper + bb_lower) / 2.0

        bb_width = bb_upper - bb_lower
        if bb_width <= 0:
            return None

        # ── Bollinger position ────────────────────────────────────────
        bb_pct = (price - bb_lower) / bb_width  # 0 = lower band, 1 = upper

        # ── Z-score approximation ─────────────────────────────────────
        half_width = bb_width / 2.0
        std_est = half_width / 2.0  # BB uses 2σ by default
        z_score = (price - bb_mid) / std_est if std_est > 0 else 0.0

        # ── RSI extremes ──────────────────────────────────────────────
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        rsi_score = 0.0
        if rsi_oversold:
            rsi_score = (30 - rsi) / 30.0  # 0-1 scale
        elif rsi_overbought:
            rsi_score = (rsi - 70) / 30.0

        # ── Direction ─────────────────────────────────────────────────
        oversold = bb_pct < 0.1 or (z_score < -1.5 and rsi_oversold)
        overbought = bb_pct > 0.9 or (z_score > 1.5 and rsi_overbought)

        if oversold:
            direction = SignalDirection.LONG
            action = SignalAction.BUY_CALL
        elif overbought:
            direction = SignalDirection.SHORT
            action = SignalAction.BUY_PUT
        else:
            return None

        # ── Confidence ────────────────────────────────────────────────
        bb_extremity = max(0.0, abs(bb_pct - 0.5) - 0.3) / 0.5
        z_extremity = min(1.0, max(0.0, (abs(z_score) - 1.0) / 2.0))

        confidence = (
            bb_extremity * 0.35
            + z_extremity * 0.30
            + rsi_score * 0.35
        )
        confidence = min(1.0, max(0.0, confidence))

        if confidence < 0.25:
            return None

        # ── Price levels ──────────────────────────────────────────────
        if direction == SignalDirection.LONG:
            stop_loss = bb_lower - atr * 0.5
            take_profit = bb_mid
        else:
            stop_loss = bb_upper + atr * 0.5
            take_profit = bb_mid

        stop_dist = abs(price - stop_loss)
        target_dist = abs(take_profit - price)
        rr = target_dist / stop_dist if stop_dist > 0 else 0.0

        return AgentSignal(
            agent_name=self._name,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            action=action,
            direction=direction,
            confidence=round(confidence, 4),
            entry_price=round(price, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            risk_reward_ratio=round(rr, 2),
            reasoning=(
                f"MeanReversion: RSI={rsi:.1f}, BB%={bb_pct:.2f}, "
                f"Z={z_score:+.2f}"
            ),
            metadata={
                "rsi": rsi,
                "bb_pct": bb_pct,
                "z_score": z_score,
                "bb_mid": bb_mid,
            },
        )
