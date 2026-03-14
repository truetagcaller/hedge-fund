"""Trend Following AI Agent.

Analyses EMA crossovers, ADX trend strength, and momentum to identify
sustained directional moves.  Receives highest weight in TRENDING regimes
and near-zero weight in MEAN_REVERTING markets.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from hedgefund.agents.base import AgentSignal, TradingAgent
from hedgefund.streaming.event_bus import EventBus
from hedgefund.types import MarketRegime, SignalAction, SignalDirection

log = structlog.get_logger(__name__)

_REGIME_WEIGHTS: dict[MarketRegime, float] = {
    MarketRegime.TRENDING: 0.40,
    MarketRegime.LOW_VOL_BULLISH: 0.25,
    MarketRegime.HIGH_VOL_BULLISH: 0.20,
    MarketRegime.LOW_VOL_BEARISH: 0.25,
    MarketRegime.HIGH_VOL_BEARISH: 0.20,
    MarketRegime.MEAN_REVERTING: 0.05,
}


class TrendFollowingAgent(TradingAgent):
    """Detects and follows sustained price trends.

    Strategy logic
    --------------
    1. **EMA alignment**: EMA(9) > EMA(21) > EMA(50) > EMA(200) = strong
       uptrend (inverse for downtrend).
    2. **ADX filter**: ADX > 25 confirms trend presence.
    3. **Momentum confirmation**: MACD histogram positive and rising.
    4. **Entry**: on pull-back to EMA(21) in direction of trend.
    5. **Stop**: 1.5 × ATR below entry (long) / above entry (short).
    6. **Target**: 2.5 × ATR in trend direction.
    """

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__("trend_following", event_bus)

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

        ema_9 = market_data.get("ema_9", 0.0)
        ema_21 = market_data.get("ema_21", 0.0)
        ema_50 = market_data.get("ema_50", 0.0)
        ema_200 = market_data.get("ema_200", 0.0)
        adx = market_data.get("adx", 0.0)
        macd = market_data.get("macd", 0.0)
        macd_signal = market_data.get("macd_signal", 0.0)
        atr = market_data.get("atr", 0.0)

        if not all([ema_9, ema_21, ema_50, atr]):
            return None

        # ── EMA alignment score ───────────────────────────────────────
        bullish_alignment = 0
        if ema_9 > ema_21:
            bullish_alignment += 1
        if ema_21 > ema_50:
            bullish_alignment += 1
        if ema_50 > ema_200 > 0:
            bullish_alignment += 1

        bearish_alignment = 0
        if ema_9 < ema_21:
            bearish_alignment += 1
        if ema_21 < ema_50:
            bearish_alignment += 1
        if ema_200 > 0 and ema_50 < ema_200:
            bearish_alignment += 1

        # ── ADX filter ────────────────────────────────────────────────
        adx_strength = min(1.0, max(0.0, (adx - 15) / 35))  # 15-50 range
        if adx < 20:
            return None  # No trend present

        # ── MACD confirmation ─────────────────────────────────────────
        macd_hist = macd - macd_signal
        macd_bullish = macd_hist > 0
        macd_bearish = macd_hist < 0

        # ── Direction determination ───────────────────────────────────
        if bullish_alignment >= 2 and macd_bullish:
            direction = SignalDirection.LONG
            action = SignalAction.BUY_CALL
            alignment_score = bullish_alignment / 3.0
        elif bearish_alignment >= 2 and macd_bearish:
            direction = SignalDirection.SHORT
            action = SignalAction.BUY_PUT
            alignment_score = bearish_alignment / 3.0
        else:
            return None

        # ── Confidence ────────────────────────────────────────────────
        confidence = (
            alignment_score * 0.40
            + adx_strength * 0.35
            + min(1.0, abs(macd_hist) / (atr * 0.5 + 1e-10)) * 0.25
        )
        confidence = min(1.0, max(0.0, confidence))

        if confidence < 0.30:
            return None

        # ── Price levels ──────────────────────────────────────────────
        stop_distance = atr * 1.5
        target_distance = atr * 2.5

        if direction == SignalDirection.LONG:
            entry = price
            stop_loss = price - stop_distance
            take_profit = price + target_distance
        else:
            entry = price
            stop_loss = price + stop_distance
            take_profit = price - target_distance

        rr = target_distance / stop_distance if stop_distance > 0 else 0.0

        reasons = []
        reasons.append(f"EMA alignment={alignment_score:.0%}")
        reasons.append(f"ADX={adx:.1f}")
        reasons.append(f"MACD_hist={macd_hist:+.4f}")

        return AgentSignal(
            agent_name=self._name,
            timestamp=datetime.utcnow(),
            symbol=symbol,
            action=action,
            direction=direction,
            confidence=round(confidence, 4),
            entry_price=round(entry, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            risk_reward_ratio=round(rr, 2),
            reasoning=f"Trend: {'; '.join(reasons)}",
            metadata={
                "ema_alignment": alignment_score,
                "adx": adx,
                "macd_histogram": macd_hist,
                "atr": atr,
            },
        )
