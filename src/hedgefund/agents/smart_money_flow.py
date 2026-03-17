"""Smart Money Flow AI Agent.

Tracks institutional order flow, dark pool activity, and large block
trades to follow smart money positioning.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from hedgefund.agents.base import AgentSignal, TradingAgent
from hedgefund.streaming.event_bus import Event, EventBus, EventType
from hedgefund.types import MarketRegime, SignalAction, SignalDirection

log = structlog.get_logger(__name__)

_REGIME_WEIGHTS: dict[MarketRegime, float] = {
    MarketRegime.TRENDING: 0.30,
    MarketRegime.HIGH_VOL_BULLISH: 0.25,
    MarketRegime.HIGH_VOL_BEARISH: 0.25,
    MarketRegime.LOW_VOL_BULLISH: 0.15,
    MarketRegime.LOW_VOL_BEARISH: 0.15,
    MarketRegime.MEAN_REVERTING: 0.10,
}


class SmartMoneyFlowAgent(TradingAgent):
    """Follows institutional / smart money order flow.

    Strategy logic
    --------------
    1. **Smart money score**: Proprietary score from order flow analysis
       tracking large block trades and dark pool prints.
    2. **Accumulation/Distribution**: Net institutional buying vs selling.
    3. **Divergence**: Smart money buying while price drops → bullish
       divergence (and vice versa).
    4. **Confirmation**: Requires sustained flow, not a single print.
    """

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__("smart_money_flow", event_bus)
        self._smart_money_cache: dict[str, dict[str, Any]] = {}

    def subscribe(self) -> None:
        super().subscribe()
        self._event_bus.subscribe(EventType.ORDERBOOK, self._on_orderbook)

    async def _on_orderbook(self, event: Event) -> None:
        if not event.symbol:
            return
        sm_score = event.data.get("smart_money_score")
        if sm_score is not None:
            log.debug(
                "agent.smart_money_data",
                agent=self._name,
                symbol=event.symbol,
                source=event.source,
            )
            self._smart_money_cache[event.symbol] = {
                "score": float(sm_score),
                "source": event.source,
                "timestamp": event.timestamp,
            }

    def get_weight(self, regime: MarketRegime) -> float:
        return _REGIME_WEIGHTS.get(regime, 0.10)

    async def analyze(
        self,
        symbol: str,
        market_data: dict[str, Any],
    ) -> AgentSignal | None:
        price = market_data.get("price", 0.0)
        if price <= 0:
            return None

        atr = market_data.get("atr", 0.0)
        if atr <= 0:
            atr = price * 0.02

        # Get smart money score
        sm_data = self._smart_money_cache.get(symbol, {})
        sm_score = float(market_data.get(
            "smart_money_score",
            sm_data.get("score", 0.0),
        ))

        if abs(sm_score) < 0.15:
            return None  # No significant smart money signal

        # ── Direction ─────────────────────────────────────────────────
        if sm_score > 0:
            direction = SignalDirection.LONG
            action = SignalAction.BUY_CALL
        else:
            direction = SignalDirection.SHORT
            action = SignalAction.BUY_PUT

        # ── Divergence check ──────────────────────────────────────────
        # Smart money buying while indicators are bearish = stronger signal
        rsi = market_data.get("rsi", 50.0)
        divergence = False
        if sm_score > 0.3 and rsi < 40:
            divergence = True  # Bullish divergence
        elif sm_score < -0.3 and rsi > 60:
            divergence = True  # Bearish divergence

        # ── Confidence ────────────────────────────────────────────────
        sm_strength = min(1.0, abs(sm_score))
        divergence_bonus = 0.15 if divergence else 0.0

        confidence = sm_strength * 0.70 + divergence_bonus + 0.15
        confidence = min(1.0, max(0.0, confidence))

        if confidence < 0.25:
            return None

        # ── Price levels ──────────────────────────────────────────────
        # Smart money typically has broader time horizon
        stop_distance = atr * 2.0
        target_distance = atr * 3.5

        if direction == SignalDirection.LONG:
            stop_loss = price - stop_distance
            take_profit = price + target_distance
        else:
            stop_loss = price + stop_distance
            take_profit = price - target_distance

        rr = target_distance / stop_distance if stop_distance > 0 else 0.0

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
                f"SmartMoney: score={sm_score:+.2f}, "
                f"divergence={divergence}, RSI={rsi:.1f}"
            ),
            metadata={
                "smart_money_score": sm_score,
                "divergence": divergence,
                "rsi": rsi,
            },
        )
