"""Liquidity Sweep AI Agent.

Detects liquidity sweeps from order book imbalances — large resting
orders being filled rapidly, signalling institutional activity.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from hedgefund.agents.base import AgentSignal, TradingAgent
from hedgefund.streaming.event_bus import Event, EventBus, EventType
from hedgefund.types import MarketRegime, SignalAction, SignalDirection

log = structlog.get_logger(__name__)

_REGIME_WEIGHTS: dict[MarketRegime, float] = {
    MarketRegime.HIGH_VOL_BULLISH: 0.25,
    MarketRegime.HIGH_VOL_BEARISH: 0.25,
    MarketRegime.TRENDING: 0.20,
    MarketRegime.LOW_VOL_BULLISH: 0.15,
    MarketRegime.LOW_VOL_BEARISH: 0.15,
    MarketRegime.MEAN_REVERTING: 0.10,
}


class LiquiditySweepAgent(TradingAgent):
    """Detects liquidity sweeps and generates momentum signals.

    Strategy logic
    --------------
    1. **Order book imbalance**: bid_volume >> ask_volume (or vice versa).
    2. **Volume spike**: Unusually high volume relative to average.
    3. **Sweep direction**: Large buy sweep → bullish momentum;
       large sell sweep → bearish momentum.
    4. **Confirmation**: Price moves in direction of sweep.
    """

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__("liquidity_sweep", event_bus)
        self._latest_orderbook: dict[str, dict[str, Any]] = {}

    def subscribe(self) -> None:
        super().subscribe()
        self._event_bus.subscribe(EventType.ORDERBOOK, self._on_orderbook)

    async def _on_orderbook(self, event: Event) -> None:
        if not event.symbol:
            return
        log.debug(
            "agent.orderbook_received",
            agent=self._name,
            symbol=event.symbol,
            source=event.source,
        )
        self._latest_orderbook[event.symbol] = {
            "bid_volume": event.data.get("bid_volume", 0),
            "ask_volume": event.data.get("ask_volume", 0),
            "imbalance_score": event.data.get("imbalance_score", 0.0),
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

        # Get order book data
        ob_data = self._latest_orderbook.get(symbol, {})
        bid_vol = float(market_data.get("bid_volume", ob_data.get("bid_volume", 0)))
        ask_vol = float(market_data.get("ask_volume", ob_data.get("ask_volume", 0)))
        imbalance = float(market_data.get(
            "imbalance_score",
            ob_data.get("imbalance_score", 0.0),
        ))

        total_vol = bid_vol + ask_vol
        if total_vol <= 0:
            return None

        # ── Imbalance calculation ─────────────────────────────────────
        if imbalance == 0.0 and total_vol > 0:
            imbalance = (bid_vol - ask_vol) / total_vol

        # Need significant imbalance
        if abs(imbalance) < 0.20:
            return None

        # ── Volume spike detection ────────────────────────────────────
        volume = market_data.get("volume", 0)
        avg_volume = market_data.get("avg_volume", 0)
        volume_spike = False
        if avg_volume > 0 and volume > 0:
            volume_spike = volume > avg_volume * 2.0

        # ── Direction ─────────────────────────────────────────────────
        if imbalance > 0.20:
            # More bids → buying pressure → bullish sweep
            direction = SignalDirection.LONG
            action = SignalAction.BUY_CALL
        else:
            # More asks → selling pressure → bearish sweep
            direction = SignalDirection.SHORT
            action = SignalAction.BUY_PUT

        # ── Confidence ────────────────────────────────────────────────
        imb_score = min(1.0, abs(imbalance))
        vol_score = 0.3 if volume_spike else 0.0

        confidence = imb_score * 0.65 + vol_score * 0.35
        confidence = min(1.0, max(0.0, confidence))

        if confidence < 0.25:
            return None

        # ── Price levels ──────────────────────────────────────────────
        stop_distance = atr * 1.5
        target_distance = atr * 2.5

        if direction == SignalDirection.LONG:
            stop_loss = price - stop_distance
            take_profit = price + target_distance
        else:
            stop_loss = price + stop_distance
            take_profit = price - target_distance

        rr = target_distance / stop_distance if stop_distance > 0 else 0.0

        return AgentSignal(
            agent_name=self._name,
            timestamp=datetime.utcnow(),
            symbol=symbol,
            action=action,
            direction=direction,
            confidence=round(confidence, 4),
            entry_price=round(price, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            risk_reward_ratio=round(rr, 2),
            reasoning=(
                f"LiqSweep: imbalance={imbalance:+.2f}, "
                f"bid_vol={bid_vol:.0f}, ask_vol={ask_vol:.0f}, "
                f"vol_spike={volume_spike}"
            ),
            metadata={
                "imbalance": imbalance,
                "bid_volume": bid_vol,
                "ask_volume": ask_vol,
                "volume_spike": volume_spike,
            },
        )
