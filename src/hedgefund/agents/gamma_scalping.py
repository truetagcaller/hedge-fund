"""Gamma Scalping AI Agent.

Targets elevated gamma exposure near option expiration and delta-hedging
opportunities.  Most active when dealer gamma positioning creates
predictable price pinning or expansion.
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
    MarketRegime.HIGH_VOL_BULLISH: 0.30,
    MarketRegime.HIGH_VOL_BEARISH: 0.30,
    MarketRegime.MEAN_REVERTING: 0.25,
    MarketRegime.LOW_VOL_BULLISH: 0.10,
    MarketRegime.LOW_VOL_BEARISH: 0.10,
    MarketRegime.TRENDING: 0.10,
}


class GammaScalpingAgent(TradingAgent):
    """Exploits gamma exposure and delta hedging flows.

    Strategy logic
    --------------
    1. **Negative GEX**: Dealers are short gamma → expect amplified moves
       away from strikes → momentum entry.
    2. **Positive GEX**: Dealers are long gamma → expect pinning near max
       pain → mean reversion entry.
    3. **High gamma** near expiry creates rapid delta changes that market
       makers must hedge, amplifying directional moves.
    4. **Max pain gravitation**: Price tends to drift toward max pain as
       expiry approaches.
    """

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__("gamma_scalping", event_bus)

    def get_weight(self, regime: MarketRegime) -> float:
        return _REGIME_WEIGHTS.get(regime, 0.10)

    async def analyze(
        self,
        symbol: str,
        market_data: dict[str, Any],
    ) -> AgentSignal | None:
        price = market_data.get("price", 0.0)
        gamma = market_data.get("gamma", 0.0)
        delta = market_data.get("delta", 0.0)
        gex = market_data.get("gex", 0.0)
        max_pain = market_data.get("max_pain", 0.0)
        atr = market_data.get("atr", 0.0)
        iv = market_data.get("iv", 0.0)

        if price <= 0:
            return None
        if atr <= 0:
            atr = price * 0.02

        # ── GEX regime ────────────────────────────────────────────────
        # Negative GEX → amplified moves (momentum)
        # Positive GEX → suppressed moves (pinning)
        gex_negative = gex < 0
        gex_magnitude = abs(gex)

        # ── Max pain gravitation ──────────────────────────────────────
        max_pain_valid = max_pain > 0
        if max_pain_valid:
            distance_to_mp = (max_pain - price) / price  # positive = price below MP
        else:
            distance_to_mp = 0.0

        # ── Gamma exposure scoring ────────────────────────────────────
        gamma_significant = abs(gamma) > 0.01

        if not gamma_significant and gex_magnitude < 1e6:
            return None  # Not enough gamma activity

        # ── Direction logic ───────────────────────────────────────────
        if gex_negative:
            # Dealers short gamma — momentum trade
            # Direction based on current delta tilt
            if delta > 0.1:
                direction = SignalDirection.LONG
                action = SignalAction.BUY_CALL
            elif delta < -0.1:
                direction = SignalDirection.SHORT
                action = SignalAction.BUY_PUT
            else:
                return None  # No clear direction
        else:
            # Dealers long gamma — pinning trade toward max pain
            if max_pain_valid and abs(distance_to_mp) > 0.005:
                if distance_to_mp > 0:
                    direction = SignalDirection.LONG
                    action = SignalAction.BUY_CALL
                else:
                    direction = SignalDirection.SHORT
                    action = SignalAction.BUY_PUT
            else:
                return None  # Price already at max pain

        # ── Confidence ────────────────────────────────────────────────
        gex_score = min(1.0, gex_magnitude / 5e9) if gex_magnitude > 0 else 0.0
        gamma_score = min(1.0, abs(gamma) / 0.1)
        mp_score = min(1.0, abs(distance_to_mp) / 0.02) if max_pain_valid else 0.0

        confidence = (
            gex_score * 0.40
            + gamma_score * 0.30
            + mp_score * 0.30
        )
        confidence = min(1.0, max(0.0, confidence))

        if confidence < 0.25:
            return None

        # ── Price levels ──────────────────────────────────────────────
        if gex_negative:
            # Momentum — wider stops
            stop_distance = atr * 2.0
            target_distance = atr * 3.0
        else:
            # Pinning — tighter range
            stop_distance = atr * 1.0
            target_distance = abs(max_pain - price) if max_pain_valid else atr * 1.5

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
                f"GammaScalp: GEX={gex:.0f}, gamma={gamma:.4f}, "
                f"delta={delta:+.3f}, max_pain={max_pain:.2f}"
            ),
            metadata={
                "gex": gex,
                "gamma": gamma,
                "delta": delta,
                "max_pain": max_pain,
                "gex_negative": gex_negative,
            },
        )
