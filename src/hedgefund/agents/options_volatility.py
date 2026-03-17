"""Options Volatility AI Agent.

Analyses implied volatility rank, IV/HV spread, put-call ratio, and
volatility skew to generate options-specific trading signals.
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
    MarketRegime.HIGH_VOL_BULLISH: 0.35,
    MarketRegime.HIGH_VOL_BEARISH: 0.35,
    MarketRegime.LOW_VOL_BULLISH: 0.15,
    MarketRegime.LOW_VOL_BEARISH: 0.15,
    MarketRegime.TRENDING: 0.15,
    MarketRegime.MEAN_REVERTING: 0.20,
}


class OptionsVolatilityAgent(TradingAgent):
    """Generates signals from options-market volatility data.

    Strategy logic
    --------------
    1. **IV rank**: IV percentile relative to 1-year range.
       High IV → sell premium; Low IV → buy premium.
    2. **IV/HV spread**: IV >> HV suggests options over-priced.
    3. **Put-call ratio**: Extreme PCR values signal crowd positioning.
    4. **GEX (Gamma Exposure)**: Negative GEX → volatile moves expected.
    """

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__("options_volatility", event_bus)

    def get_weight(self, regime: MarketRegime) -> float:
        return _REGIME_WEIGHTS.get(regime, 0.15)

    async def analyze(
        self,
        symbol: str,
        market_data: dict[str, Any],
    ) -> AgentSignal | None:
        price = market_data.get("price", 0.0)
        iv = market_data.get("iv", 0.0)
        hv = market_data.get("hv", 0.0)
        pcr = market_data.get("pcr", 0.0)
        gex = market_data.get("gex", 0.0)
        atr = market_data.get("atr", 0.0)

        if price <= 0 or iv <= 0:
            return None

        # ── IV/HV spread ──────────────────────────────────────────────
        iv_hv_ratio = iv / hv if hv > 0 else 1.0
        iv_overpriced = iv_hv_ratio > 1.3
        iv_underpriced = iv_hv_ratio < 0.8

        # ── PCR analysis ──────────────────────────────────────────────
        # PCR > 1.2 = extreme bearish positioning (contrarian bullish)
        # PCR < 0.5 = extreme bullish positioning (contrarian bearish)
        pcr_bullish = pcr > 1.2
        pcr_bearish = pcr < 0.5 and pcr > 0

        # ── GEX analysis ──────────────────────────────────────────────
        # Negative GEX = dealers short gamma → amplified moves
        negative_gex = gex < 0

        # ── Direction scoring ─────────────────────────────────────────
        bullish_score = 0.0
        bearish_score = 0.0

        if pcr_bullish:
            bullish_score += 0.35
        if pcr_bearish:
            bearish_score += 0.35

        if iv_underpriced:
            # IV cheap → buy options in the trending direction
            bullish_score += 0.15
        if iv_overpriced:
            # IV expensive → sell premium or go contrarian
            bearish_score += 0.10

        if negative_gex:
            # Amplified moves expected — favour momentum
            bullish_score += 0.10
            bearish_score += 0.10

        # Net direction
        net = bullish_score - bearish_score
        if abs(net) < 0.10:
            return None

        if net > 0:
            direction = SignalDirection.LONG
            action = SignalAction.BUY_CALL
            strength = bullish_score
        else:
            direction = SignalDirection.SHORT
            action = SignalAction.BUY_PUT
            strength = bearish_score

        # ── Confidence ────────────────────────────────────────────────
        pcr_extremity = min(1.0, abs(pcr - 0.85) / 0.6) if pcr > 0 else 0.0
        iv_signal = min(1.0, abs(iv_hv_ratio - 1.0) / 0.5)
        gex_signal = min(1.0, abs(gex) / 1e9) if gex != 0 else 0.0

        confidence = (
            pcr_extremity * 0.40
            + iv_signal * 0.35
            + gex_signal * 0.15
            + strength * 0.10
        )
        confidence = min(1.0, max(0.0, confidence))

        if confidence < 0.25:
            return None

        # ── Price levels ──────────────────────────────────────────────
        if atr <= 0:
            atr = price * 0.02

        stop_distance = atr * 1.5
        target_distance = atr * 2.0

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
                f"OptionsVol: IV={iv:.2f}, HV={hv:.2f}, "
                f"IV/HV={iv_hv_ratio:.2f}, PCR={pcr:.2f}, GEX={gex:.0f}"
            ),
            metadata={
                "iv": iv,
                "hv": hv,
                "iv_hv_ratio": iv_hv_ratio,
                "pcr": pcr,
                "gex": gex,
            },
        )
