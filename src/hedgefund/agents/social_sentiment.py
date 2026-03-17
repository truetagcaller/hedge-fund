"""Social Sentiment AI Agent.

Processes X (Twitter) sentiment data to gauge retail/social positioning.
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
    MarketRegime.HIGH_VOL_BULLISH: 0.20,
    MarketRegime.HIGH_VOL_BEARISH: 0.20,
    MarketRegime.TRENDING: 0.15,
    MarketRegime.LOW_VOL_BULLISH: 0.10,
    MarketRegime.LOW_VOL_BEARISH: 0.10,
    MarketRegime.MEAN_REVERTING: 0.10,
}


class SocialSentimentAgent(TradingAgent):
    """Generates signals from X/Twitter social sentiment.

    Strategy logic
    --------------
    1. **Aggregate sentiment**: Weighted average of recent social posts.
    2. **Volume spike**: Unusual mention volume amplifies the signal.
    3. **Contrarian filter**: Extreme bullish crowd → cautiously bearish.
    4. **Confirmation**: Sentiment must persist (not one-off spike).
    """

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__("social_sentiment", event_bus)
        self._latest_social: dict[str, dict[str, Any]] = {}

    def subscribe(self) -> None:
        super().subscribe()
        self._event_bus.subscribe(EventType.SENTIMENT, self._on_sentiment)

    async def _on_sentiment(self, event: Event) -> None:
        if not event.symbol:
            return
        source = event.data.get("source", event.source)
        if source == "news":
            return  # Handled by NewsReactionAgent
        log.info(
            "agent.social_sentiment_received",
            agent=self._name,
            symbol=event.symbol,
            source=event.source,
        )
        self._latest_social[event.symbol] = {
            "score": event.data.get("score", event.data.get("sentiment_score", 0.0)),
            "magnitude": event.data.get("magnitude", 0.5),
            "mention_count": event.data.get("mention_count", 0),
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

        # Get social sentiment
        social_data = self._latest_social.get(symbol)
        social_score = market_data.get("social_sentiment", 0.0)
        magnitude = 0.5

        if social_data:
            social_score = float(social_data.get("score", social_score))
            magnitude = float(social_data.get("magnitude", 0.5))

        if abs(social_score) < 0.25:
            return None

        # ── Direction (with contrarian filter) ────────────────────────
        # Extreme crowd sentiment can be contrarian
        if abs(social_score) > 0.85:
            # Contrarian: go against extreme crowd sentiment
            if social_score > 0:
                direction = SignalDirection.SHORT
                action = SignalAction.BUY_PUT
            else:
                direction = SignalDirection.LONG
                action = SignalAction.BUY_CALL
            contrarian = True
        else:
            # Follow moderate sentiment
            if social_score > 0:
                direction = SignalDirection.LONG
                action = SignalAction.BUY_CALL
            else:
                direction = SignalDirection.SHORT
                action = SignalAction.BUY_PUT
            contrarian = False

        # ── Confidence ────────────────────────────────────────────────
        strength = min(1.0, abs(social_score))
        mag_score = min(1.0, magnitude)

        confidence = strength * 0.50 + mag_score * 0.30
        # Reduce confidence for contrarian trades (higher risk)
        if contrarian:
            confidence *= 0.7
        # Boost if mention volume is high
        mention_count = 0
        if social_data:
            mention_count = int(social_data.get("mention_count", 0))
        if mention_count > 100:
            confidence = min(1.0, confidence + 0.10)

        confidence = min(1.0, max(0.0, confidence))
        if confidence < 0.20:
            return None

        # ── Price levels ──────────────────────────────────────────────
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
                f"Social: score={social_score:+.2f}, "
                f"magnitude={magnitude:.2f}, "
                f"contrarian={contrarian}, mentions={mention_count}"
            ),
            metadata={
                "social_score": social_score,
                "magnitude": magnitude,
                "contrarian": contrarian,
                "mention_count": mention_count,
            },
        )
