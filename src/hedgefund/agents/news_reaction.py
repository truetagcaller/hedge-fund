"""News Reaction AI Agent.

Reacts to incoming news sentiment by generating directional signals.
Strongest in high-volatility regimes where news drives rapid price moves.
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
    MarketRegime.HIGH_VOL_BULLISH: 0.35,
    MarketRegime.HIGH_VOL_BEARISH: 0.35,
    MarketRegime.TRENDING: 0.20,
    MarketRegime.LOW_VOL_BULLISH: 0.10,
    MarketRegime.LOW_VOL_BEARISH: 0.10,
    MarketRegime.MEAN_REVERTING: 0.10,
}


class NewsReactionAgent(TradingAgent):
    """Generates trading signals from news sentiment.

    Strategy logic
    --------------
    1. **Sentiment magnitude**: Only act on strong sentiment (|score| > 0.3).
    2. **Direction**: Positive sentiment → bullish; negative → bearish.
    3. **Confidence**: Proportional to sentiment magnitude and source
       reliability.
    4. **Speed premium**: Recent news gets higher weight (decays over time).
    """

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__("news_reaction", event_bus)
        self._latest_sentiment: dict[str, dict[str, Any]] = {}

    def subscribe(self) -> None:
        """Subscribe to both TICK and NEWS events."""
        super().subscribe()
        self._event_bus.subscribe(EventType.NEWS, self._on_news)

    async def _on_news(self, event: Event) -> None:
        """Cache latest news sentiment per symbol."""
        if not event.source or not event.symbol:
            return
        log.info(
            "agent.news_received",
            agent=self._name,
            symbol=event.symbol,
            source=event.source,
        )
        self._latest_sentiment[event.symbol] = {
            "score": event.data.get("sentiment_score", event.data.get("score", 0.0)),
            "magnitude": event.data.get("magnitude", 0.5),
            "headline": event.data.get("headline", ""),
            "source": event.source,
            "timestamp": event.timestamp,
        }

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

        atr = market_data.get("atr", 0.0)
        if atr <= 0:
            atr = price * 0.02

        # Use cached news sentiment or market_data fallback
        news_data = self._latest_sentiment.get(symbol)
        sentiment_score = market_data.get("news_sentiment", 0.0)
        magnitude = 0.5

        if news_data:
            sentiment_score = float(news_data.get("score", sentiment_score))
            magnitude = float(news_data.get("magnitude", 0.5))

        if abs(sentiment_score) < 0.3:
            return None  # Sentiment not strong enough

        # ── Direction ─────────────────────────────────────────────────
        if sentiment_score > 0:
            direction = SignalDirection.LONG
            action = SignalAction.BUY_CALL
        else:
            direction = SignalDirection.SHORT
            action = SignalAction.BUY_PUT

        # ── Confidence ────────────────────────────────────────────────
        sentiment_strength = min(1.0, abs(sentiment_score))
        magnitude_score = min(1.0, magnitude)

        confidence = sentiment_strength * 0.60 + magnitude_score * 0.40
        confidence = min(1.0, max(0.0, confidence))

        if confidence < 0.25:
            return None

        # ── Price levels ──────────────────────────────────────────────
        # News-driven moves tend to be sharp — use wider stops
        stop_distance = atr * 2.0
        target_distance = atr * 3.0

        if direction == SignalDirection.LONG:
            stop_loss = price - stop_distance
            take_profit = price + target_distance
        else:
            stop_loss = price + stop_distance
            take_profit = price - target_distance

        rr = target_distance / stop_distance if stop_distance > 0 else 0.0
        headline = ""
        if news_data:
            headline = str(news_data.get("headline", ""))[:100]

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
                f"News: sentiment={sentiment_score:+.2f}, "
                f"magnitude={magnitude:.2f}, headline={headline!r}"
            ),
            metadata={
                "sentiment_score": sentiment_score,
                "magnitude": magnitude,
                "headline": headline,
            },
        )
