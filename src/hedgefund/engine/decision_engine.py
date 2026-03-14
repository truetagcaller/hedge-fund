"""AI Trading Decision Engine.

Combines all signal sources (technical indicators, options market signals,
order book imbalance, smart money detection, news/social sentiment, and
market regime) into a single composite score per symbol and produces
actionable :class:`TradeDecision` objects when confidence thresholds are met.

Every input arrives via :class:`EventBus` subscriptions -- the engine is
entirely event-driven and maintains a rolling :class:`SignalState` per symbol.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

import structlog

from hedgefund.streaming.event_bus import Event, EventBus, EventType
from hedgefund.types import SignalAction, SignalDirection

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------

class TradeAction(Enum):
    BUY_CALL = "BUY_CALL"
    BUY_PUT = "BUY_PUT"
    SELL_CALL = "SELL_CALL"
    SELL_PUT = "SELL_PUT"
    NO_TRADE = "NO_TRADE"


@dataclass(slots=True)
class TradeDecision:
    """Actionable trading decision produced by the decision engine."""

    decision_id: str
    timestamp: datetime
    symbol: str
    action: TradeAction
    entry_price: float
    stop_loss: float
    target_price: float
    confidence: float
    risk_reward_ratio: float
    reasoning: str
    signal_breakdown: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_id() -> str:
        return f"DEC-{uuid.uuid4().hex[:12].upper()}"


# ---------------------------------------------------------------------------
# Per-symbol signal state
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SignalState:
    """Aggregated signal state maintained for a single symbol."""

    symbol: str

    # Individual source scores: -1.0 (bearish) .. +1.0 (bullish)
    technical: float = 0.0
    options: float = 0.0
    orderbook: float = 0.0
    smart_money: float = 0.0
    news: float = 0.0
    social: float = 0.0
    regime: float = 0.0

    # Latest raw data snapshots
    last_price: float = 0.0
    atr: float = 0.0
    iv: float = 0.0
    pcr: float = 0.0
    gex: float = 0.0
    max_pain: float = 0.0

    # Timestamps for freshness tracking
    technical_ts: float = 0.0
    options_ts: float = 0.0
    orderbook_ts: float = 0.0
    smart_money_ts: float = 0.0
    news_ts: float = 0.0
    social_ts: float = 0.0
    regime_ts: float = 0.0
    price_ts: float = 0.0

    def all_scores(self) -> Dict[str, float]:
        return {
            "technical": self.technical,
            "options": self.options,
            "orderbook": self.orderbook,
            "smart_money": self.smart_money,
            "news": self.news,
            "social": self.social,
            "regime": self.regime,
        }


# ---------------------------------------------------------------------------
# Default signal weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "technical": 0.25,
    "options": 0.20,
    "orderbook": 0.15,
    "smart_money": 0.15,
    "news": 0.10,
    "social": 0.10,
    "regime": 0.05,
}


class TradingDecisionEngine:
    """AI-driven decision engine that fuses all signal sources.

    Parameters
    ----------
    event_bus:
        Central event bus for subscribing to signal events.
    weights:
        Signal source weights.  Must sum to 1.0 (normalized if not).
    min_confidence:
        Minimum composite confidence to emit a decision (0-1).
    cooldown_seconds:
        Minimum seconds between decisions for the same symbol.
    max_concurrent_per_symbol:
        Maximum outstanding (un-executed) decisions per symbol.
    atr_stop_multiplier:
        Multiplier applied to ATR for stop-loss calculation.
    min_risk_reward:
        Minimum risk-reward ratio enforced on every decision.
    stale_seconds:
        Signal data older than this is treated as score 0.0.
    """

    def __init__(
        self,
        event_bus: EventBus,
        *,
        weights: Dict[str, float] | None = None,
        min_confidence: float = 0.65,
        cooldown_seconds: float = 900.0,
        max_concurrent_per_symbol: int = 1,
        atr_stop_multiplier: float = 1.5,
        min_risk_reward: float = 2.0,
        stale_seconds: float = 300.0,
    ) -> None:
        self._bus = event_bus

        # Normalize weights
        raw_w = weights or dict(DEFAULT_WEIGHTS)
        total = sum(raw_w.values())
        self._weights = {k: v / total for k, v in raw_w.items()} if total > 0 else dict(DEFAULT_WEIGHTS)

        self._min_confidence = min_confidence
        self._cooldown_seconds = cooldown_seconds
        self._max_concurrent = max_concurrent_per_symbol
        self._atr_stop_mult = atr_stop_multiplier
        self._min_rr = min_risk_reward
        self._stale_seconds = stale_seconds

        # State
        self._signal_states: Dict[str, SignalState] = {}
        self._last_decision_time: Dict[str, float] = {}  # symbol -> monotonic ts
        self._active_decisions: Dict[str, int] = {}  # symbol -> count
        self._decision_history: list[TradeDecision] = []

        self._subscribe()

    # ------------------------------------------------------------------
    # EventBus subscriptions
    # ------------------------------------------------------------------

    def _subscribe(self) -> None:
        """Register handlers for all relevant event types."""
        self._bus.subscribe(EventType.TICK, self._on_tick)
        self._bus.subscribe(EventType.OPTIONS_CHAIN, self._on_options)
        self._bus.subscribe(EventType.ORDERBOOK, self._on_orderbook)
        self._bus.subscribe(EventType.SIGNAL, self._on_signal)
        self._bus.subscribe(EventType.NEWS, self._on_news)
        self._bus.subscribe(EventType.SENTIMENT, self._on_sentiment)
        self._bus.subscribe(EventType.RISK_UPDATE, self._on_risk_update)
        log.info("decision_engine.subscribed")

    def _ensure_state(self, symbol: str) -> SignalState:
        if symbol not in self._signal_states:
            self._signal_states[symbol] = SignalState(symbol=symbol)
        return self._signal_states[symbol]

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_tick(self, event: Event) -> None:
        """Handle price tick events -- update technical + price data."""
        state = self._ensure_state(event.symbol)
        data = event.data
        now = time.monotonic()

        state.last_price = data.get("price", data.get("close", state.last_price))
        state.atr = data.get("atr", state.atr)
        state.price_ts = now

        # Technical score may arrive embedded in tick data
        if "technical_score" in data:
            state.technical = float(data["technical_score"])
            state.technical_ts = now

    async def _on_options(self, event: Event) -> None:
        """Handle options chain / greeks events."""
        state = self._ensure_state(event.symbol)
        data = event.data
        now = time.monotonic()

        state.iv = data.get("iv", state.iv)
        state.pcr = data.get("pcr", state.pcr)
        state.gex = data.get("gex", state.gex)
        state.max_pain = data.get("max_pain", state.max_pain)

        if "options_score" in data:
            state.options = float(data["options_score"])
        else:
            # Derive a basic score from PCR: PCR > 1 is bearish, < 0.7 is bullish
            pcr = state.pcr
            if pcr > 0:
                state.options = max(-1.0, min(1.0, (0.85 - pcr) * 2.0))
        state.options_ts = now

    async def _on_orderbook(self, event: Event) -> None:
        """Handle order-book imbalance events."""
        state = self._ensure_state(event.symbol)
        data = event.data
        now = time.monotonic()

        if "imbalance_score" in data:
            state.orderbook = float(data["imbalance_score"])
        elif "bid_volume" in data and "ask_volume" in data:
            bid_v = float(data["bid_volume"])
            ask_v = float(data["ask_volume"])
            total = bid_v + ask_v
            if total > 0:
                state.orderbook = max(-1.0, min(1.0, (bid_v - ask_v) / total))
        state.orderbook_ts = now

        # Smart money detection can be embedded
        if "smart_money_score" in data:
            state.smart_money = float(data["smart_money_score"])
            state.smart_money_ts = now

    async def _on_signal(self, event: Event) -> None:
        """Handle explicit signal events from other subsystems."""
        state = self._ensure_state(event.symbol)
        data = event.data
        now = time.monotonic()

        source = data.get("source", "")
        score = data.get("score", 0.0)

        if source == "technical":
            state.technical = float(score)
            state.technical_ts = now
        elif source == "options":
            state.options = float(score)
            state.options_ts = now
        elif source == "orderbook":
            state.orderbook = float(score)
            state.orderbook_ts = now
        elif source == "smart_money":
            state.smart_money = float(score)
            state.smart_money_ts = now
        elif source == "regime":
            state.regime = float(score)
            state.regime_ts = now

    async def _on_news(self, event: Event) -> None:
        """Handle news sentiment events."""
        state = self._ensure_state(event.symbol)
        data = event.data
        now = time.monotonic()

        state.news = float(data.get("sentiment_score", data.get("score", 0.0)))
        state.news_ts = now

    async def _on_sentiment(self, event: Event) -> None:
        """Handle social / X sentiment events."""
        state = self._ensure_state(event.symbol)
        data = event.data
        now = time.monotonic()

        source = data.get("source", "social")
        score = float(data.get("score", data.get("sentiment_score", 0.0)))

        if source == "news":
            state.news = score
            state.news_ts = now
        else:
            state.social = score
            state.social_ts = now

    async def _on_risk_update(self, event: Event) -> None:
        """Handle regime / risk state updates."""
        state = self._ensure_state(event.symbol)
        data = event.data
        now = time.monotonic()

        if "regime_score" in data:
            state.regime = float(data["regime_score"])
            state.regime_ts = now

    # ------------------------------------------------------------------
    # Composite score computation
    # ------------------------------------------------------------------

    def _compute_composite(self, state: SignalState) -> float:
        """Weighted combination of all signal sources.

        Stale signals (older than ``_stale_seconds``) are treated as 0.0
        and their weight is redistributed proportionally.
        """
        now = time.monotonic()
        scores = state.all_scores()
        timestamps = {
            "technical": state.technical_ts,
            "options": state.options_ts,
            "orderbook": state.orderbook_ts,
            "smart_money": state.smart_money_ts,
            "news": state.news_ts,
            "social": state.social_ts,
            "regime": state.regime_ts,
        }

        active_weight = 0.0
        weighted_sum = 0.0
        for key, weight in self._weights.items():
            ts = timestamps.get(key, 0.0)
            if ts > 0 and (now - ts) <= self._stale_seconds:
                weighted_sum += scores[key] * weight
                active_weight += weight

        if active_weight <= 0:
            return 0.0

        # Normalize so active weights sum to their proportion
        return weighted_sum / active_weight

    # ------------------------------------------------------------------
    # Decision generation
    # ------------------------------------------------------------------

    def generate_decision(self, symbol: str) -> Optional[TradeDecision]:
        """Generate a trading decision for *symbol* based on current state.

        Returns ``None`` if confidence is below threshold, the symbol is in
        cooldown, or we already have the maximum concurrent decisions.
        """
        state = self._signal_states.get(symbol)
        if state is None:
            return None

        # Cooldown check
        now = time.monotonic()
        last_ts = self._last_decision_time.get(symbol, 0.0)
        if (now - last_ts) < self._cooldown_seconds:
            return None

        # Max concurrent check
        active_count = self._active_decisions.get(symbol, 0)
        if active_count >= self._max_concurrent:
            return None

        # Need a valid price
        if state.last_price <= 0:
            return None

        composite = self._compute_composite(state)
        confidence = abs(composite)

        if confidence < self._min_confidence:
            return None

        # Determine direction
        if composite > 0:
            direction = SignalDirection.LONG
        elif composite < 0:
            direction = SignalDirection.SHORT
        else:
            return None

        # Compute stop and target
        atr = state.atr if state.atr > 0 else state.last_price * 0.02
        stop_distance = atr * self._atr_stop_mult
        target_distance = stop_distance * self._min_rr

        price = state.last_price
        if direction == SignalDirection.LONG:
            action = TradeAction.BUY_CALL
            stop_loss = price - stop_distance
            target_price = price + target_distance
        else:
            action = TradeAction.BUY_PUT
            stop_loss = price + stop_distance
            target_price = price - target_distance

        rr_ratio = target_distance / stop_distance if stop_distance > 0 else 0.0
        if rr_ratio < self._min_rr:
            return None

        # Build reasoning
        scores = state.all_scores()
        top_contributors = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
        reasons = []
        for src, sc in top_contributors:
            if abs(sc) > 0.05:
                bias = "bullish" if sc > 0 else "bearish"
                reasons.append(f"{src}={sc:+.2f} ({bias})")
        reasoning = f"Composite={composite:+.3f}. " + "; ".join(reasons)

        decision = TradeDecision(
            decision_id=TradeDecision.generate_id(),
            timestamp=datetime.utcnow(),
            symbol=symbol,
            action=action,
            entry_price=round(price, 4),
            stop_loss=round(stop_loss, 4),
            target_price=round(target_price, 4),
            confidence=round(confidence, 4),
            risk_reward_ratio=round(rr_ratio, 2),
            reasoning=reasoning,
            signal_breakdown=dict(scores),
            metadata={
                "composite_score": round(composite, 4),
                "atr": round(atr, 4),
                "iv": round(state.iv, 4),
            },
        )

        self._last_decision_time[symbol] = now
        self._active_decisions[symbol] = active_count + 1
        self._decision_history.append(decision)

        log.info(
            "decision_engine.decision_generated",
            symbol=symbol,
            action=action.value,
            confidence=decision.confidence,
            composite=composite,
            rr=rr_ratio,
        )
        return decision

    # ------------------------------------------------------------------
    # Decision lifecycle management
    # ------------------------------------------------------------------

    def mark_decision_complete(self, symbol: str) -> None:
        """Decrease active decision count when a decision is executed or expires."""
        count = self._active_decisions.get(symbol, 0)
        if count > 0:
            self._active_decisions[symbol] = count - 1

    def get_tracked_symbols(self) -> list[str]:
        """Return all symbols currently being tracked."""
        return list(self._signal_states.keys())

    def get_signal_state(self, symbol: str) -> Optional[SignalState]:
        """Return the current signal state for a symbol."""
        return self._signal_states.get(symbol)

    def get_decision_history(self, limit: int = 50) -> list[TradeDecision]:
        """Return recent decision history."""
        return list(self._decision_history[-limit:])

    async def on_event(self, event: Event) -> None:
        """Unified event handler -- routes to the appropriate handler.

        This can be used as a single subscription point if preferred.
        """
        handlers = {
            EventType.TICK: self._on_tick,
            EventType.OPTIONS_CHAIN: self._on_options,
            EventType.ORDERBOOK: self._on_orderbook,
            EventType.SIGNAL: self._on_signal,
            EventType.NEWS: self._on_news,
            EventType.SENTIMENT: self._on_sentiment,
            EventType.RISK_UPDATE: self._on_risk_update,
        }
        handler = handlers.get(event.event_type)
        if handler is not None:
            await handler(event)
