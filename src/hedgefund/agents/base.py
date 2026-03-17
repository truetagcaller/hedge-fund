"""Abstract base class for all AI trading agents.

Every agent must verify its data source before generating signals.
NO synthetic, mock, or random data is ever permitted.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from hedgefund.streaming.event_bus import Event, EventBus, EventType
from hedgefund.types import MarketRegime, SignalAction, SignalDirection

log = structlog.get_logger(__name__)

# Sources that are NEVER allowed
_FORBIDDEN_SOURCES = frozenset({
    "synthetic", "mock", "random", "generated", "simulated",
    "fake", "sample", "test", "dummy", "placeholder",
})


@dataclass(slots=True)
class AgentSignal:
    """Signal produced by a single AI trading agent."""

    agent_name: str
    timestamp: datetime
    symbol: str
    action: SignalAction
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)
    data_source: str = ""
    is_live_data: bool = False


class TradingAgent(abc.ABC):
    """Base class for all AI trading agents.

    Subclasses implement :meth:`analyze` and :meth:`get_weight` to provide
    strategy-specific logic.  The base class enforces data source verification
    and activation gating so that **no agent can produce signals from
    unverified or synthetic data**.

    Parameters
    ----------
    name:
        Unique agent identifier.
    event_bus:
        Central event bus for subscribing to market events.
    """

    def __init__(self, name: str, event_bus: EventBus) -> None:
        self._name = name
        self._event_bus = event_bus
        self._active = False
        self._data_source_verified = False
        self._verified_source: str = ""
        self._last_data_timestamp: datetime | None = None
        self._signals_generated: int = 0
        self._last_signal_time: float = 0.0

    # ── Properties ────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def data_source_verified(self) -> bool:
        return self._data_source_verified

    @property
    def last_data_timestamp(self) -> datetime | None:
        return self._last_data_timestamp

    @property
    def signals_generated(self) -> int:
        return self._signals_generated

    # ── Activation / deactivation ─────────────────────────────────────

    def activate(self) -> None:
        """Activate the agent.  Only succeeds if data source is verified."""
        if not self._data_source_verified:
            log.warning(
                "agent.activate_blocked",
                agent=self._name,
                reason="data_source_not_verified",
            )
            return
        self._active = True
        log.info("agent.activated", agent=self._name)

    def deactivate(self) -> None:
        """Deactivate the agent — it will stop producing signals."""
        self._active = False
        log.info("agent.deactivated", agent=self._name)

    # ── Data source verification ──────────────────────────────────────

    def verify_data_source(self, source: str) -> bool:
        """Validate that *source* represents a real data feed.

        Returns ``True`` if the source is accepted; ``False`` if it is
        synthetic or otherwise forbidden.
        """
        if not source:
            log.warning("agent.empty_source", agent=self._name)
            return False

        if source.lower() in _FORBIDDEN_SOURCES:
            log.error(
                "agent.forbidden_source",
                agent=self._name,
                source=source,
            )
            return False

        self._data_source_verified = True
        self._verified_source = source
        log.info(
            "agent.data_source_verified",
            agent=self._name,
            source=source,
        )
        return True

    # ── Signal generation (guarded) ───────────────────────────────────

    async def generate_signal(
        self,
        symbol: str,
        market_data: dict[str, Any],
    ) -> AgentSignal | None:
        """Public entry point — guards then delegates to :meth:`analyze`."""
        if not self._active:
            return None
        if not self._data_source_verified:
            return None

        # Validate data origin
        source = market_data.get("source", "")
        if not source or source.lower() in _FORBIDDEN_SOURCES:
            log.warning(
                "agent.rejected_data",
                agent=self._name,
                source=source,
                reason="unverified_or_forbidden_source",
            )
            return None

        self._last_data_timestamp = datetime.now(timezone.utc)

        signal = await self.analyze(symbol, market_data)
        if signal is not None:
            signal.data_source = source
            signal.is_live_data = True
            signal.agent_name = self._name
            self._signals_generated += 1
            self._last_signal_time = time.monotonic()
            log.info(
                "agent.signal_generated",
                agent=self._name,
                symbol=symbol,
                action=signal.action.value,
                confidence=signal.confidence,
                source=source,
            )
        return signal

    # ── EventBus handler ──────────────────────────────────────────────

    async def on_tick(self, event: Event) -> None:
        """Default tick handler — subclasses may override."""
        log.debug(
            "agent.tick_received",
            agent=self._name,
            symbol=event.symbol,
            source=event.source,
        )

    def subscribe(self) -> None:
        """Subscribe to relevant EventBus events."""
        self._event_bus.subscribe(EventType.TICK, self.on_tick)

    # ── Abstract interface ────────────────────────────────────────────

    @abc.abstractmethod
    async def analyze(
        self,
        symbol: str,
        market_data: dict[str, Any],
    ) -> AgentSignal | None:
        """Produce a signal from real market data.

        Must return ``None`` if the data is insufficient or no trade is
        warranted.  Implementations must **never** generate synthetic data.
        """
        ...

    @abc.abstractmethod
    def get_weight(self, regime: MarketRegime) -> float:
        """Return this agent's importance weight for the given regime.

        Values should be in [0.0, 1.0].  The
        :class:`~hedgefund.engine.signal_fusion.SignalFusionEngine`
        normalises across all agents.
        """
        ...

    # ── Status ────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "active": self._active,
            "data_source_verified": self._data_source_verified,
            "verified_source": self._verified_source,
            "last_data_timestamp": (
                self._last_data_timestamp.isoformat()
                if self._last_data_timestamp
                else None
            ),
            "signals_generated": self._signals_generated,
        }
