"""Central async event bus for the hedge fund trading system.

Implements a publish/subscribe pattern with asyncio queues, supporting
multiple concurrent consumers, optional symbol filtering, backpressure
handling, and real-time processing metrics.
"""

from __future__ import annotations

import asyncio
import enum
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional

from hedgefund.logger import get_logger

log = get_logger(__name__)


class EventType(enum.Enum):
    """All event types flowing through the system."""

    TICK = "TICK"
    ORDERBOOK = "ORDERBOOK"
    OPTIONS_CHAIN = "OPTIONS_CHAIN"
    FILL = "FILL"
    SIGNAL = "SIGNAL"
    NEWS = "NEWS"
    SENTIMENT = "SENTIMENT"
    RISK_UPDATE = "RISK_UPDATE"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"


@dataclass(slots=True)
class Event:
    """A single event flowing through the bus.

    Attributes:
        event_type: Category of the event.
        timestamp: When the event was created.
        symbol: Ticker symbol the event relates to (empty string for system events).
        data: Arbitrary payload dict.
        source: Originating component name.
    """

    event_type: EventType
    timestamp: datetime
    symbol: str
    data: Dict[str, Any]
    source: str


# Type alias for subscriber handlers.
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


@dataclass
class _Subscription:
    """Internal representation of a subscriber."""

    handler: EventHandler
    event_type: EventType
    symbol_filter: Optional[str] = None


class EventBus:
    """Async publish/subscribe event bus with backpressure and metrics.

    Usage::

        bus = EventBus()
        await bus.start()

        bus.subscribe(EventType.TICK, my_handler, symbol="AAPL")
        await bus.publish(Event(...))

        stats = bus.get_stats()
        await bus.stop()
    """

    def __init__(self, queue_size: int = 10_000) -> None:
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=queue_size)
        self._queue_size = queue_size
        self._subscriptions: Dict[EventType, List[_Subscription]] = defaultdict(list)
        self._consumers: List[asyncio.Task[None]] = []
        self._running = False

        # Metrics
        self._events_processed: int = 0
        self._events_dropped: int = 0
        self._started_at: float = 0.0
        self._latency_sum: float = 0.0
        self._latency_count: int = 0

    # ── Subscription management ───────────────────────────────────────────

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
        symbol: Optional[str] = None,
    ) -> None:
        """Register *handler* for events of *event_type*.

        Args:
            event_type: The type of event to listen for.
            handler: Async callable invoked with each matching :class:`Event`.
            symbol: If provided, only events for this symbol are delivered.
        """
        sub = _Subscription(
            handler=handler,
            event_type=event_type,
            symbol_filter=symbol,
        )
        self._subscriptions[event_type].append(sub)
        log.info(
            "event_bus_subscribe",
            event_type=event_type.value,
            symbol=symbol,
            handler=handler.__qualname__,
        )

    def unsubscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """Remove *handler* from *event_type* subscribers."""
        subs = self._subscriptions.get(event_type, [])
        self._subscriptions[event_type] = [
            s for s in subs if s.handler is not handler
        ]

    # ── Publishing ────────────────────────────────────────────────────────

    async def publish(self, event: Event) -> None:
        """Publish an event to the bus.

        If the internal queue is full, the oldest event is dropped and a
        warning is emitted (backpressure / drop-oldest policy).
        """
        if not self._running:
            return

        if self._queue.full():
            # Drop oldest to make room
            try:
                self._queue.get_nowait()
                self._events_dropped += 1
                log.warning(
                    "event_bus_backpressure",
                    dropped_total=self._events_dropped,
                    queue_size=self._queue_size,
                )
            except asyncio.QueueEmpty:
                pass

        await self._queue.put(event)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self, num_consumers: int = 3) -> None:
        """Start the event bus with *num_consumers* concurrent processing tasks."""
        if self._running:
            return

        self._running = True
        self._started_at = time.monotonic()

        for i in range(num_consumers):
            task = asyncio.create_task(
                self._consumer_loop(), name=f"event-bus-consumer-{i}"
            )
            self._consumers.append(task)

        log.info("event_bus_started", consumers=num_consumers, queue_size=self._queue_size)

    async def stop(self) -> None:
        """Stop all consumers and drain remaining events."""
        if not self._running:
            return

        self._running = False

        # Cancel all consumer tasks
        for task in self._consumers:
            task.cancel()

        results = await asyncio.gather(*self._consumers, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                log.error("event_bus_consumer_error", error=str(r))

        self._consumers.clear()
        log.info(
            "event_bus_stopped",
            events_processed=self._events_processed,
            events_dropped=self._events_dropped,
        )

    # ── Metrics ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return event processing metrics."""
        elapsed = time.monotonic() - self._started_at if self._started_at else 0.0
        eps = self._events_processed / elapsed if elapsed > 0 else 0.0
        avg_latency = (
            (self._latency_sum / self._latency_count) * 1000.0
            if self._latency_count > 0
            else 0.0
        )

        return {
            "events_processed": self._events_processed,
            "events_dropped": self._events_dropped,
            "events_per_second": round(eps, 2),
            "latency_ms": round(avg_latency, 3),
            "queue_depth": self._queue.qsize(),
            "subscriber_count": sum(len(v) for v in self._subscriptions.values()),
            "running": self._running,
        }

    # ── Internal ──────────────────────────────────────────────────────────

    async def _consumer_loop(self) -> None:
        """Continuously dequeue events and dispatch to matching handlers."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            dispatch_start = time.monotonic()

            subs = self._subscriptions.get(event.event_type, [])
            tasks = []
            for sub in subs:
                if sub.symbol_filter and sub.symbol_filter != event.symbol:
                    continue
                tasks.append(self._safe_dispatch(sub.handler, event))

            if tasks:
                await asyncio.gather(*tasks)

            latency = time.monotonic() - dispatch_start
            self._latency_sum += latency
            self._latency_count += 1
            self._events_processed += 1

    @staticmethod
    async def _safe_dispatch(handler: EventHandler, event: Event) -> None:
        """Invoke a handler, catching and logging any exceptions."""
        try:
            await handler(event)
        except Exception:
            log.exception(
                "event_handler_error",
                handler=handler.__qualname__,
                event_type=event.event_type.value,
                symbol=event.symbol,
            )
