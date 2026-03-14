"""Unified market data feed manager.

Aggregates WebSocket feeds from multiple brokers, publishes tick and order
book data to the central :class:`EventBus`, and handles reconnection with
exponential backoff.
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from hedgefund.logger import get_logger
from hedgefund.streaming.event_bus import Event, EventBus, EventType

if TYPE_CHECKING:
    from hedgefund.streaming.websocket_feeds import BaseWebSocketFeed

log = get_logger(__name__)


class FeedType(enum.Enum):
    """Types of data feeds that can be subscribed."""

    TICK = "TICK"
    ORDERBOOK = "ORDERBOOK"
    OPTIONS = "OPTIONS"


@dataclass
class _FeedEntry:
    """Tracks a single symbol's active feed."""

    symbol: str
    feed_types: Set[FeedType]
    feed: BaseWebSocketFeed
    task: Optional[asyncio.Task[None]] = None
    connected: bool = False
    last_tick_time: float = 0.0
    reconnect_attempts: int = 0


class MarketFeedManager:
    """Aggregates WebSocket feeds and publishes data to the EventBus.

    Usage::

        manager = MarketFeedManager(event_bus=bus, feed_factory=my_factory)
        await manager.start()
        await manager.subscribe("AAPL", feed_types=[FeedType.TICK, FeedType.ORDERBOOK])
        tick = manager.get_latest_tick("AAPL")
        await manager.unsubscribe("AAPL")
        await manager.stop()
    """

    def __init__(
        self,
        event_bus: EventBus,
        feed_factory: Any = None,
        health_check_interval: float = 30.0,
        max_reconnect_delay: float = 60.0,
    ) -> None:
        self._event_bus = event_bus
        self._feed_factory = feed_factory
        self._feeds: Dict[str, _FeedEntry] = {}
        self._latest_ticks: Dict[str, Dict[str, Any]] = {}
        self._order_books: Dict[str, Dict[str, Any]] = {}
        self._health_check_interval = health_check_interval
        self._max_reconnect_delay = max_reconnect_delay
        self._health_task: Optional[asyncio.Task[None]] = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the feed manager and its health monitoring loop."""
        self._running = True
        self._health_task = asyncio.create_task(
            self._health_check_loop(), name="feed-health-check"
        )
        log.info("market_feed_manager_started")

    async def stop(self) -> None:
        """Stop all feeds and tear down the manager."""
        self._running = False

        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

        # Stop all active feeds
        symbols = list(self._feeds.keys())
        for symbol in symbols:
            await self._stop_feed(symbol)

        log.info("market_feed_manager_stopped", feeds_closed=len(symbols))

    # ── Subscription management ───────────────────────────────────────────

    async def subscribe(
        self,
        symbol: str,
        feed_types: Optional[List[FeedType]] = None,
    ) -> None:
        """Subscribe to market data for *symbol*.

        Args:
            symbol: Ticker symbol to subscribe.
            feed_types: Types of feeds to start. Defaults to ``[TICK]``.
        """
        if feed_types is None:
            feed_types = [FeedType.TICK]

        if symbol in self._feeds:
            # Update feed types for existing subscription
            entry = self._feeds[symbol]
            entry.feed_types.update(feed_types)
            log.info("market_feed_updated", symbol=symbol, feed_types=[f.value for f in entry.feed_types])
            return

        feed = self._create_feed(symbol, feed_types)
        entry = _FeedEntry(
            symbol=symbol,
            feed_types=set(feed_types),
            feed=feed,
        )
        self._feeds[symbol] = entry
        entry.task = asyncio.create_task(
            self._run_feed(entry), name=f"feed-{symbol}"
        )
        log.info("market_feed_subscribed", symbol=symbol, feed_types=[f.value for f in feed_types])

    async def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from market data for *symbol*."""
        await self._stop_feed(symbol)
        self._latest_ticks.pop(symbol, None)
        self._order_books.pop(symbol, None)
        log.info("market_feed_unsubscribed", symbol=symbol)

    # ── Data access ───────────────────────────────────────────────────────

    def get_latest_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the most recent tick data for *symbol*, or ``None``."""
        return self._latest_ticks.get(symbol)

    def get_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the current order book snapshot for *symbol*, or ``None``."""
        return self._order_books.get(symbol)

    def get_subscribed_symbols(self) -> List[str]:
        """Return all currently subscribed symbols."""
        return list(self._feeds.keys())

    # ── Internal ──────────────────────────────────────────────────────────

    def _create_feed(self, symbol: str, feed_types: List[FeedType]) -> BaseWebSocketFeed:
        """Create a feed instance using the factory, or fall back to a stub."""
        if self._feed_factory is not None:
            return self._feed_factory(symbol, feed_types)

        # Import here to avoid circular imports at module level
        from hedgefund.streaming.websocket_feeds import GenericWebSocketFeed

        return GenericWebSocketFeed(
            symbol=symbol,
            event_bus=self._event_bus,
            poll_interval=2.0,
        )

    async def _run_feed(self, entry: _FeedEntry) -> None:
        """Run a feed with automatic reconnection and exponential backoff."""
        while self._running and entry.symbol in self._feeds:
            try:
                async with entry.feed:
                    entry.connected = True
                    entry.reconnect_attempts = 0
                    log.info("feed_connected", symbol=entry.symbol)

                    await entry.feed.run(self._make_callbacks(entry))

            except asyncio.CancelledError:
                break
            except Exception:
                entry.connected = False
                entry.reconnect_attempts += 1
                delay = min(
                    2.0 ** entry.reconnect_attempts,
                    self._max_reconnect_delay,
                )
                log.warning(
                    "feed_disconnected_reconnecting",
                    symbol=entry.symbol,
                    attempt=entry.reconnect_attempts,
                    delay_s=delay,
                )
                await asyncio.sleep(delay)

        entry.connected = False

    def _make_callbacks(self, entry: _FeedEntry) -> Dict[str, Any]:
        """Build the callback dict passed to the feed's run loop."""

        async def on_tick(data: Dict[str, Any]) -> None:
            data["received_at"] = time.time()
            self._latest_ticks[entry.symbol] = data
            entry.last_tick_time = time.time()
            await self._event_bus.publish(
                Event(
                    event_type=EventType.TICK,
                    timestamp=datetime.utcnow(),
                    symbol=entry.symbol,
                    data=data,
                    source="market_feed",
                )
            )

        async def on_orderbook(data: Dict[str, Any]) -> None:
            self._order_books[entry.symbol] = data
            await self._event_bus.publish(
                Event(
                    event_type=EventType.ORDERBOOK,
                    timestamp=datetime.utcnow(),
                    symbol=entry.symbol,
                    data=data,
                    source="market_feed",
                )
            )

        async def on_options(data: Dict[str, Any]) -> None:
            await self._event_bus.publish(
                Event(
                    event_type=EventType.OPTIONS_CHAIN,
                    timestamp=datetime.utcnow(),
                    symbol=entry.symbol,
                    data=data,
                    source="market_feed",
                )
            )

        return {
            "on_tick": on_tick,
            "on_orderbook": on_orderbook,
            "on_options": on_options,
        }

    async def _stop_feed(self, symbol: str) -> None:
        """Cancel and remove a feed for *symbol*."""
        entry = self._feeds.pop(symbol, None)
        if entry is None:
            return

        if entry.task is not None and not entry.task.done():
            entry.task.cancel()
            try:
                await entry.task
            except asyncio.CancelledError:
                pass

    async def _health_check_loop(self) -> None:
        """Periodically check feed health and log stale connections."""
        while self._running:
            try:
                await asyncio.sleep(self._health_check_interval)
                now = time.time()
                for symbol, entry in list(self._feeds.items()):
                    stale_seconds = now - entry.last_tick_time if entry.last_tick_time else 0.0
                    if entry.connected and entry.last_tick_time and stale_seconds > self._health_check_interval * 2:
                        log.warning(
                            "feed_stale",
                            symbol=symbol,
                            seconds_since_tick=round(stale_seconds, 1),
                        )
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("feed_health_check_error")
