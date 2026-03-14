"""Data Source Validator — ensures only real data enters the system.

Validates that market data, news, and sentiment feeds are connected and
originate from verified sources before any trading logic activates.

If no valid data source is connected the system must:
1. Pause all AI agents.
2. Disable signal generation.
3. Disable paper trading.
4. Show warnings in the dashboard.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from hedgefund.streaming.event_bus import Event, EventBus, EventType

log = structlog.get_logger(__name__)

_FORBIDDEN_SOURCES = frozenset({
    "synthetic", "mock", "random", "generated", "simulated",
    "fake", "sample", "test", "dummy", "placeholder",
})


class DataSourceStatus(str, enum.Enum):
    CONNECTED = "connected"
    NOT_CONNECTED = "not_connected"
    NOT_CONFIGURED = "not_configured"
    ERROR = "error"


@dataclass(slots=True)
class DataSourceReport:
    """Snapshot of all data source statuses."""

    market_feed: DataSourceStatus
    market_feed_source: str
    broker_connection: DataSourceStatus
    broker_name: str
    news_api: DataSourceStatus
    news_api_source: str
    x_sentiment: DataSourceStatus
    x_api_source: str
    last_market_data_timestamp: datetime | None
    last_news_timestamp: datetime | None
    last_sentiment_timestamp: datetime | None
    is_any_source_connected: bool
    is_trading_ready: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_feed": {
                "status": self.market_feed.value,
                "source": self.market_feed_source,
                "last_timestamp": (
                    self.last_market_data_timestamp.isoformat()
                    if self.last_market_data_timestamp
                    else None
                ),
            },
            "broker": {
                "status": self.broker_connection.value,
                "name": self.broker_name,
            },
            "news_api": {
                "status": self.news_api.value,
                "source": self.news_api_source,
            },
            "x_sentiment": {
                "status": self.x_sentiment.value,
                "source": self.x_api_source,
            },
            "is_trading_ready": self.is_trading_ready,
            "is_any_source_connected": self.is_any_source_connected,
            "warnings": self.warnings,
        }


class DataSourceValidator:
    """Tracks and validates all external data sources.

    Must be subscribed to the :class:`EventBus` so it can automatically
    detect when real data starts flowing.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._market_feed_status = DataSourceStatus.NOT_CONNECTED
        self._broker_status = DataSourceStatus.NOT_CONNECTED
        self._news_status = DataSourceStatus.NOT_CONFIGURED
        self._x_status = DataSourceStatus.NOT_CONFIGURED
        self._last_market_ts: datetime | None = None
        self._last_news_ts: datetime | None = None
        self._last_sentiment_ts: datetime | None = None
        self._market_feed_source: str = ""
        self._broker_name: str = ""
        self._news_source: str = ""
        self._x_source: str = ""

    # ── EventBus integration ──────────────────────────────────────────

    def subscribe(self) -> None:
        """Subscribe to events to automatically track data source status."""
        self._event_bus.subscribe(EventType.TICK, self._on_tick)
        self._event_bus.subscribe(EventType.NEWS, self._on_news)
        self._event_bus.subscribe(EventType.SENTIMENT, self._on_sentiment)
        log.info("data_source_validator.subscribed")

    async def _on_tick(self, event: Event) -> None:
        if event.source and event.source.lower() not in _FORBIDDEN_SOURCES:
            self._market_feed_status = DataSourceStatus.CONNECTED
            self._market_feed_source = event.source
            self._last_market_ts = event.timestamp
            log.info(
                "data_source_validator.market_data_received",
                source=event.source,
                symbol=event.symbol,
            )

    async def _on_news(self, event: Event) -> None:
        if event.source and event.source.lower() not in _FORBIDDEN_SOURCES:
            self._news_status = DataSourceStatus.CONNECTED
            self._news_source = event.source
            self._last_news_ts = event.timestamp
            log.info(
                "data_source_validator.news_received",
                source=event.source,
            )

    async def _on_sentiment(self, event: Event) -> None:
        source = event.data.get("source", event.source)
        if source and source.lower() not in _FORBIDDEN_SOURCES:
            if source in ("twitter", "x", "social"):
                self._x_status = DataSourceStatus.CONNECTED
                self._x_source = source
            self._last_sentiment_ts = event.timestamp
            log.info(
                "data_source_validator.sentiment_received",
                source=source,
            )

    # ── Manual status updates ─────────────────────────────────────────

    def update_broker_status(
        self, status: DataSourceStatus, broker_name: str,
    ) -> None:
        self._broker_status = status
        self._broker_name = broker_name
        log.info(
            "data_source_validator.broker_status",
            status=status.value,
            broker=broker_name,
        )

    def update_market_feed_status(
        self, status: DataSourceStatus, source: str,
    ) -> None:
        self._market_feed_status = status
        self._market_feed_source = source

    def update_news_status(
        self, status: DataSourceStatus, source: str,
    ) -> None:
        self._news_status = status
        self._news_source = source

    def update_x_status(
        self, status: DataSourceStatus, source: str,
    ) -> None:
        self._x_status = status
        self._x_source = source

    # ── Queries ───────────────────────────────────────────────────────

    def is_trading_allowed(self) -> bool:
        """True only if market feed AND broker are connected."""
        return (
            self._market_feed_status == DataSourceStatus.CONNECTED
            and self._broker_status == DataSourceStatus.CONNECTED
        )

    def get_report(self) -> DataSourceReport:
        """Build a complete status report."""
        is_any = any(
            s == DataSourceStatus.CONNECTED
            for s in [
                self._market_feed_status,
                self._broker_status,
                self._news_status,
                self._x_status,
            ]
        )

        warnings: list[str] = []
        if self._market_feed_status != DataSourceStatus.CONNECTED:
            warnings.append("DATA SOURCE NOT CONNECTED — Market feed required")
        if self._broker_status != DataSourceStatus.CONNECTED:
            warnings.append("DATA SOURCE NOT CONNECTED — Broker not connected")
        if self._news_status == DataSourceStatus.NOT_CONFIGURED:
            warnings.append("News API not configured")
        if self._x_status == DataSourceStatus.NOT_CONFIGURED:
            warnings.append("X (Twitter) API not configured")

        return DataSourceReport(
            market_feed=self._market_feed_status,
            market_feed_source=self._market_feed_source,
            broker_connection=self._broker_status,
            broker_name=self._broker_name,
            news_api=self._news_status,
            news_api_source=self._news_source,
            x_sentiment=self._x_status,
            x_api_source=self._x_source,
            last_market_data_timestamp=self._last_market_ts,
            last_news_timestamp=self._last_news_ts,
            last_sentiment_timestamp=self._last_sentiment_ts,
            is_any_source_connected=is_any,
            is_trading_ready=self.is_trading_allowed(),
            warnings=warnings,
        )

    def get_missing_credentials_message(self) -> str:
        """User-facing message about what needs to be configured."""
        lines = []
        if self._market_feed_status != DataSourceStatus.CONNECTED:
            lines.append(
                "Market data source not configured.\n"
                "Please provide one of the following:\n"
                "  - Zerodha API key\n"
                "  - Binance API key\n"
                "  - Market data provider API"
            )
        if self._broker_status != DataSourceStatus.CONNECTED:
            lines.append(
                "Broker not connected.\n"
                "Please configure one of:\n"
                "  - Zerodha Kite\n"
                "  - Binance\n"
                "  - Groww\n"
                "  - INDMoney"
            )
        if self._news_status == DataSourceStatus.NOT_CONFIGURED:
            lines.append("News API key not configured.")
        if self._x_status == DataSourceStatus.NOT_CONFIGURED:
            lines.append("X (Twitter) API credentials not configured.")

        if not lines:
            return "All data sources connected."
        return "\n\n".join(lines)

    def validate_data_origin(self, data: dict[str, Any]) -> bool:
        """Verify that *data* includes valid source metadata.

        Returns ``False`` for synthetic/mock/random sources.
        """
        source = data.get("source", "")
        if not source:
            return False
        if source.lower() in _FORBIDDEN_SOURCES:
            return False
        return True
