"""Database write guard to prevent synthetic/mock data from being persisted.

All writes to MongoDB should be routed through ``GuardedMongoWriter`` (or
validated manually via ``WriteGuard``) to ensure only real market data,
real trades, and real signals are stored.
"""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger(__name__)


class ForbiddenWriteError(Exception):
    """Raised when attempting to write non-real data to the database."""


class WriteGuard:
    """Validates all database writes to prevent synthetic data storage.

    Every write must pass through the appropriate ``validate_*`` method
    before being persisted.  The guard rejects records whose ``source``
    (or equivalent field) matches a known synthetic/mock origin.
    """

    FORBIDDEN_SOURCES: set[str] = {
        "synthetic",
        "mock",
        "random",
        "generated",
        "simulated",
        "fake",
        "sample",
    }

    # Only these top-level write types are accepted.
    ALLOWED_WRITE_TYPES: set[str] = {
        "market_data",
        "trade",
        "signal",
        "sentiment",
        "backtest",
    }

    # ── validators ───────────────────────────────────────────────────────

    @classmethod
    def _check_forbidden_source(cls, source: str, context: str) -> None:
        """Raise if *source* is a forbidden synthetic origin."""
        if source.lower().strip() in cls.FORBIDDEN_SOURCES:
            raise ForbiddenWriteError(
                f"Forbidden data source for {context}: {source!r}"
            )

    @classmethod
    def validate_market_data(cls, data: dict[str, Any]) -> None:
        """Validate market data before writing to DB.

        Required fields: ``source``, ``timestamp``, ``symbol``.
        """
        if "source" not in data:
            raise ForbiddenWriteError("Market data missing 'source' field.")
        cls._check_forbidden_source(data["source"], "market_data")

        if "timestamp" not in data:
            raise ForbiddenWriteError("Market data missing 'timestamp' field.")
        if "symbol" not in data:
            raise ForbiddenWriteError("Market data missing 'symbol' field.")

    @classmethod
    def validate_trade(cls, data: dict[str, Any]) -> None:
        """Validate a trade record before writing.

        Required fields: ``broker_id``, ``source``.
        """
        if "broker_id" not in data:
            raise ForbiddenWriteError("Trade record missing 'broker_id' field.")
        if "source" not in data:
            raise ForbiddenWriteError("Trade record missing 'source' field.")
        cls._check_forbidden_source(data["source"], "trade")

    @classmethod
    def validate_signal(cls, data: dict[str, Any]) -> None:
        """Validate a signal before writing.

        Required fields: ``data_source``.
        """
        if "data_source" not in data:
            raise ForbiddenWriteError("Signal missing 'data_source' field.")
        cls._check_forbidden_source(data["data_source"], "signal")

    @classmethod
    def validate_sentiment(cls, data: dict[str, Any]) -> None:
        """Validate sentiment data before writing.

        Required fields: ``source`` (e.g. ``"twitter"``, ``"news_api"``).
        """
        if "source" not in data:
            raise ForbiddenWriteError("Sentiment data missing 'source' field.")
        cls._check_forbidden_source(data["source"], "sentiment")

    @classmethod
    def validate_backtest(cls, data: dict[str, Any]) -> None:
        """Validate backtest results before writing.

        Backtests are allowed when backed by historical data, but not when
        the underlying data source is purely random/synthetic.

        Required fields: ``data_source``.
        """
        if "data_source" not in data:
            raise ForbiddenWriteError(
                "Backtest results missing 'data_source' field."
            )
        cls._check_forbidden_source(data["data_source"], "backtest")


class GuardedMongoWriter:
    """Wraps MongoDB writes with :class:`WriteGuard` validation.

    Parameters
    ----------
    db:
        A :class:`hedgefund.auth.database.MongoDB` instance (or any object
        exposing the required collection properties).
    """

    def __init__(self, db: Any) -> None:
        self._db = db

    # ── guarded inserts ──────────────────────────────────────────────────

    async def insert_market_data(self, data: dict[str, Any]) -> None:
        """Validate and insert market data."""
        WriteGuard.validate_market_data(data)
        collection = getattr(self._db, "market_data", None)
        if collection is None:
            # Fall back to a generic collection name via the motor db
            db_ref = getattr(self._db, "_db", None)
            if db_ref is not None:
                collection = db_ref["market_data"]
            else:
                raise RuntimeError("MongoDB instance has no market_data collection.")
        await collection.insert_one(data)
        log.debug(
            "write_guard.market_data_inserted",
            symbol=data.get("symbol"),
            source=data.get("source"),
        )

    async def insert_trade(self, data: dict[str, Any]) -> None:
        """Validate and insert a trade record."""
        WriteGuard.validate_trade(data)
        await self._db.trades.insert_one(data)
        log.debug(
            "write_guard.trade_inserted",
            broker_id=data.get("broker_id"),
            source=data.get("source"),
        )

    async def insert_signal(self, data: dict[str, Any]) -> None:
        """Validate and insert a signal."""
        WriteGuard.validate_signal(data)
        await self._db.signals.insert_one(data)
        log.debug(
            "write_guard.signal_inserted",
            data_source=data.get("data_source"),
        )

    async def insert_sentiment(self, data: dict[str, Any]) -> None:
        """Validate and insert sentiment data."""
        WriteGuard.validate_sentiment(data)
        await self._db.sentiment_data.insert_one(data)
        log.debug(
            "write_guard.sentiment_inserted",
            source=data.get("source"),
        )

    async def insert_backtest(self, data: dict[str, Any]) -> None:
        """Validate and insert backtest results."""
        WriteGuard.validate_backtest(data)
        await self._db.backtests.insert_one(data)
        log.debug(
            "write_guard.backtest_inserted",
            data_source=data.get("data_source"),
        )
