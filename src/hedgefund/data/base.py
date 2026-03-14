"""Abstract interfaces for all data providers.

Every concrete provider inherits from one of these ABCs, ensuring a uniform
contract across Yahoo, Polygon, or any future data source.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from hedgefund.types import OHLCV, OptionQuote, SentimentResult


class DataProvider(abc.ABC):
    """Async provider for OHLCV market data."""

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Tear down connection gracefully."""

    @abc.abstractmethod
    async def get_snapshot(
        self,
        symbol: str,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Return the latest OHLCV bars as a DataFrame.

        Columns: ``open, high, low, close, volume`` with a
        :class:`~pandas.DatetimeIndex`.
        """

    @abc.abstractmethod
    async def get_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for the given range."""

    @abc.abstractmethod
    async def stream(
        self,
        symbols: List[str],
        timeframe: str = "1m",
    ) -> AsyncIterator[OHLCV]:
        """Yield real-time OHLCV bars as they arrive."""
        ...  # pragma: no cover – abstract async generator

    async def __aenter__(self) -> "DataProvider":
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.disconnect()


class OptionsChainProvider(abc.ABC):
    """Async provider for full options chains with greeks."""

    @abc.abstractmethod
    async def get_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None,
    ) -> List[OptionQuote]:
        """Return options quotes for *symbol*, optionally filtered by expiration."""

    @abc.abstractmethod
    async def get_expirations(self, symbol: str) -> List[str]:
        """Return available expiration dates (ISO-8601 strings)."""


class NewsProvider(abc.ABC):
    """Async provider for financial news articles."""

    @abc.abstractmethod
    async def fetch_news(
        self,
        symbols: List[str],
        max_items: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return recent news items relevant to *symbols*.

        Each dict must contain at least ``title``, ``url``, ``published_at``,
        and ``source``.
        """

    @abc.abstractmethod
    async def stream_news(
        self,
        symbols: List[str],
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield news items as they become available."""
        ...  # pragma: no cover


class SocialFeedProvider(abc.ABC):
    """Async provider for social-media sentiment data."""

    @abc.abstractmethod
    async def fetch_posts(
        self,
        symbols: List[str],
        max_items: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return recent social posts mentioning *symbols*."""

    @abc.abstractmethod
    async def get_sentiment(
        self,
        symbol: str,
    ) -> SentimentResult:
        """Compute aggregated sentiment for *symbol*."""
