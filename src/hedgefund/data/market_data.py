"""Yahoo Finance data provider.

Wraps :pypi:`yfinance` behind the :class:`~hedgefund.data.base.DataProvider`
interface.  All blocking I/O is offloaded to a thread-pool so the event loop
stays responsive.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from hedgefund.data.base import DataProvider
from hedgefund.exceptions import DataConnectionError, DataValidationError
from hedgefund.logger import get_logger
from hedgefund.types import OHLCV

log = get_logger(__name__)

# Map our timeframe strings to yfinance ``interval`` values.
_INTERVAL_MAP: Dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "1d": "1d",
}

# yfinance enforces maximum periods per interval.  These are conservative
# defaults to avoid server-side 422 errors.
_MAX_PERIOD_MAP: Dict[str, str] = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "1h": "730d",
    "1d": "max",
}


class YahooFinanceProvider(DataProvider):
    """Concrete :class:`DataProvider` backed by Yahoo Finance.

    Parameters
    ----------
    proxy:
        Optional HTTP proxy URL forwarded to yfinance.
    """

    def __init__(self, proxy: Optional[str] = None) -> None:
        self._proxy = proxy
        self._connected = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def connect(self) -> None:
        self._connected = True
        log.info("yahoo_connected")

    async def disconnect(self) -> None:
        self._connected = False
        log.info("yahoo_disconnected")

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalise column names and validate the result is non-empty."""
        if df is None or df.empty:
            raise DataValidationError(f"No data returned for {symbol}")
        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise DataValidationError(f"Missing columns for {symbol}: {missing}")
        return df

    def _resolve_interval(self, timeframe: str) -> str:
        interval = _INTERVAL_MAP.get(timeframe)
        if interval is None:
            raise DataValidationError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Choose from {list(_INTERVAL_MAP)}"
            )
        return interval

    # ── public API ────────────────────────────────────────────────────────

    async def get_snapshot(
        self,
        symbol: str,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Return the most recent OHLCV bars for *symbol*."""
        interval = self._resolve_interval(timeframe)
        period = _MAX_PERIOD_MAP.get(timeframe, "5d")
        # Cap snapshot to a small window for speed.
        if period == "max":
            period = "5d"

        log.debug("yahoo_snapshot", symbol=symbol, interval=interval, period=period)

        try:
            ticker = yf.Ticker(symbol)
            df: pd.DataFrame = await asyncio.to_thread(
                ticker.history,
                period=period,
                interval=interval,
                proxy=self._proxy,
            )
        except Exception as exc:
            raise DataConnectionError(f"Yahoo snapshot failed for {symbol}: {exc}") from exc

        return self._validate_dataframe(df, symbol)

    async def get_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars between *start* and *end*."""
        interval = self._resolve_interval(timeframe)

        log.debug(
            "yahoo_historical",
            symbol=symbol,
            start=str(start.date()),
            end=str(end.date()),
            interval=interval,
        )

        try:
            ticker = yf.Ticker(symbol)
            df: pd.DataFrame = await asyncio.to_thread(
                ticker.history,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
                proxy=self._proxy,
            )
        except Exception as exc:
            raise DataConnectionError(f"Yahoo historical failed for {symbol}: {exc}") from exc

        return self._validate_dataframe(df, symbol)

    async def stream(
        self,
        symbols: List[str],
        timeframe: str = "1m",
    ) -> AsyncIterator[OHLCV]:
        """Poll Yahoo Finance at regular intervals to simulate streaming.

        Yahoo Finance does not offer a true WebSocket feed, so we poll at an
        interval matching *timeframe* and yield new bars as they appear.
        """
        interval = self._resolve_interval(timeframe)

        # Determine sleep duration (seconds) between polls.
        _poll_seconds: Dict[str, int] = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "1d": 86400,
        }
        sleep_sec = _poll_seconds.get(timeframe, 60)
        last_timestamps: Dict[str, datetime | None] = {s: None for s in symbols}

        log.info("yahoo_stream_start", symbols=symbols, interval=interval)

        while True:
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    df: pd.DataFrame = await asyncio.to_thread(
                        ticker.history,
                        period="1d",
                        interval=interval,
                        proxy=self._proxy,
                    )
                    if df is None or df.empty:
                        continue

                    df.columns = [c.lower() for c in df.columns]
                    last_ts = last_timestamps[symbol]

                    for ts, row in df.iterrows():
                        ts_dt = ts.to_pydatetime()  # type: ignore[union-attr]
                        if last_ts is not None and ts_dt <= last_ts:
                            continue
                        yield OHLCV(
                            timestamp=ts_dt,
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                            volume=int(row["volume"]),
                        )
                        last_timestamps[symbol] = ts_dt

                except Exception:
                    log.exception("yahoo_stream_error", symbol=symbol)

            await asyncio.sleep(sleep_sec)
