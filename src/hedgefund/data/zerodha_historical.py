"""Historical OHLCV candle data and instrument search via Zerodha Kite Connect.

Provides async access to Kite's historical data API for chart candles and
the instrument master list for symbol resolution.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any

import httpx

from hedgefund.logger import get_logger

log = get_logger(__name__)

_KITE_BASE_URL = "https://api.kite.trade"
_SOURCE = "Zerodha Kite API"

_VALID_INTERVALS = frozenset(
    {"minute", "3minute", "5minute", "15minute", "30minute", "60minute", "day"}
)


class ZerodhaHistorical:
    """Fetches historical OHLCV candles from Kite Connect.

    Usage::

        hist = ZerodhaHistorical(api_key="xxx", access_token="yyy")
        candles = await hist.get_candles(
            instrument_token="256265",  # NIFTY 50
            interval="15minute",
            from_date="2026-03-01",
            to_date="2026-03-14",
        )
    """

    def __init__(self, api_key: str, access_token: str) -> None:
        self._api_key = api_key
        self._access_token = access_token
        self._client: httpx.AsyncClient | None = None
        self._instruments_cache: list[dict[str, Any]] | None = None

    def _headers(self) -> dict[str, str]:
        return {
            "X-Kite-Version": "3",
            "Authorization": f"token {self._api_key}:{self._access_token}",
        }

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- Historical candles -------------------------------------------------

    async def get_candles(
        self,
        instrument_token: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> list[dict[str, Any]]:
        """Fetch historical OHLCV candles.

        Args:
            instrument_token: Numeric Kite instrument token (e.g. ``"256265"``).
            interval: One of ``minute``, ``3minute``, ``5minute``, ``15minute``,
                ``30minute``, ``60minute``, ``day``.
            from_date: Start date in ``YYYY-MM-DD`` or ``YYYY-MM-DD+HH:MM:SS``
                format.
            to_date: End date in the same format.

        Returns:
            List of dicts with keys ``timestamp``, ``open``, ``high``, ``low``,
            ``close``, ``volume``, ``source``.
        """
        if interval not in _VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. "
                f"Must be one of: {', '.join(sorted(_VALID_INTERVALS))}"
            )

        client = await self._ensure_client()
        path = f"/instruments/historical/{instrument_token}/{interval}"
        url = f"{_KITE_BASE_URL}{path}"
        params = {"from": from_date, "to": to_date}

        resp = await client.get(url, headers=self._headers(), params=params)

        if resp.status_code == 403:
            log.error(
                "kite_historical.auth_error",
                status=resp.status_code,
                source=_SOURCE,
            )
            raise PermissionError(
                "Kite API token expired or invalid. Re-authenticate via Zerodha OAuth."
            )

        if resp.status_code == 429:
            log.warning("kite_historical.rate_limited", source=_SOURCE)
            raise RuntimeError("Kite API rate limit exceeded")

        if resp.status_code != 200:
            log.error(
                "kite_historical.api_error",
                status=resp.status_code,
                body=resp.text[:200],
                source=_SOURCE,
            )
            raise RuntimeError(
                f"Kite historical API error {resp.status_code}: {resp.text[:200]}"
            )

        body = resp.json()
        if body.get("status") == "error":
            raise RuntimeError(
                f"Kite API error ({body.get('error_type')}): {body.get('message')}"
            )

        raw_candles: list[list[Any]] = body.get("data", {}).get("candles", [])

        candles: list[dict[str, Any]] = []
        for c in raw_candles:
            # Kite returns: [timestamp, open, high, low, close, volume]
            if len(c) < 6:
                continue
            candles.append(
                {
                    "timestamp": c[0],
                    "open": c[1],
                    "high": c[2],
                    "low": c[3],
                    "close": c[4],
                    "volume": c[5],
                    "source": _SOURCE,
                }
            )

        log.info(
            "kite_historical.candles_fetched",
            message="Market data received from Zerodha Kite API",
            instrument_token=instrument_token,
            interval=interval,
            candles=len(candles),
            source=_SOURCE,
        )
        return candles

    # -- Instrument list ----------------------------------------------------

    async def get_instruments(self, exchange: str = "") -> list[dict[str, Any]]:
        """Fetch the full instrument list from Kite.

        Args:
            exchange: Optional exchange filter (``NSE``, ``NFO``, ``BSE``,
                ``BFO``, ``MCX``, ``CDS``).  Empty string returns all.

        Returns:
            List of instrument dicts with keys like ``instrument_token``,
            ``exchange_token``, ``tradingsymbol``, ``name``, ``exchange``,
            ``lot_size``, ``instrument_type``, ``expiry``, ``strike``, etc.
        """
        client = await self._ensure_client()
        path = f"/instruments/{exchange}" if exchange else "/instruments"
        url = f"{_KITE_BASE_URL}{path}"

        resp = await client.get(url, headers=self._headers())

        if resp.status_code == 403:
            raise PermissionError("Kite API token expired or invalid")
        if resp.status_code != 200:
            raise RuntimeError(
                f"Kite instruments API error {resp.status_code}: {resp.text[:200]}"
            )

        # Response is CSV, not JSON
        text = resp.text
        reader = csv.DictReader(io.StringIO(text))
        instruments: list[dict[str, Any]] = []
        for row in reader:
            instruments.append({**row, "source": _SOURCE})

        self._instruments_cache = instruments

        log.info(
            "kite_historical.instruments_fetched",
            message="Market data received from Zerodha Kite API",
            exchange=exchange or "ALL",
            count=len(instruments),
            source=_SOURCE,
        )
        return instruments

    async def search_instrument(
        self,
        query: str,
        exchange: str = "NSE",
    ) -> list[dict[str, Any]]:
        """Search instruments by name or trading symbol.

        Fetches the instrument list for the given exchange (cached after first
        call) and filters locally by substring match against ``tradingsymbol``
        and ``name`` fields.

        Args:
            query: Search string (case-insensitive).
            exchange: Exchange to search within (default ``NSE``).

        Returns:
            List of matching instrument dicts, max 50 results.
        """
        # Use cache if available and exchange matches
        if self._instruments_cache and any(
            inst.get("exchange") == exchange for inst in self._instruments_cache
        ):
            instruments = self._instruments_cache
        else:
            instruments = await self.get_instruments(exchange)

        query_lower = query.lower()
        matches: list[dict[str, Any]] = []
        for inst in instruments:
            if inst.get("exchange") != exchange:
                continue
            symbol = (inst.get("tradingsymbol") or "").lower()
            name = (inst.get("name") or "").lower()
            if query_lower in symbol or query_lower in name:
                matches.append(inst)
                if len(matches) >= 50:
                    break

        log.info(
            "kite_historical.instrument_search",
            message="Market data received from Zerodha Kite API",
            query=query,
            exchange=exchange,
            results=len(matches),
            source=_SOURCE,
        )
        return matches
