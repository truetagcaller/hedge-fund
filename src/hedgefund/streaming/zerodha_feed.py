"""Live market data feed from Zerodha Kite Connect Quote API.

Polls the Kite REST Quote API at configurable intervals and publishes
TICK events to the central EventBus.  This avoids the ``kiteconnect``
Python package dependency by using plain HTTP requests.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

from hedgefund.logger import get_logger
from hedgefund.streaming.event_bus import Event, EventBus, EventType

log = get_logger(__name__)

_KITE_BASE_URL = "https://api.kite.trade"

_DEFAULT_INSTRUMENTS: list[str] = [
    "NSE:NIFTY 50",
    "NSE:NIFTY BANK",
    "BSE:SENSEX",
    "NSE:RELIANCE",
    "NSE:TCS",
    "NSE:INFY",
    "NSE:HDFCBANK",
    "NSE:ICICIBANK",
]

_SOURCE = "Zerodha Kite API"


class ZerodhaMarketFeed:
    """Fetches live quotes and publishes TICK events to EventBus.

    Since Kite WebSocket (kiteticker) requires the kiteconnect Python package,
    we use REST polling of the Quote API as a reliable alternative.
    Polls every 2 seconds for subscribed instruments.
    """

    def __init__(
        self,
        api_key: str,
        access_token: str,
        event_bus: EventBus,
        symbols: list[str] | None = None,
        poll_interval: float = 2.0,
    ) -> None:
        self._api_key = api_key
        self._access_token = access_token
        self._event_bus = event_bus
        self._instruments: list[str] = list(symbols or _DEFAULT_INSTRUMENTS)
        self._poll_interval = poll_interval
        self._running = False
        self._poll_task: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None

    # -- HTTP helpers -------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "X-Kite-Version": "3",
            "Authorization": f"token {self._api_key}:{self._access_token}",
        }

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def _kite_get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue a GET request to the Kite API and return the JSON body."""
        client = await self._ensure_client()
        url = f"{_KITE_BASE_URL}{path}"
        resp = await client.get(url, headers=self._headers(), params=params)

        if resp.status_code == 403:
            log.error(
                "kite_api_auth_error",
                status=resp.status_code,
                detail="Token expired or invalid. Re-authenticate via Zerodha OAuth.",
                source=_SOURCE,
            )
            raise PermissionError("Kite API token expired or invalid")

        if resp.status_code == 429:
            log.warning("kite_api_rate_limited", source=_SOURCE)
            raise RuntimeError("Kite API rate limit exceeded")

        if resp.status_code != 200:
            body = resp.text
            log.error(
                "kite_api_error",
                status=resp.status_code,
                body=body[:200],
                source=_SOURCE,
            )
            raise RuntimeError(f"Kite API error {resp.status_code}: {body[:200]}")

        data: dict[str, Any] = resp.json()
        if data.get("status") == "error":
            raise RuntimeError(
                f"Kite API error ({data.get('error_type')}): {data.get('message')}"
            )

        return data.get("data", data)

    # -- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Start the polling loop."""
        if self._running:
            return
        self._running = True
        self._poll_task = asyncio.create_task(
            self._poll_loop(), name="zerodha-market-feed"
        )
        log.info(
            "zerodha_feed.started",
            instruments=len(self._instruments),
            poll_interval=self._poll_interval,
            source=_SOURCE,
        )

    async def stop(self) -> None:
        """Stop polling and close the HTTP client."""
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        log.info("zerodha_feed.stopped", source=_SOURCE)

    # -- Subscription -------------------------------------------------------

    async def subscribe(self, instruments: list[str]) -> None:
        """Add instruments to the subscription list.

        Format: ``'NSE:NIFTY 50'``, ``'NSE:RELIANCE'``
        """
        added: list[str] = []
        for inst in instruments:
            if inst not in self._instruments:
                self._instruments.append(inst)
                added.append(inst)
        if added:
            log.info("zerodha_feed.subscribed", instruments=added, source=_SOURCE)

    async def unsubscribe(self, instruments: list[str]) -> None:
        """Remove instruments from the subscription list."""
        for inst in instruments:
            if inst in self._instruments:
                self._instruments.remove(inst)
        log.info("zerodha_feed.unsubscribed", instruments=instruments, source=_SOURCE)

    # -- Polling loop -------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Continuously poll quotes and publish TICK events."""
        while self._running:
            try:
                await self._poll_quotes()
            except asyncio.CancelledError:
                break
            except PermissionError:
                log.error(
                    "zerodha_feed.auth_failed",
                    detail="Stopping feed due to authentication failure.",
                    source=_SOURCE,
                )
                self._running = False
                break
            except Exception:
                log.warning("zerodha_feed.poll_error", exc_info=True, source=_SOURCE)

            await asyncio.sleep(self._poll_interval)

    async def _poll_quotes(self) -> None:
        """Fetch quotes for all subscribed instruments and publish TICK events."""
        if not self._instruments:
            return

        quotes = await self._fetch_quotes(self._instruments)
        now = datetime.now(timezone.utc)

        for key, q in quotes.items():
            symbol = key  # e.g. "NSE:RELIANCE"
            ohlc = q.get("ohlc", {})
            tick_data: dict[str, Any] = {
                "last_price": q.get("last_price"),
                "volume": q.get("volume"),
                "open": ohlc.get("open"),
                "high": ohlc.get("high"),
                "low": ohlc.get("low"),
                "close": ohlc.get("close"),
                "change": q.get("net_change"),
                "change_percent": q.get("change", 0),  # Kite "change" is %
                "buy_quantity": q.get("buy_quantity"),
                "sell_quantity": q.get("sell_quantity"),
                "average_price": q.get("average_price"),
                "last_quantity": q.get("last_quantity"),
                "last_trade_time": q.get("last_trade_time"),
                "oi": q.get("oi"),
                "oi_day_high": q.get("oi_day_high"),
                "oi_day_low": q.get("oi_day_low"),
                "depth": q.get("depth"),
                "instrument_token": q.get("instrument_token"),
                "source": _SOURCE,
                "is_live": True,
            }

            event = Event(
                event_type=EventType.TICK,
                timestamp=now,
                symbol=symbol,
                data=tick_data,
                source=_SOURCE,
            )
            await self._event_bus.publish(event)

        log.info(
            "zerodha_feed.market_data_received",
            message="Market data received from Zerodha Kite API",
            instruments=len(quotes),
            source=_SOURCE,
        )

    # -- Public API methods -------------------------------------------------

    async def _fetch_quotes(self, instruments: list[str]) -> dict[str, Any]:
        """Call Kite Quote API for a list of instruments."""
        # Build query string: ?i=NSE:NIFTY+50&i=NSE:RELIANCE
        params: list[tuple[str, str]] = [("i", inst) for inst in instruments]
        client = await self._ensure_client()
        url = f"{_KITE_BASE_URL}/quote"
        resp = await client.get(url, headers=self._headers(), params=params)

        if resp.status_code == 403:
            raise PermissionError("Kite API token expired or invalid")
        if resp.status_code == 429:
            raise RuntimeError("Kite API rate limit exceeded")
        if resp.status_code != 200:
            raise RuntimeError(f"Kite API error {resp.status_code}: {resp.text[:200]}")

        body = resp.json()
        if body.get("status") == "error":
            raise RuntimeError(
                f"Kite API error ({body.get('error_type')}): {body.get('message')}"
            )

        return body.get("data", {})

    async def get_ltp(self, instruments: list[str]) -> dict[str, Any]:
        """Get last traded prices for instruments.

        Returns dict keyed by instrument name, e.g.
        ``{"NSE:RELIANCE": {"instrument_token": ..., "last_price": ...}}``
        """
        client = await self._ensure_client()
        params = [("i", inst) for inst in instruments]
        resp = await client.get(
            f"{_KITE_BASE_URL}/quote/ltp",
            headers=self._headers(),
            params=params,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Kite LTP error: {resp.text[:200]}")
        data = resp.json().get("data", {})

        log.info(
            "zerodha_feed.ltp_fetched",
            message="Market data received from Zerodha Kite API",
            instruments=len(instruments),
            source=_SOURCE,
        )
        result: dict[str, Any] = {}
        for key, val in data.items():
            result[key] = {**val, "source": _SOURCE, "is_live": True}
        return result

    async def get_quote(self, instruments: list[str]) -> dict[str, Any]:
        """Get full quotes with depth for instruments."""
        quotes = await self._fetch_quotes(instruments)
        log.info(
            "zerodha_feed.quote_fetched",
            message="Market data received from Zerodha Kite API",
            instruments=len(instruments),
            source=_SOURCE,
        )
        result: dict[str, Any] = {}
        for key, val in quotes.items():
            result[key] = {**val, "source": _SOURCE, "is_live": True}
        return result

    async def get_ohlc(self, instruments: list[str]) -> dict[str, Any]:
        """Get OHLC quotes for instruments."""
        params: list[tuple[str, str]] = [("i", inst) for inst in instruments]
        client = await self._ensure_client()
        resp = await client.get(
            f"{_KITE_BASE_URL}/quote/ohlc",
            headers=self._headers(),
            params=params,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Kite OHLC error: {resp.text[:200]}")

        body = resp.json()
        if body.get("status") == "error":
            raise RuntimeError(
                f"Kite API error ({body.get('error_type')}): {body.get('message')}"
            )

        data = body.get("data", {})
        log.info(
            "zerodha_feed.ohlc_fetched",
            message="Market data received from Zerodha Kite API",
            instruments=len(instruments),
            source=_SOURCE,
        )
        result: dict[str, Any] = {}
        for key, val in data.items():
            result[key] = {**val, "source": _SOURCE, "is_live": True}
        return result
