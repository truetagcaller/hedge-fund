"""IndMoney broker adapter (limited / read-only).

Implements the :class:`~hedgefund.execution.base.Broker` interface for
IndMoney.

.. warning::

    IndMoney does **not** provide an official public API for trading.  This
    adapter uses token-based authentication against internal endpoints that
    may change without notice.  **Only read-only portfolio operations are
    supported** -- ``submit_order``, ``cancel_order``, and ``stream_fills``
    raise :class:`NotImplementedError`.

Authentication:
    Obtain a session token by inspecting network requests in an authenticated
    browser or mobile session and pass it via :class:`IndMoneyConfig`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

import httpx

from hedgefund.exceptions import (
    BrokerConnectionError,
    ExecutionError,
)
from hedgefund.execution.base import Broker
from hedgefund.logger import get_logger
from hedgefund.types import (
    Greeks,
    OptionContract,
    OptionType,
    Order,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    Position,
    Side,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal API base URL (subject to change)
# ---------------------------------------------------------------------------

_INDMONEY_BASE_URL = "https://api.indmoney.com"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndMoneyConfig:
    """Connection parameters for the IndMoney adapter.

    Attributes:
        token: Bearer / session token extracted from an authenticated
            browser or mobile session.
        session_id: Optional session identifier cookie value for endpoints
            that require it.
    """

    token: str = ""
    session_id: str = ""


# ---------------------------------------------------------------------------
# Broker implementation
# ---------------------------------------------------------------------------


class IndMoneyBroker(Broker):
    """Read-only broker adapter for IndMoney.

    This adapter can fetch portfolio and position data but **cannot** place
    or cancel orders because IndMoney does not expose a public trading API.

    Usage::

        cfg = IndMoneyConfig(token="<token>")
        async with IndMoneyBroker(cfg) as broker:
            snapshot = await broker.get_portfolio()
    """

    def __init__(self, config: IndMoneyConfig) -> None:
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._connected: bool = False

    # -- helpers -----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._config.token}",
            "User-Agent": (
                "Mozilla/5.0 (Linux; Android 13) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Mobile Safari/537.36"
            ),
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._config.session_id:
            headers["Cookie"] = f"session_id={self._config.session_id}"
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Issue an HTTP request to IndMoney's internal API.

        Raises :class:`ExecutionError` on non-2xx responses.
        """
        if self._client is None:
            raise BrokerConnectionError("HTTP client not initialised; call connect().")

        url = f"{_INDMONEY_BASE_URL}{path}"
        response = await self._client.request(
            method,
            url,
            headers=self._headers(),
            params=params,
            json=json_body,
        )

        if response.status_code == 401:
            raise BrokerConnectionError(
                "IndMoney session expired or invalid.  Obtain a fresh token "
                "from an authenticated session and update IndMoneyConfig.token."
            )

        if response.status_code >= 400:
            logger.error(
                "indmoney_api_error",
                path=path,
                status=response.status_code,
                body=response.text[:500],
            )
            raise ExecutionError(
                f"IndMoney API error {response.status_code}: {response.text[:200]}"
            )

        if not response.content:
            return {}

        return response.json()

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        """Validate the session by fetching the user profile."""
        if not self._config.token:
            raise BrokerConnectionError(
                "IndMoneyConfig.token is empty.  Extract a session token from "
                "the IndMoney app or an authenticated browser session."
            )

        self._client = httpx.AsyncClient(timeout=30.0)

        try:
            # Attempt to fetch user profile / dashboard to validate token.
            profile = await self._request("GET", "/user/profile")
            user_name = profile.get("name", profile.get("userName", "unknown"))
            logger.info("indmoney_connected", user=user_name)
            self._connected = True
        except Exception as exc:
            await self.disconnect()
            raise BrokerConnectionError(f"IndMoney connection failed: {exc}") from exc

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        self._connected = False
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("indmoney_disconnected")

    # -- orders (not supported) --------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """Not supported -- IndMoney has no public trading API.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "IndMoney does not expose a public trading API.  Order placement "
            "is not supported.  Use the IndMoney app to trade."
        )

    async def cancel_order(self, order_id: str) -> Order:
        """Not supported -- IndMoney has no public trading API.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "IndMoney does not expose a public trading API.  Order "
            "cancellation is not supported.  Use the IndMoney app to "
            "manage orders."
        )

    # -- positions / portfolio ---------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Fetch the user's investment holdings from IndMoney.

        IndMoney aggregates stocks, mutual funds, US stocks, and fixed
        deposits.  This method attempts to fetch each category and merges
        them into a flat list of :class:`Position` objects.
        """
        positions: list[Position] = []

        # Indian stocks
        positions.extend(await self._fetch_stocks())

        # Mutual funds
        positions.extend(await self._fetch_mutual_funds())

        # US stocks
        positions.extend(await self._fetch_us_stocks())

        return positions

    async def _fetch_stocks(self) -> list[Position]:
        """Fetch Indian equity holdings."""
        positions: list[Position] = []
        try:
            data = await self._request("GET", "/stocks/holdings")
            holdings = data if isinstance(data, list) else data.get("holdings", data.get("data", []))
            for h in (holdings if isinstance(holdings, list) else []):
                symbol = h.get("symbol", h.get("tradingSymbol", "UNKNOWN"))
                qty = int(h.get("quantity", h.get("qty", 0)))
                if qty == 0:
                    continue

                avg_price = float(h.get("avgPrice", h.get("averagePrice", 0)))
                current_price = float(h.get("ltp", h.get("currentPrice", h.get("lastPrice", 0))))
                pnl = (current_price - avg_price) * qty

                positions.append(
                    Position(
                        contract=OptionContract(
                            symbol=symbol,
                            underlying=symbol,
                            option_type=OptionType.CALL,
                            strike=0,
                            expiration=date.today(),
                            multiplier=1,
                        ),
                        quantity=qty,
                        avg_entry=avg_price,
                        current_price=current_price,
                        greeks=Greeks(delta=0, gamma=0, theta=0, vega=0),
                        unrealized_pnl=pnl,
                    )
                )
        except (ExecutionError, BrokerConnectionError):
            logger.warning("indmoney_stocks_fetch_failed")

        return positions

    async def _fetch_mutual_funds(self) -> list[Position]:
        """Fetch mutual fund investments."""
        positions: list[Position] = []
        try:
            data = await self._request("GET", "/mutualfunds/investments")
            investments = data if isinstance(data, list) else data.get("investments", data.get("data", []))
            for inv in (investments if isinstance(investments, list) else []):
                name = inv.get("schemeName", inv.get("fundName", "MF_UNKNOWN"))
                units = float(inv.get("units", 0))
                if units <= 0:
                    continue

                nav = float(inv.get("nav", inv.get("currentNav", 0)))
                avg_nav = float(inv.get("avgNav", inv.get("purchaseNav", 0)))
                invested = float(inv.get("investedAmount", inv.get("investedValue", 0)))
                current_val = float(inv.get("currentValue", inv.get("marketValue", 0)))
                pnl = current_val - invested if invested else 0

                positions.append(
                    Position(
                        contract=OptionContract(
                            symbol=name,
                            underlying=name,
                            option_type=OptionType.CALL,
                            strike=0,
                            expiration=date.today(),
                            multiplier=1,
                        ),
                        quantity=max(1, int(units)),
                        avg_entry=avg_nav,
                        current_price=nav,
                        greeks=Greeks(delta=0, gamma=0, theta=0, vega=0),
                        unrealized_pnl=pnl,
                    )
                )
        except (ExecutionError, BrokerConnectionError):
            logger.warning("indmoney_mf_fetch_failed")

        return positions

    async def _fetch_us_stocks(self) -> list[Position]:
        """Fetch US equity holdings."""
        positions: list[Position] = []
        try:
            data = await self._request("GET", "/us-stocks/holdings")
            holdings = data if isinstance(data, list) else data.get("holdings", data.get("data", []))
            for h in (holdings if isinstance(holdings, list) else []):
                symbol = h.get("ticker", h.get("symbol", "US_UNKNOWN"))
                qty_raw = h.get("quantity", h.get("shares", 0))
                qty = int(float(qty_raw)) if float(qty_raw) >= 1 else 1
                if float(qty_raw) <= 0:
                    continue

                avg_price = float(h.get("avgPrice", h.get("averageCost", 0)))
                current_price = float(h.get("currentPrice", h.get("ltp", 0)))
                pnl = (current_price - avg_price) * float(qty_raw)

                positions.append(
                    Position(
                        contract=OptionContract(
                            symbol=f"US:{symbol}",
                            underlying=symbol,
                            option_type=OptionType.CALL,
                            strike=0,
                            expiration=date.today(),
                            multiplier=1,
                        ),
                        quantity=qty,
                        avg_entry=avg_price,
                        current_price=current_price,
                        greeks=Greeks(delta=0, gamma=0, theta=0, vega=0),
                        unrealized_pnl=pnl,
                    )
                )
        except (ExecutionError, BrokerConnectionError):
            logger.warning("indmoney_us_stocks_fetch_failed")

        return positions

    async def get_portfolio(self) -> PortfolioSnapshot:
        """Build a portfolio snapshot from IndMoney holdings.

        Because IndMoney does not expose a cash/margin endpoint, ``cash``
        defaults to ``0.0`` and ``net_liquidation`` is the sum of current
        market values across all asset categories.
        """
        positions = await self.get_positions()

        total_value = sum(
            p.current_price * p.quantity * p.contract.multiplier for p in positions
        )
        total_pnl = sum(p.unrealized_pnl for p in positions)

        return PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            cash=0.0,
            net_liquidation=total_value,
            positions=positions,
            total_pnl=total_pnl,
        )

    # -- streaming (not supported) -----------------------------------------

    async def stream_fills(self) -> AsyncIterator[Order]:
        """Not supported -- IndMoney has no public streaming API.

        This generator yields nothing and logs a warning.  It will block
        indefinitely (sleeping) to satisfy the interface contract until
        ``disconnect`` is called.
        """
        logger.warning(
            "indmoney_stream_fills_unsupported",
            msg="IndMoney does not provide a streaming API. stream_fills will idle.",
        )
        while self._connected:
            await asyncio.sleep(60.0)
        return  # type: ignore[return-value]
        yield  # pragma: no cover  -- makes this a generator
