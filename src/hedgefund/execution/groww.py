"""Groww broker adapter (limited / read-only).

Implements the :class:`~hedgefund.execution.base.Broker` interface for Groww.

.. warning::

    Groww does **not** provide an official public trading API.  This adapter
    uses reverse-engineered internal endpoints that may change without notice.
    Order placement and cancellation are **not supported** and will raise
    :class:`NotImplementedError`.  Only portfolio-reading operations
    (``get_positions``, ``get_portfolio``) are functional.

Authentication:
    Log in to Groww in a browser, extract the session token (``access_token``
    or ``Authorization`` header value) from the browser's developer tools, and
    pass it as ``token`` in :class:`GrowwConfig`.
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

_GROWW_BASE_URL = "https://groww.in/v1/api"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GrowwConfig:
    """Connection parameters for the Groww adapter.

    Attributes:
        email: Groww account email (used for logging/identification only).
        token: Bearer / session token extracted from an authenticated browser
            session.  Typically the ``Authorization`` header value.
        session_id: Optional session-id cookie value for additional
            authentication if required by certain endpoints.
    """

    email: str = ""
    token: str = ""
    session_id: str = ""


# ---------------------------------------------------------------------------
# Broker implementation
# ---------------------------------------------------------------------------


class GrowwBroker(Broker):
    """Read-only broker adapter for Groww.

    This adapter can fetch portfolio and position data but **cannot** place
    or cancel orders because Groww does not expose a public trading API.

    Usage::

        cfg = GrowwConfig(token="Bearer <token>", email="user@example.com")
        async with GrowwBroker(cfg) as broker:
            positions = await broker.get_positions()
    """

    def __init__(self, config: GrowwConfig) -> None:
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._connected: bool = False

    # -- helpers -----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Authorization": self._config.token,
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Origin": "https://groww.in",
            "Referer": "https://groww.in/",
        }
        if self._config.session_id:
            headers["Cookie"] = f"sessionId={self._config.session_id}"
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Issue an HTTP request to Groww's internal API.

        Raises :class:`ExecutionError` on non-2xx responses.
        """
        if self._client is None:
            raise BrokerConnectionError("HTTP client not initialised; call connect().")

        url = f"{_GROWW_BASE_URL}{path}"
        response = await self._client.request(
            method,
            url,
            headers=self._headers(),
            params=params,
        )

        if response.status_code == 401:
            raise BrokerConnectionError(
                "Groww session expired or invalid.  Obtain a fresh token from "
                "the browser and update GrowwConfig.token."
            )

        if response.status_code >= 400:
            logger.error(
                "groww_api_error",
                path=path,
                status=response.status_code,
                body=response.text[:500],
            )
            raise ExecutionError(
                f"Groww API error {response.status_code}: {response.text[:200]}"
            )

        if not response.content:
            return {}

        return response.json()

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        """Validate the session by fetching the user profile."""
        if not self._config.token:
            raise BrokerConnectionError(
                "GrowwConfig.token is empty.  Extract the Authorization header "
                "from an authenticated browser session on groww.in."
            )

        self._client = httpx.AsyncClient(timeout=30.0)

        try:
            profile = await self._request("GET", "/user/v1/user/profile")
            logger.info(
                "groww_connected",
                email=self._config.email or profile.get("email", "unknown"),
            )
            self._connected = True
        except Exception as exc:
            await self.disconnect()
            raise BrokerConnectionError(f"Groww connection failed: {exc}") from exc

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        self._connected = False
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("groww_disconnected")

    # -- orders (not supported) --------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """Not supported -- Groww has no public trading API.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Groww does not expose a public trading API.  Order placement is "
            "not supported.  Use the Groww mobile app or website to trade."
        )

    async def cancel_order(self, order_id: str) -> Order:
        """Not supported -- Groww has no public trading API.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Groww does not expose a public trading API.  Order cancellation "
            "is not supported.  Use the Groww mobile app or website to manage orders."
        )

    # -- positions / portfolio ---------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Fetch the user's stock and mutual-fund holdings from Groww.

        Returns a list of :class:`Position` objects.  Since Groww is
        primarily an equity/mutual-fund platform, positions are mapped with
        generic ``OptionContract`` wrappers (strike=0, type=CALL as
        placeholder).
        """
        positions: list[Position] = []

        # Stocks
        try:
            holdings = await self._request("GET", "/stocks/v1/holdings")
            for h in holdings if isinstance(holdings, list) else holdings.get("holdings", []):
                symbol = h.get("tradingSymbol", h.get("symbol", "UNKNOWN"))
                qty = int(h.get("quantity", 0))
                if qty == 0:
                    continue

                avg_price = float(h.get("avgPrice", h.get("averagePrice", 0)))
                current_price = float(h.get("ltp", h.get("lastPrice", 0)))
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
        except ExecutionError:
            logger.warning("groww_holdings_fetch_failed")

        # Mutual funds
        try:
            mf_data = await self._request("GET", "/mf/v1/investments")
            for inv in mf_data if isinstance(mf_data, list) else mf_data.get("investments", []):
                scheme_name = inv.get("schemeName", inv.get("fundName", "MF_UNKNOWN"))
                units = float(inv.get("units", 0))
                if units <= 0:
                    continue

                nav = float(inv.get("nav", inv.get("currentNav", 0)))
                avg_nav = float(inv.get("avgNav", inv.get("averageNav", 0)))
                invested = float(inv.get("investedValue", 0))
                current_val = float(inv.get("currentValue", 0))
                pnl = current_val - invested if invested else 0

                positions.append(
                    Position(
                        contract=OptionContract(
                            symbol=scheme_name,
                            underlying=scheme_name,
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
        except ExecutionError:
            logger.warning("groww_mf_fetch_failed")

        return positions

    async def get_portfolio(self) -> PortfolioSnapshot:
        """Build a portfolio snapshot from Groww holdings.

        Because Groww does not expose a margin/cash endpoint, ``cash`` is
        reported as ``0.0`` and ``net_liquidation`` is the sum of current
        market values.
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
        """Not supported -- Groww has no public streaming API.

        This generator yields nothing and logs a warning.  It will block
        indefinitely (sleeping) to satisfy the interface contract until
        ``disconnect`` is called.
        """
        logger.warning(
            "groww_stream_fills_unsupported",
            msg="Groww does not provide a streaming API. stream_fills will idle.",
        )
        while self._connected:
            await asyncio.sleep(60.0)
        # Explicit return so the function is recognised as an async generator.
        return  # type: ignore[return-value]
        yield  # pragma: no cover  # noqa: E501  -- makes this a generator
