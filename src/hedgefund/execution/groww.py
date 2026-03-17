"""Groww broker adapter.

Implements the :class:`~hedgefund.execution.base.Broker` interface for Groww
using the official Groww Trading API (https://groww.in/trade-api/docs).

Supports portfolio reading (holdings, positions) and order management
(place, cancel).  Market data is available via LTP/quote/OHLC endpoints.

Authentication:
    Obtain an API key (JWT access token) and API secret from the Groww
    developer portal.  Note that API keys reset daily at 6 AM IST.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
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
# Official Groww Trading API base URL
# ---------------------------------------------------------------------------

_GROWW_BASE_URL = "https://api.groww.in/v1"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GrowwConfig:
    """Connection parameters for the Groww adapter.

    Attributes:
        api_key: JWT access token from the Groww developer portal.
            Used as the ``Authorization: Bearer <token>`` header.
        api_secret: API secret from the Groww developer portal.
    """

    api_key: str = ""
    api_secret: str = ""


# ---------------------------------------------------------------------------
# Broker implementation
# ---------------------------------------------------------------------------


class GrowwBroker(Broker):
    """Broker adapter for the Groww Trading API.

    Supports portfolio reading (holdings, positions) and order management.

    Usage::

        cfg = GrowwConfig(api_key="<jwt>", api_secret="<secret>")
        async with GrowwBroker(cfg) as broker:
            positions = await broker.get_positions()
    """

    def __init__(self, config: GrowwConfig) -> None:
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._connected: bool = False
        self._user_info: dict[str, Any] = {}
        self._access_token: str = ""

    # -- helpers -----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        token = self._access_token or self._config.api_key
        auth_value = token if token.startswith("Bearer ") else f"Bearer {token}"
        return {
            "Authorization": auth_value,
            "X-API-VERSION": "1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Issue an HTTP request to Groww's official API.

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
            json=json_body,
        )

        if response.status_code == 401:
            raise BrokerConnectionError(
                "Groww API key expired or invalid. API keys reset daily at 6 AM IST. "
                "Obtain a fresh key from the Groww developer portal."
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
        """Exchange API key + secret for access token, then validate."""
        if not self._config.api_key:
            raise BrokerConnectionError(
                "GrowwConfig.api_key is empty. Obtain an API key from the "
                "Groww developer portal."
            )

        # Exchange api_key + api_secret for an access token
        try:
            access_token = await asyncio.to_thread(
                self._exchange_token,
                self._config.api_key,
                self._config.api_secret,
            )
            self._access_token = access_token
            logger.info("groww_token_exchanged")
        except Exception as exc:
            raise BrokerConnectionError(
                f"Groww token exchange failed: {exc}. "
                "Check your API key and secret."
            ) from exc

        self._client = httpx.AsyncClient(timeout=30.0)

        try:
            result = await self._request("GET", "/user/profile")
            # Response: {"code":"200","success":{"data":{...}}}
            data = result
            if isinstance(result, dict) and "success" in result:
                data = result["success"].get("data", result)
            self._user_info = data if isinstance(data, dict) else {}
            ucc = self._user_info.get("ucc", "unknown")
            segments = self._user_info.get("activeSegments", [])
            logger.info(
                "groww_connected",
                ucc=ucc,
                segments=segments,
            )
            self._connected = True
        except Exception as exc:
            await self.disconnect()
            raise BrokerConnectionError(f"Groww connection failed: {exc}") from exc

    @staticmethod
    def _exchange_token(api_key: str, api_secret: str) -> str:
        """Exchange API key + secret for an access token (blocking call)."""
        from growwapi import GrowwAPI
        return GrowwAPI.get_access_token(api_key=api_key, secret=api_secret)

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        self._connected = False
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("groww_disconnected")

    # -- orders ------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """Place an order via the Groww Trading API.

        Uses ``POST /order/create``.
        """
        side_map = {Side.BUY: "BUY", Side.SELL: "SELL"}
        order_type_map = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
        }

        payload: dict[str, Any] = {
            "trading_symbol": order.contract.symbol,
            "exchange": "NSE",
            "transaction_type": side_map.get(order.side, "BUY"),
            "order_type": order_type_map.get(order.order_type, "MARKET"),
            "quantity": order.quantity,
            "product": "CNC",
        }
        if order.order_type == OrderType.LIMIT and order.limit_price:
            payload["price"] = order.limit_price

        try:
            result = await self._request("POST", "/order/create", json_body=payload)
            groww_order_id = ""
            if isinstance(result, dict):
                success = result.get("success", result)
                if isinstance(success, dict):
                    groww_order_id = str(
                        success.get("data", {}).get("groww_order_id", "")
                        if isinstance(success.get("data"), dict)
                        else success.get("groww_order_id", "")
                    )

            return Order(
                order_id=groww_order_id or order.order_id,
                contract=order.contract,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.limit_price,
                status=OrderStatus.SUBMITTED,
            )
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(f"Groww order placement failed: {exc}") from exc

    async def cancel_order(self, order_id: str) -> Order:
        """Cancel an order via the Groww Trading API.

        Uses ``POST /order/cancel``.
        """
        try:
            await self._request(
                "POST", "/order/cancel", json_body={"groww_order_id": order_id}
            )
            return Order(
                order_id=order_id,
                contract=OptionContract(
                    symbol="UNKNOWN",
                    underlying="UNKNOWN",
                    option_type=OptionType.CALL,
                    strike=0,
                    expiration=datetime.now(timezone.utc).date(),
                    multiplier=1,
                ),
                side=Side.BUY,
                quantity=0,
                order_type=OrderType.MARKET,
                status=OrderStatus.CANCELLED,
            )
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(f"Groww order cancel failed: {exc}") from exc

    # -- positions / portfolio ---------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Fetch the user's holdings from Groww.

        Uses ``GET /holdings/user`` for stock holdings and
        ``GET /positions/user`` for intraday/derivative positions.
        """
        positions: list[Position] = []

        # Stock holdings
        try:
            result = await self._request("GET", "/holdings/user")
            holdings_list = _extract_list(result, "holdings")
            for h in holdings_list:
                symbol = h.get("trading_symbol", h.get("tradingSymbol", "UNKNOWN"))
                qty = int(h.get("quantity", 0))
                if qty == 0:
                    continue

                avg_price = float(h.get("average_price", h.get("avgPrice", 0)))
                # Holdings API may not include LTP; default to avg_price
                current_price = float(
                    h.get("ltp", h.get("lastPrice", avg_price))
                )
                pnl = (current_price - avg_price) * qty

                positions.append(
                    Position(
                        contract=OptionContract(
                            symbol=symbol,
                            underlying=symbol,
                            option_type=OptionType.CALL,
                            strike=0,
                            expiration=datetime.now(timezone.utc).date(),
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

        # Intraday / derivative positions
        try:
            result = await self._request("GET", "/positions/user")
            pos_list = _extract_list(result, "positions")
            for p in pos_list:
                symbol = p.get(
                    "trading_symbol", p.get("tradingSymbol", "UNKNOWN")
                )
                qty = int(p.get("quantity", 0))
                if qty == 0:
                    continue

                net_price = float(p.get("net_price", p.get("netPrice", 0)))
                credit_price = float(
                    p.get("credit_price", p.get("creditPrice", 0))
                )
                debit_price = float(
                    p.get("debit_price", p.get("debitPrice", 0))
                )
                realised_pnl = float(
                    p.get("realised_pnl", p.get("realisedPnl", 0))
                )

                positions.append(
                    Position(
                        contract=OptionContract(
                            symbol=symbol,
                            underlying=symbol,
                            option_type=OptionType.CALL,
                            strike=0,
                            expiration=datetime.now(timezone.utc).date(),
                            multiplier=1,
                        ),
                        quantity=abs(qty),
                        avg_entry=debit_price if debit_price else net_price,
                        current_price=credit_price if credit_price else net_price,
                        greeks=Greeks(delta=0, gamma=0, theta=0, vega=0),
                        unrealized_pnl=realised_pnl,
                    )
                )
        except ExecutionError:
            logger.warning("groww_positions_fetch_failed")

        return positions

    async def get_portfolio(self) -> PortfolioSnapshot:
        """Build a portfolio snapshot from Groww holdings + positions.

        Cash is reported as ``0.0`` since the Groww API does not expose a
        margin/cash endpoint directly.
        """
        positions = await self.get_positions()

        total_value = sum(
            p.current_price * p.quantity * p.contract.multiplier for p in positions
        )
        total_pnl = sum(p.unrealized_pnl for p in positions)

        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            cash=0.0,
            net_liquidation=total_value,
            positions=positions,
            total_pnl=total_pnl,
        )

    # -- streaming (not supported) -----------------------------------------

    async def stream_fills(self) -> AsyncIterator[Order]:
        """Not supported -- Groww does not provide a streaming API.

        This generator yields nothing and idles until ``disconnect`` is called.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_list(result: Any, key: str) -> list[dict[str, Any]]:
    """Extract a list from various Groww API response shapes.

    Handles: ``[...]``, ``{"key": [...]}``,
    ``{"success": {"data": {"key": [...]}}}`` etc.
    """
    if isinstance(result, list):
        return result
    if not isinstance(result, dict):
        return []
    # Official format: {"code":"200","success":{"data":{...}}}
    success = result.get("success", result)
    if isinstance(success, dict):
        data = success.get("data", success)
        if isinstance(data, dict):
            items = data.get(key, [])
            if isinstance(items, list):
                return items
        if isinstance(data, list):
            return data
    # Flat format
    items = result.get(key, [])
    return items if isinstance(items, list) else []
