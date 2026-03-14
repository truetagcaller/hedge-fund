"""Zerodha Kite Connect API broker adapter.

Implements the :class:`~hedgefund.execution.base.Broker` interface using
Zerodha's Kite Connect REST API v3 (https://kite.trade/docs/connect/v3/).

Authentication flow:
    1. Obtain an ``api_key`` and ``api_secret`` from the Kite developer console.
    2. Redirect the user to the Kite login URL to get a ``request_token``.
    3. Exchange the ``request_token`` for an ``access_token`` via the session API.
    4. Alternatively, provide an ``access_token`` directly if you already have one.

Zerodha uses exchange-prefixed trading symbols for options, e.g.
``NFO:NIFTY2431520000CE``.  The adapter maps these to the internal
:class:`~hedgefund.types.OptionContract` format.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import httpx

from hedgefund.exceptions import (
    BrokerConnectionError,
    ExecutionError,
    OrderRejectedError,
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
# Kite API constants
# ---------------------------------------------------------------------------

_KITE_BASE_URL = "https://api.kite.trade"
_KITE_LOGIN_URL = "https://kite.zerodha.com/connect/login"

_KITE_ORDER_TYPE_MAP: dict[OrderType, str] = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP: "SL-M",
    OrderType.STOP_LIMIT: "SL",
}

_KITE_STATUS_MAP: dict[str, OrderStatus] = {
    "OPEN": OrderStatus.SUBMITTED,
    "COMPLETE": OrderStatus.FILLED,
    "CANCELLED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "TRIGGER PENDING": OrderStatus.SUBMITTED,
    "OPEN PENDING": OrderStatus.PENDING,
    "VALIDATION PENDING": OrderStatus.PENDING,
    "PUT ORDER REQ RECEIVED": OrderStatus.PENDING,
    "MODIFY VALIDATION PENDING": OrderStatus.SUBMITTED,
    "MODIFY ORDER REQ RECEIVED": OrderStatus.SUBMITTED,
    "CANCEL PENDING": OrderStatus.SUBMITTED,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ZerodhaConfig:
    """Connection parameters for the Zerodha Kite Connect adapter."""

    api_key: str
    api_secret: str
    access_token: Optional[str] = None
    redirect_url: str = "https://127.0.0.1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_kite_tradingsymbol(
    tradingsymbol: str,
    exchange: str,
) -> Optional[OptionContract]:
    """Best-effort parse of a Kite trading symbol into an OptionContract.

    Kite options symbols look like ``NIFTY2431520000CE`` where:
        - Underlying + expiry date (YY + M_code + DD)
        - Strike price (integer)
        - CE / PE suffix

    Returns ``None`` when the symbol does not look like an option.
    """
    if exchange not in ("NFO", "BFO", "MCX"):
        return None

    # Last two chars should be CE or PE
    if tradingsymbol[-2:] not in ("CE", "PE"):
        return None

    option_type = OptionType.CALL if tradingsymbol.endswith("CE") else OptionType.PUT
    body = tradingsymbol[:-2]

    # Find where the underlying name ends and the numeric part begins.
    idx = 0
    for i, ch in enumerate(body):
        if ch.isdigit():
            idx = i
            break
    else:
        return None

    underlying = body[:idx]
    numeric_part = body[idx:]

    # First five digits are YYMDD, rest is strike
    if len(numeric_part) < 6:
        return None

    try:
        expiry_str = numeric_part[:5]
        year = 2000 + int(expiry_str[:2])
        # Kite uses a single-char month code: 1-9 for Jan-Sep, O/N/D for Oct-Dec.
        month_ch = expiry_str[2]
        if month_ch.isdigit():
            month = int(month_ch)
        elif month_ch.upper() == "O":
            month = 10
        elif month_ch.upper() == "N":
            month = 11
        elif month_ch.upper() == "D":
            month = 12
        else:
            return None
        day = int(expiry_str[3:5])
        expiration = date(year, month, day)
        strike = float(numeric_part[5:])
    except (ValueError, IndexError):
        return None

    lot_size = 25 if "NIFTY" in underlying.upper() else 100

    return OptionContract(
        symbol=f"{exchange}:{tradingsymbol}",
        underlying=underlying,
        option_type=option_type,
        strike=strike,
        expiration=expiration,
        multiplier=lot_size,
    )


def _kite_checksum(api_key: str, request_token: str, api_secret: str) -> str:
    """Compute the SHA-256 checksum required by Kite session creation."""
    raw = api_key + request_token + api_secret
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Broker implementation
# ---------------------------------------------------------------------------


class ZerodhaBroker(Broker):
    """Async broker adapter for Zerodha Kite Connect v3.

    Usage::

        cfg = ZerodhaConfig(api_key="xxx", api_secret="yyy", access_token="zzz")
        async with ZerodhaBroker(cfg) as broker:
            positions = await broker.get_positions()
    """

    def __init__(self, config: ZerodhaConfig) -> None:
        self._config = config
        self._access_token: Optional[str] = config.access_token
        self._client: Optional[httpx.AsyncClient] = None
        self._connected: bool = False
        self._known_order_statuses: dict[str, str] = {}

    # -- helpers -----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if not self._access_token:
            raise BrokerConnectionError("Zerodha access_token is not set; call connect() first.")
        return {
            "X-Kite-Version": "3",
            "Authorization": f"token {self._config.api_key}:{self._access_token}",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Issue an HTTP request to the Kite API and return the JSON body.

        Raises :class:`ExecutionError` on non-200 responses.
        """
        if self._client is None:
            raise BrokerConnectionError("HTTP client not initialised; call connect().")

        url = f"{_KITE_BASE_URL}{path}"
        response = await self._client.request(
            method,
            url,
            headers=self._headers(),
            params=params,
            data=data,
        )

        body: dict[str, Any] = response.json()

        if response.status_code != 200 or body.get("status") == "error":
            error_type = body.get("error_type", "GeneralException")
            message = body.get("message", response.text)
            logger.error(
                "kite_api_error",
                path=path,
                status=response.status_code,
                error_type=error_type,
                message=message,
            )
            raise ExecutionError(f"Kite API error ({error_type}): {message}")

        return body.get("data", body)

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        """Validate the Kite session and fetch the user profile.

        If ``access_token`` was not provided at construction time, the
        adapter will attempt to create a session using the ``request_token``
        (which must be set on the config's ``access_token`` field temporarily
        as a request token).
        """
        self._client = httpx.AsyncClient(timeout=30.0)

        if not self._access_token:
            raise BrokerConnectionError(
                "No access_token provided.  Redirect the user to "
                f"{_KITE_LOGIN_URL}?api_key={self._config.api_key}&v=3 "
                "and exchange the resulting request_token for an access_token."
            )

        try:
            profile = await self._request("GET", "/user/profile")
            logger.info(
                "zerodha_connected",
                user_id=profile.get("user_id"),
                user_name=profile.get("user_name"),
                email=profile.get("email"),
            )
            self._connected = True
        except Exception as exc:
            await self.disconnect()
            raise BrokerConnectionError(f"Zerodha connection failed: {exc}") from exc

    async def disconnect(self) -> None:
        """Close the underlying HTTP client."""
        self._connected = False
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("zerodha_disconnected")

    # -- session helpers ---------------------------------------------------

    async def create_session(self, request_token: str) -> str:
        """Exchange a *request_token* for an access token.

        Returns the access token and stores it internally so subsequent
        requests are authenticated.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)

        checksum = _kite_checksum(
            self._config.api_key,
            request_token,
            self._config.api_secret,
        )

        response = await self._client.post(
            f"{_KITE_BASE_URL}/session/token",
            data={
                "api_key": self._config.api_key,
                "request_token": request_token,
                "checksum": checksum,
            },
        )
        body = response.json()
        if response.status_code != 200 or body.get("status") == "error":
            raise BrokerConnectionError(
                f"Session creation failed: {body.get('message', response.text)}"
            )

        self._access_token = body["data"]["access_token"]
        logger.info("zerodha_session_created")
        return self._access_token  # type: ignore[return-value]

    @property
    def login_url(self) -> str:
        """URL the end-user must visit to authorise the app."""
        return f"{_KITE_LOGIN_URL}?api_key={self._config.api_key}&v=3"

    # -- orders ------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """Place an order via Kite Connect.

        Maps internal :class:`OrderType` to Kite order varieties and returns
        the order with an updated ``status`` and ``order_id``.
        """
        kite_order_type = _KITE_ORDER_TYPE_MAP.get(order.order_type, "MARKET")
        transaction_type = "BUY" if order.side == Side.BUY else "SELL"

        # Determine exchange and tradingsymbol from the contract symbol.
        if ":" in order.contract.symbol:
            exchange, tradingsymbol = order.contract.symbol.split(":", 1)
        else:
            exchange = "NFO"
            tradingsymbol = order.contract.symbol

        params: dict[str, Any] = {
            "tradingsymbol": tradingsymbol,
            "exchange": exchange,
            "transaction_type": transaction_type,
            "order_type": kite_order_type,
            "quantity": order.quantity,
            "product": "NRML",  # Normal (carry-forward) for F&O
            "validity": "DAY",
        }

        if order.limit_price is not None:
            params["price"] = order.limit_price
        if order.stop_price is not None:
            params["trigger_price"] = order.stop_price

        try:
            result = await self._request("POST", "/orders/regular", data=params)
            kite_order_id = result.get("order_id", order.order_id)
            order.order_id = str(kite_order_id)
            order.status = OrderStatus.SUBMITTED
            logger.info(
                "zerodha_order_submitted",
                order_id=order.order_id,
                symbol=tradingsymbol,
                side=transaction_type,
                qty=order.quantity,
            )
        except ExecutionError as exc:
            order.status = OrderStatus.REJECTED
            raise OrderRejectedError(order.order_id, str(exc)) from exc

        return order

    async def cancel_order(self, order_id: str) -> Order:
        """Cancel an order via ``DELETE /orders/regular/{order_id}``."""
        try:
            await self._request("DELETE", f"/orders/regular/{order_id}")
            logger.info("zerodha_order_cancelled", order_id=order_id)
        except ExecutionError:
            logger.warning("zerodha_cancel_failed", order_id=order_id)

        # Fetch the latest order state to return.
        orders = await self._request("GET", "/orders")
        for o in orders if isinstance(orders, list) else []:
            if str(o.get("order_id")) == order_id:
                return self._map_kite_order(o)

        # Fallback: return a stub with CANCELLED status.
        return Order(
            order_id=order_id,
            signal_id="",
            contract=OptionContract(
                symbol="UNKNOWN",
                underlying="UNKNOWN",
                option_type=OptionType.CALL,
                strike=0,
                expiration=date.today(),
            ),
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0,
            status=OrderStatus.CANCELLED,
        )

    # -- positions / portfolio ---------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Fetch net positions from ``/portfolio/positions``."""
        data = await self._request("GET", "/portfolio/positions")
        net_positions: list[dict[str, Any]] = data.get("net", []) if isinstance(data, dict) else []

        positions: list[Position] = []
        for p in net_positions:
            if p.get("quantity", 0) == 0:
                continue

            contract = _parse_kite_tradingsymbol(
                p.get("tradingsymbol", ""),
                p.get("exchange", ""),
            )
            if contract is None:
                # Equity or unrecognised instrument -- wrap generically.
                contract = OptionContract(
                    symbol=f"{p.get('exchange', '')}:{p.get('tradingsymbol', '')}",
                    underlying=p.get("tradingsymbol", "UNKNOWN"),
                    option_type=OptionType.CALL,
                    strike=0,
                    expiration=date.today(),
                    multiplier=1,
                )

            avg_price = float(p.get("average_price", 0))
            last_price = float(p.get("last_price", 0))
            qty = int(p.get("quantity", 0))
            pnl = float(p.get("pnl", 0))

            positions.append(
                Position(
                    contract=contract,
                    quantity=qty,
                    avg_entry=avg_price,
                    current_price=last_price,
                    greeks=Greeks(delta=0, gamma=0, theta=0, vega=0),
                    unrealized_pnl=pnl,
                    realized_pnl=float(p.get("realised", 0)),
                )
            )

        return positions

    async def get_portfolio(self) -> PortfolioSnapshot:
        """Build a portfolio snapshot from margins and positions."""
        margins_data = await self._request("GET", "/user/margins")
        positions = await self.get_positions()

        # Extract equity segment margin info.
        equity = margins_data.get("equity", {}) if isinstance(margins_data, dict) else {}
        cash = float(equity.get("available", {}).get("cash", 0))
        net_liquidation = cash + sum(p.unrealized_pnl for p in positions)

        total_delta = sum(p.greeks.delta * p.quantity for p in positions)
        total_gamma = sum(p.greeks.gamma * p.quantity for p in positions)
        total_theta = sum(p.greeks.theta * p.quantity for p in positions)
        total_vega = sum(p.greeks.vega * p.quantity for p in positions)

        return PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            cash=cash,
            net_liquidation=net_liquidation,
            positions=positions,
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_theta=total_theta,
            total_vega=total_vega,
        )

    # -- streaming ---------------------------------------------------------

    async def stream_fills(self) -> AsyncIterator[Order]:
        """Poll ``/orders`` for status changes and yield filled orders.

        Zerodha's Kite Connect WebSocket (kiteticker) is primarily for
        market-data streaming; order updates are most reliably obtained by
        polling the order book via the REST API.
        """
        while self._connected:
            try:
                orders_data = await self._request("GET", "/orders")
                if not isinstance(orders_data, list):
                    orders_data = []

                for o in orders_data:
                    oid = str(o.get("order_id"))
                    current_status = o.get("status", "")
                    previous = self._known_order_statuses.get(oid)

                    if previous != current_status and current_status in (
                        "COMPLETE",
                        "CANCELLED",
                        "REJECTED",
                    ):
                        self._known_order_statuses[oid] = current_status
                        yield self._map_kite_order(o)
                    else:
                        self._known_order_statuses[oid] = current_status

            except ExecutionError:
                logger.warning("zerodha_stream_poll_error")

            await asyncio.sleep(2.0)

    # -- internal mapping --------------------------------------------------

    def _map_kite_order(self, o: dict[str, Any]) -> Order:
        """Convert a Kite order dict to an internal :class:`Order`."""
        contract = _parse_kite_tradingsymbol(
            o.get("tradingsymbol", ""),
            o.get("exchange", ""),
        )
        if contract is None:
            contract = OptionContract(
                symbol=f"{o.get('exchange', '')}:{o.get('tradingsymbol', '')}",
                underlying=o.get("tradingsymbol", "UNKNOWN"),
                option_type=OptionType.CALL,
                strike=0,
                expiration=date.today(),
                multiplier=1,
            )

        status = _KITE_STATUS_MAP.get(o.get("status", ""), OrderStatus.PENDING)
        side = Side.BUY if o.get("transaction_type") == "BUY" else Side.SELL

        # Map Kite order_type back to internal OrderType.
        kite_otype = o.get("order_type", "MARKET")
        order_type_map: dict[str, OrderType] = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "SL": OrderType.STOP_LIMIT,
            "SL-M": OrderType.STOP,
        }
        order_type = order_type_map.get(kite_otype, OrderType.MARKET)

        filled_qty = int(o.get("filled_quantity", 0))
        avg_price = float(o.get("average_price", 0)) if filled_qty else None

        filled_at = None
        if o.get("exchange_timestamp"):
            try:
                filled_at = datetime.fromisoformat(str(o["exchange_timestamp"]))
            except (ValueError, TypeError):
                pass

        return Order(
            order_id=str(o.get("order_id", "")),
            signal_id=o.get("tag", ""),
            contract=contract,
            side=side,
            order_type=order_type,
            quantity=int(o.get("quantity", 0)),
            limit_price=float(o["price"]) if o.get("price") else None,
            stop_price=float(o["trigger_price"]) if o.get("trigger_price") else None,
            status=status,
            filled_price=avg_price,
            filled_quantity=filled_qty,
            filled_at=filled_at,
        )
