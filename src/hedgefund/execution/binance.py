"""Binance REST + WebSocket broker adapter.

Implements the :class:`~hedgefund.execution.base.Broker` interface for
Binance Spot, Futures (USDM), and European-style options (EAPI).

Authentication uses HMAC-SHA256 signed requests with an ``api_key`` and
``api_secret`` pair generated on the Binance dashboard.

Rate-limit handling:
    The adapter reads the ``X-MBX-USED-WEIGHT-*`` response headers and
    backs off automatically when usage exceeds 80 % of the limit.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional
from urllib.parse import urlencode

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
# API base URLs
# ---------------------------------------------------------------------------

_SPOT_BASE = "https://api.binance.com"
_SPOT_TESTNET = "https://testnet.binance.vision"

_FUTURES_BASE = "https://fapi.binance.com"
_FUTURES_TESTNET = "https://testnet.binancefuture.com"

_OPTIONS_BASE = "https://eapi.binance.com"

_WS_SPOT = "wss://stream.binance.com:9443/ws"
_WS_SPOT_TESTNET = "wss://testnet.binance.vision/ws"
_WS_FUTURES = "wss://fstream.binance.com/ws"
_WS_FUTURES_TESTNET = "wss://stream.binancefuture.com/ws"

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

_ORDER_TYPE_MAP: dict[OrderType, str] = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP: "STOP_LOSS",
    OrderType.STOP_LIMIT: "STOP_LOSS_LIMIT",
}

_FUTURES_ORDER_TYPE_MAP: dict[OrderType, str] = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP: "STOP_MARKET",
    OrderType.STOP_LIMIT: "STOP",
}

_STATUS_MAP: dict[str, OrderStatus] = {
    "NEW": OrderStatus.SUBMITTED,
    "PARTIALLY_FILLED": OrderStatus.PARTIAL,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.CANCELLED,
    "PENDING_CANCEL": OrderStatus.SUBMITTED,
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BinanceConfig:
    """Connection parameters for the Binance adapter."""

    api_key: str
    api_secret: str
    testnet: bool = False
    futures_enabled: bool = False


# ---------------------------------------------------------------------------
# Broker implementation
# ---------------------------------------------------------------------------


class BinanceBroker(Broker):
    """Async broker adapter for Binance Spot, Futures, and Options.

    Usage::

        cfg = BinanceConfig(api_key="...", api_secret="...", testnet=True)
        async with BinanceBroker(cfg) as broker:
            snap = await broker.get_portfolio()
    """

    def __init__(self, config: BinanceConfig) -> None:
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._connected: bool = False
        self._listen_key: Optional[str] = None
        self._listen_key_task: Optional[asyncio.Task[None]] = None

        # Resolve base URLs depending on testnet flag.
        if config.testnet:
            self._spot_base = _SPOT_TESTNET
            self._futures_base = _FUTURES_TESTNET
            self._ws_spot = _WS_SPOT_TESTNET
            self._ws_futures = _WS_FUTURES_TESTNET
        else:
            self._spot_base = _SPOT_BASE
            self._futures_base = _FUTURES_BASE
            self._ws_spot = _WS_SPOT
            self._ws_futures = _WS_FUTURES

    # -- signing -----------------------------------------------------------

    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:
        """Add ``timestamp`` and HMAC-SHA256 ``signature`` to *params*."""
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        sig = hmac.new(
            self._config.api_secret.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = sig
        return params

    def _headers(self) -> dict[str, str]:
        return {"X-MBX-APIKEY": self._config.api_key}

    # -- HTTP helpers with rate-limit handling -----------------------------

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[dict[str, Any]] = None,
        signed: bool = True,
        max_retries: int = 3,
    ) -> Any:
        """Issue an HTTP request with optional signing and retry on rate limits."""
        if self._client is None:
            raise BrokerConnectionError("HTTP client not initialised; call connect().")

        params = dict(params) if params else {}
        if signed:
            params = self._sign(params)

        for attempt in range(1, max_retries + 1):
            response = await self._client.request(
                method,
                url,
                params=params,
                headers=self._headers(),
            )

            # Rate-limit back-off: HTTP 429 or 418.
            if response.status_code in (429, 418):
                retry_after = int(response.headers.get("Retry-After", 5))
                logger.warning(
                    "binance_rate_limited",
                    retry_after=retry_after,
                    attempt=attempt,
                )
                if attempt < max_retries:
                    await asyncio.sleep(retry_after)
                    continue
                raise ExecutionError(
                    f"Binance rate limit exceeded after {max_retries} retries."
                )

            if response.status_code >= 400:
                body = response.json() if response.content else {}
                code = body.get("code", response.status_code)
                msg = body.get("msg", response.text)
                logger.error(
                    "binance_api_error",
                    url=url,
                    status=response.status_code,
                    code=code,
                    msg=msg,
                )
                raise ExecutionError(f"Binance API error {code}: {msg}")

            # Proactive back-off when we approach the weight limit.
            used_weight = int(response.headers.get("X-MBX-USED-WEIGHT-1m", 0))
            if used_weight > 960:  # 80% of the 1200 default limit
                logger.warning("binance_weight_high", used_weight=used_weight)
                await asyncio.sleep(2.0)

            return response.json()

        raise ExecutionError("Request failed after retries")

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        """Test connectivity and fetch account information."""
        self._client = httpx.AsyncClient(timeout=30.0)

        try:
            # Ping to verify network.
            await self._request("GET", f"{self._spot_base}/api/v3/ping", signed=False)

            # Verify credentials with account info.
            if self._config.futures_enabled:
                account = await self._request("GET", f"{self._futures_base}/fapi/v2/account")
                logger.info(
                    "binance_futures_connected",
                    total_wallet=account.get("totalWalletBalance"),
                )
            else:
                account = await self._request("GET", f"{self._spot_base}/api/v3/account")
                logger.info(
                    "binance_spot_connected",
                    maker_commission=account.get("makerCommission"),
                )

            self._connected = True
        except Exception as exc:
            await self.disconnect()
            raise BrokerConnectionError(f"Binance connection failed: {exc}") from exc

    async def disconnect(self) -> None:
        """Close the HTTP client and cancel any listen-key keep-alive."""
        self._connected = False

        if self._listen_key_task and not self._listen_key_task.done():
            self._listen_key_task.cancel()
            self._listen_key_task = None

        if self._listen_key:
            try:
                if self._config.futures_enabled:
                    url = f"{self._futures_base}/fapi/v1/listenKey"
                else:
                    url = f"{self._spot_base}/api/v3/userDataStream"
                await self._request(
                    "DELETE", url, params={"listenKey": self._listen_key}, signed=False
                )
            except Exception:
                pass
            self._listen_key = None

        if self._client:
            await self._client.aclose()
            self._client = None

        logger.info("binance_disconnected")

    # -- orders ------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """Place an order on Binance Spot, Futures, or Options."""
        is_option = order.contract.expiration != date.today() or order.contract.strike > 0
        is_futures = self._config.futures_enabled and not is_option

        side_str = "BUY" if order.side == Side.BUY else "SELL"

        if is_option:
            return await self._submit_option_order(order, side_str)
        elif is_futures:
            return await self._submit_futures_order(order, side_str)
        else:
            return await self._submit_spot_order(order, side_str)

    async def _submit_spot_order(self, order: Order, side_str: str) -> Order:
        binance_type = _ORDER_TYPE_MAP.get(order.order_type, "MARKET")
        params: dict[str, Any] = {
            "symbol": order.contract.underlying.replace("/", ""),
            "side": side_str,
            "type": binance_type,
            "quantity": str(order.quantity),
            "newClientOrderId": order.order_id,
        }
        if order.order_type == OrderType.LIMIT:
            params["timeInForce"] = "GTC"
            params["price"] = str(order.limit_price)
        if order.stop_price is not None:
            params["stopPrice"] = str(order.stop_price)
        if order.order_type == OrderType.STOP_LIMIT:
            params["timeInForce"] = "GTC"
            params["price"] = str(order.limit_price)

        try:
            result = await self._request(
                "POST", f"{self._spot_base}/api/v3/order", params=params
            )
            return self._map_binance_order(result, order)
        except ExecutionError as exc:
            order.status = OrderStatus.REJECTED
            raise OrderRejectedError(order.order_id, str(exc)) from exc

    async def _submit_futures_order(self, order: Order, side_str: str) -> Order:
        binance_type = _FUTURES_ORDER_TYPE_MAP.get(order.order_type, "MARKET")
        params: dict[str, Any] = {
            "symbol": order.contract.underlying.replace("/", ""),
            "side": side_str,
            "type": binance_type,
            "quantity": str(order.quantity),
            "newClientOrderId": order.order_id,
        }
        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            params["timeInForce"] = "GTC"
            params["price"] = str(order.limit_price)
        if order.stop_price is not None:
            params["stopPrice"] = str(order.stop_price)

        try:
            result = await self._request(
                "POST", f"{self._futures_base}/fapi/v1/order", params=params
            )
            return self._map_binance_order(result, order)
        except ExecutionError as exc:
            order.status = OrderStatus.REJECTED
            raise OrderRejectedError(order.order_id, str(exc)) from exc

    async def _submit_option_order(self, order: Order, side_str: str) -> Order:
        """Submit a European-style option order via Binance EAPI."""
        binance_type = _ORDER_TYPE_MAP.get(order.order_type, "MARKET")
        # Binance options symbol format: BTC-240315-50000-C
        opt_type = "C" if order.contract.option_type == OptionType.CALL else "P"
        exp_str = order.contract.expiration.strftime("%y%m%d")
        symbol = (
            f"{order.contract.underlying}-{exp_str}-"
            f"{int(order.contract.strike)}-{opt_type}"
        )

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side_str,
            "type": binance_type,
            "quantity": str(order.quantity),
            "clientOrderId": order.order_id,
        }
        if order.limit_price is not None:
            params["price"] = str(order.limit_price)
            params["timeInForce"] = "GTC"

        try:
            result = await self._request(
                "POST", f"{_OPTIONS_BASE}/eapi/v1/order", params=params
            )
            return self._map_binance_order(result, order)
        except ExecutionError as exc:
            order.status = OrderStatus.REJECTED
            raise OrderRejectedError(order.order_id, str(exc)) from exc

    async def cancel_order(self, order_id: str) -> Order:
        """Cancel an open order by ``origClientOrderId``."""
        try:
            if self._config.futures_enabled:
                # Need symbol -- try to cancel across known symbols.
                result = await self._request(
                    "DELETE",
                    f"{self._futures_base}/fapi/v1/order",
                    params={"origClientOrderId": order_id},
                )
            else:
                result = await self._request(
                    "DELETE",
                    f"{self._spot_base}/api/v3/order",
                    params={"origClientOrderId": order_id},
                )
            logger.info("binance_order_cancelled", order_id=order_id)
        except ExecutionError:
            logger.warning("binance_cancel_failed", order_id=order_id)
            result = {}

        if result:
            return self._map_binance_order_from_dict(result)

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
        """Fetch current positions from the account endpoint."""
        positions: list[Position] = []

        if self._config.futures_enabled:
            account = await self._request("GET", f"{self._futures_base}/fapi/v2/account")
            for p in account.get("positions", []):
                amt = float(p.get("positionAmt", 0))
                if amt == 0:
                    continue
                symbol = p.get("symbol", "UNKNOWN")
                entry = float(p.get("entryPrice", 0))
                mark = float(p.get("markPrice", 0))
                unrealised = float(p.get("unrealizedProfit", 0))
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
                        quantity=int(abs(amt)),
                        avg_entry=entry,
                        current_price=mark,
                        greeks=Greeks(delta=0, gamma=0, theta=0, vega=0),
                        unrealized_pnl=unrealised,
                    )
                )
        else:
            account = await self._request("GET", f"{self._spot_base}/api/v3/account")
            for bal in account.get("balances", []):
                free = float(bal.get("free", 0))
                locked = float(bal.get("locked", 0))
                total = free + locked
                if total <= 0:
                    continue
                asset = bal["asset"]
                positions.append(
                    Position(
                        contract=OptionContract(
                            symbol=asset,
                            underlying=f"{asset}USDT",
                            option_type=OptionType.CALL,
                            strike=0,
                            expiration=date.today(),
                            multiplier=1,
                        ),
                        quantity=int(total) if total >= 1 else 1,
                        avg_entry=0,
                        current_price=0,
                        greeks=Greeks(delta=0, gamma=0, theta=0, vega=0),
                        unrealized_pnl=0,
                    )
                )

        return positions

    async def get_portfolio(self) -> PortfolioSnapshot:
        """Aggregate balances into a portfolio snapshot valued in USDT."""
        positions = await self.get_positions()

        if self._config.futures_enabled:
            account = await self._request("GET", f"{self._futures_base}/fapi/v2/account")
            cash = float(account.get("availableBalance", 0))
            net_liq = float(account.get("totalWalletBalance", 0))
        else:
            account = await self._request("GET", f"{self._spot_base}/api/v3/account")
            # Sum all USDT-equivalent balances.
            cash = 0.0
            for bal in account.get("balances", []):
                if bal["asset"] == "USDT":
                    cash = float(bal.get("free", 0)) + float(bal.get("locked", 0))
                    break
            net_liq = cash  # Simplified; a full implementation would price each asset.

        return PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            cash=cash,
            net_liquidation=net_liq,
            positions=positions,
        )

    # -- streaming ---------------------------------------------------------

    async def stream_fills(self) -> AsyncIterator[Order]:
        """Stream fill events via the Binance User Data Stream WebSocket.

        The adapter creates a ``listenKey``, opens a WebSocket, and yields
        :class:`Order` objects whenever an ``executionReport`` (spot) or
        ``ORDER_TRADE_UPDATE`` (futures) event is received.

        Falls back to REST polling if ``websockets`` is not installed.
        """
        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("binance_websockets_unavailable_falling_back_to_polling")
            async for order in self._poll_fills():
                yield order
            return

        listen_key = await self._create_listen_key()
        self._listen_key = listen_key
        self._listen_key_task = asyncio.create_task(self._keepalive_listen_key())

        if self._config.futures_enabled:
            ws_url = f"{self._ws_futures}/{listen_key}"
        else:
            ws_url = f"{self._ws_spot}/{listen_key}"

        try:
            async with websockets.connect(ws_url) as ws:
                while self._connected:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    except asyncio.TimeoutError:
                        continue

                    msg = json.loads(raw)
                    event_type = msg.get("e", "")

                    if event_type == "executionReport":
                        status_str = msg.get("X", "")
                        if status_str in ("FILLED", "PARTIALLY_FILLED"):
                            yield self._map_ws_execution_report(msg)

                    elif event_type == "ORDER_TRADE_UPDATE":
                        order_data = msg.get("o", {})
                        status_str = order_data.get("X", "")
                        if status_str in ("FILLED", "PARTIALLY_FILLED"):
                            yield self._map_ws_futures_order(order_data)

        except Exception as exc:
            logger.error("binance_ws_error", error=str(exc))

    async def _poll_fills(self) -> AsyncIterator[Order]:
        """Fallback polling for fills when WebSocket is unavailable."""
        seen: set[str] = set()
        while self._connected:
            try:
                if self._config.futures_enabled:
                    orders = await self._request(
                        "GET", f"{self._futures_base}/fapi/v1/allOrders"
                    )
                else:
                    orders = await self._request(
                        "GET", f"{self._spot_base}/api/v3/allOrders",
                        params={"limit": 50},
                    )

                for o in (orders if isinstance(orders, list) else []):
                    oid = str(o.get("clientOrderId", o.get("orderId")))
                    status = o.get("status", "")
                    if status in ("FILLED", "PARTIALLY_FILLED") and oid not in seen:
                        seen.add(oid)
                        yield self._map_binance_order_from_dict(o)

            except ExecutionError:
                logger.warning("binance_poll_error")

            await asyncio.sleep(3.0)

    # -- listen key management ---------------------------------------------

    async def _create_listen_key(self) -> str:
        if self._config.futures_enabled:
            url = f"{self._futures_base}/fapi/v1/listenKey"
        else:
            url = f"{self._spot_base}/api/v3/userDataStream"

        result = await self._request("POST", url, signed=False)
        return result["listenKey"]

    async def _keepalive_listen_key(self) -> None:
        """Ping the listen key every 30 minutes to keep it alive."""
        while self._connected and self._listen_key:
            await asyncio.sleep(1800)
            try:
                if self._config.futures_enabled:
                    url = f"{self._futures_base}/fapi/v1/listenKey"
                else:
                    url = f"{self._spot_base}/api/v3/userDataStream"
                await self._request(
                    "PUT", url, params={"listenKey": self._listen_key}, signed=False
                )
            except Exception:
                logger.warning("binance_listen_key_keepalive_failed")

    # -- mapping helpers ---------------------------------------------------

    def _map_binance_order(self, result: dict[str, Any], order: Order) -> Order:
        """Update an existing Order with data from a Binance response."""
        status = _STATUS_MAP.get(result.get("status", ""), OrderStatus.SUBMITTED)
        order.status = status
        order.order_id = str(result.get("clientOrderId", order.order_id))

        filled_qty = int(float(result.get("executedQty", 0)))
        order.filled_quantity = filled_qty
        if filled_qty > 0:
            cum_quote = float(result.get("cummulativeQuoteQty", 0))
            order.filled_price = cum_quote / filled_qty if filled_qty else None
        if status == OrderStatus.FILLED:
            order.filled_at = datetime.utcnow()

        logger.info(
            "binance_order_submitted",
            order_id=order.order_id,
            status=status.value,
        )
        return order

    def _map_binance_order_from_dict(self, o: dict[str, Any]) -> Order:
        """Create an Order from a raw Binance order dict."""
        symbol = o.get("symbol", "UNKNOWN")
        side = Side.BUY if o.get("side") == "BUY" else Side.SELL
        status = _STATUS_MAP.get(o.get("status", ""), OrderStatus.PENDING)

        type_rev: dict[str, OrderType] = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP_LOSS": OrderType.STOP,
            "STOP_LOSS_LIMIT": OrderType.STOP_LIMIT,
            "STOP_MARKET": OrderType.STOP,
            "STOP": OrderType.STOP_LIMIT,
        }
        order_type = type_rev.get(o.get("type", "MARKET"), OrderType.MARKET)

        filled_qty = int(float(o.get("executedQty", 0)))
        cum_quote = float(o.get("cummulativeQuoteQty", 0))
        filled_price = (cum_quote / filled_qty) if filled_qty > 0 else None

        return Order(
            order_id=str(o.get("clientOrderId", o.get("orderId", ""))),
            signal_id="",
            contract=OptionContract(
                symbol=symbol,
                underlying=symbol,
                option_type=OptionType.CALL,
                strike=0,
                expiration=date.today(),
                multiplier=1,
            ),
            side=side,
            order_type=order_type,
            quantity=int(float(o.get("origQty", 0))),
            limit_price=float(o["price"]) if float(o.get("price", 0)) > 0 else None,
            stop_price=float(o["stopPrice"]) if float(o.get("stopPrice", 0)) > 0 else None,
            status=status,
            filled_price=filled_price,
            filled_quantity=filled_qty,
            filled_at=datetime.utcfromtimestamp(o["updateTime"] / 1000) if o.get("updateTime") else None,
        )

    def _map_ws_execution_report(self, msg: dict[str, Any]) -> Order:
        """Map a spot ``executionReport`` WebSocket event to an Order."""
        side = Side.BUY if msg.get("S") == "BUY" else Side.SELL
        status = _STATUS_MAP.get(msg.get("X", ""), OrderStatus.SUBMITTED)

        filled_qty = int(float(msg.get("z", 0)))
        cum_quote = float(msg.get("Z", 0))
        filled_price = (cum_quote / filled_qty) if filled_qty > 0 else None

        return Order(
            order_id=str(msg.get("c", "")),
            signal_id="",
            contract=OptionContract(
                symbol=msg.get("s", "UNKNOWN"),
                underlying=msg.get("s", "UNKNOWN"),
                option_type=OptionType.CALL,
                strike=0,
                expiration=date.today(),
                multiplier=1,
            ),
            side=side,
            order_type=OrderType.MARKET,
            quantity=int(float(msg.get("q", 0))),
            status=status,
            filled_price=filled_price,
            filled_quantity=filled_qty,
            filled_at=datetime.utcfromtimestamp(msg["T"] / 1000) if msg.get("T") else None,
        )

    def _map_ws_futures_order(self, o: dict[str, Any]) -> Order:
        """Map a futures ``ORDER_TRADE_UPDATE`` payload to an Order."""
        side = Side.BUY if o.get("S") == "BUY" else Side.SELL
        status = _STATUS_MAP.get(o.get("X", ""), OrderStatus.SUBMITTED)

        filled_qty = int(float(o.get("z", 0)))
        avg_price = float(o.get("ap", 0))

        return Order(
            order_id=str(o.get("c", "")),
            signal_id="",
            contract=OptionContract(
                symbol=o.get("s", "UNKNOWN"),
                underlying=o.get("s", "UNKNOWN"),
                option_type=OptionType.CALL,
                strike=0,
                expiration=date.today(),
                multiplier=1,
            ),
            side=side,
            order_type=OrderType.MARKET,
            quantity=int(float(o.get("q", 0))),
            status=status,
            filled_price=avg_price if avg_price > 0 else None,
            filled_quantity=filled_qty,
            filled_at=datetime.utcnow(),
        )
