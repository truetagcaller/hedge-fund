"""WebSocket feed implementations for each supported broker.

Provides concrete feeds for Zerodha (Kite), Binance, and a generic
REST-polling fallback for brokers without WebSocket support (Groww,
IndMoney).
"""

from __future__ import annotations

import abc
import asyncio
import json
import struct
import time
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

from hedgefund.logger import get_logger
from hedgefund.streaming.event_bus import Event, EventBus, EventType

log = get_logger(__name__)

# Type for async callbacks used by feeds.
FeedCallback = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]


class BaseWebSocketFeed(abc.ABC):
    """Base class for all WebSocket / streaming feeds.

    Implementations must support the async context manager protocol and
    provide a :meth:`run` method that streams data until cancelled.
    """

    def __init__(self, symbol: str, event_bus: EventBus) -> None:
        self.symbol = symbol
        self._event_bus = event_bus
        self._connected = False
        self._connect_time: float = 0.0
        self._messages_received: int = 0
        self._latency_sum: float = 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def avg_latency_ms(self) -> float:
        if self._messages_received == 0:
            return 0.0
        return (self._latency_sum / self._messages_received) * 1000.0

    # ── Async context manager ─────────────────────────────────────────────

    async def __aenter__(self) -> BaseWebSocketFeed:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.disconnect()

    # ── Abstract interface ────────────────────────────────────────────────

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish the WebSocket / polling connection."""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close the connection."""

    @abc.abstractmethod
    async def run(self, callbacks: Dict[str, Any]) -> None:
        """Stream data indefinitely, invoking callbacks on new data.

        The *callbacks* dict may contain keys ``on_tick``, ``on_orderbook``,
        and ``on_options``, each mapping to an async callable.
        """

    def _record_message(self, latency: float = 0.0) -> None:
        """Track message count and latency for metrics."""
        self._messages_received += 1
        self._latency_sum += latency


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Zerodha (Kite) WebSocket Feed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ZerodhaWebSocketFeed(BaseWebSocketFeed):
    """WebSocket feed for Zerodha Kite Connect.

    Uses the Kite binary WebSocket protocol at ``wss://ws.kite.trade``.
    Supports three modes:

    - ``ltp``: Last traded price only.
    - ``quote``: LTP + OHLC + volume.
    - ``full``: Quote + 5-level order book depth.

    Args:
        symbol: Trading symbol.
        event_bus: Central event bus for publishing events.
        api_key: Kite Connect API key.
        access_token: Session access token.
        instrument_token: Numeric instrument token for the symbol.
        mode: Subscription mode (``ltp``, ``quote``, ``full``).
    """

    WS_URL = "wss://ws.kite.trade"

    # Kite binary packet sizes per mode
    _MODE_LTP = "ltp"
    _MODE_QUOTE = "quote"
    _MODE_FULL = "full"
    _PACKET_SIZES = {_MODE_LTP: 8, _MODE_QUOTE: 44, _MODE_FULL: 184}

    def __init__(
        self,
        symbol: str,
        event_bus: EventBus,
        api_key: str = "",
        access_token: str = "",
        instrument_token: int = 0,
        mode: str = "full",
    ) -> None:
        super().__init__(symbol, event_bus)
        self._api_key = api_key
        self._access_token = access_token
        self._instrument_token = instrument_token
        self._mode = mode
        self._ws: Any = None
        self._ping_task: Optional[asyncio.Task[None]] = None

    async def connect(self) -> None:
        """Connect to Kite WebSocket."""
        try:
            import websockets  # type: ignore[import-untyped]

            url = f"{self.WS_URL}?api_key={self._api_key}&access_token={self._access_token}"
            self._ws = await websockets.connect(url)
            self._connected = True
            self._connect_time = time.time()

            # Subscribe to instrument
            await self._send_subscribe()

            # Start heartbeat
            self._ping_task = asyncio.create_task(
                self._ping_loop(), name=f"kite-ping-{self.symbol}"
            )

            log.info(
                "zerodha_ws_connected",
                symbol=self.symbol,
                instrument_token=self._instrument_token,
                mode=self._mode,
            )
        except ImportError:
            log.error("websockets_not_installed", msg="pip install websockets")
            raise
        except Exception:
            log.exception("zerodha_ws_connect_failed", symbol=self.symbol)
            raise

    async def disconnect(self) -> None:
        """Disconnect from Kite WebSocket."""
        self._connected = False
        if self._ping_task is not None:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
            self._ping_task = None

        if self._ws is not None:
            await self._ws.close()
            self._ws = None

        log.info("zerodha_ws_disconnected", symbol=self.symbol)

    async def run(self, callbacks: Dict[str, Any]) -> None:
        """Receive binary packets and dispatch parsed data to callbacks."""
        on_tick = callbacks.get("on_tick")
        on_orderbook = callbacks.get("on_orderbook")

        while self._connected and self._ws is not None:
            try:
                message = await self._ws.recv()

                if isinstance(message, bytes) and len(message) > 2:
                    packets = self._parse_binary(message)
                    for packet in packets:
                        self._record_message()
                        if on_tick and "last_price" in packet:
                            await on_tick(packet)
                        if on_orderbook and "depth" in packet:
                            await on_orderbook(packet)
                elif isinstance(message, str):
                    # Text frames are typically heartbeat/control messages
                    log.debug("zerodha_text_message", data=message[:200])

            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("zerodha_ws_recv_error", symbol=self.symbol)
                break

    def _parse_binary(self, data: bytes) -> List[Dict[str, Any]]:
        """Parse Kite binary protocol packets.

        Kite sends a 2-byte header with the number of packets, followed by
        each packet prefixed with a 2-byte length.
        """
        packets: List[Dict[str, Any]] = []
        if len(data) < 2:
            return packets

        num_packets = struct.unpack(">H", data[:2])[0]
        offset = 2

        for _ in range(num_packets):
            if offset + 2 > len(data):
                break
            pkt_len = struct.unpack(">H", data[offset : offset + 2])[0]
            offset += 2

            if offset + pkt_len > len(data):
                break

            pkt_data = data[offset : offset + pkt_len]
            offset += pkt_len

            parsed = self._parse_packet(pkt_data)
            if parsed:
                packets.append(parsed)

        return packets

    def _parse_packet(self, pkt: bytes) -> Optional[Dict[str, Any]]:
        """Parse a single binary packet depending on its size (mode)."""
        if len(pkt) < 8:
            return None

        token = struct.unpack(">I", pkt[0:4])[0]
        last_price = struct.unpack(">i", pkt[4:8])[0] / 100.0

        result: Dict[str, Any] = {
            "instrument_token": token,
            "last_price": last_price,
            "symbol": self.symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Quote mode: OHLC + volume
        if len(pkt) >= 44:
            result.update({
                "high": struct.unpack(">i", pkt[8:12])[0] / 100.0,
                "low": struct.unpack(">i", pkt[12:16])[0] / 100.0,
                "open": struct.unpack(">i", pkt[16:20])[0] / 100.0,
                "close": struct.unpack(">i", pkt[20:24])[0] / 100.0,
                "volume": struct.unpack(">I", pkt[24:28])[0],
            })

        # Full mode: 5-level depth
        if len(pkt) >= 184:
            depth = {"buy": [], "sell": []}
            depth_offset = 28
            for side_key in ("buy", "sell"):
                for _ in range(5):
                    if depth_offset + 12 <= len(pkt):
                        qty = struct.unpack(">I", pkt[depth_offset : depth_offset + 4])[0]
                        price = struct.unpack(">i", pkt[depth_offset + 4 : depth_offset + 8])[0] / 100.0
                        orders = struct.unpack(">H", pkt[depth_offset + 8 : depth_offset + 10])[0]
                        depth[side_key].append({
                            "quantity": qty,
                            "price": price,
                            "orders": orders,
                        })
                        depth_offset += 12
            result["depth"] = depth

        return result

    async def _send_subscribe(self) -> None:
        """Send subscription message for the instrument token."""
        if self._ws is None:
            return
        msg = json.dumps({
            "a": "subscribe",
            "v": [self._instrument_token],
        })
        await self._ws.send(msg)

        # Set mode
        mode_msg = json.dumps({
            "a": "mode",
            "v": [self._mode, [self._instrument_token]],
        })
        await self._ws.send(mode_msg)

    async def _ping_loop(self) -> None:
        """Send periodic pings to keep the connection alive."""
        while self._connected:
            try:
                await asyncio.sleep(15)
                if self._ws is not None:
                    await self._ws.ping()
            except asyncio.CancelledError:
                break
            except Exception:
                log.warning("zerodha_ping_failed", symbol=self.symbol)
                break


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Binance WebSocket Feed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BinanceWebSocketFeed(BaseWebSocketFeed):
    """WebSocket feed for Binance exchange.

    Supports individual and combined streams:
    - ``{symbol}@trade`` for individual trades.
    - ``{symbol}@depth`` for order book updates.
    - ``{symbol}@kline_{interval}`` for kline/candlestick data.
    - User data stream for order fills.

    Args:
        symbol: Trading pair (e.g. ``btcusdt``).
        event_bus: Central event bus.
        streams: List of stream names to subscribe (e.g. ``["trade", "depth"]``).
        user_data_stream_key: Listen key for user data stream (fills).
    """

    BASE_URL = "wss://stream.binance.com:9443"

    def __init__(
        self,
        symbol: str,
        event_bus: EventBus,
        streams: Optional[List[str]] = None,
        user_data_stream_key: Optional[str] = None,
    ) -> None:
        super().__init__(symbol, event_bus)
        self._streams = streams or ["trade", "depth@100ms"]
        self._user_data_key = user_data_stream_key
        self._ws: Any = None
        self._user_ws: Any = None
        self._user_ws_task: Optional[asyncio.Task[None]] = None

    async def connect(self) -> None:
        """Connect to Binance combined WebSocket stream."""
        try:
            import websockets  # type: ignore[import-untyped]

            sym_lower = self.symbol.lower()
            stream_names = [f"{sym_lower}@{s}" for s in self._streams]
            combined = "/".join(stream_names)
            url = f"{self.BASE_URL}/stream?streams={combined}"

            self._ws = await websockets.connect(url)
            self._connected = True
            self._connect_time = time.time()

            # Optionally connect user data stream for fills
            if self._user_data_key:
                user_url = f"{self.BASE_URL}/ws/{self._user_data_key}"
                self._user_ws = await websockets.connect(user_url)
                self._user_ws_task = asyncio.create_task(
                    self._user_data_loop(), name=f"binance-user-{self.symbol}"
                )

            log.info(
                "binance_ws_connected",
                symbol=self.symbol,
                streams=stream_names,
            )
        except ImportError:
            log.error("websockets_not_installed", msg="pip install websockets")
            raise
        except Exception:
            log.exception("binance_ws_connect_failed", symbol=self.symbol)
            raise

    async def disconnect(self) -> None:
        """Disconnect from Binance WebSocket."""
        self._connected = False

        if self._user_ws_task is not None:
            self._user_ws_task.cancel()
            try:
                await self._user_ws_task
            except asyncio.CancelledError:
                pass
            self._user_ws_task = None

        if self._user_ws is not None:
            await self._user_ws.close()
            self._user_ws = None

        if self._ws is not None:
            await self._ws.close()
            self._ws = None

        log.info("binance_ws_disconnected", symbol=self.symbol)

    async def run(self, callbacks: Dict[str, Any]) -> None:
        """Receive JSON messages from the combined stream and dispatch."""
        on_tick = callbacks.get("on_tick")
        on_orderbook = callbacks.get("on_orderbook")

        while self._connected and self._ws is not None:
            try:
                raw = await self._ws.recv()

                msg = json.loads(raw)
                stream = msg.get("stream", "")
                data = msg.get("data", {})

                if "@trade" in stream:
                    tick = self._parse_trade(data)
                    self._record_message()
                    if on_tick:
                        await on_tick(tick)

                elif "@depth" in stream:
                    book = self._parse_depth(data)
                    self._record_message()
                    if on_orderbook:
                        await on_orderbook(book)

                elif "@kline" in stream:
                    kline = self._parse_kline(data)
                    self._record_message()
                    if on_tick:
                        await on_tick(kline)

            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("binance_ws_recv_error", symbol=self.symbol)
                break

    def _parse_trade(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a Binance trade event."""
        return {
            "symbol": self.symbol,
            "last_price": float(data.get("p", 0)),
            "quantity": float(data.get("q", 0)),
            "trade_time": data.get("T", 0),
            "buyer_maker": data.get("m", False),
            "trade_id": data.get("t", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _parse_depth(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a Binance depth update."""
        bids = [
            {"price": float(b[0]), "quantity": float(b[1])}
            for b in data.get("b", [])
        ]
        asks = [
            {"price": float(a[0]), "quantity": float(a[1])}
            for a in data.get("a", [])
        ]
        return {
            "symbol": self.symbol,
            "bids": bids,
            "asks": asks,
            "last_update_id": data.get("u", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _parse_kline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a Binance kline/candlestick event."""
        k = data.get("k", {})
        return {
            "symbol": self.symbol,
            "open": float(k.get("o", 0)),
            "high": float(k.get("h", 0)),
            "low": float(k.get("l", 0)),
            "close": float(k.get("c", 0)),
            "last_price": float(k.get("c", 0)),
            "volume": float(k.get("v", 0)),
            "interval": k.get("i", ""),
            "is_closed": k.get("x", False),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _user_data_loop(self) -> None:
        """Listen to user data stream for order fills."""
        while self._connected and self._user_ws is not None:
            try:
                raw = await self._user_ws.recv()
                msg = json.loads(raw)
                event_type = msg.get("e", "")

                if event_type == "executionReport":
                    fill_data = {
                        "symbol": msg.get("s", ""),
                        "order_id": msg.get("i", ""),
                        "side": msg.get("S", ""),
                        "order_type": msg.get("o", ""),
                        "status": msg.get("X", ""),
                        "price": float(msg.get("p", 0)),
                        "quantity": float(msg.get("q", 0)),
                        "filled_quantity": float(msg.get("z", 0)),
                        "commission": float(msg.get("n", 0)),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await self._event_bus.publish(
                        Event(
                            event_type=EventType.FILL,
                            timestamp=datetime.now(timezone.utc),
                            symbol=self.symbol,
                            data=fill_data,
                            source="binance_user_data",
                        )
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("binance_user_data_error", symbol=self.symbol)
                break


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generic REST Polling Feed (Groww, IndMoney, etc.)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class GenericWebSocketFeed(BaseWebSocketFeed):
    """Fallback feed for brokers without native WebSocket support.

    Polls a REST API at a configurable interval and converts the responses
    into the standard event stream format.

    Args:
        symbol: Trading symbol.
        event_bus: Central event bus.
        poll_interval: Seconds between REST polls (1-5 recommended).
        rest_client: Optional async callable ``(symbol) -> dict`` for fetching data.
    """

    def __init__(
        self,
        symbol: str,
        event_bus: EventBus,
        poll_interval: float = 2.0,
        rest_client: Optional[Callable[..., Coroutine[Any, Any, Dict[str, Any]]]] = None,
    ) -> None:
        super().__init__(symbol, event_bus)
        self._poll_interval = max(1.0, min(poll_interval, 10.0))
        self._rest_client = rest_client

    async def connect(self) -> None:
        """Mark feed as connected (no real WebSocket connection)."""
        self._connected = True
        self._connect_time = time.time()
        log.info(
            "generic_feed_connected",
            symbol=self.symbol,
            poll_interval=self._poll_interval,
        )

    async def disconnect(self) -> None:
        """Mark feed as disconnected."""
        self._connected = False
        log.info("generic_feed_disconnected", symbol=self.symbol)

    async def run(self, callbacks: Dict[str, Any]) -> None:
        """Poll the REST API and invoke callbacks with the results."""
        on_tick = callbacks.get("on_tick")

        while self._connected:
            try:
                poll_start = time.monotonic()

                if self._rest_client is not None:
                    data = await self._rest_client(self.symbol)
                else:
                    # Stub: produce an empty tick so the pipeline works
                    data = {
                        "symbol": self.symbol,
                        "last_price": 0.0,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": "generic_poll",
                    }

                latency = time.monotonic() - poll_start
                self._record_message(latency)

                if on_tick and data:
                    await on_tick(data)

                await asyncio.sleep(self._poll_interval)

            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("generic_feed_poll_error", symbol=self.symbol)
                await asyncio.sleep(self._poll_interval)
