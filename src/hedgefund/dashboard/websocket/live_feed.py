"""WebSocket connection manager for real-time dashboard updates.

Broadcasts portfolio snapshots, P&L, positions, signals, and risk metrics
to all connected clients. Implements heartbeat pings and graceful
disconnect handling.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from typing import Any, Dict, Set

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from hedgefund.types import PortfolioSnapshot, TradeSignal

log = structlog.get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections with heartbeat and broadcast capabilities."""

    def __init__(self, heartbeat_interval: int = 15) -> None:
        self._connections: Set[WebSocket] = set()
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    async def start(self) -> None:
        """Start the heartbeat background loop."""
        self._running = True
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="ws-heartbeat"
        )
        log.info("ws_manager_started", heartbeat_interval=self._heartbeat_interval)

    async def shutdown(self) -> None:
        """Stop heartbeat and close all connections."""
        self._running = False

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Close every active connection
        for ws in list(self._connections):
            await self._safe_close(ws)
        self._connections.clear()

        log.info("ws_manager_shutdown")

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self._connections.add(websocket)
        log.info("ws_client_connected", total=self.connection_count)

    def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a disconnected client."""
        self._connections.discard(websocket)
        log.info("ws_client_disconnected", total=self.connection_count)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Send a JSON message to every connected client.

        Disconnected or errored clients are removed silently.
        """
        if not self._connections:
            return

        stale: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json(message)
                else:
                    stale.append(ws)
            except Exception:
                stale.append(ws)

        for ws in stale:
            self._connections.discard(ws)

    # ── Typed broadcast helpers ──────────────────────────────────────────

    async def broadcast_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        """Push a portfolio snapshot to all clients."""
        positions = []
        for p in snapshot.positions:
            pos_dict = asdict(p)
            pos_dict["market_value"] = p.market_value
            pos_dict["notional_value"] = p.notional_value
            positions.append(pos_dict)

        await self.broadcast({
            "type": "portfolio",
            "data": {
                "timestamp": snapshot.timestamp.isoformat(),
                "cash": snapshot.cash,
                "net_liquidation": snapshot.net_liquidation,
                "total_market_value": snapshot.total_market_value,
                "position_count": snapshot.position_count,
                "daily_pnl": snapshot.daily_pnl,
                "total_pnl": snapshot.total_pnl,
                "drawdown_pct": snapshot.drawdown_pct,
                "high_water_mark": snapshot.high_water_mark,
                "positions": positions,
            },
        })

    async def broadcast_risk(self, snapshot: PortfolioSnapshot) -> None:
        """Push risk metrics to all clients."""
        await self.broadcast({
            "type": "risk",
            "data": {
                "timestamp": snapshot.timestamp.isoformat(),
                "total_delta": snapshot.total_delta,
                "total_gamma": snapshot.total_gamma,
                "total_theta": snapshot.total_theta,
                "total_vega": snapshot.total_vega,
                "drawdown_pct": snapshot.drawdown_pct,
                "daily_pnl": snapshot.daily_pnl,
            },
        })

    async def broadcast_signals(self, signals: list[TradeSignal]) -> None:
        """Push active signals to all clients."""
        await self.broadcast({
            "type": "signals",
            "data": [
                {
                    "signal_id": s.signal_id,
                    "timestamp": s.timestamp.isoformat(),
                    "underlying": s.underlying,
                    "action": s.action.value,
                    "direction": s.direction.value,
                    "confidence": s.confidence,
                    "strategy_name": s.strategy_name,
                    "entry_price": s.entry_price,
                    "stop_loss": s.stop_loss,
                    "target_price": s.target_price,
                    "risk_reward_ratio": s.risk_reward_ratio,
                    "reasoning": s.reasoning,
                }
                for s in signals
            ],
        })

    async def broadcast_trade(self, trade: Dict[str, Any]) -> None:
        """Push a new trade execution event."""
        await self.broadcast({"type": "trade", "data": trade})

    async def broadcast_news(self, news_items: list[Dict[str, Any]]) -> None:
        """Push news feed updates to all clients."""
        await self.broadcast({"type": "news", "data": news_items})

    async def broadcast_sentiment(self, sentiments: Dict[str, Any]) -> None:
        """Push sentiment data to all clients."""
        await self.broadcast({"type": "sentiment", "data": sentiments})

    async def broadcast_order_book(self, symbol: str, book: Dict[str, Any]) -> None:
        """Push order book snapshot to all clients."""
        await self.broadcast({"type": "order_book", "data": {"symbol": symbol, **book}})

    async def broadcast_smart_money(self, data: Dict[str, Any]) -> None:
        """Push smart money detection signals to all clients."""
        await self.broadcast({"type": "smart_money", "data": data})

    async def broadcast_engine_status(self, status: Dict[str, Any]) -> None:
        """Push AI engine status to all clients."""
        await self.broadcast({"type": "engine_status", "data": status})

    async def broadcast_decision(self, decision: Dict[str, Any]) -> None:
        """Push AI trading decision to all clients."""
        await self.broadcast({"type": "decision", "data": decision})

    # ── Internal ─────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Periodically send pings to detect stale connections."""
        while self._running:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                if not self._connections:
                    continue

                stale: list[WebSocket] = []
                for ws in list(self._connections):
                    try:
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json({
                                "type": "heartbeat",
                                "ts": time.time(),
                            })
                        else:
                            stale.append(ws)
                    except Exception:
                        stale.append(ws)

                for ws in stale:
                    self._connections.discard(ws)

                if stale:
                    log.debug("ws_stale_connections_removed", count=len(stale))
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("ws_heartbeat_error")

    async def _safe_close(self, ws: WebSocket) -> None:
        """Close a WebSocket ignoring errors."""
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close()
        except Exception:  # noqa: S110
                log.debug("unexpected_error", exc_info=True)


async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket route handler -- delegates to the app-level ConnectionManager."""
    manager: ConnectionManager = websocket.app.state.ws_manager
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()

            # Clients can send subscription preferences or pong frames
            msg_type = data.get("type", "")
            if msg_type == "pong":
                continue
            elif msg_type == "subscribe":
                # Future: per-client channel subscriptions
                log.debug("ws_subscribe_request", channels=data.get("channels"))
            else:
                log.debug("ws_unknown_message", data=data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        log.exception("ws_handler_error")
        manager.disconnect(websocket)
