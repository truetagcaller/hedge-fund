"""Paper trading broker for backtesting and simulation.

Provides a full :class:`Broker` implementation with:
    * Simulated fills (market orders at mid + slippage, limit orders when
      price crosses the limit).
    * Position tracking with running P&L.
    * Commission simulation.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import structlog

from hedgefund.types import (
    Greeks,
    OptionContract,
    Order,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    Position,
    Side,
)
from hedgefund.execution.base import Broker

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class _PendingOrder:
    """Internal representation of an in-flight limit/stop order."""

    order: Order
    submitted_at: datetime


class PaperBroker(Broker):
    """Simulated broker for backtesting and paper trading.

    Parameters
    ----------
    initial_cash:
        Starting account cash.
    slippage_bps:
        Slippage applied to market-order fills (in basis points).
    commission_per_contract:
        Flat commission per contract per fill.
    fill_ratio:
        Fraction of limit-order fills to simulate (1.0 = always fill when
        price crosses; <1.0 models partial-fill scenarios).
    """

    def __init__(
        self,
        *,
        initial_cash: float = 10_000_000.0,
        slippage_bps: int = 5,
        commission_per_contract: float = 0.65,
        fill_ratio: float = 1.0,
    ) -> None:
        self._initial_cash = initial_cash
        self._cash = initial_cash
        self._slippage_pct = slippage_bps / 10_000
        self._commission = commission_per_contract
        self._fill_ratio = fill_ratio

        # Position tracking: keyed by contract OSI symbol
        self._positions: dict[str, Position] = {}
        # Pending (working) orders
        self._pending: dict[str, _PendingOrder] = {}
        # All submitted orders (for audit trail)
        self._order_history: list[Order] = []
        # Fill queue consumed by stream_fills
        self._fill_queue: asyncio.Queue[Order] = asyncio.Queue()
        # Realised P&L
        self._realized_pnl: float = 0.0
        # Connected flag
        self._connected: bool = False

    # ── Broker lifecycle ──────────────────────────────────────────────────

    async def connect(self) -> None:
        self._connected = True
        logger.info("paper_broker.connected", cash=self._cash)

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("paper_broker.disconnected")

    # ── Order submission ──────────────────────────────────────────────────

    async def submit_order(self, order: Order) -> Order:
        """Submit an order for simulated execution.

        Market orders are filled immediately.  Limit and stop orders are
        queued and filled when :meth:`process_tick` is called with a
        qualifying price.
        """
        self._check_connected()
        order.status = OrderStatus.SUBMITTED
        self._order_history.append(order)

        if order.order_type == OrderType.MARKET:
            return await self._fill_market(order)

        # Queue limit / stop / stop-limit orders
        self._pending[order.order_id] = _PendingOrder(
            order=order,
            submitted_at=datetime.utcnow(),
        )
        logger.info(
            "paper_broker.order_queued",
            order_id=order.order_id,
            type=order.order_type.value,
            limit=order.limit_price,
            stop=order.stop_price,
        )
        return order

    async def cancel_order(self, order_id: str) -> Order:
        pending = self._pending.pop(order_id, None)
        if pending is None:
            logger.warning("paper_broker.cancel_not_found", order_id=order_id)
            # Return a best-effort lookup from history
            for o in reversed(self._order_history):
                if o.order_id == order_id:
                    return o
            raise KeyError(f"Unknown order {order_id}")

        pending.order.status = OrderStatus.CANCELLED
        logger.info("paper_broker.order_cancelled", order_id=order_id)
        return pending.order

    # ── Queries ───────────────────────────────────────────────────────────

    async def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    async def get_portfolio(self) -> PortfolioSnapshot:
        positions = list(self._positions.values())
        market_value = sum(p.market_value for p in positions)
        unrealized = sum(p.unrealized_pnl for p in positions)
        nlv = self._cash + market_value

        return PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            cash=self._cash,
            net_liquidation=nlv,
            positions=positions,
            total_delta=sum(p.greeks.delta * p.quantity * p.contract.multiplier for p in positions),
            total_gamma=sum(p.greeks.gamma * p.quantity * p.contract.multiplier for p in positions),
            total_theta=sum(p.greeks.theta * p.quantity * p.contract.multiplier for p in positions),
            total_vega=sum(p.greeks.vega * p.quantity * p.contract.multiplier for p in positions),
            daily_pnl=unrealized + self._realized_pnl,
            total_pnl=nlv - self._initial_cash,
        )

    async def stream_fills(self) -> AsyncIterator[Order]:
        """Yield filled orders as they happen."""
        while self._connected:
            try:
                order = await asyncio.wait_for(self._fill_queue.get(), timeout=1.0)
                yield order
            except asyncio.TimeoutError:
                continue

    # ── Tick processing (drives limit/stop fills) ─────────────────────────

    async def process_tick(
        self,
        contract: OptionContract,
        bid: float,
        ask: float,
        timestamp: datetime | None = None,
    ) -> list[Order]:
        """Simulate the passage of a market tick.

        Checks all pending orders against the new bid/ask to determine
        whether any should fill.  Also updates the current price on
        matching positions.

        Returns a list of orders that were filled on this tick.
        """
        ts = timestamp or datetime.utcnow()
        mid = (bid + ask) / 2.0
        filled: list[Order] = []

        # Update position mark
        key = contract.osi_symbol
        if key in self._positions:
            pos = self._positions[key]
            old_price = pos.current_price
            pos.current_price = mid
            pos.unrealized_pnl = (
                (mid - pos.avg_entry) * pos.quantity * contract.multiplier
            )

        # Check pending orders
        to_remove: list[str] = []
        for oid, pending in self._pending.items():
            o = pending.order
            if o.contract.osi_symbol != contract.osi_symbol:
                continue

            should_fill = False
            fill_price = mid

            if o.order_type == OrderType.LIMIT:
                if o.side == Side.BUY and ask <= (o.limit_price or 0):
                    should_fill = True
                    fill_price = o.limit_price or ask
                elif o.side == Side.SELL and bid >= (o.limit_price or 0):
                    should_fill = True
                    fill_price = o.limit_price or bid

            elif o.order_type == OrderType.STOP:
                if o.side == Side.BUY and ask >= (o.stop_price or 0):
                    should_fill = True
                    fill_price = ask * (1 + self._slippage_pct)
                elif o.side == Side.SELL and bid <= (o.stop_price or 0):
                    should_fill = True
                    fill_price = bid * (1 - self._slippage_pct)

            elif o.order_type == OrderType.STOP_LIMIT:
                # Stop triggers, then behaves as limit
                if o.side == Side.BUY and ask >= (o.stop_price or 0):
                    if ask <= (o.limit_price or 0):
                        should_fill = True
                        fill_price = o.limit_price or ask
                elif o.side == Side.SELL and bid <= (o.stop_price or 0):
                    if bid >= (o.limit_price or 0):
                        should_fill = True
                        fill_price = o.limit_price or bid

            if should_fill:
                await self._execute_fill(o, fill_price, ts)
                to_remove.append(oid)
                filled.append(o)

        for oid in to_remove:
            self._pending.pop(oid, None)

        return filled

    # ── Internal fill logic ───────────────────────────────────────────────

    async def _fill_market(self, order: Order) -> Order:
        """Immediately fill a market order at mid +/- slippage."""
        # For market orders we assume the signal's entry price as the
        # reference mid, since we may not have a live quote.
        mid = order.limit_price or 0.0
        if mid <= 0:
            # Fallback: use a nominal price (caller should set limit_price
            # to the mid for market orders in backtest mode).
            mid = 1.0

        if order.side == Side.BUY:
            fill_price = mid * (1 + self._slippage_pct)
        else:
            fill_price = mid * (1 - self._slippage_pct)

        fill_price = max(0.01, fill_price)
        await self._execute_fill(order, fill_price, datetime.utcnow())
        return order

    async def _execute_fill(
        self,
        order: Order,
        fill_price: float,
        ts: datetime,
    ) -> None:
        """Apply a fill to the order and update positions / cash."""
        commission = self._commission * order.quantity
        cost = fill_price * order.quantity * order.contract.multiplier

        if order.side == Side.BUY:
            self._cash -= cost + commission
        else:
            self._cash += cost - commission

        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.filled_at = ts
        order.status = OrderStatus.FILLED
        order.commission = commission

        self._update_position(order)

        logger.info(
            "paper_broker.fill",
            order_id=order.order_id,
            side=order.side.value,
            qty=order.quantity,
            price=fill_price,
            commission=commission,
            cash_remaining=self._cash,
        )

        await self._fill_queue.put(order)

    def _update_position(self, order: Order) -> None:
        """Update or create a position based on a filled order."""
        key = order.contract.osi_symbol
        existing = self._positions.get(key)

        signed_qty = order.quantity if order.side == Side.BUY else -order.quantity
        fill = order.filled_price or 0.0

        if existing is None:
            self._positions[key] = Position(
                contract=order.contract,
                quantity=signed_qty,
                avg_entry=fill,
                current_price=fill,
                greeks=Greeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0),
                unrealized_pnl=0.0,
            )
            return

        new_qty = existing.quantity + signed_qty

        if new_qty == 0:
            # Position closed
            realized = (fill - existing.avg_entry) * abs(signed_qty) * order.contract.multiplier
            if order.side == Side.SELL:
                self._realized_pnl += realized
            else:
                self._realized_pnl -= realized
            del self._positions[key]
            return

        # Position increased or partially reduced
        if (existing.quantity > 0 and signed_qty > 0) or (
            existing.quantity < 0 and signed_qty < 0
        ):
            # Adding to position: weighted average entry
            total_cost = existing.avg_entry * abs(existing.quantity) + fill * abs(signed_qty)
            existing.avg_entry = total_cost / abs(new_qty)
        else:
            # Partial close: realise P&L on the closed portion
            closed_qty = min(abs(existing.quantity), abs(signed_qty))
            realized = (fill - existing.avg_entry) * closed_qty * order.contract.multiplier
            if signed_qty < 0:
                self._realized_pnl += realized
            else:
                self._realized_pnl -= realized

        existing.quantity = new_qty
        existing.unrealized_pnl = (
            (existing.current_price - existing.avg_entry)
            * new_qty
            * order.contract.multiplier
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _check_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("PaperBroker is not connected. Call connect() first.")

    # ── Inspection (useful for tests) ─────────────────────────────────────

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def pending_orders(self) -> dict[str, _PendingOrder]:
        return dict(self._pending)

    @property
    def order_history(self) -> list[Order]:
        return list(self._order_history)
