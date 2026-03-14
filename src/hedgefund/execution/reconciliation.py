"""Position reconciliation engine.

Compares local state with broker state, detects fills, updates positions,
and handles partial fills.  Designed to run periodically (e.g. every few
seconds) to keep the local book in sync.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

import structlog

from hedgefund.types import Order, OrderStatus, Position, PortfolioSnapshot
from hedgefund.execution.base import Broker

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ReconciliationDiff:
    """Delta between local and broker state."""

    new_fills: list[Order]
    partial_fills: list[Order]
    position_mismatches: list[PositionMismatch]
    cash_difference: float
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class PositionMismatch:
    """Describes a discrepancy for a single position."""

    symbol: str
    local_quantity: int
    broker_quantity: int
    local_avg_entry: float
    broker_avg_entry: float

    @property
    def quantity_diff(self) -> int:
        return self.broker_quantity - self.local_quantity


class ReconciliationEngine:
    """Synchronises local state with broker state.

    Parameters
    ----------
    broker:
        Broker instance to query for authoritative state.
    reconcile_interval:
        Minimum seconds between automatic reconciliation cycles.
    tolerance_cash:
        Acceptable cash rounding difference (in dollars).
    """

    def __init__(
        self,
        broker: Broker,
        *,
        reconcile_interval: float = 5.0,
        tolerance_cash: float = 0.01,
    ) -> None:
        self._broker = broker
        self._interval = reconcile_interval
        self._tolerance = tolerance_cash

        # Local books
        self._local_positions: dict[str, Position] = {}
        self._local_cash: float = 0.0
        self._pending_orders: dict[str, Order] = {}
        self._last_reconcile: Optional[datetime] = None
        self._running: bool = False

    # ── Public API ────────────────────────────────────────────────────────

    def track_order(self, order: Order) -> None:
        """Register an order so reconciliation can detect its fill."""
        self._pending_orders[order.order_id] = order
        logger.debug("reconciliation.tracking_order", order_id=order.order_id)

    def update_local_positions(self, positions: Sequence[Position]) -> None:
        """Bulk-set the local position book (e.g. after a restart)."""
        self._local_positions = {p.contract.osi_symbol: p for p in positions}
        logger.info(
            "reconciliation.local_positions_set",
            count=len(self._local_positions),
        )

    def update_local_cash(self, cash: float) -> None:
        """Set the local cash balance."""
        self._local_cash = cash

    async def reconcile(self) -> ReconciliationDiff:
        """Perform a single reconciliation cycle.

        Fetches broker state, diffs against local books, and returns
        all detected changes.
        """
        ts = datetime.utcnow()
        broker_portfolio = await self._broker.get_portfolio()
        broker_positions = await self._broker.get_positions()

        new_fills = self._detect_fills(broker_portfolio)
        partial_fills = self._detect_partial_fills()
        mismatches = self._diff_positions(broker_positions)
        cash_diff = broker_portfolio.cash - self._local_cash

        # Apply broker state as truth
        self._apply_broker_state(broker_positions, broker_portfolio.cash)
        self._last_reconcile = ts

        diff = ReconciliationDiff(
            new_fills=new_fills,
            partial_fills=partial_fills,
            position_mismatches=mismatches,
            cash_difference=cash_diff,
            timestamp=ts,
        )

        if new_fills or partial_fills or mismatches:
            logger.info(
                "reconciliation.diff_detected",
                new_fills=len(new_fills),
                partial_fills=len(partial_fills),
                mismatches=len(mismatches),
                cash_diff=cash_diff,
            )
        else:
            logger.debug("reconciliation.in_sync")

        return diff

    async def run_loop(self) -> None:
        """Continuously reconcile at the configured interval.

        Runs until :meth:`stop` is called.
        """
        self._running = True
        logger.info("reconciliation.loop_started", interval=self._interval)

        while self._running:
            try:
                await self.reconcile()
            except Exception:
                logger.exception("reconciliation.loop_error")
            await asyncio.sleep(self._interval)

        logger.info("reconciliation.loop_stopped")

    def stop(self) -> None:
        """Signal the reconciliation loop to stop."""
        self._running = False

    # ── Fill detection ────────────────────────────────────────────────────

    def _detect_fills(self, broker_portfolio: PortfolioSnapshot) -> list[Order]:
        """Detect orders that have been fully filled since last check."""
        filled: list[Order] = []
        to_remove: list[str] = []

        for oid, order in self._pending_orders.items():
            if order.status == OrderStatus.FILLED:
                filled.append(order)
                to_remove.append(oid)

        for oid in to_remove:
            self._pending_orders.pop(oid, None)

        return filled

    def _detect_partial_fills(self) -> list[Order]:
        """Detect orders with partial fills."""
        partials: list[Order] = []
        for order in self._pending_orders.values():
            if order.status == OrderStatus.PARTIAL and order.filled_quantity > 0:
                partials.append(order)
        return partials

    # ── Position diffing ──────────────────────────────────────────────────

    def _diff_positions(
        self,
        broker_positions: Sequence[Position],
    ) -> list[PositionMismatch]:
        """Compare local positions with broker positions."""
        broker_map = {p.contract.osi_symbol: p for p in broker_positions}
        mismatches: list[PositionMismatch] = []

        # Check all symbols in either book
        all_symbols = set(self._local_positions) | set(broker_map)

        for sym in all_symbols:
            local = self._local_positions.get(sym)
            broker = broker_map.get(sym)

            local_qty = local.quantity if local else 0
            broker_qty = broker.quantity if broker else 0
            local_entry = local.avg_entry if local else 0.0
            broker_entry = broker.avg_entry if broker else 0.0

            if local_qty != broker_qty or abs(local_entry - broker_entry) > self._tolerance:
                mismatches.append(
                    PositionMismatch(
                        symbol=sym,
                        local_quantity=local_qty,
                        broker_quantity=broker_qty,
                        local_avg_entry=local_entry,
                        broker_avg_entry=broker_entry,
                    )
                )

        return mismatches

    # ── State application ─────────────────────────────────────────────────

    def _apply_broker_state(
        self,
        broker_positions: Sequence[Position],
        broker_cash: float,
    ) -> None:
        """Overwrite local state with broker's authoritative view."""
        self._local_positions = {p.contract.osi_symbol: p for p in broker_positions}
        self._local_cash = broker_cash
