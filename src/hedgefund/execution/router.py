"""Smart order router.

Routes orders to the appropriate broker, handles retries, and implements
TWAP / VWAP execution algorithms for large orders.  Pre-trade risk checks
run before any order reaches the broker.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Sequence

import structlog

from hedgefund.types import Order, OrderStatus, OrderType, Side, TradeSignal, PortfolioSnapshot
from hedgefund.execution.base import Broker
from hedgefund.risk.validators import PreTradeValidator, ValidationResult

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RoutingResult:
    """Outcome of a routed order or execution algorithm."""

    success: bool
    orders: list[Order]
    message: str = ""
    total_filled: int = 0
    avg_fill_price: float = 0.0


class SmartOrderRouter:
    """Routes and executes orders with retry logic and algorithmic execution.

    Parameters
    ----------
    broker:
        The broker to route orders to.
    validator:
        Optional pre-trade validator.  If provided, every order is checked
        before submission.
    max_retries:
        Number of retries on transient failures.
    retry_delay:
        Base delay between retries (exponential back-off is applied).
    """

    def __init__(
        self,
        broker: Broker,
        validator: PreTradeValidator | None = None,
        *,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self._broker = broker
        self._validator = validator
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    # ── Single order submission ───────────────────────────────────────────

    async def route_order(
        self,
        order: Order,
        signal: TradeSignal | None = None,
        portfolio: PortfolioSnapshot | None = None,
    ) -> RoutingResult:
        """Route a single order through pre-trade checks and submit.

        If a ``signal`` and ``portfolio`` are provided and a validator is
        configured, pre-trade checks run first.
        """
        # Pre-trade risk check
        if self._validator and signal and portfolio:
            result = await self._validator.validate(order, signal, portfolio)
            if not result.approved:
                logger.warning(
                    "router.order_rejected_by_validator",
                    order_id=order.order_id,
                    check=result.failed_check,
                    message=result.message,
                )
                order.status = OrderStatus.REJECTED
                return RoutingResult(
                    success=False,
                    orders=[order],
                    message=f"Pre-trade check failed: {result.message}",
                )

        # Submit with retries
        submitted = await self._submit_with_retry(order)
        if submitted.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            return RoutingResult(
                success=True,
                orders=[submitted],
                total_filled=submitted.filled_quantity,
                avg_fill_price=submitted.filled_price or 0.0,
            )

        if submitted.status == OrderStatus.REJECTED:
            return RoutingResult(
                success=False,
                orders=[submitted],
                message="Order rejected by broker",
            )

        # SUBMITTED but not yet filled (limit/stop orders)
        return RoutingResult(
            success=True,
            orders=[submitted],
            message="Order submitted, awaiting fill",
        )

    # ── Multi-order batch ─────────────────────────────────────────────────

    async def route_orders(
        self,
        orders: Sequence[Order],
        signal: TradeSignal | None = None,
        portfolio: PortfolioSnapshot | None = None,
    ) -> list[RoutingResult]:
        """Submit multiple orders (e.g. legs of a spread)."""
        results = []
        for order in orders:
            result = await self.route_order(order, signal, portfolio)
            results.append(result)
            if not result.success:
                logger.warning(
                    "router.batch_leg_failed",
                    order_id=order.order_id,
                    message=result.message,
                )
                # Cancel previously-submitted legs on failure
                await self._cancel_filled_legs(results)
                break
        return results

    # ── TWAP execution ────────────────────────────────────────────────────

    async def execute_twap(
        self,
        order: Order,
        duration: timedelta,
        num_slices: int = 5,
        signal: TradeSignal | None = None,
        portfolio: PortfolioSnapshot | None = None,
    ) -> RoutingResult:
        """Time-Weighted Average Price execution.

        Splits *order* into ``num_slices`` child orders submitted at
        equal time intervals over *duration*.
        """
        if num_slices < 1:
            num_slices = 1

        slice_qty = order.quantity // num_slices
        remainder = order.quantity % num_slices
        interval = duration.total_seconds() / num_slices

        filled_orders: list[Order] = []
        total_filled = 0
        total_cost = 0.0

        for i in range(num_slices):
            qty = slice_qty + (1 if i < remainder else 0)
            if qty <= 0:
                continue

            child = Order(
                order_id=Order.generate_id(),
                signal_id=order.signal_id,
                contract=order.contract,
                side=order.side,
                order_type=order.order_type,
                quantity=qty,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
            )

            result = await self.route_order(child, signal, portfolio)
            filled_orders.extend(result.orders)
            total_filled += result.total_filled
            total_cost += result.avg_fill_price * result.total_filled

            if i < num_slices - 1:
                await asyncio.sleep(interval)

        avg_price = total_cost / total_filled if total_filled > 0 else 0.0

        logger.info(
            "router.twap_complete",
            order_id=order.order_id,
            slices=num_slices,
            total_filled=total_filled,
            avg_price=avg_price,
        )
        return RoutingResult(
            success=total_filled > 0,
            orders=filled_orders,
            total_filled=total_filled,
            avg_fill_price=avg_price,
            message=f"TWAP: {total_filled}/{order.quantity} filled across {num_slices} slices",
        )

    # ── VWAP execution ────────────────────────────────────────────────────

    async def execute_vwap(
        self,
        order: Order,
        volume_profile: list[float],
        signal: TradeSignal | None = None,
        portfolio: PortfolioSnapshot | None = None,
    ) -> RoutingResult:
        """Volume-Weighted Average Price execution.

        Distributes the order according to *volume_profile*, a list of
        relative volume weights for each time bucket (need not sum to 1).
        """
        total_weight = sum(volume_profile)
        if total_weight <= 0:
            return RoutingResult(
                success=False,
                orders=[order],
                message="Invalid volume profile (zero total weight)",
            )

        filled_orders: list[Order] = []
        total_filled = 0
        total_cost = 0.0
        remaining = order.quantity

        for weight in volume_profile:
            fraction = weight / total_weight
            qty = max(1, round(order.quantity * fraction))
            qty = min(qty, remaining)
            if qty <= 0:
                continue

            child = Order(
                order_id=Order.generate_id(),
                signal_id=order.signal_id,
                contract=order.contract,
                side=order.side,
                order_type=order.order_type,
                quantity=qty,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
            )

            result = await self.route_order(child, signal, portfolio)
            filled_orders.extend(result.orders)
            total_filled += result.total_filled
            total_cost += result.avg_fill_price * result.total_filled
            remaining -= qty

        avg_price = total_cost / total_filled if total_filled > 0 else 0.0

        logger.info(
            "router.vwap_complete",
            order_id=order.order_id,
            buckets=len(volume_profile),
            total_filled=total_filled,
            avg_price=avg_price,
        )
        return RoutingResult(
            success=total_filled > 0,
            orders=filled_orders,
            total_filled=total_filled,
            avg_fill_price=avg_price,
            message=f"VWAP: {total_filled}/{order.quantity} filled across {len(volume_profile)} buckets",
        )

    # ── Retry logic ───────────────────────────────────────────────────────

    async def _submit_with_retry(self, order: Order) -> Order:
        """Submit with exponential-backoff retries on transient errors."""
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return await self._broker.submit_order(order)
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        "router.retry",
                        order_id=order.order_id,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)

        logger.error(
            "router.retries_exhausted",
            order_id=order.order_id,
            error=str(last_exc),
        )
        order.status = OrderStatus.REJECTED
        return order

    # ── Cleanup helpers ───────────────────────────────────────────────────

    async def _cancel_filled_legs(self, results: list[RoutingResult]) -> None:
        """Best-effort cancel of previously submitted legs on batch failure."""
        for result in results:
            for o in result.orders:
                if o.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
                    try:
                        await self._broker.cancel_order(o.order_id)
                    except Exception:
                        logger.exception(
                            "router.cancel_leg_failed",
                            order_id=o.order_id,
                        )
