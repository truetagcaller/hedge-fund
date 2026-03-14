"""Abstract broker interface.

Every broker implementation (paper, Alpaca, IBKR, etc.) must satisfy this
contract so the rest of the system can remain broker-agnostic.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import Optional, Sequence

from hedgefund.types import Order, OrderStatus, Position, PortfolioSnapshot


class Broker(abc.ABC):
    """Async broker interface.

    Lifecycle: ``connect`` -> use -> ``disconnect``.
    Implementations may also be used as async context managers.
    """

    # ── Connection lifecycle ──────────────────────────────────────────────

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish a connection to the broker."""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close the broker connection."""

    async def __aenter__(self) -> Broker:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        await self.disconnect()

    # ── Order management ──────────────────────────────────────────────────

    @abc.abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """Submit *order* to the exchange.

        Returns the order with updated ``status`` (at minimum ``SUBMITTED``).
        Raises :class:`~hedgefund.exceptions.OrderRejectedError` on failure.
        """

    @abc.abstractmethod
    async def cancel_order(self, order_id: str) -> Order:
        """Request cancellation of an open order.

        Returns the order with ``status`` set to ``CANCELLED`` (or the
        current status if cancellation was not possible).
        """

    # ── Position / portfolio queries ──────────────────────────────────────

    @abc.abstractmethod
    async def get_positions(self) -> list[Position]:
        """Return all currently open positions."""

    @abc.abstractmethod
    async def get_portfolio(self) -> PortfolioSnapshot:
        """Return a point-in-time snapshot of the portfolio."""

    # ── Streaming ─────────────────────────────────────────────────────────

    @abc.abstractmethod
    async def stream_fills(self) -> AsyncIterator[Order]:
        """Yield orders as they are filled or partially filled.

        The iterator should run indefinitely until the connection is closed
        or ``disconnect`` is called.
        """
        # Make this an async generator so subclasses can ``yield`` directly.
        yield  # type: ignore[misc]
