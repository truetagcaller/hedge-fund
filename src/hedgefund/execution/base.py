"""Abstract broker interface.

Every broker implementation (paper, Alpaca, IBKR, etc.) must satisfy this
contract so the rest of the system can remain broker-agnostic.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import Optional

from hedgefund.types import (
    Order,
    OptionQuote,
    Position,
    PortfolioSnapshot,
)


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

    # ── Balance / market data ──────────────────────────────────────────

    async def get_balance(self) -> dict[str, float]:
        """Return available balance as ``{cash, margin_available, margin_used}``.

        Default implementation derives values from :meth:`get_portfolio`.
        Broker adapters may override for a more efficient API call.
        """
        snap = await self.get_portfolio()
        return {
            "cash": snap.cash,
            "margin_available": snap.cash,
            "margin_used": snap.net_liquidation - snap.cash,
        }

    async def get_market_data(self, symbol: str) -> dict[str, object]:
        """Return live quote data for *symbol*.

        Broker adapters that support market-data feeds should override this.
        The default implementation returns an empty dict.
        """
        return {}

    async def get_options_chain(
        self, underlying: str, expiry: str | None = None,
    ) -> list[OptionQuote]:
        """Return option chain for *underlying* (optionally filtered by *expiry*).

        Broker adapters that support options should override this.
        The default implementation returns an empty list.
        """
        return []

    async def get_orders(self, status_filter: Optional[str] = None) -> list[Order]:
        """Return orders, optionally filtered by status.

        Broker adapters should override for real order listing.
        Default returns an empty list.
        """
        return []

    async def modify_order(
        self, order_id: str, *, quantity: int | None = None,
        price: float | None = None,
    ) -> Order:
        """Modify an open order's quantity or price.

        Not all brokers support modification — default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support order modification."
        )

    # ── Streaming ─────────────────────────────────────────────────────────

    @abc.abstractmethod
    async def stream_fills(self) -> AsyncIterator[Order]:
        """Yield orders as they are filled or partially filled.

        The iterator should run indefinitely until the connection is closed
        or ``disconnect`` is called.
        """
        # Make this an async generator so subclasses can ``yield`` directly.
        yield  # type: ignore[misc]
