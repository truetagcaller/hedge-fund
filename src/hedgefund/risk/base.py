"""Abstract base class for risk management."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

from hedgefund.types import Order, PortfolioSnapshot, TradeSignal


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    """Result of a risk check."""

    approved: bool
    reason: str
    adjusted_quantity: Optional[int] = None


class RiskManager(abc.ABC):
    """Abstract interface that every risk-management implementation must satisfy.

    The three methods form a pipeline:
        1. ``validate_signal``  -- gate-check *before* sizing
        2. ``size_position``    -- compute appropriate quantity
        3. ``check_circuit_breakers`` -- portfolio-wide safety checks
    """

    @abc.abstractmethod
    async def validate_signal(
        self,
        signal: TradeSignal,
        portfolio: PortfolioSnapshot,
    ) -> RiskVerdict:
        """Decide whether *signal* is permissible given the current portfolio.

        Returns a :class:`RiskVerdict` indicating approval or rejection with a
        human-readable reason.
        """

    @abc.abstractmethod
    async def size_position(
        self,
        signal: TradeSignal,
        portfolio: PortfolioSnapshot,
    ) -> int:
        """Return the number of contracts to trade for *signal*.

        Must return 0 if the signal should be skipped.
        """

    @abc.abstractmethod
    async def check_circuit_breakers(
        self,
        portfolio: PortfolioSnapshot,
    ) -> RiskVerdict:
        """Evaluate portfolio-wide circuit breakers.

        If the verdict is not approved, the caller must halt all new order
        submissions until conditions improve.
        """
