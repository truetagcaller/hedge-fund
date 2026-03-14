"""Position sizing algorithms.

Three strategies are provided:
    * **fixed_fraction** -- risk a fixed percentage of capital per trade.
    * **kelly_criterion** -- quarter-Kelly sizing derived from win rate and payoff.
    * **volatility_adjusted** -- ATR-based sizing that scales inversely with
      realised market volatility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SizingResult:
    """Output of a position-sizing calculation."""

    quantity: int
    method: str
    risk_per_contract: float
    total_risk: float
    fraction_of_capital: float


class PositionSizer:
    """Computes trade size using one of three algorithms.

    Parameters
    ----------
    default_risk_pct:
        Fraction of capital risked per trade (default 0.01 = 1 %).
    kelly_fraction:
        Fraction of full Kelly to use (default 0.25 = quarter Kelly).
    max_position_pct:
        Hard cap on any single position as a fraction of capital.
    min_contracts:
        Minimum contracts to open (if the sizing says less, return 0).
    """

    def __init__(
        self,
        *,
        default_risk_pct: float = 0.01,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.05,
        min_contracts: int = 1,
    ) -> None:
        self._default_risk_pct = default_risk_pct
        self._kelly_fraction = kelly_fraction
        self._max_position_pct = max_position_pct
        self._min_contracts = min_contracts

    # ── Public API ────────────────────────────────────────────────────────

    def fixed_fraction(
        self,
        capital: float,
        risk_per_contract: float,
        multiplier: int = 100,
        risk_pct: float | None = None,
    ) -> SizingResult:
        """Risk a fixed percentage of *capital* per trade.

        Parameters
        ----------
        capital:
            Current portfolio equity.
        risk_per_contract:
            Dollar risk per contract (e.g. entry - stop_loss).
        multiplier:
            Contract multiplier (100 for standard equity options).
        risk_pct:
            Override for per-trade risk fraction.
        """
        pct = risk_pct if risk_pct is not None else self._default_risk_pct
        dollar_risk = capital * pct
        risk_per = risk_per_contract * multiplier
        if risk_per <= 0:
            logger.warning("position_sizer.zero_risk_per_contract")
            return SizingResult(0, "fixed_fraction", 0.0, 0.0, 0.0)

        raw = dollar_risk / risk_per
        quantity = self._clamp(raw, capital, risk_per, multiplier)

        return SizingResult(
            quantity=quantity,
            method="fixed_fraction",
            risk_per_contract=risk_per,
            total_risk=quantity * risk_per,
            fraction_of_capital=(quantity * risk_per) / capital if capital > 0 else 0.0,
        )

    def kelly_criterion(
        self,
        capital: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        risk_per_contract: float,
        multiplier: int = 100,
    ) -> SizingResult:
        """Quarter-Kelly sizing.

        Parameters
        ----------
        capital:
            Current portfolio equity.
        win_rate:
            Historical win rate (0-1).
        avg_win:
            Average winning trade profit (absolute).
        avg_loss:
            Average losing trade loss (absolute, positive number).
        risk_per_contract:
            Dollar risk per contract.
        multiplier:
            Contract multiplier.
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            logger.warning(
                "position_sizer.invalid_kelly_inputs",
                win_rate=win_rate,
                avg_loss=avg_loss,
            )
            return SizingResult(0, "kelly_criterion", 0.0, 0.0, 0.0)

        payoff_ratio = avg_win / avg_loss
        # Full Kelly: f* = (p * b - q) / b  where p = win_rate, q = 1-p, b = payoff
        full_kelly = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
        fractional_kelly = full_kelly * self._kelly_fraction

        if fractional_kelly <= 0:
            logger.info(
                "position_sizer.negative_kelly",
                full_kelly=full_kelly,
            )
            return SizingResult(0, "kelly_criterion", 0.0, 0.0, 0.0)

        dollar_risk = capital * fractional_kelly
        risk_per = risk_per_contract * multiplier
        if risk_per <= 0:
            return SizingResult(0, "kelly_criterion", 0.0, 0.0, 0.0)

        raw = dollar_risk / risk_per
        quantity = self._clamp(raw, capital, risk_per, multiplier)

        return SizingResult(
            quantity=quantity,
            method="kelly_criterion",
            risk_per_contract=risk_per,
            total_risk=quantity * risk_per,
            fraction_of_capital=(quantity * risk_per) / capital if capital > 0 else 0.0,
        )

    def volatility_adjusted(
        self,
        capital: float,
        atr: float,
        entry_price: float,
        multiplier: int = 100,
        atr_multiplier: float = 2.0,
        risk_pct: float | None = None,
    ) -> SizingResult:
        """ATR-based sizing that scales inversely with market volatility.

        The stop distance is ``atr * atr_multiplier``.  Higher ATR (more
        volatile) produces fewer contracts; lower ATR produces more.

        Parameters
        ----------
        capital:
            Current portfolio equity.
        atr:
            Current Average True Range of the underlying.
        entry_price:
            Planned entry price of the option.
        multiplier:
            Contract multiplier.
        atr_multiplier:
            Number of ATRs used as the stop distance.
        risk_pct:
            Override for per-trade risk fraction.
        """
        pct = risk_pct if risk_pct is not None else self._default_risk_pct

        if atr <= 0 or entry_price <= 0:
            logger.warning(
                "position_sizer.invalid_vol_inputs",
                atr=atr,
                entry_price=entry_price,
            )
            return SizingResult(0, "volatility_adjusted", 0.0, 0.0, 0.0)

        stop_distance = atr * atr_multiplier
        risk_per = stop_distance * multiplier
        dollar_risk = capital * pct
        raw = dollar_risk / risk_per
        quantity = self._clamp(raw, capital, entry_price * multiplier, multiplier)

        return SizingResult(
            quantity=quantity,
            method="volatility_adjusted",
            risk_per_contract=risk_per,
            total_risk=quantity * risk_per,
            fraction_of_capital=(quantity * risk_per) / capital if capital > 0 else 0.0,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _clamp(
        self,
        raw_qty: float,
        capital: float,
        price_per_contract: float,
        multiplier: int,
    ) -> int:
        """Floor the quantity and enforce the max-position-pct cap."""
        qty = max(0, int(math.floor(raw_qty)))

        # Enforce maximum position size as fraction of capital
        if capital > 0 and price_per_contract > 0:
            max_qty = int(math.floor(capital * self._max_position_pct / price_per_contract))
            qty = min(qty, max_qty)

        if qty < self._min_contracts:
            return 0

        return qty
