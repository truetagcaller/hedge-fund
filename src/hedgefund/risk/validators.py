"""Pre-trade validation using a chain-of-responsibility pattern.

An ordered sequence of risk checks runs before any order is submitted.
Each check returns a ``ValidationResult``; if any check fails, the order
is rejected and subsequent checks are skipped.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional, Sequence

import structlog

from hedgefund.types import (
    Greeks,
    OptionQuote,
    Order,
    PortfolioSnapshot,
    TradeSignal,
)
from hedgefund.risk.drawdown import DrawdownMonitor
from hedgefund.risk.limits import RiskLimits

logger = structlog.get_logger(__name__)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of the full validation chain."""

    approved: bool
    failed_check: Optional[str] = None
    message: str = ""
    checks_run: int = 0


# ── Abstract check ────────────────────────────────────────────────────────────

class RiskCheck(abc.ABC):
    """Single link in the validation chain."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable check name for logging."""

    @abc.abstractmethod
    async def execute(self, ctx: ValidationContext) -> bool:
        """Return ``True`` if the check passes, ``False`` otherwise.

        Implementations should set ``ctx.failure_message`` when returning
        ``False``.
        """


# ── Shared context passed through the chain ───────────────────────────────────

@dataclass(slots=True)
class ValidationContext:
    """Mutable bag of data flowing through the chain."""

    order: Order
    signal: TradeSignal
    portfolio: PortfolioSnapshot
    quote: Optional[OptionQuote] = None
    drawdown_monitor: Optional[DrawdownMonitor] = None
    risk_limits: Optional[RiskLimits] = None
    failure_message: str = ""


# ── Concrete checks ──────────────────────────────────────────────────────────

class CapitalAvailableCheck(RiskCheck):
    """Verify that sufficient buying power exists for the order."""

    @property
    def name(self) -> str:
        return "capital_available"

    async def execute(self, ctx: ValidationContext) -> bool:
        required = (
            ctx.order.quantity
            * ctx.order.contract.multiplier
            * (ctx.order.limit_price or ctx.signal.entry_price)
        )
        if required > ctx.portfolio.cash:
            ctx.failure_message = (
                f"Insufficient capital: need ${required:,.2f}, "
                f"available ${ctx.portfolio.cash:,.2f}"
            )
            return False
        return True


class PositionLimitCheck(RiskCheck):
    """Ensure we have not hit the maximum concurrent positions."""

    def __init__(self, max_positions: int = 20) -> None:
        self._max = max_positions

    @property
    def name(self) -> str:
        return "position_limit"

    async def execute(self, ctx: ValidationContext) -> bool:
        if ctx.portfolio.position_count >= self._max:
            ctx.failure_message = (
                f"Position limit reached: {ctx.portfolio.position_count}/{self._max}"
            )
            return False
        return True


class DrawdownStatusCheck(RiskCheck):
    """Reject orders when the drawdown circuit breaker is tripped."""

    @property
    def name(self) -> str:
        return "drawdown_status"

    async def execute(self, ctx: ValidationContext) -> bool:
        if ctx.drawdown_monitor is None:
            return True
        if not ctx.drawdown_monitor.is_trading_allowed:
            ctx.failure_message = (
                f"Trading halted: drawdown state is {ctx.drawdown_monitor.state.value}"
            )
            return False
        return True


class SpreadWidthCheck(RiskCheck):
    """Reject orders when the bid-ask spread is too wide.

    Parameters
    ----------
    max_spread_pct:
        Maximum spread as a percentage of the mid price (default 0.05 = 5 %).
    """

    def __init__(self, max_spread_pct: float = 0.05) -> None:
        self._max_spread_pct = max_spread_pct

    @property
    def name(self) -> str:
        return "spread_width"

    async def execute(self, ctx: ValidationContext) -> bool:
        if ctx.quote is None:
            # No quote available -- skip check
            return True
        mid = ctx.quote.mid_price
        if mid <= 0:
            ctx.failure_message = "Invalid mid price (zero or negative)"
            return False
        spread_pct = ctx.quote.spread / mid
        if spread_pct > self._max_spread_pct:
            ctx.failure_message = (
                f"Spread {spread_pct:.4%} exceeds limit {self._max_spread_pct:.4%}"
            )
            return False
        return True


class LiquidityCheck(RiskCheck):
    """Ensure there is adequate volume and open interest.

    Parameters
    ----------
    min_volume:
        Minimum daily volume (default 100).
    min_open_interest:
        Minimum open interest (default 500).
    """

    def __init__(
        self,
        min_volume: int = 100,
        min_open_interest: int = 500,
    ) -> None:
        self._min_volume = min_volume
        self._min_oi = min_open_interest

    @property
    def name(self) -> str:
        return "liquidity"

    async def execute(self, ctx: ValidationContext) -> bool:
        if ctx.quote is None:
            return True
        issues: list[str] = []
        if ctx.quote.volume < self._min_volume:
            issues.append(f"Volume {ctx.quote.volume} < {self._min_volume}")
        if ctx.quote.open_interest < self._min_oi:
            issues.append(f"OI {ctx.quote.open_interest} < {self._min_oi}")
        if issues:
            ctx.failure_message = "; ".join(issues)
            return False
        return True


class GreeksLimitCheck(RiskCheck):
    """Ensure the order will not push portfolio Greeks beyond limits."""

    @property
    def name(self) -> str:
        return "greeks_limits"

    async def execute(self, ctx: ValidationContext) -> bool:
        if ctx.risk_limits is None:
            return True
        result = ctx.risk_limits.check_greeks_exposure(ctx.portfolio)
        if not result.passed:
            ctx.failure_message = result.message
            return False
        return True


# ── Validator (chain runner) ──────────────────────────────────────────────────

class PreTradeValidator:
    """Runs an ordered chain of :class:`RiskCheck` instances.

    Checks execute sequentially.  The first failure short-circuits the
    remaining chain.

    If no custom ``checks`` sequence is provided, a sensible default chain
    is constructed.
    """

    def __init__(
        self,
        checks: Sequence[RiskCheck] | None = None,
        *,
        drawdown_monitor: DrawdownMonitor | None = None,
        risk_limits: RiskLimits | None = None,
        max_positions: int = 20,
        max_spread_pct: float = 0.05,
        min_volume: int = 100,
        min_open_interest: int = 500,
    ) -> None:
        if checks is not None:
            self._checks: list[RiskCheck] = list(checks)
        else:
            self._checks = [
                CapitalAvailableCheck(),
                PositionLimitCheck(max_positions),
                DrawdownStatusCheck(),
                SpreadWidthCheck(max_spread_pct),
                LiquidityCheck(min_volume, min_open_interest),
                GreeksLimitCheck(),
            ]

        self._drawdown_monitor = drawdown_monitor
        self._risk_limits = risk_limits

    async def validate(
        self,
        order: Order,
        signal: TradeSignal,
        portfolio: PortfolioSnapshot,
        quote: OptionQuote | None = None,
    ) -> ValidationResult:
        """Execute the validation chain.

        Returns
        -------
        ValidationResult
            ``approved=True`` if every check passed, otherwise details of
            the first failure.
        """
        ctx = ValidationContext(
            order=order,
            signal=signal,
            portfolio=portfolio,
            quote=quote,
            drawdown_monitor=self._drawdown_monitor,
            risk_limits=self._risk_limits,
        )

        checks_run = 0
        for check in self._checks:
            checks_run += 1
            try:
                passed = await check.execute(ctx)
            except Exception:
                logger.exception("pre_trade_validator.check_error", check=check.name)
                return ValidationResult(
                    approved=False,
                    failed_check=check.name,
                    message=f"Check '{check.name}' raised an exception",
                    checks_run=checks_run,
                )

            if not passed:
                logger.warning(
                    "pre_trade_validator.rejected",
                    check=check.name,
                    message=ctx.failure_message,
                    order_id=order.order_id,
                )
                return ValidationResult(
                    approved=False,
                    failed_check=check.name,
                    message=ctx.failure_message,
                    checks_run=checks_run,
                )

            logger.debug("pre_trade_validator.check_passed", check=check.name)

        logger.info(
            "pre_trade_validator.approved",
            order_id=order.order_id,
            checks_run=checks_run,
        )
        return ValidationResult(approved=True, checks_run=checks_run)
