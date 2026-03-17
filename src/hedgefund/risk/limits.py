"""Risk limit enforcement.

Each limit check returns a ``(passed, message)`` tuple so callers can
collect all violations before deciding how to proceed.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from hedgefund.types import PortfolioSnapshot

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LimitCheckResult:
    """Outcome of a single risk-limit check."""

    passed: bool
    message: str


class RiskLimits:
    """Enforces per-trade, daily, and portfolio-wide risk limits.

    All thresholds can be overridden at construction time.  Defaults match
    the values in ``RiskConfig``.

    Parameters
    ----------
    risk_per_trade_pct:
        Maximum capital risked on a single trade (default 0.01 = 1 %).
    max_daily_loss_pct:
        Maximum daily portfolio loss (default 0.03 = 3 %).
    max_drawdown_pct:
        Maximum peak-to-trough drawdown (default 0.10 = 10 %).
    max_concurrent_positions:
        Maximum number of open positions (default 20).
    max_sector_exposure_pct:
        Maximum portfolio weight in a single sector (default 0.25 = 25 %).
    max_delta_exposure:
        Absolute portfolio delta cap.
    max_gamma_exposure:
        Absolute portfolio gamma cap.
    max_vega_exposure:
        Absolute portfolio vega cap.
    """

    def __init__(
        self,
        *,
        risk_per_trade_pct: float = 0.01,
        max_daily_loss_pct: float = 0.03,
        max_drawdown_pct: float = 0.10,
        max_concurrent_positions: int = 20,
        max_sector_exposure_pct: float = 0.25,
        max_delta_exposure: float = 500.0,
        max_gamma_exposure: float = 100.0,
        max_vega_exposure: float = 50_000.0,
    ) -> None:
        self._risk_per_trade_pct = risk_per_trade_pct
        self._max_daily_loss_pct = max_daily_loss_pct
        self._max_drawdown_pct = max_drawdown_pct
        self._max_concurrent_positions = max_concurrent_positions
        self._max_sector_exposure_pct = max_sector_exposure_pct
        self._max_delta = max_delta_exposure
        self._max_gamma = max_gamma_exposure
        self._max_vega = max_vega_exposure

    # ── Individual checks ─────────────────────────────────────────────────

    def check_per_trade_risk(
        self,
        trade_risk: float,
        portfolio_value: float,
    ) -> LimitCheckResult:
        """Ensure a single trade does not risk more than the per-trade limit."""
        if portfolio_value <= 0:
            return LimitCheckResult(False, "Portfolio value is zero or negative")

        pct = trade_risk / portfolio_value
        if pct > self._risk_per_trade_pct:
            return LimitCheckResult(
                False,
                f"Per-trade risk {pct:.4%} exceeds limit {self._risk_per_trade_pct:.4%}",
            )
        return LimitCheckResult(True, "Per-trade risk within limit")

    def check_daily_loss(
        self,
        daily_pnl: float,
        portfolio_value: float,
    ) -> LimitCheckResult:
        """Ensure cumulative daily loss has not exceeded the daily limit."""
        if portfolio_value <= 0:
            return LimitCheckResult(False, "Portfolio value is zero or negative")

        loss_pct = abs(min(0.0, daily_pnl)) / portfolio_value
        if loss_pct >= self._max_daily_loss_pct:
            return LimitCheckResult(
                False,
                f"Daily loss {loss_pct:.4%} exceeds limit {self._max_daily_loss_pct:.4%}",
            )
        return LimitCheckResult(True, "Daily loss within limit")

    def check_max_drawdown(
        self,
        drawdown_pct: float,
    ) -> LimitCheckResult:
        """Ensure portfolio drawdown has not breached the maximum."""
        if drawdown_pct >= self._max_drawdown_pct:
            return LimitCheckResult(
                False,
                f"Drawdown {drawdown_pct:.4%} exceeds limit {self._max_drawdown_pct:.4%}",
            )
        return LimitCheckResult(True, "Drawdown within limit")

    def check_concurrent_positions(
        self,
        current_count: int,
    ) -> LimitCheckResult:
        """Ensure we are not exceeding the max number of concurrent positions."""
        if current_count >= self._max_concurrent_positions:
            return LimitCheckResult(
                False,
                f"Position count {current_count} reaches limit {self._max_concurrent_positions}",
            )
        return LimitCheckResult(True, "Position count within limit")

    def check_sector_exposure(
        self,
        sector_value: float,
        portfolio_value: float,
    ) -> LimitCheckResult:
        """Ensure no single sector exceeds the concentration limit."""
        if portfolio_value <= 0:
            return LimitCheckResult(False, "Portfolio value is zero or negative")

        pct = sector_value / portfolio_value
        if pct > self._max_sector_exposure_pct:
            return LimitCheckResult(
                False,
                f"Sector exposure {pct:.4%} exceeds limit {self._max_sector_exposure_pct:.4%}",
            )
        return LimitCheckResult(True, "Sector exposure within limit")

    def check_greeks_exposure(
        self,
        portfolio: PortfolioSnapshot,
    ) -> LimitCheckResult:
        """Ensure aggregate Greeks are within acceptable bounds."""
        violations: list[str] = []

        if abs(portfolio.total_delta) > self._max_delta:
            violations.append(
                f"Delta {portfolio.total_delta:.1f} exceeds +/-{self._max_delta:.1f}"
            )
        if abs(portfolio.total_gamma) > self._max_gamma:
            violations.append(
                f"Gamma {portfolio.total_gamma:.1f} exceeds +/-{self._max_gamma:.1f}"
            )
        if abs(portfolio.total_vega) > self._max_vega:
            violations.append(
                f"Vega {portfolio.total_vega:.1f} exceeds +/-{self._max_vega:.1f}"
            )

        if violations:
            return LimitCheckResult(False, "; ".join(violations))
        return LimitCheckResult(True, "Greeks exposure within limits")

    # ── Convenience: run all checks ───────────────────────────────────────

    def check_all(
        self,
        *,
        trade_risk: float,
        portfolio: PortfolioSnapshot,
        sector_value: float = 0.0,
    ) -> list[LimitCheckResult]:
        """Run every limit check and return all results.

        Callers can then decide whether to reject the trade based on any
        failures.
        """
        portfolio_value = portfolio.net_liquidation
        results = [
            self.check_per_trade_risk(trade_risk, portfolio_value),
            self.check_daily_loss(portfolio.daily_pnl, portfolio_value),
            self.check_max_drawdown(portfolio.drawdown_pct),
            self.check_concurrent_positions(portfolio.position_count),
            self.check_sector_exposure(sector_value, portfolio_value),
            self.check_greeks_exposure(portfolio),
        ]

        failures = [r for r in results if not r.passed]
        if failures:
            logger.warning(
                "risk_limits.violations",
                count=len(failures),
                messages=[f.message for f in failures],
            )
        else:
            logger.debug("risk_limits.all_passed")

        return results
