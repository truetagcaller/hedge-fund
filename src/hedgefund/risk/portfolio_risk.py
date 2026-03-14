"""Portfolio-level risk analytics.

Provides aggregate Greeks, Value-at-Risk (historical and parametric),
stress testing, and correlation-adjusted risk measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import structlog

from hedgefund.types import Greeks, Position, PortfolioSnapshot

logger = structlog.get_logger(__name__)


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class AggregateGreeks:
    """Portfolio-wide Greeks totals."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@dataclass(frozen=True, slots=True)
class VaRResult:
    """Value-at-Risk output."""

    var_95: float
    var_99: float
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    method: str


@dataclass(frozen=True, slots=True)
class StressScenario:
    """Definition of a single stress test scenario."""

    name: str
    underlying_shift_pct: float = 0.0  # e.g. -0.10 for -10%
    vol_shift_abs: float = 0.0  # absolute IV change, e.g. +0.05
    rate_shift_bps: float = 0.0  # interest-rate shift in basis points


@dataclass(frozen=True, slots=True)
class StressResult:
    """P&L impact of a stress scenario."""

    scenario: str
    pnl_impact: float
    delta_impact: float
    gamma_impact: float
    vega_impact: float


@dataclass(frozen=True, slots=True)
class CorrelationRisk:
    """Correlation-adjusted portfolio risk metrics."""

    undiversified_var: float
    diversified_var: float
    diversification_benefit: float  # 1 - (diversified / undiversified)


class PortfolioRiskManager:
    """Aggregate portfolio risk analytics.

    Parameters
    ----------
    confidence_levels:
        VaR confidence levels (default 95 % and 99 %).
    lookback_days:
        Number of historical return observations for historical VaR.
    """

    def __init__(
        self,
        *,
        confidence_levels: tuple[float, float] = (0.95, 0.99),
        lookback_days: int = 252,
    ) -> None:
        self._confidence_levels = confidence_levels
        self._lookback_days = lookback_days

    # ── Aggregate Greeks ──────────────────────────────────────────────────

    def aggregate_greeks(self, positions: Sequence[Position]) -> AggregateGreeks:
        """Sum Greeks across all positions, weighted by quantity and multiplier."""
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0

        for pos in positions:
            mult = pos.contract.multiplier * pos.quantity
            total_delta += pos.greeks.delta * mult
            total_gamma += pos.greeks.gamma * mult
            total_theta += pos.greeks.theta * mult
            total_vega += pos.greeks.vega * mult
            total_rho += pos.greeks.rho * mult

        result = AggregateGreeks(
            delta=total_delta,
            gamma=total_gamma,
            theta=total_theta,
            vega=total_vega,
            rho=total_rho,
        )
        logger.debug(
            "portfolio_risk.aggregate_greeks",
            delta=result.delta,
            gamma=result.gamma,
            theta=result.theta,
            vega=result.vega,
        )
        return result

    # ── Value at Risk ─────────────────────────────────────────────────────

    def historical_var(
        self,
        portfolio_value: float,
        historical_returns: np.ndarray,
    ) -> VaRResult:
        """Compute VaR from a historical return distribution.

        Parameters
        ----------
        portfolio_value:
            Current net liquidation value.
        historical_returns:
            Array of daily portfolio return observations (e.g. [-0.02, 0.01, ...]).
        """
        if len(historical_returns) < 30:
            logger.warning(
                "portfolio_risk.insufficient_history",
                observations=len(historical_returns),
            )
            return VaRResult(var_95=0.0, var_99=0.0, cvar_95=0.0, method="historical")

        sorted_returns = np.sort(historical_returns)
        n = len(sorted_returns)

        idx_95 = int(math.floor(n * (1 - self._confidence_levels[0])))
        idx_99 = int(math.floor(n * (1 - self._confidence_levels[1])))

        var_95 = abs(sorted_returns[idx_95]) * portfolio_value
        var_99 = abs(sorted_returns[idx_99]) * portfolio_value

        # CVaR: mean of all returns worse than the VaR threshold
        tail = sorted_returns[: idx_95 + 1]
        cvar_95 = abs(float(np.mean(tail))) * portfolio_value if len(tail) > 0 else var_95

        result = VaRResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            method="historical",
        )
        logger.info("portfolio_risk.historical_var", var_95=var_95, var_99=var_99)
        return result

    def parametric_var(
        self,
        portfolio_value: float,
        mean_return: float,
        std_return: float,
    ) -> VaRResult:
        """Compute VaR assuming normally-distributed returns.

        Parameters
        ----------
        portfolio_value:
            Current net liquidation value.
        mean_return:
            Estimated daily mean return.
        std_return:
            Estimated daily return standard deviation.
        """
        if std_return <= 0:
            return VaRResult(var_95=0.0, var_99=0.0, cvar_95=0.0, method="parametric")

        # z-scores for 95% and 99%
        z_95 = 1.6449
        z_99 = 2.3263

        var_95 = (mean_return - z_95 * std_return) * portfolio_value * -1
        var_99 = (mean_return - z_99 * std_return) * portfolio_value * -1

        # CVaR for normal: mu + sigma * phi(z) / (1 - alpha)
        # phi(z_95) ≈ 0.1031
        phi_z95 = 0.1031
        cvar_95 = (mean_return - std_return * phi_z95 / 0.05) * portfolio_value * -1

        var_95 = max(0.0, var_95)
        var_99 = max(0.0, var_99)
        cvar_95 = max(0.0, cvar_95)

        result = VaRResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            method="parametric",
        )
        logger.info("portfolio_risk.parametric_var", var_95=var_95, var_99=var_99)
        return result

    # ── Stress Testing ────────────────────────────────────────────────────

    def stress_test(
        self,
        positions: Sequence[Position],
        scenarios: Sequence[StressScenario] | None = None,
    ) -> list[StressResult]:
        """Run stress scenarios against current positions.

        Each scenario applies parallel shifts to the underlying price,
        implied volatility, and/or interest rates, then estimates the
        P&L impact using first- and second-order Greek sensitivities.
        """
        if scenarios is None:
            scenarios = self._default_scenarios()

        results: list[StressResult] = []
        for scenario in scenarios:
            pnl = 0.0
            delta_impact = 0.0
            gamma_impact = 0.0
            vega_impact = 0.0

            for pos in positions:
                mult = pos.contract.multiplier * pos.quantity
                g = pos.greeks
                underlying_price = pos.contract.strike  # approximation

                # Price change from underlying shift
                dp = underlying_price * scenario.underlying_shift_pct
                # First-order: delta * dP
                delta_pnl = g.delta * dp * mult
                # Second-order: 0.5 * gamma * dP^2
                gamma_pnl = 0.5 * g.gamma * (dp ** 2) * mult
                # Vega impact from vol shift
                vega_pnl = g.vega * scenario.vol_shift_abs * mult
                # Rho impact from rate shift (convert bps to decimal)
                rho_pnl = g.rho * (scenario.rate_shift_bps / 10_000) * mult

                pos_pnl = delta_pnl + gamma_pnl + vega_pnl + rho_pnl
                pnl += pos_pnl
                delta_impact += delta_pnl
                gamma_impact += gamma_pnl
                vega_impact += vega_pnl

            result = StressResult(
                scenario=scenario.name,
                pnl_impact=pnl,
                delta_impact=delta_impact,
                gamma_impact=gamma_impact,
                vega_impact=vega_impact,
            )
            results.append(result)
            logger.info(
                "portfolio_risk.stress_test",
                scenario=scenario.name,
                pnl_impact=pnl,
            )

        return results

    # ── Correlation-adjusted risk ─────────────────────────────────────────

    def correlation_adjusted_risk(
        self,
        position_vars: np.ndarray,
        correlation_matrix: np.ndarray,
    ) -> CorrelationRisk:
        """Compute diversified vs undiversified portfolio VaR.

        Parameters
        ----------
        position_vars:
            1-D array of individual position VaR values.
        correlation_matrix:
            N x N correlation matrix for the positions.
        """
        if len(position_vars) == 0:
            return CorrelationRisk(0.0, 0.0, 0.0)

        undiversified = float(np.sum(position_vars))

        # Diversified VaR = sqrt(w^T . C . w) where w = position VaRs, C = corr
        try:
            variance = float(position_vars @ correlation_matrix @ position_vars)
            diversified = math.sqrt(max(0.0, variance))
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.error("portfolio_risk.correlation_calc_failed", error=str(exc))
            diversified = undiversified

        benefit = 1.0 - (diversified / undiversified) if undiversified > 0 else 0.0

        result = CorrelationRisk(
            undiversified_var=undiversified,
            diversified_var=diversified,
            diversification_benefit=benefit,
        )
        logger.info(
            "portfolio_risk.correlation_risk",
            undiversified=undiversified,
            diversified=diversified,
            benefit=f"{benefit:.2%}",
        )
        return result

    # ── Default scenarios ─────────────────────────────────────────────────

    @staticmethod
    def _default_scenarios() -> list[StressScenario]:
        return [
            StressScenario(name="market_crash", underlying_shift_pct=-0.10, vol_shift_abs=0.15),
            StressScenario(name="sharp_rally", underlying_shift_pct=0.10, vol_shift_abs=-0.05),
            StressScenario(name="vol_spike", vol_shift_abs=0.20),
            StressScenario(name="vol_crush", vol_shift_abs=-0.10),
            StressScenario(name="rate_hike", rate_shift_bps=50),
            StressScenario(name="rate_cut", rate_shift_bps=-50),
            StressScenario(
                name="black_swan",
                underlying_shift_pct=-0.20,
                vol_shift_abs=0.30,
                rate_shift_bps=-100,
            ),
        ]
