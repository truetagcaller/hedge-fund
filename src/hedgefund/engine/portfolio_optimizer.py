"""Portfolio Optimizer — capital allocation across strategies.

Implements three allocation methods:
- **Mean-Variance Optimization** (Markowitz efficient frontier)
- **Kelly Criterion** (fractional Kelly sizing)
- **Risk Parity** (equal risk contribution)

All methods respect the risk limits defined in :class:`RiskConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from hedgefund.config.schema import RiskConfig

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """Output of a portfolio allocation calculation."""

    method: str
    allocations: dict[str, float]  # symbol -> fraction of capital
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)


class PortfolioOptimizer:
    """Compute optimal capital allocation across symbols.

    Parameters
    ----------
    risk_config:
        Risk configuration with position limits.
    risk_free_rate:
        Annualised risk-free rate for Sharpe calculations.
    kelly_fraction:
        Fraction of full Kelly to use (default quarter-Kelly).
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        *,
        risk_free_rate: float = 0.05,
        kelly_fraction: float = 0.25,
    ) -> None:
        self._risk_config = risk_config
        self._rf = risk_free_rate
        self._kelly_frac = kelly_fraction

    # ── Dispatch ──────────────────────────────────────────────────────

    def optimize(
        self,
        method: str,
        *,
        returns: np.ndarray | None = None,
        symbols: list[str] | None = None,
        win_rates: dict[str, float] | None = None,
        avg_wins: dict[str, float] | None = None,
        avg_losses: dict[str, float] | None = None,
    ) -> AllocationResult:
        """Dispatch to the appropriate allocation method.

        Parameters
        ----------
        method:
            One of ``"mean_variance"``, ``"kelly"``, ``"risk_parity"``.
        returns:
            2-D array of historical returns (rows=time, cols=assets).
        symbols:
            Asset names corresponding to columns in *returns*.
        win_rates / avg_wins / avg_losses:
            Per-symbol stats required for Kelly criterion.
        """
        symbols = symbols or []

        if method == "mean_variance":
            if returns is None or len(symbols) == 0:
                return self._empty_result("mean_variance", symbols)
            return self.mean_variance_optimize(returns, symbols)

        if method == "kelly":
            if not (win_rates and avg_wins and avg_losses and symbols):
                return self._empty_result("kelly", symbols)
            return self.kelly_criterion_allocate(
                symbols, win_rates, avg_wins, avg_losses,
            )

        if method == "risk_parity":
            if returns is None or len(symbols) == 0:
                return self._empty_result("risk_parity", symbols)
            return self.risk_parity_allocate(returns, symbols)

        log.warning("portfolio_optimizer.unknown_method", method=method)
        return self._empty_result(method, symbols)

    # ── Mean-Variance (Markowitz) ─────────────────────────────────────

    def mean_variance_optimize(
        self,
        returns: np.ndarray,
        symbols: list[str],
    ) -> AllocationResult:
        """Markowitz mean-variance: maximise Sharpe ratio.

        Constraints:
        - Sum of weights = 1.
        - Each weight in [0, max_single_position_pct].
        """
        n = len(symbols)
        if n == 0 or returns.shape[0] < 2:
            return self._empty_result("mean_variance", symbols)

        try:
            from scipy.optimize import minimize
        except ImportError:
            log.warning("portfolio_optimizer.scipy_unavailable")
            return self._equal_weight("mean_variance", symbols)

        mu = np.mean(returns, axis=0) * 252  # annualise
        cov = np.cov(returns, rowvar=False) * 252
        max_w = self._risk_config.max_single_position_pct

        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = w @ mu
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 0.0
            return -(port_ret - self._rf) / port_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, max_w)] * n
        x0 = np.ones(n) / n

        result = minimize(
            neg_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        if not result.success:
            log.warning("portfolio_optimizer.mvo_failed", msg=result.message)
            return self._equal_weight("mean_variance", symbols)

        weights = result.x
        port_ret = float(weights @ mu)
        port_vol = float(np.sqrt(weights @ cov @ weights))
        sharpe = (port_ret - self._rf) / port_vol if port_vol > 0 else 0.0

        alloc = {sym: round(float(w), 6) for sym, w in zip(symbols, weights)}

        return AllocationResult(
            method="mean_variance",
            allocations=alloc,
            expected_return=round(port_ret, 6),
            expected_risk=round(port_vol, 6),
            sharpe_ratio=round(sharpe, 4),
        )

    # ── Kelly Criterion ───────────────────────────────────────────────

    def kelly_criterion_allocate(
        self,
        symbols: list[str],
        win_rates: dict[str, float],
        avg_wins: dict[str, float],
        avg_losses: dict[str, float],
    ) -> AllocationResult:
        """Fractional Kelly sizing per symbol."""
        allocations: dict[str, float] = {}
        max_w = self._risk_config.max_single_position_pct

        for sym in symbols:
            wr = win_rates.get(sym, 0.0)
            aw = avg_wins.get(sym, 0.0)
            al = avg_losses.get(sym, 0.0)

            if al <= 0 or wr <= 0 or wr >= 1:
                allocations[sym] = 0.0
                continue

            payoff = aw / al
            full_kelly = (wr * payoff - (1 - wr)) / payoff
            frac_kelly = full_kelly * self._kelly_frac

            if frac_kelly <= 0:
                allocations[sym] = 0.0
            else:
                allocations[sym] = round(min(frac_kelly, max_w), 6)

        # Normalize if total > 1
        total = sum(allocations.values())
        if total > 1.0:
            allocations = {
                s: round(w / total, 6) for s, w in allocations.items()
            }
            total = 1.0

        return AllocationResult(
            method="kelly",
            allocations=allocations,
            expected_return=0.0,
            expected_risk=0.0,
            sharpe_ratio=0.0,
            metadata={"kelly_fraction": self._kelly_frac},
        )

    # ── Risk Parity ───────────────────────────────────────────────────

    def risk_parity_allocate(
        self,
        returns: np.ndarray,
        symbols: list[str],
    ) -> AllocationResult:
        """Equal risk contribution: weight inversely proportional to vol."""
        n = len(symbols)
        if n == 0 or returns.shape[0] < 2:
            return self._empty_result("risk_parity", symbols)

        vols = np.std(returns, axis=0) * np.sqrt(252)
        if np.any(vols <= 0):
            return self._equal_weight("risk_parity", symbols)

        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()

        # Cap at max position
        max_w = self._risk_config.max_single_position_pct
        weights = np.minimum(weights, max_w)
        weights = weights / weights.sum()  # re-normalize

        mu = np.mean(returns, axis=0) * 252
        cov = np.cov(returns, rowvar=False) * 252
        port_ret = float(weights @ mu)
        port_vol = float(np.sqrt(weights @ cov @ weights))
        sharpe = (port_ret - self._rf) / port_vol if port_vol > 0 else 0.0

        alloc = {sym: round(float(w), 6) for sym, w in zip(symbols, weights)}

        return AllocationResult(
            method="risk_parity",
            allocations=alloc,
            expected_return=round(port_ret, 6),
            expected_risk=round(port_vol, 6),
            sharpe_ratio=round(sharpe, 4),
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _equal_weight(self, method: str, symbols: list[str]) -> AllocationResult:
        n = len(symbols)
        if n == 0:
            return self._empty_result(method, symbols)
        w = round(1.0 / n, 6)
        return AllocationResult(
            method=method,
            allocations={s: w for s in symbols},
            expected_return=0.0,
            expected_risk=0.0,
            sharpe_ratio=0.0,
            metadata={"fallback": "equal_weight"},
        )

    @staticmethod
    def _empty_result(method: str, symbols: list[str]) -> AllocationResult:
        return AllocationResult(
            method=method,
            allocations={s: 0.0 for s in symbols},
            expected_return=0.0,
            expected_risk=0.0,
            sharpe_ratio=0.0,
            metadata={"error": "insufficient_data"},
        )
