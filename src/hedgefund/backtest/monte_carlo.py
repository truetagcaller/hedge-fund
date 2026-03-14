"""Monte Carlo simulation for backtest robustness analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class MonteCarloConfig:
    """Parameters for the Monte Carlo simulator."""

    n_simulations: int = 10_000
    block_size: int = 5  # for block bootstrap (preserves autocorrelation)
    confidence_levels: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
    initial_capital: float = 100_000.0
    risk_free_rate: float = 0.05
    ruin_threshold: float = 0.5  # fraction of capital at which ruin is declared
    random_seed: int | None = None


@dataclass(slots=True)
class MonteCarloResult:
    """Aggregated results from a Monte Carlo simulation."""

    n_simulations: int
    n_trades: int

    # Terminal wealth distribution.
    terminal_wealth_mean: float = 0.0
    terminal_wealth_median: float = 0.0
    terminal_wealth_ci: dict[float, float] = field(default_factory=dict)

    # Return distribution.
    total_return_mean: float = 0.0
    total_return_ci: dict[float, float] = field(default_factory=dict)

    # Drawdown distribution.
    max_drawdown_mean: float = 0.0
    max_drawdown_median: float = 0.0
    max_drawdown_ci: dict[float, float] = field(default_factory=dict)

    # Sharpe distribution.
    sharpe_mean: float = 0.0
    sharpe_ci: dict[float, float] = field(default_factory=dict)

    # Ruin probability.
    ruin_probability: float = 0.0

    # Raw simulation paths (optional, for plotting).
    equity_paths: np.ndarray | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "n_simulations": self.n_simulations,
            "n_trades": self.n_trades,
            "terminal_wealth_mean": self.terminal_wealth_mean,
            "terminal_wealth_median": self.terminal_wealth_median,
            "total_return_mean": self.total_return_mean,
            "max_drawdown_mean": self.max_drawdown_mean,
            "sharpe_mean": self.sharpe_mean,
            "ruin_probability": self.ruin_probability,
        }


class MonteCarloSimulator:
    """Bootstrap resampling of trade returns for statistical inference.

    Supports:
    - IID bootstrap (random sampling with replacement).
    - Block bootstrap (preserves short-term autocorrelation).
    - Confidence intervals for key metrics.
    - Drawdown distribution analysis.
    - Ruin probability estimation.
    """

    def __init__(self, config: MonteCarloConfig | None = None) -> None:
        self.config = config or MonteCarloConfig()
        self._log = log.bind(component="monte_carlo")

    def run(
        self,
        trade_returns: np.ndarray,
        *,
        store_paths: bool = False,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation on observed trade returns.

        Args:
            trade_returns: 1-D array of per-trade percentage returns.
            store_paths: If True, keep all simulated equity curves in the
                result (can be memory-intensive for large *n_simulations*).

        Returns:
            :class:`MonteCarloResult` with confidence intervals and
            distribution statistics.
        """
        returns = np.asarray(trade_returns, dtype=np.float64).ravel()
        n_trades = len(returns)
        cfg = self.config
        rng = np.random.default_rng(cfg.random_seed)

        self._log.info(
            "simulation_started",
            n_simulations=cfg.n_simulations,
            n_trades=n_trades,
            block_size=cfg.block_size,
        )

        terminal_wealths = np.empty(cfg.n_simulations)
        max_drawdowns = np.empty(cfg.n_simulations)
        sharpes = np.empty(cfg.n_simulations)
        equity_paths_list: list[np.ndarray] | None = [] if store_paths else None

        for i in range(cfg.n_simulations):
            sampled = self._bootstrap_sample(returns, n_trades, rng)
            equity = self._build_equity_curve(sampled, cfg.initial_capital)

            terminal_wealths[i] = equity[-1]
            max_drawdowns[i] = self._max_drawdown(equity)
            sharpes[i] = self._sharpe_ratio(sampled)

            if equity_paths_list is not None:
                equity_paths_list.append(equity)

        # Ruin probability.
        ruin_level = cfg.initial_capital * cfg.ruin_threshold
        ruin_count = 0
        for i in range(cfg.n_simulations):
            sampled = self._bootstrap_sample(returns, n_trades, rng)
            eq = self._build_equity_curve(sampled, cfg.initial_capital)
            if eq.min() <= ruin_level:
                ruin_count += 1

        total_returns = (terminal_wealths - cfg.initial_capital) / cfg.initial_capital

        result = MonteCarloResult(
            n_simulations=cfg.n_simulations,
            n_trades=n_trades,
            terminal_wealth_mean=float(terminal_wealths.mean()),
            terminal_wealth_median=float(np.median(terminal_wealths)),
            terminal_wealth_ci=self._confidence_intervals(terminal_wealths),
            total_return_mean=float(total_returns.mean()),
            total_return_ci=self._confidence_intervals(total_returns),
            max_drawdown_mean=float(max_drawdowns.mean()),
            max_drawdown_median=float(np.median(max_drawdowns)),
            max_drawdown_ci=self._confidence_intervals(max_drawdowns),
            sharpe_mean=float(sharpes.mean()),
            sharpe_ci=self._confidence_intervals(sharpes),
            ruin_probability=ruin_count / cfg.n_simulations,
            equity_paths=(
                np.array(equity_paths_list) if equity_paths_list else None
            ),
        )

        self._log.info("simulation_complete", **result.summary())
        return result

    # ── Bootstrap methods ─────────────────────────────────────────────

    def _bootstrap_sample(
        self,
        returns: np.ndarray,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw a bootstrap sample of length *n* from *returns*.

        Uses block bootstrap if ``block_size > 1``.
        """
        bs = self.config.block_size
        if bs <= 1:
            idx = rng.integers(0, len(returns), size=n)
            return returns[idx]

        # Block bootstrap.
        n_blocks = (n + bs - 1) // bs
        max_start = len(returns) - bs
        if max_start <= 0:
            # Fall back to IID if not enough data for blocks.
            idx = rng.integers(0, len(returns), size=n)
            return returns[idx]

        starts = rng.integers(0, max_start + 1, size=n_blocks)
        blocks = [returns[s : s + bs] for s in starts]
        return np.concatenate(blocks)[:n]

    # ── Equity curve and metrics ──────────────────────────────────────

    @staticmethod
    def _build_equity_curve(
        trade_returns: np.ndarray, initial_capital: float
    ) -> np.ndarray:
        """Compound trade returns into an equity curve."""
        growth = np.cumprod(1.0 + trade_returns)
        return np.insert(growth * initial_capital, 0, initial_capital)

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
        return float(dd.max())

    def _sharpe_ratio(self, returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns - self.config.risk_free_rate / 252.0
        std = returns.std()
        if std < 1e-10:
            return 0.0
        return float(excess.mean() / std * np.sqrt(252))

    def _confidence_intervals(
        self, values: np.ndarray
    ) -> dict[float, float]:
        return {
            level: float(np.percentile(values, level * 100))
            for level in self.config.confidence_levels
        }
