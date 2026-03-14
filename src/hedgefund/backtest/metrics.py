"""Comprehensive performance metrics for backtesting and live trading."""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from hedgefund.types import BacktestMetrics

log = structlog.get_logger(__name__)


class MetricsCalculator:
    """Calculate institutional-grade performance metrics.

    Can be instantiated in two ways:

    1. **Array-based** (used by the backtest engine)::

           calc = MetricsCalculator(returns_array, equity_array)
           metrics = calc.compute_all()

    2. **DataFrame-based** (legacy / convenience)::

           calc = MetricsCalculator()
           metrics = calc.calculate(equity_series, trades_df)
    """

    TRADING_DAYS_PER_YEAR = 252
    RISK_FREE_RATE = 0.05  # 5 % annual risk-free rate

    def __init__(
        self,
        returns: np.ndarray | None = None,
        equity_curve: np.ndarray | None = None,
        *,
        risk_free_rate: float = 0.05,
        trading_days_per_year: int = 252,
    ) -> None:
        self._returns: np.ndarray | None = (
            np.asarray(returns, dtype=np.float64).ravel()
            if returns is not None
            else None
        )
        self._equity: np.ndarray | None = (
            np.asarray(equity_curve, dtype=np.float64).ravel()
            if equity_curve is not None
            else None
        )
        self._rf = risk_free_rate
        self._tdays = trading_days_per_year

    # ──────────────────────────────────────────────────────────────────
    # Array-based API
    # ──────────────────────────────────────────────────────────────────

    # ── Ratio metrics ─────────────────────────────────────────────────

    def sharpe_ratio(self) -> float:
        """Annualised Sharpe ratio (excess return / volatility)."""
        r = self._require_returns()
        if len(r) < 2:
            return 0.0
        excess = r - self._rf / self._tdays
        std = r.std()
        if std < 1e-10:
            return 0.0
        return float(excess.mean() / std * np.sqrt(self._tdays))

    def sortino_ratio(self) -> float:
        """Annualised Sortino ratio (excess return / downside deviation)."""
        r = self._require_returns()
        if len(r) < 2:
            return 0.0
        excess = r - self._rf / self._tdays
        downside = r[r < 0]
        if len(downside) == 0 or downside.std() < 1e-10:
            return 0.0
        return float(excess.mean() / downside.std() * np.sqrt(self._tdays))

    def calmar_ratio(self) -> float:
        """Annualised return / max drawdown."""
        mdd = self.max_drawdown()
        if mdd < 1e-10:
            return 0.0
        return self.annualized_return() / mdd

    # ── Return metrics ────────────────────────────────────────────────

    def total_return(self) -> float:
        """Total cumulative return over the period."""
        eq = self._require_equity()
        if len(eq) < 2:
            return 0.0
        return float((eq[-1] - eq[0]) / eq[0])

    def annualized_return(self) -> float:
        """Compound annual growth rate (CAGR)."""
        r = self._require_returns()
        n_days = len(r)
        if n_days < 2:
            return 0.0
        total = 1.0 + self.total_return()
        if total <= 0:
            return -1.0
        years = n_days / self._tdays
        return float(total ** (1.0 / max(years, 1e-6)) - 1.0)

    # ── Drawdown metrics ──────────────────────────────────────────────

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown as a positive fraction."""
        eq = self._require_equity()
        if len(eq) < 2:
            return 0.0
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.where(peak > 0, peak, 1.0)
        return float(dd.max())

    def drawdown_array(self) -> np.ndarray:
        """Full drawdown time series as an array."""
        eq = self._require_equity()
        peak = np.maximum.accumulate(eq)
        return (peak - eq) / np.where(peak > 0, peak, 1.0)

    def recovery_factor_from_arrays(self) -> float:
        """Total return / max drawdown (array-based)."""
        mdd = self.max_drawdown()
        if mdd < 1e-10:
            return 0.0
        return self.total_return() / mdd

    # ── Trade-level metrics (array) ───────────────────────────────────

    def win_rate_from_returns(self, trade_returns: np.ndarray | None = None) -> float:
        tr = self._trade_returns(trade_returns)
        if len(tr) == 0:
            return 0.0
        return float((tr > 0).sum() / len(tr))

    def profit_factor_from_returns(self, trade_returns: np.ndarray | None = None) -> float:
        tr = self._trade_returns(trade_returns)
        gross_profit = float(tr[tr > 0].sum())
        gross_loss = float(abs(tr[tr < 0].sum()))
        if gross_loss < 1e-10:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def expectancy_from_returns(self, trade_returns: np.ndarray | None = None) -> float:
        tr = self._trade_returns(trade_returns)
        return float(tr.mean()) if len(tr) > 0 else 0.0

    def avg_win_from_returns(self, trade_returns: np.ndarray | None = None) -> float:
        tr = self._trade_returns(trade_returns)
        wins = tr[tr > 0]
        return float(wins.mean()) if len(wins) > 0 else 0.0

    def avg_loss_from_returns(self, trade_returns: np.ndarray | None = None) -> float:
        tr = self._trade_returns(trade_returns)
        losses = tr[tr < 0]
        return float(losses.mean()) if len(losses) > 0 else 0.0

    def consecutive_wins_from_returns(self, trade_returns: np.ndarray | None = None) -> int:
        return self._max_streak(self._trade_returns(trade_returns), positive=True)

    def consecutive_losses_from_returns(self, trade_returns: np.ndarray | None = None) -> int:
        return self._max_streak(self._trade_returns(trade_returns), positive=False)

    # ── Aggregate (array-based) ───────────────────────────────────────

    def compute_all(
        self,
        total_trades: int = 0,
        winning_trades: int = 0,
        trade_returns: np.ndarray | None = None,
    ) -> BacktestMetrics:
        """Compute the full :class:`BacktestMetrics` dataclass."""
        tr = self._trade_returns(trade_returns)
        if total_trades == 0:
            total_trades = len(tr)
        if winning_trades == 0 and len(tr) > 0:
            winning_trades = int((tr > 0).sum())
        losing_trades = total_trades - winning_trades

        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=self.win_rate_from_returns(tr),
            total_return=self.total_return(),
            annualized_return=self.annualized_return(),
            sharpe_ratio=self.sharpe_ratio(),
            sortino_ratio=self.sortino_ratio(),
            max_drawdown=self.max_drawdown(),
            profit_factor=self.profit_factor_from_returns(tr),
            avg_win=self.avg_win_from_returns(tr),
            avg_loss=self.avg_loss_from_returns(tr),
            expectancy=self.expectancy_from_returns(tr),
            calmar_ratio=self.calmar_ratio(),
        )

    # ──────────────────────────────────────────────────────────────────
    # DataFrame-based API (legacy / convenience)
    # ──────────────────────────────────────────────────────────────────

    def calculate(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        risk_free_rate: float | None = None,
    ) -> BacktestMetrics:
        """Calculate all performance metrics from equity curve and trade log.

        Args:
            equity_curve: Time-indexed series of portfolio values.
            trades: DataFrame with columns: pnl, pnl_pct, entry_time, exit_time.
            risk_free_rate: Override for annual risk-free rate.
        """
        rfr = risk_free_rate if risk_free_rate is not None else self.RISK_FREE_RATE

        if trades.empty:
            return self._empty_metrics()

        returns = equity_curve.pct_change().dropna()
        winners = trades[trades["pnl"] > 0]
        losers = trades[trades["pnl"] <= 0]

        total_trades = len(trades)
        winning_trades = len(winners)
        losing_trades = len(losers)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        avg_win = float(winners["pnl"].mean()) if len(winners) > 0 else 0.0
        avg_loss = float(abs(losers["pnl"].mean())) if len(losers) > 0 else 0.0

        total_return_val = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0
        n_days = max((equity_curve.index[-1] - equity_curve.index[0]).days, 1)
        annualized = (1 + total_return_val) ** (365.0 / n_days) - 1.0

        sharpe = self._sharpe_ratio_series(returns, rfr)
        sortino = self._sortino_ratio_series(returns, rfr)
        max_dd = self._max_drawdown_series(equity_curve)
        pf = self._profit_factor_df(trades)
        exp = self._expectancy_calc(win_rate, avg_win, avg_loss)
        calmar = annualized / abs(max_dd) if max_dd != 0 else 0.0

        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return_val,
            annualized_return=annualized,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            profit_factor=pf,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=exp,
            calmar_ratio=calmar,
        )

    # ── Series-based helper methods ───────────────────────────────────

    def _sharpe_ratio_series(self, returns: pd.Series, risk_free_rate: float) -> float:
        if returns.std() == 0 or len(returns) < 2:
            return 0.0
        daily_rf = (1 + risk_free_rate) ** (1 / self.TRADING_DAYS_PER_YEAR) - 1
        excess = returns - daily_rf
        return float(excess.mean() / excess.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR))

    def _sortino_ratio_series(self, returns: pd.Series, risk_free_rate: float) -> float:
        daily_rf = (1 + risk_free_rate) ** (1 / self.TRADING_DAYS_PER_YEAR) - 1
        excess = returns - daily_rf
        downside = excess[excess < 0]
        if len(downside) < 2 or downside.std() == 0:
            return 0.0
        return float(excess.mean() / downside.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR))

    @staticmethod
    def _max_drawdown_series(equity_curve: pd.Series) -> float:
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return float(drawdown.min())

    @staticmethod
    def _profit_factor_df(trades: pd.DataFrame) -> float:
        gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades[trades["pnl"] <= 0]["pnl"].sum())
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    @staticmethod
    def _expectancy_calc(win_rate: float, avg_win: float, avg_loss: float) -> float:
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # ── Convenience methods (DataFrame) ───────────────────────────────

    def rolling_sharpe(self, returns: pd.Series, window: int = 63) -> pd.Series:
        """Rolling Sharpe ratio (default: quarterly)."""
        daily_rf = (1 + self.RISK_FREE_RATE) ** (1 / self.TRADING_DAYS_PER_YEAR) - 1
        excess = returns - daily_rf
        rolling_mean = excess.rolling(window).mean()
        rolling_std = excess.rolling(window).std()
        return rolling_mean / rolling_std * np.sqrt(self.TRADING_DAYS_PER_YEAR)

    def drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown at each point in time."""
        peak = equity_curve.expanding().max()
        return (equity_curve - peak) / peak

    def consecutive_wins_losses(self, trades: pd.DataFrame) -> dict[str, int]:
        """Calculate max consecutive wins and losses from a trades DataFrame."""
        if trades.empty:
            return {"max_consecutive_wins": 0, "max_consecutive_losses": 0}

        is_win = (trades["pnl"] > 0).astype(int)
        groups = (is_win != is_win.shift()).cumsum()

        win_streaks = is_win.groupby(groups).sum()
        loss_streaks = (1 - is_win).groupby(groups).sum()

        return {
            "max_consecutive_wins": int(win_streaks.max()) if len(win_streaks) > 0 else 0,
            "max_consecutive_losses": int(loss_streaks.max()) if len(loss_streaks) > 0 else 0,
        }

    def monthly_returns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """Calculate monthly returns heatmap data."""
        monthly = equity_curve.resample("ME").last().pct_change()
        if monthly.empty:
            return pd.DataFrame()

        result = pd.DataFrame(
            {
                "year": monthly.index.year,
                "month": monthly.index.month,
                "return": monthly.values,
            }
        )
        return result.pivot(index="year", columns="month", values="return")

    def recovery_factor(self, equity_curve: pd.Series) -> float:
        """Net profit / max drawdown (DataFrame-based)."""
        net_profit = equity_curve.iloc[-1] - equity_curve.iloc[0]
        max_dd_abs = abs(self._max_drawdown_series(equity_curve) * equity_curve.max())
        if max_dd_abs == 0:
            return 0.0
        return float(net_profit / max_dd_abs)

    # ── Internal ──────────────────────────────────────────────────────

    def _require_returns(self) -> np.ndarray:
        if self._returns is None:
            raise ValueError("MetricsCalculator was not initialised with a returns array.")
        return self._returns

    def _require_equity(self) -> np.ndarray:
        if self._equity is None:
            raise ValueError("MetricsCalculator was not initialised with an equity curve array.")
        return self._equity

    def _trade_returns(self, tr: np.ndarray | None) -> np.ndarray:
        if tr is not None:
            return np.asarray(tr, dtype=np.float64).ravel()
        return self._require_returns()

    @staticmethod
    def _max_streak(arr: np.ndarray, *, positive: bool) -> int:
        if len(arr) == 0:
            return 0
        mask = arr > 0 if positive else arr < 0
        max_run = 0
        current = 0
        for val in mask:
            if val:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return max_run

    @staticmethod
    def _empty_metrics() -> BacktestMetrics:
        return BacktestMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            expectancy=0.0,
        )
