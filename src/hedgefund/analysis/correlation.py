"""Cross-asset correlation analysis.

Rolling correlation matrices, correlation breakdown detection, and
cross-asset momentum signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


@dataclass
class CorrelationTracker:
    """Track and analyse rolling correlations across multiple assets.

    Parameters
    ----------
    window : int
        Default rolling window (bars) for correlation calculations.
    breakdown_threshold : float
        Absolute change in correlation (vs. long-term average) that
        constitutes a "breakdown".
    long_term_window : int
        Window for computing the long-term baseline correlation.
    """

    window: int = 21
    breakdown_threshold: float = 0.30
    long_term_window: int = 252

    # ── Rolling correlation matrix ───────────────────────────────────────

    def rolling_correlation(
        self,
        returns: pd.DataFrame,
        window: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Compute pair-wise rolling correlations for each asset pair.

        Parameters
        ----------
        returns : DataFrame
            Each column is the return series of one asset.
        window : int, optional
            Override the default rolling window.

        Returns
        -------
        dict mapping ``"ASSET_A__ASSET_B"`` → Series of rolling correlation.
        """
        w = window or self.window
        result: dict[str, pd.DataFrame] = {}
        cols = list(returns.columns)

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                key = f"{a}__{b}"
                result[key] = returns[a].rolling(window=w).corr(returns[b]).to_frame(name="correlation")

        return result

    def correlation_matrix(
        self,
        returns: pd.DataFrame,
        window: int | None = None,
    ) -> pd.DataFrame:
        """Return the most-recent rolling correlation matrix as a square
        DataFrame (one value per pair, using the last available bar).
        """
        w = window or self.window
        # Use the last `w` rows
        tail = returns.iloc[-w:]
        return tail.corr()

    # ── Correlation breakdown detection ──────────────────────────────────

    def detect_breakdowns(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Identify pairs where the short-term correlation has diverged
        significantly from the long-term average.

        Returns a DataFrame with columns
        ``pair, short_term_corr, long_term_corr, deviation, is_breakdown``.
        """
        cols = list(returns.columns)
        records: list[dict] = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                rolling = returns[a].rolling(window=self.window).corr(returns[b])
                long_rolling = returns[a].rolling(window=self.long_term_window).corr(returns[b])

                short_val = rolling.iloc[-1] if not rolling.empty else np.nan
                long_val = long_rolling.iloc[-1] if not long_rolling.empty else np.nan

                if np.isnan(short_val) or np.isnan(long_val):
                    continue

                deviation = short_val - long_val
                records.append(
                    {
                        "pair": f"{a}__{b}",
                        "short_term_corr": short_val,
                        "long_term_corr": long_val,
                        "deviation": deviation,
                        "is_breakdown": abs(deviation) >= self.breakdown_threshold,
                    }
                )

        result = pd.DataFrame(records)
        if not result.empty:
            breakdown_count = result["is_breakdown"].sum()
            if breakdown_count:
                log.info("correlation_breakdowns_detected", count=int(breakdown_count))
        return result

    # ── Cross-asset momentum ─────────────────────────────────────────────

    @staticmethod
    def cross_asset_momentum(
        returns: pd.DataFrame,
        lookback: int = 21,
    ) -> pd.DataFrame:
        """Rank assets by cumulative return over *lookback* bars.

        Returns a DataFrame with ``asset, cumulative_return, rank``.
        """
        tail = returns.iloc[-lookback:]
        cum = (1.0 + tail).prod() - 1.0

        result = (
            cum.reset_index()
            .rename(columns={"index": "asset", 0: "cumulative_return"})
        )
        result["rank"] = result["cumulative_return"].rank(ascending=False).astype(int)
        result = result.sort_values("rank").reset_index(drop=True)
        return result

    @staticmethod
    def relative_strength(
        asset_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 21,
    ) -> pd.Series:
        """Rolling relative-strength ratio: cumulative asset return /
        cumulative benchmark return over *window* bars.
        """
        asset_cum = (1.0 + asset_returns).rolling(window=window).apply(np.prod, raw=True)
        bench_cum = (1.0 + benchmark_returns).rolling(window=window).apply(np.prod, raw=True)
        return asset_cum / bench_cum.replace(0, np.nan)

    # ── Eigen-decomposition for PCA risk ─────────────────────────────────

    @staticmethod
    def pca_variance_explained(
        returns: pd.DataFrame,
        n_components: int = 3,
    ) -> dict[str, float | list[float]]:
        """Run PCA on the return correlation matrix and return the variance
        explained by the top *n_components*.

        Useful for detecting concentration risk (when one factor dominates).
        """
        corr = returns.corr()
        eigenvalues = np.linalg.eigvalsh(corr.to_numpy())
        eigenvalues = np.sort(eigenvalues)[::-1]  # descending
        total = eigenvalues.sum()
        if total == 0:
            return {"explained_ratios": [], "top_n_total": 0.0}

        ratios = (eigenvalues / total).tolist()
        top_n = sum(ratios[:n_components])
        return {
            "explained_ratios": ratios[:n_components],
            "top_n_total": top_n,
        }
