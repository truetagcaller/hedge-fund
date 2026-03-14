"""Volatility feature transformers.

Implements historical volatility (multiple windows), realised volatility,
Parkinson, Yang-Zhang estimators, vol-of-vol, IV rank, and IV percentile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hedgefund.features.base import FeatureTransformer


class VolatilityFeatures(FeatureTransformer):
    """Comprehensive volatility feature set.

    Parameters
    ----------
    windows : tuple of int
        Rolling windows (in bars) for historical volatility.
    annualisation_factor : float
        Bars per year for annualising.  Default 252 (daily bars).
    iv_lookback : int
        Lookback window (in bars) for IV rank / IV percentile.
    """

    def __init__(
        self,
        windows: tuple[int, ...] = (5, 10, 21, 63),
        annualisation_factor: float = 252.0,
        iv_lookback: int = 252,
    ) -> None:
        self._windows = windows
        self._ann = annualisation_factor
        self._iv_lookback = iv_lookback

    @property
    def name(self) -> str:
        return "VolatilityFeatures"

    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        log_ret = np.log(df["close"] / df["close"].shift(1))
        df["log_return"] = log_ret

        df = self._historical_vol(df, log_ret)
        df = self._realized_vol(df, log_ret)
        df = self._parkinson(df)
        df = self._yang_zhang(df)
        df = self._vol_of_vol(df)

        # IV-based features only when an 'iv' column is present.
        if "iv" in df.columns:
            df = self._iv_rank(df)
            df = self._iv_percentile(df)

        return df

    # ── Historical (close-to-close) volatility ───────────────────────────

    def _historical_vol(self, df: pd.DataFrame, log_ret: pd.Series) -> pd.DataFrame:
        for w in self._windows:
            df[f"hvol_{w}"] = log_ret.rolling(window=w).std() * np.sqrt(self._ann)
        return df

    # ── Realised volatility (sum of squared returns) ─────────────────────

    def _realized_vol(self, df: pd.DataFrame, log_ret: pd.Series) -> pd.DataFrame:
        sq = log_ret ** 2
        for w in self._windows:
            df[f"rvol_{w}"] = np.sqrt(sq.rolling(window=w).sum() * (self._ann / w))
        return df

    # ── Parkinson estimator (uses high/low) ──────────────────────────────

    def _parkinson(self, df: pd.DataFrame) -> pd.DataFrame:
        hl_ratio = np.log(df["high"] / df["low"])
        factor = 1.0 / (4.0 * np.log(2.0))
        for w in self._windows:
            df[f"parkinson_{w}"] = np.sqrt(
                factor * (hl_ratio ** 2).rolling(window=w).mean() * self._ann
            )
        return df

    # ── Yang-Zhang estimator ─────────────────────────────────────────────

    def _yang_zhang(self, df: pd.DataFrame) -> pd.DataFrame:
        """Yang-Zhang (2000) volatility estimator combining overnight,
        open-to-close, and Rogers-Satchell components."""
        for w in self._windows:
            log_oc = np.log(df["open"] / df["close"].shift(1))  # overnight
            log_co = np.log(df["close"] / df["open"])  # open-to-close

            # Rogers-Satchell
            log_ho = np.log(df["high"] / df["open"])
            log_lo = np.log(df["low"] / df["open"])
            log_hc = np.log(df["high"] / df["close"])
            log_lc = np.log(df["low"] / df["close"])
            rs = (log_ho * log_hc + log_lo * log_lc).rolling(window=w).mean()

            var_oc = log_oc.rolling(window=w).var(ddof=1)
            var_co = log_co.rolling(window=w).var(ddof=1)

            k = 0.34 / (1.34 + (w + 1) / (w - 1))
            yz_var = var_oc + k * var_co + (1.0 - k) * rs
            df[f"yang_zhang_{w}"] = np.sqrt(yz_var.clip(lower=0.0) * self._ann)
        return df

    # ── Vol-of-vol ───────────────────────────────────────────────────────

    def _vol_of_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling standard deviation of the shortest-window historical vol."""
        base_col = f"hvol_{self._windows[0]}"
        if base_col in df.columns:
            for w in self._windows:
                df[f"vov_{w}"] = df[base_col].rolling(window=w).std()
        return df

    # ── IV Rank ──────────────────────────────────────────────────────────

    def _iv_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """IV Rank = (current IV - 52-week low) / (52-week high - 52-week low)."""
        lb = self._iv_lookback
        iv = df["iv"]
        rolling_min = iv.rolling(window=lb, min_periods=1).min()
        rolling_max = iv.rolling(window=lb, min_periods=1).max()
        iv_range = (rolling_max - rolling_min).replace(0, np.nan)
        df["iv_rank"] = (iv - rolling_min) / iv_range
        return df

    # ── IV Percentile ────────────────────────────────────────────────────

    def _iv_percentile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Percentage of days in the lookback where IV was lower than today."""
        lb = self._iv_lookback
        iv = df["iv"]

        def _pct(window: pd.Series) -> float:
            current = window.iloc[-1]
            return float((window.iloc[:-1] < current).sum()) / max(len(window) - 1, 1)

        df["iv_percentile"] = iv.rolling(window=lb, min_periods=2).apply(_pct, raw=False)
        return df
