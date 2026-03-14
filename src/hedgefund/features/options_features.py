"""Options-flow and sentiment features derived from option chain data.

All transformers expect a DataFrame where each row represents one option
contract in the chain (both calls and puts), with columns documented in
``required_columns``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import structlog

from hedgefund.features.base import FeatureTransformer
from hedgefund.types import OptionType

log = structlog.get_logger(__name__)


class OptionsFeatures(FeatureTransformer):
    """Derive trading-relevant features from an option chain snapshot.

    Produced columns (summary level — one value per underlying/expiry):
    * ``put_call_ratio``
    * ``oi_change_rate``
    * ``max_pain``
    * ``total_gex``
    * ``dealer_delta``
    * ``unusual_activity`` (bool)
    * ``iv_crush``  (bool)
    * ``iv_expansion`` (bool)
    * ``skew_25d``
    """

    # Thresholds
    UNUSUAL_VOLUME_MULTIPLE: float = 2.0
    IV_CRUSH_THRESHOLD: float = -0.10  # -10 %
    IV_EXPANSION_THRESHOLD: float = 0.10  # +10 %

    @property
    def name(self) -> str:
        return "OptionsFeatures"

    def required_columns(self) -> list[str]:
        return [
            "strike",
            "option_type",
            "volume",
            "open_interest",
            "iv",
            "delta",
            "gamma",
            "mid_price",
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich the chain DataFrame with derived option features."""
        df = df.copy()
        df = self._normalise_option_type(df)

        calls = df[df["_otype"] == OptionType.CALL]
        puts = df[df["_otype"] == OptionType.PUT]

        df["put_call_ratio"] = self._put_call_ratio(calls, puts)
        df["oi_change_rate"] = self._oi_change_rate(df)
        df["max_pain"] = self._max_pain(df)
        df["total_gex"] = self._gamma_exposure(df)
        df["dealer_delta"] = self._dealer_positioning(df)
        df["unusual_activity"] = self._unusual_activity(df)
        df = self._iv_crush_expansion(df)
        df["skew_25d"] = self._skew_25d(df)

        df.drop(columns=["_otype"], inplace=True, errors="ignore")
        return df

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_option_type(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure an internal ``_otype`` column of :class:`OptionType`."""
        def _to_otype(val):
            if isinstance(val, OptionType):
                return val
            return OptionType(str(val).upper())

        df["_otype"] = df["option_type"].apply(_to_otype)
        return df

    # ── Put/Call ratio ───────────────────────────────────────────────────

    @staticmethod
    def _put_call_ratio(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
        total_call_vol = calls["volume"].sum()
        total_put_vol = puts["volume"].sum()
        if total_call_vol == 0:
            return np.nan
        return total_put_vol / total_call_vol

    # ── OI change rate ───────────────────────────────────────────────────

    @staticmethod
    def _oi_change_rate(df: pd.DataFrame) -> float:
        """Approximate OI change rate as volume / open_interest."""
        total_oi = df["open_interest"].sum()
        if total_oi == 0:
            return np.nan
        return df["volume"].sum() / total_oi

    # ── Max pain ─────────────────────────────────────────────────────────

    @staticmethod
    def _max_pain(df: pd.DataFrame) -> Optional[float]:
        """Find the strike at which total option holder loss is maximised
        (i.e. the strike where total intrinsic-value payout is minimised).
        """
        strikes = df["strike"].unique()
        if len(strikes) == 0:
            return None

        calls = df[df["_otype"] == OptionType.CALL]
        puts = df[df["_otype"] == OptionType.PUT]

        min_pain = np.inf
        pain_strike: Optional[float] = None

        for test_price in strikes:
            call_pain = (
                calls.apply(
                    lambda r, tp=test_price: max(tp - r["strike"], 0.0) * r["open_interest"],
                    axis=1,
                ).sum()
            )
            put_pain = (
                puts.apply(
                    lambda r, tp=test_price: max(r["strike"] - tp, 0.0) * r["open_interest"],
                    axis=1,
                ).sum()
            )
            total = call_pain + put_pain
            if total < min_pain:
                min_pain = total
                pain_strike = test_price

        return pain_strike

    # ── Gamma Exposure (GEX) ─────────────────────────────────────────────

    @staticmethod
    def _gamma_exposure(df: pd.DataFrame) -> float:
        """Net GEX = sum(gamma * OI * 100 * spot_proxy * sign).

        Calls contribute positive dealer gamma; puts negative
        (assuming dealers are short options to retail).
        """
        sign = df["_otype"].apply(lambda ot: 1.0 if ot == OptionType.CALL else -1.0)
        gex = df["gamma"] * df["open_interest"] * 100.0 * df["strike"] * sign
        return float(gex.sum())

    # ── Dealer positioning estimate ──────────────────────────────────────

    @staticmethod
    def _dealer_positioning(df: pd.DataFrame) -> float:
        """Estimate net dealer delta assuming dealers are short options.

        dealer_delta = -sum(delta * OI * 100)
        """
        return float(-(df["delta"] * df["open_interest"] * 100.0).sum())

    # ── Unusual activity ─────────────────────────────────────────────────

    def _unusual_activity(self, df: pd.DataFrame) -> bool:
        """Flag when any contract's volume exceeds OI by a configurable multiple."""
        oi_safe = df["open_interest"].replace(0, np.nan)
        ratio = df["volume"] / oi_safe
        return bool((ratio > self.UNUSUAL_VOLUME_MULTIPLE).any())

    # ── IV crush / expansion ─────────────────────────────────────────────

    def _iv_crush_expansion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect IV crush or expansion from a ``prev_iv`` column if available."""
        if "prev_iv" not in df.columns:
            df["iv_crush"] = False
            df["iv_expansion"] = False
            return df

        iv_change = (df["iv"] - df["prev_iv"]) / df["prev_iv"].replace(0, np.nan)
        df["iv_change_pct"] = iv_change
        df["iv_crush"] = iv_change < self.IV_CRUSH_THRESHOLD
        df["iv_expansion"] = iv_change > self.IV_EXPANSION_THRESHOLD
        return df

    # ── 25-delta skew ────────────────────────────────────────────────────

    @staticmethod
    def _skew_25d(df: pd.DataFrame) -> float:
        """Measure 25-delta put/call IV skew.

        skew = IV(25d put) - IV(25d call)
        """
        calls = df[df["_otype"] == OptionType.CALL].copy()
        puts = df[df["_otype"] == OptionType.PUT].copy()

        if calls.empty or puts.empty:
            return np.nan

        # Find the contract closest to |delta| = 0.25
        calls["abs_delta_diff"] = (calls["delta"].abs() - 0.25).abs()
        puts["abs_delta_diff"] = (puts["delta"].abs() - 0.25).abs()

        call_25d_iv = calls.loc[calls["abs_delta_diff"].idxmin(), "iv"]
        put_25d_iv = puts.loc[puts["abs_delta_diff"].idxmin(), "iv"]

        return float(put_25d_iv - call_25d_iv)
