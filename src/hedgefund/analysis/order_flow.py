"""Order-flow analysis: volume profile, VWAP deviation, block detection,
smart money footprints, delta, and cumulative delta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


@dataclass
class OrderFlowAnalyzer:
    """Stateless analyser for order-flow and volume-based signals.

    Column conventions
    ------------------
    OHLCV bars : ``open, high, low, close, volume``
    Tick/trade data : ``price, volume, timestamp``
    With tick rule applied : ``buy_volume, sell_volume``
    """

    # Default number of bins for volume profile
    profile_bins: int = 50
    # Minimum shares for a trade to be a "block"
    block_threshold: int = 10_000
    # Rolling window (in bars) for deviation metrics
    rolling_window: int = 20

    # ── Volume Profile ───────────────────────────────────────────────────

    def volume_profile(
        self,
        df: pd.DataFrame,
        bins: int | None = None,
    ) -> pd.DataFrame:
        """Build a volume-at-price histogram from OHLCV data.

        Returns a DataFrame with ``price_level, volume, pct_of_total``.
        """
        bins = bins or self.profile_bins
        typical = (df["high"] + df["low"] + df["close"]) / 3.0

        price_min, price_max = float(typical.min()), float(typical.max())
        if price_min == price_max:
            return pd.DataFrame(
                {"price_level": [price_min], "volume": [df["volume"].sum()], "pct_of_total": [1.0]}
            )

        edges = np.linspace(price_min, price_max, bins + 1)
        bucket = np.digitize(typical.to_numpy(), edges) - 1
        bucket = np.clip(bucket, 0, bins - 1)

        vol_arr = df["volume"].to_numpy(dtype=np.float64)
        profile = np.zeros(bins, dtype=np.float64)
        for i in range(len(bucket)):
            profile[bucket[i]] += vol_arr[i]

        mid_prices = (edges[:-1] + edges[1:]) / 2.0
        total = profile.sum()
        return pd.DataFrame(
            {
                "price_level": mid_prices,
                "volume": profile,
                "pct_of_total": profile / max(total, 1.0),
            }
        )

    def point_of_control(self, df: pd.DataFrame) -> float:
        """Price level with the highest traded volume (POC)."""
        vp = self.volume_profile(df)
        idx = vp["volume"].idxmax()
        return float(vp.loc[idx, "price_level"])

    def value_area(
        self, df: pd.DataFrame, pct: float = 0.70
    ) -> tuple[float, float]:
        """Return (VAL, VAH) encompassing *pct* of total volume around the POC."""
        vp = self.volume_profile(df).sort_values("volume", ascending=False)
        total = vp["volume"].sum()
        target = total * pct

        cum = 0.0
        selected_prices: list[float] = []
        for _, row in vp.iterrows():
            cum += row["volume"]
            selected_prices.append(row["price_level"])
            if cum >= target:
                break

        return float(min(selected_prices)), float(max(selected_prices))

    # ── VWAP deviation ───────────────────────────────────────────────────

    @staticmethod
    def vwap_deviation(df: pd.DataFrame) -> pd.DataFrame:
        """Add columns for cumulative VWAP and price deviation from it."""
        df = df.copy()
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_tp_vol = (typical * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum().replace(0, np.nan)
        vwap = cum_tp_vol / cum_vol

        df["vwap"] = vwap
        df["vwap_deviation"] = df["close"] - vwap
        df["vwap_deviation_pct"] = df["vwap_deviation"] / vwap.replace(0, np.nan)
        return df

    # ── Large block detection ────────────────────────────────────────────

    def detect_blocks(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Flag trades whose volume exceeds ``block_threshold``.

        Expects ``price, volume, timestamp``.
        """
        trades = trades.copy()
        trades["is_block"] = trades["volume"] >= self.block_threshold
        trades["block_value"] = np.where(
            trades["is_block"], trades["price"] * trades["volume"], 0.0
        )
        return trades

    def block_summary(self, trades: pd.DataFrame) -> dict:
        """Return summary statistics for detected block trades."""
        blocks = self.detect_blocks(trades)
        block_rows = blocks[blocks["is_block"]]
        return {
            "block_count": len(block_rows),
            "block_total_volume": int(block_rows["volume"].sum()),
            "block_total_value": float(block_rows["block_value"].sum()),
            "block_avg_size": float(block_rows["volume"].mean()) if len(block_rows) else 0.0,
            "block_pct_of_volume": float(
                block_rows["volume"].sum() / max(blocks["volume"].sum(), 1)
            ),
        }

    # ── Smart money footprints ───────────────────────────────────────────

    def smart_money_footprint(self, df: pd.DataFrame) -> pd.DataFrame:
        """Heuristic: large-volume bars on narrow range suggest informed flow.

        Adds ``smart_money_score`` (0-1) based on volume/range ratio relative
        to recent history.
        """
        df = df.copy()
        bar_range = (df["high"] - df["low"]).replace(0, np.nan)
        vr_ratio = df["volume"] / bar_range

        rolling_mean = vr_ratio.rolling(window=self.rolling_window, min_periods=1).mean()
        rolling_std = vr_ratio.rolling(window=self.rolling_window, min_periods=1).std().replace(0, np.nan)

        z_score = (vr_ratio - rolling_mean) / rolling_std
        # Sigmoid normalisation into [0, 1]
        df["smart_money_score"] = 1.0 / (1.0 + np.exp(-z_score.fillna(0)))
        return df

    # ── Delta (buying vs selling pressure) ───────────────────────────────

    @staticmethod
    def compute_delta(df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-bar delta and cumulative delta.

        Expects ``buy_volume`` and ``sell_volume`` columns (e.g. from tick
        rule classification).
        """
        df = df.copy()
        df["delta"] = df["buy_volume"] - df["sell_volume"]
        df["cumulative_delta"] = df["delta"].cumsum()
        return df

    @staticmethod
    def estimate_delta_from_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Estimate buy/sell split from OHLCV when tick data is unavailable.

        Uses the close position within the high-low range as a proxy for the
        fraction of volume that was buyer-initiated.
        """
        df = df.copy()
        bar_range = (df["high"] - df["low"]).replace(0, np.nan)
        buy_pct = (df["close"] - df["low"]) / bar_range
        buy_pct = buy_pct.fillna(0.5).clip(0.0, 1.0)

        df["est_buy_volume"] = (df["volume"] * buy_pct).astype(int)
        df["est_sell_volume"] = df["volume"] - df["est_buy_volume"]
        df["est_delta"] = df["est_buy_volume"] - df["est_sell_volume"]
        df["est_cumulative_delta"] = df["est_delta"].cumsum()
        return df

    # ── Divergence detection ─────────────────────────────────────────────

    @staticmethod
    def delta_price_divergence(
        df: pd.DataFrame,
        window: int = 10,
    ) -> pd.DataFrame:
        """Detect divergence between price trend and cumulative delta trend.

        A positive divergence (delta rising while price falling) may signal
        accumulation; negative divergence may signal distribution.
        """
        df = df.copy()
        delta_col = "cumulative_delta" if "cumulative_delta" in df.columns else "est_cumulative_delta"
        if delta_col not in df.columns:
            log.warning("delta_divergence_no_delta_column")
            return df

        price_change = df["close"].diff(window)
        delta_change = df[delta_col].diff(window)

        df["delta_price_divergence"] = np.sign(delta_change) - np.sign(price_change)
        return df
