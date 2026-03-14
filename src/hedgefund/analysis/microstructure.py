"""Market microstructure analysis.

Provides order book imbalance, bid-ask spread analysis, tick-level metrics,
and volume-weighted indicators for gauging short-term supply/demand dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


@dataclass
class MarketMicrostructure:
    """Stateless analyser — every method takes a DataFrame and returns an
    enriched copy (or a scalar summary).

    Expected column conventions
    ---------------------------
    Order book snapshots:
        ``bid_price, bid_size, ask_price, ask_size``
    Trade/tick data:
        ``price, volume, timestamp``
    OHLCV bars:
        ``open, high, low, close, volume``
    """

    # ── Order-book imbalance ─────────────────────────────────────────────

    @staticmethod
    def order_book_imbalance(df: pd.DataFrame) -> pd.DataFrame:
        """Compute bid/ask imbalance ratio per row.

        imbalance = (bid_size - ask_size) / (bid_size + ask_size)
        Ranges from -1 (all ask) to +1 (all bid).
        """
        df = df.copy()
        total = df["bid_size"] + df["ask_size"]
        df["book_imbalance"] = (df["bid_size"] - df["ask_size"]) / total.replace(0, np.nan)
        return df

    @staticmethod
    def multi_level_imbalance(
        bids: pd.DataFrame,
        asks: pd.DataFrame,
        levels: int = 5,
    ) -> float:
        """Weighted imbalance across multiple price levels.

        ``bids`` / ``asks`` each have columns ``price, size`` sorted by
        level (best first).  Levels are weighted inversely by distance.
        """
        n = min(levels, len(bids), len(asks))
        if n == 0:
            return 0.0

        weights = 1.0 / np.arange(1, n + 1, dtype=np.float64)
        bid_pressure = float((bids["size"].iloc[:n].to_numpy() * weights).sum())
        ask_pressure = float((asks["size"].iloc[:n].to_numpy() * weights).sum())
        total = bid_pressure + ask_pressure
        if total == 0.0:
            return 0.0
        return (bid_pressure - ask_pressure) / total

    # ── Bid-ask spread analysis ──────────────────────────────────────────

    @staticmethod
    def spread_analysis(df: pd.DataFrame) -> pd.DataFrame:
        """Compute absolute spread, relative spread, and mid-price."""
        df = df.copy()
        df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2.0
        df["spread"] = df["ask_price"] - df["bid_price"]
        df["spread_bps"] = (df["spread"] / df["mid_price"].replace(0, np.nan)) * 10_000.0
        return df

    @staticmethod
    def effective_spread(
        trades: pd.DataFrame,
        quotes: pd.DataFrame,
    ) -> pd.Series:
        """Compute effective spread: 2 * |trade_price - mid_price| for each
        trade, after aligning on the most-recent quote before each trade.

        ``trades``: ``timestamp, price``
        ``quotes``: ``timestamp, bid_price, ask_price``
        """
        quotes = quotes.sort_values("timestamp")
        trades = trades.sort_values("timestamp")

        mid = (quotes["bid_price"] + quotes["ask_price"]) / 2.0
        quotes = quotes.assign(mid_price=mid)

        merged = pd.merge_asof(
            trades[["timestamp", "price"]],
            quotes[["timestamp", "mid_price"]],
            on="timestamp",
            direction="backward",
        )
        return 2.0 * (merged["price"] - merged["mid_price"]).abs()

    # ── Tick analysis ────────────────────────────────────────────────────

    @staticmethod
    def tick_direction(df: pd.DataFrame) -> pd.DataFrame:
        """Classify each trade as uptick (+1), downtick (-1), or zero-tick (0)."""
        df = df.copy()
        diff = df["price"].diff()
        df["tick_direction"] = np.sign(diff).fillna(0).astype(int)
        return df

    @staticmethod
    def tick_rule_volume(df: pd.DataFrame) -> pd.DataFrame:
        """Estimate buy/sell volume using the tick rule.

        Upticks are classified as buyer-initiated, downticks as
        seller-initiated.  Zero-ticks inherit the previous classification.
        """
        df = df.copy()
        diff = df["price"].diff()
        direction = np.sign(diff)
        # Forward-fill zeros
        direction = direction.replace(0, np.nan).ffill().fillna(1.0)
        df["buy_volume"] = np.where(direction > 0, df["volume"], 0)
        df["sell_volume"] = np.where(direction < 0, df["volume"], 0)
        return df

    @staticmethod
    def trade_intensity(df: pd.DataFrame, window: str = "1min") -> pd.DataFrame:
        """Compute number of trades and total volume per time bucket."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        grouped = df.set_index("timestamp").resample(window).agg(
            trade_count=("price", "count"),
            bucket_volume=("volume", "sum"),
        )
        return grouped.reset_index()

    # ── Volume-weighted metrics ──────────────────────────────────────────

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.DataFrame:
        """Cumulative VWAP from OHLCV data."""
        df = df.copy()
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_tp_vol = (typical * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum().replace(0, np.nan)
        df["vwap"] = cum_tp_vol / cum_vol
        return df

    @staticmethod
    def volume_weighted_spread(df: pd.DataFrame) -> Optional[float]:
        """Single scalar: average spread weighted by total size at each level.

        Expects ``bid_price, bid_size, ask_price, ask_size``.
        """
        total_size = df["bid_size"] + df["ask_size"]
        weight_sum = total_size.sum()
        if weight_sum == 0:
            return None
        spread = df["ask_price"] - df["bid_price"]
        return float((spread * total_size).sum() / weight_sum)

    @staticmethod
    def volume_clock(df: pd.DataFrame, bucket_volume: int = 1000) -> pd.DataFrame:
        """Re-sample tick data into volume bars of fixed ``bucket_volume``.

        Returns OHLCV-style bars where each bar contains exactly
        ``bucket_volume`` shares (last bar may be partial).
        """
        df = df.copy().sort_values("timestamp").reset_index(drop=True)
        cum_vol = df["volume"].cumsum()
        df["vol_bucket"] = (cum_vol // bucket_volume).astype(int)

        bars = df.groupby("vol_bucket").agg(
            timestamp=("timestamp", "first"),
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("volume", "sum"),
            trade_count=("price", "count"),
        )
        return bars.reset_index(drop=True)
