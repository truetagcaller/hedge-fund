"""Technical indicator features — implemented from scratch using pandas/numpy.

No TA-Lib dependency; every indicator is computed directly for maximum
portability across environments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hedgefund.features.base import FeatureTransformer


class TechnicalFeatures(FeatureTransformer):
    """Compute a comprehensive set of technical indicators.

    Indicators produced
    -------------------
    * EMA 9 / 21 / 50 / 200
    * VWAP (intraday cumulative)
    * RSI(14)
    * MACD(12, 26, 9)  — line, signal, histogram
    * Bollinger Bands(20, 2)  — upper, mid, lower, %B, bandwidth
    * ATR(14)
    * Supertrend(10, 3)
    """

    # ── Configuration ────────────────────────────────────────────────────
    EMA_SPANS: tuple[int, ...] = (9, 21, 50, 200)
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    ATR_PERIOD: int = 14
    SUPERTREND_PERIOD: int = 10
    SUPERTREND_MULTIPLIER: float = 3.0

    # ── FeatureTransformer interface ─────────────────────────────────────

    @property
    def name(self) -> str:
        return "TechnicalFeatures"

    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._add_ema(df)
        df = self._add_vwap(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger(df)
        df = self._add_atr(df)
        df = self._add_supertrend(df)
        return df

    # ── EMA ──────────────────────────────────────────────────────────────

    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        for span in self.EMA_SPANS:
            df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()
        return df

    # ── VWAP ─────────────────────────────────────────────────────────────

    @staticmethod
    def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_tp_vol = (typical * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)
        return df

    # ── RSI ──────────────────────────────────────────────────────────────

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        avg_gain = gain.ewm(alpha=1.0 / self.RSI_PERIOD, min_periods=self.RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.RSI_PERIOD, min_periods=self.RSI_PERIOD, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
        return df

    # ── MACD ─────────────────────────────────────────────────────────────

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=self.MACD_FAST, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.MACD_SLOW, adjust=False).mean()
        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd_line"].ewm(span=self.MACD_SIGNAL, adjust=False).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
        return df

    # ── Bollinger Bands ──────────────────────────────────────────────────

    def _add_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        sma = df["close"].rolling(window=self.BB_PERIOD).mean()
        std = df["close"].rolling(window=self.BB_PERIOD).std(ddof=0)

        df["bb_upper"] = sma + self.BB_STD * std
        df["bb_mid"] = sma
        df["bb_lower"] = sma - self.BB_STD * std

        band_width = df["bb_upper"] - df["bb_lower"]
        df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / band_width.replace(0, np.nan)
        df["bb_bandwidth"] = band_width / sma.replace(0, np.nan)
        return df

    # ── ATR ──────────────────────────────────────────────────────────────

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr_14"] = tr.ewm(span=self.ATR_PERIOD, adjust=False).mean()
        return df

    # ── Supertrend ───────────────────────────────────────────────────────

    def _add_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.SUPERTREND_PERIOD
        multiplier = self.SUPERTREND_MULTIPLIER

        hl2 = (df["high"] + df["low"]) / 2.0

        # ATR for supertrend (may differ from the global ATR column)
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        basic_upper = hl2 + multiplier * atr
        basic_lower = hl2 - multiplier * atr

        n = len(df)
        upper_band = np.empty(n, dtype=np.float64)
        lower_band = np.empty(n, dtype=np.float64)
        supertrend = np.empty(n, dtype=np.float64)
        direction = np.empty(n, dtype=np.float64)

        close_arr = df["close"].to_numpy(dtype=np.float64)
        bu = basic_upper.to_numpy(dtype=np.float64)
        bl = basic_lower.to_numpy(dtype=np.float64)

        upper_band[0] = bu[0]
        lower_band[0] = bl[0]
        supertrend[0] = bu[0]
        direction[0] = -1.0  # start bearish

        for i in range(1, n):
            # Upper band: ratchet down
            upper_band[i] = (
                min(bu[i], upper_band[i - 1])
                if close_arr[i - 1] <= upper_band[i - 1]
                else bu[i]
            )
            # Lower band: ratchet up
            lower_band[i] = (
                max(bl[i], lower_band[i - 1])
                if close_arr[i - 1] >= lower_band[i - 1]
                else bl[i]
            )

            if supertrend[i - 1] == upper_band[i - 1]:
                # Was bearish
                if close_arr[i] > upper_band[i]:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1.0
                else:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1.0
            else:
                # Was bullish
                if close_arr[i] < lower_band[i]:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1.0
                else:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1.0

        df["supertrend"] = supertrend
        df["supertrend_direction"] = direction  # 1 = bullish, -1 = bearish
        return df
