"""Tests for technical indicators."""

import pytest
import pandas as pd
import numpy as np
from hedgefund.features.technical import TechnicalFeatures


class TestTechnicalFeatures:
    def setup_method(self):
        self.tf = TechnicalFeatures()

    def test_required_columns(self):
        required = self.tf.required_columns()
        assert "close" in required
        assert "high" in required
        assert "low" in required
        assert "volume" in required

    def test_transform_adds_ema_columns(self, sample_ohlcv_df):
        result = self.tf.transform(sample_ohlcv_df)
        assert "ema_9" in result.columns
        assert "ema_21" in result.columns
        assert "ema_50" in result.columns
        assert "ema_200" in result.columns

    def test_transform_adds_rsi(self, sample_ohlcv_df):
        result = self.tf.transform(sample_ohlcv_df)
        assert "rsi" in result.columns
        # RSI should be between 0 and 100
        rsi_valid = result["rsi"].dropna()
        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()

    def test_transform_adds_macd(self, sample_ohlcv_df):
        result = self.tf.transform(sample_ohlcv_df)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

    def test_transform_adds_bollinger(self, sample_ohlcv_df):
        result = self.tf.transform(sample_ohlcv_df)
        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns
        # Upper should always be above lower
        valid = result.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_transform_adds_atr(self, sample_ohlcv_df):
        result = self.tf.transform(sample_ohlcv_df)
        assert "atr" in result.columns
        atr_valid = result["atr"].dropna()
        assert (atr_valid >= 0).all()

    def test_transform_adds_vwap(self, sample_ohlcv_df):
        result = self.tf.transform(sample_ohlcv_df)
        assert "vwap" in result.columns

    def test_transform_preserves_original_columns(self, sample_ohlcv_df):
        original_cols = set(sample_ohlcv_df.columns)
        result = self.tf.transform(sample_ohlcv_df)
        assert original_cols.issubset(set(result.columns))

    def test_transform_preserves_index(self, sample_ohlcv_df):
        result = self.tf.transform(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)
