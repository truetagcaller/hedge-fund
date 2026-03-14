"""Tests for signal generation."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from hedgefund.types import MarketRegime, SentimentResult, SignalAction


class TestRuleBasedSignals:
    @pytest.fixture
    def bullish_df(self, sample_ohlcv_df):
        """Create a DataFrame with clear bullish signals."""
        from hedgefund.features.technical import TechnicalFeatures

        tf = TechnicalFeatures()
        return tf.transform(sample_ohlcv_df)

    def test_signal_has_required_fields(self, bullish_df, sample_sentiment):
        from hedgefund.signals.rule_signal import RuleBasedSignalGenerator

        gen = RuleBasedSignalGenerator(
            underlying="SPY",
            min_confidence=0.5,
            min_risk_reward=1.5,
        )

        import asyncio
        signals = asyncio.get_event_loop().run_until_complete(
            gen.generate(
                features=bullish_df,
                regime=MarketRegime.LOW_VOL_BULLISH,
                sentiment=sample_sentiment,
            )
        )

        for signal in signals:
            assert signal.entry_price > 0
            assert signal.stop_loss > 0
            assert signal.target_price > 0
            assert signal.risk_reward_ratio >= 1.5
            assert 0 <= signal.confidence <= 1.0
            assert signal.reasoning
            assert signal.action != SignalAction.NO_TRADE

    def test_no_trade_in_choppy_market(self, sample_ohlcv_df, sample_sentiment):
        """Choppy market should produce fewer/no signals."""
        from hedgefund.features.technical import TechnicalFeatures
        from hedgefund.signals.rule_signal import RuleBasedSignalGenerator

        # Create choppy data
        choppy = sample_ohlcv_df.copy()
        noise = np.random.normal(0, 0.01, len(choppy))
        choppy["close"] = choppy["close"].iloc[0] * (1 + noise.cumsum() * 0.001)
        choppy["open"] = choppy["close"] * (1 + np.random.normal(0, 0.0001, len(choppy)))
        choppy["high"] = choppy[["open", "close"]].max(axis=1) * 1.001
        choppy["low"] = choppy[["open", "close"]].min(axis=1) * 0.999

        tf = TechnicalFeatures()
        features = tf.transform(choppy)

        gen = RuleBasedSignalGenerator(
            underlying="SPY",
            min_confidence=0.8,  # High threshold
            min_risk_reward=2.0,
        )

        import asyncio
        signals = asyncio.get_event_loop().run_until_complete(
            gen.generate(
                features=features,
                regime=MarketRegime.MEAN_REVERTING,
                sentiment=SentimentResult("SPY", 0.0, 0.3, "test"),
            )
        )
        # Should have fewer signals with high confidence threshold
        assert len(signals) <= 5
