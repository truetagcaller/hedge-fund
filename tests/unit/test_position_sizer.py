"""Tests for position sizing algorithms."""

import pytest
from hedgefund.risk.position_sizer import PositionSizer


class TestVolatilityAdjustedSizing:
    def setup_method(self):
        self.sizer = PositionSizer(
            risk_per_trade_pct=0.01,
            kelly_fraction=0.25,
        )

    def test_higher_volatility_smaller_position(self):
        """Higher ATR should result in smaller position."""
        qty_low_vol = self.sizer.volatility_adjusted(
            capital=10_000_000,
            atr=1.0,
            price=5.0,
            multiplier=100,
        )
        qty_high_vol = self.sizer.volatility_adjusted(
            capital=10_000_000,
            atr=3.0,
            price=5.0,
            multiplier=100,
        )
        assert qty_low_vol > qty_high_vol

    def test_position_respects_capital_limit(self):
        qty = self.sizer.volatility_adjusted(
            capital=100_000,
            atr=0.5,
            price=50.0,
            multiplier=100,
        )
        # Total cost should not exceed capital
        total_cost = qty * 50.0 * 100
        assert total_cost <= 100_000

    def test_zero_atr_returns_zero(self):
        qty = self.sizer.volatility_adjusted(
            capital=10_000_000,
            atr=0.0,
            price=5.0,
            multiplier=100,
        )
        assert qty == 0
