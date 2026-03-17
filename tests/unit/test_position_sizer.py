"""Tests for position sizing algorithms."""

from hedgefund.risk.position_sizer import PositionSizer


class TestVolatilityAdjustedSizing:
    def setup_method(self):
        self.sizer = PositionSizer(
            default_risk_pct=0.01,
            kelly_fraction=0.25,
        )

    def test_higher_volatility_smaller_position(self):
        """Higher ATR should result in smaller position."""
        result_low = self.sizer.volatility_adjusted(
            capital=10_000_000,
            atr=1.0,
            entry_price=5.0,
            multiplier=100,
        )
        result_high = self.sizer.volatility_adjusted(
            capital=10_000_000,
            atr=3.0,
            entry_price=5.0,
            multiplier=100,
        )
        assert result_low.quantity > result_high.quantity

    def test_position_respects_capital_limit(self):
        result = self.sizer.volatility_adjusted(
            capital=100_000,
            atr=0.5,
            entry_price=50.0,
            multiplier=100,
        )
        # Total cost should not exceed capital
        total_cost = result.quantity * 50.0 * 100
        assert total_cost <= 100_000

    def test_zero_atr_returns_zero(self):
        result = self.sizer.volatility_adjusted(
            capital=10_000_000,
            atr=0.0,
            entry_price=5.0,
            multiplier=100,
        )
        assert result.quantity == 0
