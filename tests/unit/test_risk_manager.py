"""Tests for risk management system."""

import pytest
from datetime import datetime, timezone

from hedgefund.risk.position_sizer import PositionSizer
from hedgefund.risk.drawdown import DrawdownMonitor
from hedgefund.risk.limits import RiskLimits
from hedgefund.types import (
    TradeSignal,
    SignalAction,
    SignalDirection,
    PortfolioSnapshot,
    Position,
    OptionContract,
    OptionType,
    Greeks,
)


class TestPositionSizer:
    def setup_method(self):
        self.sizer = PositionSizer(
            default_risk_pct=0.01,
            kelly_fraction=0.25,
        )

    def test_fixed_fraction_sizing(self, sample_trade_signal, sample_portfolio):
        risk_per_contract = abs(
            sample_trade_signal.entry_price - sample_trade_signal.stop_loss
        ) * 100
        result = self.sizer.fixed_fraction(
            capital=sample_portfolio.net_liquidation,
            risk_per_contract=risk_per_contract,
        )
        # With $10M capital, 1% risk = $100K max risk
        expected_max = int(100_000 / risk_per_contract)
        assert result.quantity <= expected_max
        assert result.quantity > 0

    def test_zero_risk_returns_zero(self, sample_portfolio):
        result = self.sizer.fixed_fraction(
            capital=sample_portfolio.net_liquidation,
            risk_per_contract=0.0,
        )
        assert result.quantity == 0

    def test_kelly_criterion(self):
        result = self.sizer.kelly_criterion(
            capital=10_000_000,
            win_rate=0.6,
            avg_win=2.0,
            avg_loss=1.0,
            risk_per_contract=500,
        )
        assert result.quantity > 0


class TestDrawdownMonitor:
    def setup_method(self):
        self.monitor = DrawdownMonitor(
            max_drawdown_pct=0.10,
            recovery_threshold_pct=0.05,
        )

    def test_no_drawdown_initially(self):
        assert self.monitor.is_trading_allowed
        assert self.monitor.current_drawdown() == 0.0

    def test_update_equity(self):
        self.monitor.update(10_000_000)
        self.monitor.update(10_500_000)
        assert self.monitor.high_water_mark == 10_500_000

    def test_circuit_breaker_trips(self):
        self.monitor.update(10_000_000)
        self.monitor.update(8_900_000)  # 11% drawdown
        assert not self.monitor.is_trading_allowed

    def test_circuit_breaker_below_threshold(self):
        self.monitor.update(10_000_000)
        self.monitor.update(9_500_000)  # 5% drawdown
        assert self.monitor.is_trading_allowed


class TestRiskLimits:
    def setup_method(self):
        self.limits = RiskLimits(
            risk_per_trade_pct=0.01,
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.10,
            max_concurrent_positions=20,
            max_delta_exposure=500,
            max_gamma_exposure=100,
            max_vega_exposure=50000,
        )

    def test_trade_within_limits(self, sample_trade_signal, sample_portfolio):
        trade_risk = abs(
            sample_trade_signal.entry_price - sample_trade_signal.stop_loss
        ) * 100  # multiplier
        result = self.limits.check_per_trade_risk(
            trade_risk=trade_risk,
            portfolio_value=sample_portfolio.net_liquidation,
        )
        assert result.passed, result.message

    def test_daily_loss_exceeded(self):
        portfolio = PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            cash=9_500_000,
            net_liquidation=9_600_000,
            positions=[],
            daily_pnl=-350_000,  # 3.5% daily loss
            drawdown_pct=0.04,
            high_water_mark=10_000_000,
        )
        result = self.limits.check_daily_loss(
            daily_pnl=portfolio.daily_pnl,
            portfolio_value=portfolio.net_liquidation,
        )
        assert not result.passed

    def test_max_positions_exceeded(self, portfolio_with_positions):
        result = self.limits.check_concurrent_positions(
            current_count=portfolio_with_positions.position_count,
        )
        # With max_concurrent_positions=20 and 1 position, should pass
        assert result.passed

        # Now test with limit=1
        low_limit = RiskLimits(max_concurrent_positions=1)
        result = low_limit.check_concurrent_positions(
            current_count=portfolio_with_positions.position_count,
        )
        assert not result.passed
