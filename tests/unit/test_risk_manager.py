"""Tests for risk management system."""

import pytest
from datetime import datetime

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
            risk_per_trade_pct=0.01,
            kelly_fraction=0.25,
        )

    def test_fixed_fraction_sizing(self, sample_trade_signal, sample_portfolio):
        qty = self.sizer.fixed_fraction(
            signal=sample_trade_signal,
            portfolio=sample_portfolio,
        )
        # With $10M capital, 1% risk = $100K max risk
        # Risk per contract = |entry - stop| * multiplier
        risk_per_contract = abs(
            sample_trade_signal.entry_price - sample_trade_signal.stop_loss
        ) * 100
        expected_max = int(100_000 / risk_per_contract)
        assert qty <= expected_max
        assert qty > 0

    def test_zero_risk_returns_zero(self, sample_portfolio):
        signal = TradeSignal(
            signal_id="test",
            timestamp=datetime.utcnow(),
            underlying="SPY",
            action=SignalAction.BUY_CALL,
            direction=SignalDirection.LONG,
            confidence=0.75,
            strategy_name="test",
            entry_price=5.0,
            stop_loss=5.0,  # Zero risk!
            target_price=7.0,
            risk_reward_ratio=2.0,
            reasoning="test",
        )
        qty = self.sizer.fixed_fraction(signal=signal, portfolio=sample_portfolio)
        assert qty == 0

    def test_kelly_criterion(self):
        qty = self.sizer.kelly_criterion(
            win_rate=0.6,
            avg_win=2.0,
            avg_loss=1.0,
            capital=10_000_000,
            price_per_contract=500,
        )
        assert qty > 0


class TestDrawdownMonitor:
    def setup_method(self):
        self.monitor = DrawdownMonitor(
            max_drawdown_pct=0.10,
            recovery_threshold_pct=0.05,
        )

    def test_no_drawdown_initially(self):
        assert not self.monitor.is_circuit_breaker_active
        assert self.monitor.current_drawdown_pct == 0.0

    def test_update_equity(self):
        self.monitor.update(10_000_000)
        self.monitor.update(10_500_000)
        assert self.monitor.high_water_mark == 10_500_000

    def test_circuit_breaker_trips(self):
        self.monitor.update(10_000_000)
        self.monitor.update(8_900_000)  # 11% drawdown
        assert self.monitor.is_circuit_breaker_active

    def test_circuit_breaker_below_threshold(self):
        self.monitor.update(10_000_000)
        self.monitor.update(9_500_000)  # 5% drawdown
        assert not self.monitor.is_circuit_breaker_active


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
        passed, msg = self.limits.check_trade_risk(
            signal=sample_trade_signal,
            portfolio=sample_portfolio,
        )
        assert passed, msg

    def test_daily_loss_exceeded(self):
        portfolio = PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            cash=9_500_000,
            net_liquidation=9_600_000,
            positions=[],
            daily_pnl=-350_000,  # 3.5% daily loss
            drawdown_pct=0.04,
            high_water_mark=10_000_000,
        )
        passed, msg = self.limits.check_daily_loss(portfolio)
        assert not passed

    def test_max_positions_exceeded(self, portfolio_with_positions):
        self.limits.max_concurrent_positions = 1
        passed, msg = self.limits.check_position_count(portfolio_with_positions)
        assert not passed
