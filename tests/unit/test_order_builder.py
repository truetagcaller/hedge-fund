"""Tests for order construction."""

import pytest
from datetime import date

from hedgefund.types import (
    OptionContract,
    OptionType,
    Side,
    OrderType,
)
from hedgefund.execution.order_builder import OrderBuilder


class TestOrderBuilder:
    def setup_method(self):
        self.builder = OrderBuilder()

    def test_single_leg_order(self, sample_option_contract):
        order = self.builder.single_leg(
            contract=sample_option_contract,
            side=Side.BUY,
            quantity=10,
            signal_id="SIG-TEST001",
            order_type=OrderType.LIMIT,
            limit_price=5.30,
        )
        assert order.quantity == 10
        assert order.side == Side.BUY
        assert order.limit_price == 5.30
        assert order.order_id.startswith("ORD-")

    def test_vertical_spread(self):
        orders = self.builder.vertical_spread(
            underlying="SPY",
            option_type=OptionType.CALL,
            long_strike=500.0,
            short_strike=510.0,
            expiration=date(2025, 3, 21),
            quantity=5,
            signal_id="SIG-TEST002",
            limit_price=3.50,
        )
        assert len(orders) == 2
        assert orders[0].side == Side.BUY
        assert orders[1].side == Side.SELL

    def test_iron_condor(self):
        orders = self.builder.iron_condor(
            underlying="SPY",
            put_long_strike=480.0,
            put_short_strike=490.0,
            call_short_strike=510.0,
            call_long_strike=520.0,
            expiration=date(2025, 3, 21),
            quantity=3,
            signal_id="SIG-TEST003",
            limit_price=2.50,
        )
        assert len(orders) == 4
