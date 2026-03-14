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
        order = self.builder.build_single(
            contract=sample_option_contract,
            side=Side.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=5.30,
        )
        assert order.quantity == 10
        assert order.side == Side.BUY
        assert order.limit_price == 5.30
        assert order.order_id.startswith("ORD-")

    def test_vertical_spread(self):
        long_leg = OptionContract(
            symbol="SPY250321C00500000",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=500.0,
            expiration=date(2025, 3, 21),
        )
        short_leg = OptionContract(
            symbol="SPY250321C00510000",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=510.0,
            expiration=date(2025, 3, 21),
        )
        orders = self.builder.build_vertical_spread(
            long_contract=long_leg,
            short_contract=short_leg,
            quantity=5,
            net_debit=3.50,
        )
        assert len(orders) == 2
        assert orders[0].side == Side.BUY
        assert orders[1].side == Side.SELL

    def test_iron_condor(self):
        contracts = [
            OptionContract("", "SPY", OptionType.PUT, 480, date(2025, 3, 21)),
            OptionContract("", "SPY", OptionType.PUT, 490, date(2025, 3, 21)),
            OptionContract("", "SPY", OptionType.CALL, 510, date(2025, 3, 21)),
            OptionContract("", "SPY", OptionType.CALL, 520, date(2025, 3, 21)),
        ]
        orders = self.builder.build_iron_condor(
            put_long=contracts[0],
            put_short=contracts[1],
            call_short=contracts[2],
            call_long=contracts[3],
            quantity=3,
        )
        assert len(orders) == 4
