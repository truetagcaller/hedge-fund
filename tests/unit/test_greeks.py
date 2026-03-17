"""Tests for Black-Scholes Greeks calculations."""

import numpy as np
from hedgefund.features.greeks import GreeksCalculator, bs_price
from hedgefund.types import OptionType


class TestGreeksCalculator:
    def setup_method(self):
        self.calc = GreeksCalculator(risk_free_rate=0.05, dividend_yield=0.0)

    def test_call_price_atm(self):
        """ATM call should have meaningful value."""
        price = bs_price(
            S=100, K=100, T=0.25, r=0.05, q=0.0, sigma=0.20,
            option_type=OptionType.CALL,
        )
        assert 3.0 < price < 6.0

    def test_put_price_atm(self):
        """ATM put should have meaningful value."""
        price = bs_price(
            S=100, K=100, T=0.25, r=0.05, q=0.0, sigma=0.20,
            option_type=OptionType.PUT,
        )
        assert 2.0 < price < 5.0

    def test_put_call_parity(self):
        """Put-call parity: C - P = S - K*exp(-rT)."""
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
        call = bs_price(S, K, T, r, 0.0, sigma, OptionType.CALL)
        put = bs_price(S, K, T, r, 0.0, sigma, OptionType.PUT)
        parity = S - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 0.01

    def test_delta_call_range(self):
        """Call delta should be between 0 and 1."""
        greeks = self.calc.compute(
            S=100, K=100, T=0.25, sigma=0.20, option_type=OptionType.CALL,
        )
        assert 0.0 < greeks.delta < 1.0

    def test_delta_put_range(self):
        """Put delta should be between -1 and 0."""
        greeks = self.calc.compute(
            S=100, K=100, T=0.25, sigma=0.20, option_type=OptionType.PUT,
        )
        assert -1.0 < greeks.delta < 0.0

    def test_gamma_positive(self):
        """Gamma should always be positive."""
        greeks = self.calc.compute(
            S=100, K=100, T=0.25, sigma=0.20, option_type=OptionType.CALL,
        )
        assert greeks.gamma > 0

    def test_theta_negative_long(self):
        """Long options should have negative theta (time decay)."""
        greeks = self.calc.compute(
            S=100, K=100, T=0.25, sigma=0.20, option_type=OptionType.CALL,
        )
        assert greeks.theta < 0

    def test_vega_positive(self):
        """Vega should be positive for long options."""
        greeks = self.calc.compute(
            S=100, K=100, T=0.25, sigma=0.20, option_type=OptionType.CALL,
        )
        assert greeks.vega > 0

    def test_implied_volatility(self):
        """IV calculation should recover the original sigma."""
        S, K, T, sigma = 100, 100, 0.25, 0.25
        market_price = bs_price(S, K, T, 0.05, 0.0, sigma, OptionType.CALL)
        iv = self.calc.implied_volatility(
            market_price=market_price, S=S, K=K, T=T,
            option_type=OptionType.CALL,
        )
        assert iv is not None
        assert abs(iv - sigma) < 0.001

    def test_deep_itm_call_delta(self):
        """Deep ITM call should have delta near 1."""
        greeks = self.calc.compute(
            S=150, K=100, T=0.25, sigma=0.20, option_type=OptionType.CALL,
        )
        assert greeks.delta > 0.95

    def test_deep_otm_call_delta(self):
        """Deep OTM call should have delta near 0."""
        greeks = self.calc.compute(
            S=50, K=100, T=0.25, sigma=0.20, option_type=OptionType.CALL,
        )
        assert greeks.delta < 0.05
