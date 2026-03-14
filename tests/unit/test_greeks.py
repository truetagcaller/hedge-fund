"""Tests for Black-Scholes Greeks calculations."""

import pytest
import numpy as np
from hedgefund.features.greeks import GreeksCalculator


class TestGreeksCalculator:
    def setup_method(self):
        self.calc = GreeksCalculator()

    def test_call_price_atm(self):
        """ATM call should have meaningful value."""
        price = self.calc.black_scholes_price(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert 3.0 < price < 6.0

    def test_put_price_atm(self):
        """ATM put should have meaningful value."""
        price = self.calc.black_scholes_price(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put"
        )
        assert 2.0 < price < 5.0

    def test_put_call_parity(self):
        """Put-call parity: C - P = S - K*exp(-rT)."""
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
        call = self.calc.black_scholes_price(S, K, T, r, sigma, "call")
        put = self.calc.black_scholes_price(S, K, T, r, sigma, "put")
        parity = S - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 0.01

    def test_delta_call_range(self):
        """Call delta should be between 0 and 1."""
        greeks = self.calc.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert 0.0 < greeks.delta < 1.0

    def test_delta_put_range(self):
        """Put delta should be between -1 and 0."""
        greeks = self.calc.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put"
        )
        assert -1.0 < greeks.delta < 0.0

    def test_gamma_positive(self):
        """Gamma should always be positive."""
        greeks = self.calc.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert greeks.gamma > 0

    def test_theta_negative_long(self):
        """Long options should have negative theta (time decay)."""
        greeks = self.calc.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert greeks.theta < 0

    def test_vega_positive(self):
        """Vega should be positive for long options."""
        greeks = self.calc.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert greeks.vega > 0

    def test_implied_volatility(self):
        """IV calculation should recover the original sigma."""
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.25
        market_price = self.calc.black_scholes_price(S, K, T, r, sigma, "call")
        iv = self.calc.implied_volatility(
            market_price=market_price, S=S, K=K, T=T, r=r, option_type="call"
        )
        assert abs(iv - sigma) < 0.001

    def test_deep_itm_call_delta(self):
        """Deep ITM call should have delta near 1."""
        greeks = self.calc.calculate_greeks(
            S=150, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert greeks.delta > 0.95

    def test_deep_otm_call_delta(self):
        """Deep OTM call should have delta near 0."""
        greeks = self.calc.calculate_greeks(
            S=50, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert greeks.delta < 0.05
