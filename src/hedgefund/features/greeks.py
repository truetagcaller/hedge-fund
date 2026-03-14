"""Black-Scholes Greeks calculator with Newton-Raphson implied volatility solver.

Computes delta, gamma, theta, vega, rho, and implied volatility for European
options.  Also provides helpers for constructing an IV surface from a grid of
strikes and expirations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

import structlog

from hedgefund.types import Greeks, OptionType

log = structlog.get_logger(__name__)

# Tiny constants to prevent numerical blow-ups.
_MIN_VOL = 1e-6
_MAX_VOL = 10.0
_MIN_TIME = 1e-10


# ── Core Black-Scholes helpers ───────────────────────────────────────────────

def _d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return _d1(S, K, T, r, q, sigma) - sigma * math.sqrt(T)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """Compute Black-Scholes European option price.

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to expiry in years
    r : risk-free rate (annualised, continuous)
    q : continuous dividend yield
    sigma : volatility (annualised)
    option_type : CALL or PUT
    """
    if T < _MIN_TIME:
        # At expiry, just intrinsic
        if option_type == OptionType.CALL:
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d_1 = _d1(S, K, T, r, q, sigma)
    d_2 = d_1 - sigma * math.sqrt(T)
    df = math.exp(-r * T)
    dq = math.exp(-q * T)

    if option_type == OptionType.CALL:
        return S * dq * stats.norm.cdf(d_1) - K * df * stats.norm.cdf(d_2)
    return K * df * stats.norm.cdf(-d_2) - S * dq * stats.norm.cdf(-d_1)


# ── Greeks Calculator ────────────────────────────────────────────────────────

@dataclass
class GreeksCalculator:
    """Compute Greeks for a single European option or a full chain.

    Attributes
    ----------
    risk_free_rate : annualised risk-free rate (e.g. 0.05)
    dividend_yield : continuous dividend yield (e.g. 0.013)
    """

    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0

    # ── Single-contract Greeks ───────────────────────────────────────────

    def compute(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: OptionType,
    ) -> Greeks:
        """Return all Greeks for one contract."""
        r, q = self.risk_free_rate, self.dividend_yield
        if T < _MIN_TIME or sigma < _MIN_VOL:
            intrinsic = max(S - K, 0.0) if option_type == OptionType.CALL else max(K - S, 0.0)
            sign = 1.0 if option_type == OptionType.CALL else -1.0
            itm = (sign * (S - K)) > 0
            return Greeks(
                delta=sign if itm else 0.0,
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                rho=0.0,
                iv=sigma,
            )

        sqrt_T = math.sqrt(T)
        d1 = _d1(S, K, T, r, q, sigma)
        d2 = d1 - sigma * sqrt_T
        dq = math.exp(-q * T)
        df = math.exp(-r * T)
        n_d1 = stats.norm.pdf(d1)

        if option_type == OptionType.CALL:
            delta = dq * stats.norm.cdf(d1)
            theta = (
                -(S * dq * n_d1 * sigma) / (2.0 * sqrt_T)
                - r * K * df * stats.norm.cdf(d2)
                + q * S * dq * stats.norm.cdf(d1)
            ) / 365.0
            rho = K * T * df * stats.norm.cdf(d2) / 100.0
        else:
            delta = -dq * stats.norm.cdf(-d1)
            theta = (
                -(S * dq * n_d1 * sigma) / (2.0 * sqrt_T)
                + r * K * df * stats.norm.cdf(-d2)
                - q * S * dq * stats.norm.cdf(-d1)
            ) / 365.0
            rho = -K * T * df * stats.norm.cdf(-d2) / 100.0

        gamma = (dq * n_d1) / (S * sigma * sqrt_T)
        vega = S * dq * n_d1 * sqrt_T / 100.0

        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho, iv=sigma)

    # ── Implied volatility (Newton-Raphson) ──────────────────────────────

    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        option_type: OptionType,
        *,
        initial_guess: float = 0.3,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> Optional[float]:
        """Solve for IV using Newton-Raphson on vega.

        Returns ``None`` if the solver fails to converge.
        """
        r, q = self.risk_free_rate, self.dividend_yield

        if T < _MIN_TIME:
            return None

        # Clamp initial guess
        sigma = max(_MIN_VOL, min(initial_guess, _MAX_VOL))

        for _ in range(max_iter):
            price = bs_price(S, K, T, r, q, sigma, option_type)
            diff = price - market_price

            if abs(diff) < tol:
                return sigma

            # Vega (not divided by 100 here — we need raw sensitivity)
            sqrt_T = math.sqrt(T)
            d1 = _d1(S, K, T, r, q, sigma)
            vega = S * math.exp(-q * T) * stats.norm.pdf(d1) * sqrt_T

            if vega < 1e-12:
                # Vega too small; try bisection fallback
                return self._iv_bisection(market_price, S, K, T, option_type)

            sigma -= diff / vega
            sigma = max(_MIN_VOL, min(sigma, _MAX_VOL))

        log.warning(
            "iv_newton_no_converge",
            S=S, K=K, T=T, market_price=market_price,
        )
        return self._iv_bisection(market_price, S, K, T, option_type)

    def _iv_bisection(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        option_type: OptionType,
        *,
        tol: float = 1e-6,
        max_iter: int = 200,
    ) -> Optional[float]:
        """Fallback bisection solver when Newton-Raphson fails."""
        r, q = self.risk_free_rate, self.dividend_yield
        lo, hi = _MIN_VOL, _MAX_VOL

        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            price = bs_price(S, K, T, r, q, mid, option_type)
            if abs(price - market_price) < tol:
                return mid
            if price > market_price:
                hi = mid
            else:
                lo = mid

        log.warning(
            "iv_bisection_no_converge",
            S=S, K=K, T=T, market_price=market_price,
        )
        return None

    # ── Vectorised chain Greeks ──────────────────────────────────────────

    def compute_chain(self, df: pd.DataFrame, spot: float) -> pd.DataFrame:
        """Compute Greeks for every row in an option-chain DataFrame.

        Expected columns: ``strike``, ``time_to_expiry``, ``iv``,
        ``option_type`` (str "CALL"/"PUT" or :class:`OptionType`).

        Returns the DataFrame with additional columns for each Greek.
        """
        df = df.copy()
        results: list[Greeks] = []
        for _, row in df.iterrows():
            otype = row["option_type"]
            if isinstance(otype, str):
                otype = OptionType(otype)
            g = self.compute(
                S=spot,
                K=float(row["strike"]),
                T=float(row["time_to_expiry"]),
                sigma=float(row["iv"]),
                option_type=otype,
            )
            results.append(g)

        df["delta"] = [g.delta for g in results]
        df["gamma"] = [g.gamma for g in results]
        df["theta"] = [g.theta for g in results]
        df["vega"] = [g.vega for g in results]
        df["rho"] = [g.rho for g in results]
        return df

    # ── IV Surface construction ──────────────────────────────────────────

    def build_iv_surface(
        self,
        option_chain: pd.DataFrame,
        spot: float,
    ) -> pd.DataFrame:
        """Build an IV surface from a chain of market quotes.

        Expected columns: ``strike``, ``time_to_expiry``, ``mid_price``,
        ``option_type`` (str or :class:`OptionType`).

        Returns a DataFrame with columns
        ``strike, time_to_expiry, option_type, iv, moneyness, log_moneyness``.
        """
        records: list[dict] = []
        for _, row in option_chain.iterrows():
            otype = row["option_type"]
            if isinstance(otype, str):
                otype = OptionType(otype)

            K = float(row["strike"])
            T = float(row["time_to_expiry"])
            mid = float(row["mid_price"])

            iv = self.implied_volatility(mid, spot, K, T, otype)
            if iv is not None:
                records.append(
                    {
                        "strike": K,
                        "time_to_expiry": T,
                        "option_type": otype.value,
                        "iv": iv,
                        "moneyness": K / spot,
                        "log_moneyness": math.log(K / spot),
                    }
                )

        surface = pd.DataFrame(records)
        if surface.empty:
            log.warning("iv_surface_empty")
        return surface
