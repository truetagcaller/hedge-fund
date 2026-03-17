"""Options chain fetcher with Black-Scholes greeks calculation.

Uses Yahoo Finance (via :pypi:`yfinance`) for raw chain data and computes
greeks locally using the Black-Scholes-Merton model.
"""

from __future__ import annotations

import asyncio
import math
from datetime import date, datetime, timezone
from typing import List, Optional

import yfinance as yf
from scipy.stats import norm

from hedgefund.data.base import OptionsChainProvider
from hedgefund.exceptions import DataConnectionError, DataValidationError
from hedgefund.logger import get_logger
from hedgefund.types import Greeks, OptionContract, OptionQuote, OptionType

log = get_logger(__name__)

# Annualised risk-free rate (US 10Y proxy – override via config if desired).
_DEFAULT_RISK_FREE_RATE = 0.05


# ── Black-Scholes helpers ─────────────────────────────────────────────────────


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if option_type == OptionType.CALL else max(K - S, 0.0)
        return intrinsic

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    if option_type == OptionType.CALL:
        return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> Greeks:
    """Compute the full greek surface for a single option."""
    if T <= 0 or sigma <= 0:
        return Greeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0, iv=sigma)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)
    pdf_d1 = float(norm.pdf(d1))
    discount = math.exp(-r * T)

    if option_type == OptionType.CALL:
        delta = float(norm.cdf(d1))
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            - r * K * discount * float(norm.cdf(d2))
        ) / 365.0
        rho = K * T * discount * float(norm.cdf(d2)) / 100.0
    else:
        delta = float(norm.cdf(d1)) - 1.0
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            + r * K * discount * float(norm.cdf(-d2))
        ) / 365.0
        rho = -K * T * discount * float(norm.cdf(-d2)) / 100.0

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100.0

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho, iv=sigma)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Newton-Raphson implied volatility solver."""
    if market_price <= 0 or T <= 0:
        return 0.0

    sigma = 0.3  # initial guess

    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        # Vega (unscaled) for Newton step
        d1 = _d1(S, K, T, r, sigma)
        vega = S * float(norm.pdf(d1)) * math.sqrt(T)
        if vega < 1e-12:
            break

        sigma -= diff / vega
        sigma = max(sigma, 1e-6)

    return sigma


# ── Provider ──────────────────────────────────────────────────────────────────


class OptionsChainFetcher(OptionsChainProvider):
    """Fetch full options chains from Yahoo Finance with local greeks.

    Parameters
    ----------
    risk_free_rate:
        Annualised risk-free rate used in Black-Scholes calculations.
    """

    def __init__(self, risk_free_rate: float = _DEFAULT_RISK_FREE_RATE) -> None:
        self._r = risk_free_rate

    async def get_expirations(self, symbol: str) -> List[str]:
        """Return all available expiration dates for *symbol*."""
        try:
            ticker = yf.Ticker(symbol)
            expirations: tuple[str, ...] = await asyncio.to_thread(
                lambda: ticker.options
            )
        except Exception as exc:
            raise DataConnectionError(
                f"Failed to fetch expirations for {symbol}: {exc}"
            ) from exc

        return list(expirations)

    async def get_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None,
    ) -> List[OptionQuote]:
        """Return options quotes with computed greeks.

        If *expiration* is ``None`` the nearest expiration is used.
        """
        try:
            ticker = yf.Ticker(symbol)

            if expiration is None:
                exps = await asyncio.to_thread(lambda: ticker.options)
                if not exps:
                    raise DataValidationError(f"No options expirations for {symbol}")
                expiration = exps[0]

            chain = await asyncio.to_thread(ticker.option_chain, expiration)

            # Current underlying price
            info = await asyncio.to_thread(lambda: ticker.fast_info)
            spot = float(info.last_price)
        except DataValidationError:
            raise
        except Exception as exc:
            raise DataConnectionError(
                f"Failed to fetch option chain for {symbol} exp={expiration}: {exc}"
            ) from exc

        exp_date = date.fromisoformat(expiration)
        T = max((exp_date - datetime.now(timezone.utc).date()).days / 365.0, 1e-6)
        now = datetime.now(timezone.utc)

        quotes: List[OptionQuote] = []

        for opt_type, df in (
            (OptionType.CALL, chain.calls),
            (OptionType.PUT, chain.puts),
        ):
            for _, row in df.iterrows():
                strike = float(row["strike"])
                bid = float(row.get("bid", 0.0))
                ask = float(row.get("ask", 0.0))
                last = float(row.get("lastPrice", 0.0))
                vol = int(row.get("volume", 0) or 0)
                oi = int(row.get("openInterest", 0) or 0)

                mid = (bid + ask) / 2.0 if (bid + ask) > 0 else last
                iv = implied_volatility(mid, spot, strike, T, self._r, opt_type)
                greeks = bs_greeks(spot, strike, T, self._r, iv, opt_type)

                contract = OptionContract(
                    symbol=str(row.get("contractSymbol", "")),
                    underlying=symbol,
                    option_type=opt_type,
                    strike=strike,
                    expiration=exp_date,
                )

                quotes.append(
                    OptionQuote(
                        contract=contract,
                        bid=bid,
                        ask=ask,
                        last=last,
                        volume=vol,
                        open_interest=oi,
                        greeks=greeks,
                        timestamp=now,
                    )
                )

        log.info(
            "options_chain_fetched",
            symbol=symbol,
            expiration=expiration,
            quotes=len(quotes),
        )
        return quotes
