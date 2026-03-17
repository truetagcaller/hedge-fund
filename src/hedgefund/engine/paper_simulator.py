"""Enhanced paper trading simulator with realistic market simulation.

Wraps :class:`PaperBroker` with realistic fill simulation (latency,
slippage, partial fills, market impact) and optionally generates
synthetic market data using Geometric Brownian Motion when no real
feed is connected.
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import structlog

from hedgefund.execution.paper import PaperBroker
from hedgefund.streaming.event_bus import Event, EventBus, EventType
from hedgefund.types import (
    Greeks,
    OptionContract,
    OptionQuote,
    OptionType,
    Order,
    OrderStatus,
)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Configuration for realistic paper trading simulation."""

    # Fill latency simulation
    min_latency_ms: float = 50.0
    max_latency_ms: float = 500.0

    # Slippage simulation (normal distribution)
    slippage_mean_bps: float = 5.0
    slippage_std_bps: float = 3.0

    # Commission
    commission_per_contract: float = 0.65

    # Partial fill simulation
    partial_fill_probability: float = 0.0  # 0-1, probability of partial fill
    partial_fill_min_pct: float = 0.3  # minimum fill percentage when partial

    # Market impact
    impact_coefficient: float = 0.001  # price impact per contract
    impact_threshold_contracts: int = 50  # impact kicks in above this size

    # GBM parameters for synthetic data
    gbm_drift: float = 0.05  # annualized drift (5%)
    gbm_volatility: float = 0.20  # annualized vol (20%)
    gbm_dt: float = 1.0 / 252.0 / 390.0  # ~1 minute in trading-year fractions

    # Synthetic data generation
    tick_interval_seconds: float = 1.0
    base_spread_bps: float = 10.0
    base_volume_per_tick: int = 1000

    # BSM pricing
    risk_free_rate: float = 0.05


# ---------------------------------------------------------------------------
# Black-Scholes-Merton helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Approximation of the cumulative normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool,
) -> float:
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        # Intrinsic value only
        if is_call:
            return max(0.0, S - K)
        return max(0.0, K - S)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if is_call:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _bsm_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool,
) -> Greeks:
    """Compute Black-Scholes Greeks."""
    if T <= 0 or sigma <= 0:
        return Greeks(
            delta=1.0 if is_call and S > K else (-1.0 if not is_call and S < K else 0.0),
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            iv=sigma,
        )

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    nd1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi)

    if is_call:
        delta = _norm_cdf(d1)
    else:
        delta = _norm_cdf(d1) - 1.0

    gamma = nd1 / (S * sigma * sqrt_T) if S > 0 else 0.0
    vega = S * nd1 * sqrt_T / 100.0  # per 1% change in vol
    theta_term1 = -(S * nd1 * sigma) / (2.0 * sqrt_T)

    if is_call:
        theta_term2 = -r * K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        theta_term2 = r * K * math.exp(-r * T) * _norm_cdf(-d2)

    theta = (theta_term1 + theta_term2) / 365.0  # per calendar day

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, iv=sigma)


# ---------------------------------------------------------------------------
# Paper Trading Simulator
# ---------------------------------------------------------------------------

class PaperTradingSimulator:
    """Enhanced paper trading simulator with realistic market behavior.

    Wraps a :class:`PaperBroker` and adds:
    - Random fill latency
    - Slippage simulation (normal distribution)
    - Commission calculation
    - Partial fills
    - Market impact for large orders
    - Synthetic market data generation (GBM)
    - Options chain generation with BSM pricing

    Parameters
    ----------
    event_bus:
        Central event bus for publishing simulated ticks.
    config:
        Simulation configuration parameters.
    initial_cash:
        Starting cash for the paper broker.
    symbols:
        Symbols to generate synthetic data for.
    initial_prices:
        Initial prices for each symbol (defaults to 100.0).
    """

    def __init__(
        self,
        event_bus: EventBus,
        *,
        config: SimulationConfig | None = None,
        initial_cash: float = 10_000_000.0,
        symbols: list[str] | None = None,
        initial_prices: Dict[str, float] | None = None,
    ) -> None:
        self._bus = event_bus
        self._config = config or SimulationConfig()

        self._broker = PaperBroker(
            initial_cash=initial_cash,
            slippage_bps=0,  # We handle slippage ourselves
            commission_per_contract=0.0,  # We handle commission ourselves
            fill_ratio=1.0,
        )

        self._symbols = symbols or ["SPY"]
        self._prices: Dict[str, float] = {}
        for sym in self._symbols:
            self._prices[sym] = (initial_prices or {}).get(sym, 100.0)

        self._running = False
        self._tick_task: Optional[asyncio.Task[None]] = None
        self._tick_count = 0

    @property
    def broker(self) -> PaperBroker:
        """Access the underlying PaperBroker."""
        return self._broker

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect the broker and start synthetic data generation."""
        await self._broker.connect()
        self._running = True
        self._tick_task = asyncio.create_task(
            self._tick_loop(), name="paper-simulator-ticks"
        )
        log.info(
            "paper_simulator.started",
            symbols=self._symbols,
            initial_prices=self._prices,
        )

    async def shutdown(self) -> None:
        """Stop the simulator and disconnect the broker."""
        self._running = False
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
            self._tick_task = None
        await self._broker.disconnect()
        log.info("paper_simulator.shutdown", ticks_generated=self._tick_count)

    # ------------------------------------------------------------------
    # Order submission with realistic simulation
    # ------------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """Submit an order with simulated latency, slippage, and market impact."""
        # Simulate fill latency
        latency_ms = random.uniform(  # noqa: S311
            self._config.min_latency_ms,
            self._config.max_latency_ms,
        )
        await asyncio.sleep(latency_ms / 1000.0)

        # Apply slippage
        slippage_bps = random.gauss(  # noqa: S311
            self._config.slippage_mean_bps,
            self._config.slippage_std_bps,
        )
        slippage_pct = max(0.0, slippage_bps) / 10_000.0

        if order.limit_price is not None and order.limit_price > 0:
            from hedgefund.types import Side
            if order.side == Side.BUY:
                order.limit_price = order.limit_price * (1.0 + slippage_pct)
            else:
                order.limit_price = order.limit_price * (1.0 - slippage_pct)

        # Market impact for large orders
        if order.quantity > self._config.impact_threshold_contracts:
            excess = order.quantity - self._config.impact_threshold_contracts
            impact = excess * self._config.impact_coefficient
            if order.limit_price is not None:
                from hedgefund.types import Side
                if order.side == Side.BUY:
                    order.limit_price *= (1.0 + impact)
                else:
                    order.limit_price *= (1.0 - impact)

        # Partial fill simulation
        original_qty = order.quantity
        if (
            self._config.partial_fill_probability > 0
            and random.random() < self._config.partial_fill_probability  # noqa: S311
        ):
            fill_pct = random.uniform(self._config.partial_fill_min_pct, 1.0)  # noqa: S311
            order.quantity = max(1, int(order.quantity * fill_pct))
            log.debug(
                "paper_simulator.partial_fill",
                original_qty=original_qty,
                filled_qty=order.quantity,
            )

        # Submit to underlying broker
        result = await self._broker.submit_order(order)

        # Apply commission
        if result.status == OrderStatus.FILLED:
            commission = self._config.commission_per_contract * (result.filled_quantity or order.quantity)
            result.commission = commission

        return result

    # ------------------------------------------------------------------
    # Synthetic market data generation
    # ------------------------------------------------------------------

    async def _tick_loop(self) -> None:
        """Generate synthetic market ticks using GBM."""
        while self._running:
            try:
                for symbol in self._symbols:
                    await self._generate_tick(symbol)
                self._tick_count += 1
                await asyncio.sleep(self._config.tick_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("paper_simulator.tick_error")
                await asyncio.sleep(self._config.tick_interval_seconds)

    async def _generate_tick(self, symbol: str) -> None:
        """Generate a single synthetic tick for a symbol using GBM."""
        price = self._prices[symbol]
        cfg = self._config

        # Geometric Brownian Motion: dS = mu*S*dt + sigma*S*dW
        dt = cfg.gbm_dt
        drift = cfg.gbm_drift
        vol = cfg.gbm_volatility

        z = random.gauss(0, 1)  # noqa: S311
        new_price = price * math.exp((drift - 0.5 * vol ** 2) * dt + vol * math.sqrt(dt) * z)
        new_price = max(0.01, new_price)
        self._prices[symbol] = new_price

        # Compute realistic bid-ask
        spread = new_price * cfg.base_spread_bps / 10_000.0
        bid = new_price - spread / 2.0
        ask = new_price + spread / 2.0

        # Simulate volume
        volume = max(1, int(random.gauss(cfg.base_volume_per_tick, cfg.base_volume_per_tick * 0.3)))  # noqa: S311

        # Synthetic ATR (roughly vol * price * sqrt(dt_daily))
        atr = new_price * vol * math.sqrt(1.0 / 252.0)

        # Publish tick event
        await self._bus.publish(Event(
            event_type=EventType.TICK,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            data={
                "price": round(new_price, 4),
                "bid": round(bid, 4),
                "ask": round(ask, 4),
                "close": round(new_price, 4),
                "open": round(price, 4),
                "high": round(max(price, new_price) * (1 + random.uniform(0, 0.001)), 4),  # noqa: S311
                "low": round(min(price, new_price) * (1 - random.uniform(0, 0.001)), 4),  # noqa: S311
                "volume": volume,
                "atr": round(atr, 4),
            },
            source="paper_simulator",
        ))

        # Update positions in broker with new price
        for pos_key, pos in list(self._broker._positions.items()):
            if pos.contract.underlying == symbol:
                await self._broker.process_tick(
                    pos.contract, bid=bid, ask=ask
                )

    # ------------------------------------------------------------------
    # Options chain generation
    # ------------------------------------------------------------------

    def generate_options_chain(
        self,
        symbol: str,
        *,
        num_strikes: int = 11,
        expiration_days: list[int] | None = None,
    ) -> list[OptionQuote]:
        """Generate a synthetic options chain using BSM pricing.

        Parameters
        ----------
        symbol:
            Underlying symbol.
        num_strikes:
            Number of strikes to generate (centered around current price).
        expiration_days:
            Days to expiration for each series (default [7, 14, 30, 45]).
        """
        price = self._prices.get(symbol, 100.0)
        expirations = expiration_days or [7, 14, 30, 45]
        cfg = self._config

        quotes: list[OptionQuote] = []
        strike_step = round(price * 0.01, 2)  # ~1% spacing
        if strike_step < 0.5:
            strike_step = 0.5

        center_strike = round(price / strike_step) * strike_step
        half = num_strikes // 2

        for dte in expirations:
            T = dte / 365.0
            exp_date = (datetime.now(timezone.utc) + timedelta(days=dte)).date()

            for i in range(-half, half + 1):
                strike = center_strike + i * strike_step
                if strike <= 0:
                    continue

                for is_call in (True, False):
                    opt_type = OptionType.CALL if is_call else OptionType.PUT
                    sigma = cfg.gbm_volatility * random.uniform(0.8, 1.2)  # skew  # noqa: S311

                    theo_price = _bsm_price(price, strike, T, cfg.risk_free_rate, sigma, is_call)
                    greeks = _bsm_greeks(price, strike, T, cfg.risk_free_rate, sigma, is_call)

                    spread = max(0.01, theo_price * cfg.base_spread_bps / 10_000.0 * 5)
                    bid = max(0.01, theo_price - spread / 2.0)
                    ask = theo_price + spread / 2.0

                    oi = max(0, int(random.gauss(5000, 2000)))  # noqa: S311
                    vol = max(0, int(random.gauss(500, 200)))  # noqa: S311

                    contract = OptionContract(
                        symbol=f"{symbol}{exp_date.strftime('%y%m%d')}{'C' if is_call else 'P'}{int(strike * 100):08d}",
                        underlying=symbol,
                        option_type=opt_type,
                        strike=strike,
                        expiration=exp_date,
                    )

                    quotes.append(OptionQuote(
                        contract=contract,
                        bid=round(bid, 2),
                        ask=round(ask, 2),
                        last=round(theo_price, 2),
                        volume=vol,
                        open_interest=oi,
                        greeks=greeks,
                        timestamp=datetime.now(timezone.utc),
                    ))

        return quotes

    async def publish_options_chain(self, symbol: str) -> None:
        """Generate and publish options chain data to the event bus."""
        chain = self.generate_options_chain(symbol)

        total_call_oi = sum(q.open_interest for q in chain if q.contract.option_type == OptionType.CALL)
        total_put_oi = sum(q.open_interest for q in chain if q.contract.option_type == OptionType.PUT)
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

        price = self._prices.get(symbol, 100.0)
        # Max pain: strike with minimum total value of in-the-money options
        strikes = sorted({q.contract.strike for q in chain})
        max_pain_strike = price
        min_pain_value = float("inf")
        for s in strikes:
            pain = 0.0
            for q in chain:
                if q.contract.option_type == OptionType.CALL and price > q.contract.strike:
                    pain += (price - q.contract.strike) * q.open_interest
                elif q.contract.option_type == OptionType.PUT and price < q.contract.strike:
                    pain += (q.contract.strike - price) * q.open_interest
            if pain < min_pain_value:
                min_pain_value = pain
                max_pain_strike = s

        avg_iv = sum(q.greeks.iv for q in chain) / len(chain) if chain else 0.0

        await self._bus.publish(Event(
            event_type=EventType.OPTIONS_CHAIN,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            data={
                "pcr": round(pcr, 4),
                "max_pain": round(max_pain_strike, 2),
                "iv": round(avg_iv, 4),
                "gex": round(random.gauss(0, 1e9), 2),  # noqa: S311
                "total_call_oi": total_call_oi,
                "total_put_oi": total_put_oi,
                "chain_length": len(chain),
            },
            source="paper_simulator",
        ))

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_current_prices(self) -> Dict[str, float]:
        """Return current simulated prices for all symbols."""
        return dict(self._prices)

    def get_stats(self) -> Dict[str, Any]:
        """Return simulator statistics."""
        return {
            "running": self._running,
            "tick_count": self._tick_count,
            "symbols": self._symbols,
            "prices": {k: round(v, 4) for k, v in self._prices.items()},
            "broker_cash": self._broker.cash,
            "broker_realized_pnl": self._broker.realized_pnl,
            "pending_orders": len(self._broker.pending_orders),
        }
