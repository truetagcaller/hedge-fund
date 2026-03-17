"""Shared test fixtures for the hedgefund trading system."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timezone

from hedgefund.types import (
    OptionContract,
    OptionType,
    Greeks,
    OptionQuote,
    TradeSignal,
    SignalAction,
    SignalDirection,
    Position,
    PortfolioSnapshot,
    SentimentResult,
)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
    price = 100.0
    prices = []
    for _ in range(n):
        price *= 1 + np.random.normal(0, 0.002)
        prices.append(price)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            "close": [p * (1 + np.random.normal(0, 0.0005)) for p in prices],
            "volume": np.random.randint(1000, 100000, n),
        }
    )
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def sample_option_contract() -> OptionContract:
    return OptionContract(
        symbol="SPY250321C00500000",
        underlying="SPY",
        option_type=OptionType.CALL,
        strike=500.0,
        expiration=date(2025, 3, 21),
    )


@pytest.fixture
def sample_greeks() -> Greeks:
    return Greeks(
        delta=0.45,
        gamma=0.02,
        theta=-0.15,
        vega=0.25,
        rho=0.05,
        iv=0.22,
    )


@pytest.fixture
def sample_option_quote(sample_option_contract, sample_greeks) -> OptionQuote:
    return OptionQuote(
        contract=sample_option_contract,
        bid=5.20,
        ask=5.40,
        last=5.30,
        volume=1500,
        open_interest=25000,
        greeks=sample_greeks,
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_trade_signal(sample_option_contract) -> TradeSignal:
    return TradeSignal(
        signal_id=TradeSignal.generate_id(),
        timestamp=datetime.now(timezone.utc),
        underlying="SPY",
        action=SignalAction.BUY_CALL,
        direction=SignalDirection.LONG,
        confidence=0.75,
        strategy_name="ema_crossover",
        entry_price=5.30,
        stop_loss=4.24,
        target_price=7.42,
        risk_reward_ratio=2.0,
        reasoning="EMA 9/21 bullish crossover with RSI confirmation",
        contracts=[sample_option_contract],
    )


@pytest.fixture
def sample_portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=datetime.now(timezone.utc),
        cash=9_500_000.0,
        net_liquidation=10_000_000.0,
        positions=[],
        total_delta=0.0,
        total_gamma=0.0,
        total_theta=0.0,
        total_vega=0.0,
        daily_pnl=0.0,
        total_pnl=0.0,
        drawdown_pct=0.0,
        high_water_mark=10_000_000.0,
    )


@pytest.fixture
def sample_sentiment() -> SentimentResult:
    return SentimentResult(
        symbol="SPY",
        score=0.65,
        magnitude=0.8,
        source="news",
        headline="Markets rally on strong earnings",
    )


@pytest.fixture
def portfolio_with_positions(sample_option_contract, sample_greeks) -> PortfolioSnapshot:
    positions = [
        Position(
            contract=sample_option_contract,
            quantity=10,
            avg_entry=5.00,
            current_price=5.50,
            greeks=sample_greeks,
            unrealized_pnl=5000.0,
        ),
    ]
    return PortfolioSnapshot(
        timestamp=datetime.now(timezone.utc),
        cash=9_000_000.0,
        net_liquidation=9_505_000.0,
        positions=positions,
        total_delta=4.5,
        total_gamma=0.2,
        total_theta=-1.5,
        total_vega=2.5,
        daily_pnl=5000.0,
        total_pnl=5000.0,
        drawdown_pct=0.0495,
        high_water_mark=10_000_000.0,
    )
