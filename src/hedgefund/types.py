"""Shared domain types used across all modules.

These dataclasses form the system's lingua franca — every module communicates
through these types rather than raw dicts or ad-hoc structures.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional


# ── Enums ──────────────────────────────────────────────────────────────────────

class Side(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class OptionType(enum.Enum):
    CALL = "CALL"
    PUT = "PUT"


class OrderType(enum.Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(enum.Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class SignalAction(enum.Enum):
    BUY_CALL = "BUY_CALL"
    BUY_PUT = "BUY_PUT"
    SELL_CALL = "SELL_CALL"
    SELL_PUT = "SELL_PUT"
    SPREAD = "SPREAD"
    NO_TRADE = "NO_TRADE"


class SignalDirection(enum.Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class MarketRegime(enum.Enum):
    LOW_VOL_BULLISH = "LOW_VOL_BULLISH"
    HIGH_VOL_BULLISH = "HIGH_VOL_BULLISH"
    LOW_VOL_BEARISH = "LOW_VOL_BEARISH"
    HIGH_VOL_BEARISH = "HIGH_VOL_BEARISH"
    MEAN_REVERTING = "MEAN_REVERTING"
    TRENDING = "TRENDING"


class DataOrigin(enum.Enum):
    """Origin/source of market data."""
    LIVE_BROKER = "LIVE_BROKER"
    LIVE_API = "LIVE_API"
    HISTORICAL = "HISTORICAL"
    NOT_CONNECTED = "NOT_CONNECTED"


class Timeframe(enum.Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    D1 = "1d"


class AssetClass(enum.Enum):
    """Tradeable asset classes."""
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"
    COMMODITY = "COMMODITY"
    CRYPTO = "CRYPTO"


class TradingMode(enum.Enum):
    """Trading execution mode."""
    PAPER = "PAPER"
    LIVE = "LIVE"
    BACKTEST = "BACKTEST"


# ── Market Data ────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class OHLCV:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True, slots=True)
class OptionContract:
    symbol: str
    underlying: str
    option_type: OptionType
    strike: float
    expiration: date
    multiplier: int = 100

    @property
    def osi_symbol(self) -> str:
        """OCC standardized symbol."""
        exp = self.expiration.strftime("%y%m%d")
        cp = "C" if self.option_type == OptionType.CALL else "P"
        strike_str = f"{int(self.strike * 1000):08d}"
        return f"{self.underlying:<6}{exp}{cp}{strike_str}"


@dataclass(frozen=True, slots=True)
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float = 0.0
    iv: float = 0.0


@dataclass(slots=True)
class OptionQuote:
    contract: OptionContract
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    greeks: Greeks
    timestamp: datetime

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


# ── Trading Signals ────────────────────────────────────────────────────────────

@dataclass(slots=True)
class TradeSignal:
    signal_id: str
    timestamp: datetime
    underlying: str
    action: SignalAction
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    strategy_name: str
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float
    reasoning: str
    contracts: list[OptionContract] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_id() -> str:
        return f"SIG-{uuid.uuid4().hex[:12].upper()}"

    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_amount(self) -> float:
        return abs(self.target_price - self.entry_price)


# ── Orders ─────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Order:
    order_id: str
    signal_id: str
    contract: OptionContract
    side: Side
    order_type: OrderType
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    filled_at: Optional[datetime] = None
    commission: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def generate_id() -> str:
        return f"ORD-{uuid.uuid4().hex[:12].upper()}"


# ── Portfolio ──────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Position:
    contract: OptionContract
    quantity: int
    avg_entry: float
    current_price: float
    greeks: Greeks
    unrealized_pnl: float
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def market_value(self) -> float:
        return self.current_price * abs(self.quantity) * self.contract.multiplier

    @property
    def notional_value(self) -> float:
        return self.contract.strike * abs(self.quantity) * self.contract.multiplier


@dataclass(slots=True)
class PortfolioSnapshot:
    timestamp: datetime
    cash: float
    net_liquidation: float
    positions: list[Position]
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_theta: float = 0.0
    total_vega: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    drawdown_pct: float = 0.0
    high_water_mark: float = 0.0

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def total_market_value(self) -> float:
        return sum(p.market_value for p in self.positions)


# ── Sentiment ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SentimentResult:
    symbol: str
    score: float  # -1.0 (bearish) to +1.0 (bullish)
    magnitude: float  # 0.0 to 1.0 (strength/confidence)
    source: str
    headline: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Backtest Results ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class BacktestMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float
    calmar_ratio: float = 0.0

    @property
    def loss_rate(self) -> float:
        return 1.0 - self.win_rate


# ── Trade Record (for learning) ───────────────────────────────────────────────

@dataclass(slots=True)
class TradeRecord:
    trade_id: str
    signal_id: str
    underlying: str
    contract: OptionContract
    side: Side
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    hold_duration_minutes: int
    strategy_name: str
    regime_at_entry: MarketRegime
    sentiment_at_entry: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


# ── Execution Context ────────────────────────────────────────────────────────

@dataclass(slots=True)
class TradingExecutionContext:
    """Determines where and how trades are executed.

    Every order submission must reference an execution context so the
    broker router, risk manager, and instrument mapper all operate on
    a consistent set of parameters.
    """

    user_id: str
    active_broker: str
    asset_class: AssetClass
    trading_mode: TradingMode
    broker_account_id: str = ""
    market_segment: str = ""  # e.g. "NFO", "SPOT", "MCX"

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "active_broker": self.active_broker,
            "asset_class": self.asset_class.value,
            "trading_mode": self.trading_mode.value,
            "broker_account_id": self.broker_account_id,
            "market_segment": self.market_segment,
        }


# ── Level-4: Per-User Engine & Strategy Management ───────────────────────────

class UserEngineState(enum.Enum):
    """Lifecycle state of a per-user execution engine."""
    INITIALIZING = "INITIALIZING"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    SHUTDOWN = "SHUTDOWN"


class StrategyState(enum.Enum):
    """Runtime state of a strategy within a user engine."""
    ENABLED = "ENABLED"
    DISABLED = "DISABLED"
    PROBATION = "PROBATION"


@dataclass(slots=True)
class StrategyAllocation:
    """Capital allocation for a single strategy within a user's portfolio."""
    strategy_name: str
    user_id: str
    allocation_pct: float  # 0.0 to 1.0
    allocated_capital: float
    method: str  # "equal", "manual", "performance", "mvo", "kelly"
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        ua = self.updated_at
        return {
            "strategy_name": self.strategy_name,
            "user_id": self.user_id,
            "allocation_pct": round(self.allocation_pct, 6),
            "allocated_capital": round(self.allocated_capital, 2),
            "method": self.method,
            "updated_at": ua.isoformat() if hasattr(ua, "isoformat") else str(ua),
        }


@dataclass(slots=True)
class StrategyMetrics:
    """Performance metrics for a strategy over a time window."""
    strategy_name: str
    user_id: str
    window: str  # "7d", "30d", "all"
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_rr: float = 0.0
    total_pnl: float = 0.0
    avg_hold_minutes: float = 0.0
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "user_id": self.user_id,
            "window": self.window,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "avg_rr": round(self.avg_rr, 2),
            "total_pnl": round(self.total_pnl, 2),
            "avg_hold_minutes": round(self.avg_hold_minutes, 1),
            "computed_at": self.computed_at.isoformat()
            if hasattr(self.computed_at, "isoformat")
            else str(self.computed_at),
        }
