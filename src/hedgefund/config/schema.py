"""Pydantic models for every configuration section.

Each model maps 1-to-1 with a top-level key in ``config/base.yaml`` and carries
sensible defaults so the system can boot with *zero* user-supplied config.
"""

from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


# ── Data ──────────────────────────────────────────────────────────────────────


class PolygonConfig(BaseModel):
    base_url: str = "https://api.polygon.io"
    api_key: str = ""


class YahooConfig(BaseModel):
    enabled: bool = True


class DataConfig(BaseModel):
    cache_ttl_seconds: int = Field(30, ge=1)
    stale_data_threshold_seconds: int = Field(60, ge=1)
    symbols: List[str] = Field(
        default_factory=lambda: ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA", "AMZN", "MSFT"]
    )
    timeframes: List[str] = Field(default_factory=lambda: ["1m", "5m", "15m", "1h", "1d"])
    polygon: PolygonConfig = Field(default_factory=PolygonConfig)
    yahoo: YahooConfig = Field(default_factory=YahooConfig)


# ── Features ──────────────────────────────────────────────────────────────────


class FeaturesConfig(BaseModel):
    ema_periods: List[int] = Field(default_factory=lambda: [9, 21, 50, 200])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    volume_profile_bins: int = 50


# ── Analysis ──────────────────────────────────────────────────────────────────


class AnalysisConfig(BaseModel):
    regime_lookback_days: int = 60
    vwap_deviation_threshold: float = 0.02
    order_flow_large_trade_multiplier: float = 5.0
    gex_enabled: bool = True


# ── Sentiment ─────────────────────────────────────────────────────────────────


class SentimentWeightsConfig(BaseModel):
    news: float = 0.5
    social: float = 0.3
    options_flow: float = 0.2


class SentimentConfig(BaseModel):
    enabled: bool = True
    news_lookback_hours: int = 24
    social_lookback_hours: int = 12
    min_magnitude: float = Field(0.3, ge=0.0, le=1.0)
    weights: SentimentWeightsConfig = Field(default_factory=SentimentWeightsConfig)


# ── Signals ───────────────────────────────────────────────────────────────────


class EnsembleWeightsConfig(BaseModel):
    ml_signal: float = 0.4
    rl_signal: float = 0.3
    rule_signal: float = 0.3


class SignalConfig(BaseModel):
    min_confidence: float = Field(0.65, ge=0.0, le=1.0)
    min_risk_reward: float = Field(2.0, ge=0.0)
    ensemble_weights: EnsembleWeightsConfig = Field(default_factory=EnsembleWeightsConfig)
    cooldown_minutes: int = Field(15, ge=0)


# ── Risk ──────────────────────────────────────────────────────────────────────


class RiskConfig(BaseModel):
    initial_capital: float = 10_000_000.0
    risk_per_trade_pct: float = Field(0.01, ge=0.0, le=1.0)
    max_daily_loss_pct: float = Field(0.05, ge=0.0, le=1.0)
    max_drawdown_pct: float = Field(0.10, ge=0.0, le=1.0)
    max_concurrent_positions: int = Field(20, ge=1)
    max_single_position_pct: float = Field(0.05, ge=0.0, le=1.0)
    max_sector_exposure_pct: float = Field(0.25, ge=0.0, le=1.0)
    max_delta_exposure: float = 500.0
    max_gamma_exposure: float = 100.0
    max_vega_exposure: float = 50_000.0
    position_sizing_method: Literal["fixed_fraction", "kelly", "volatility_adjusted"] = (
        "volatility_adjusted"
    )
    kelly_fraction: float = Field(0.25, ge=0.0, le=1.0)
    atr_stop_multiplier: float = 2.0
    trailing_stop_enabled: bool = True
    trailing_stop_atr_multiplier: float = 1.5


# ── Execution ─────────────────────────────────────────────────────────────────


class ExecutionConfig(BaseModel):
    broker: Literal["paper", "alpaca", "ibkr", "zerodha", "binance", "groww", "indmoney"] = "paper"
    slippage_bps: int = Field(5, ge=0)
    max_spread_pct: float = Field(0.05, ge=0.0)
    order_timeout_seconds: int = Field(30, ge=1)
    retry_attempts: int = Field(3, ge=0)
    min_volume: int = Field(100, ge=0)
    min_open_interest: int = Field(500, ge=0)
    credentials_file: str = Field("~/.hedgefund/credentials.json", description="Path to encrypted credentials file")


# ── Learning / ML ─────────────────────────────────────────────────────────────


class LSTMConfig(BaseModel):
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = Field(0.2, ge=0.0, le=1.0)
    sequence_length: int = 60
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 64


class TransformerConfig(BaseModel):
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    learning_rate: float = 0.0001


class RLConfig(BaseModel):
    algorithm: str = "PPO"
    learning_rate: float = 0.0003
    gamma: float = Field(0.99, ge=0.0, le=1.0)
    n_steps: int = 2048
    batch_size: int = 64
    total_timesteps: int = 1_000_000


class RandomForestConfig(BaseModel):
    n_estimators: int = 500
    max_depth: int = 10
    min_samples_split: int = 20


class LearningConfig(BaseModel):
    enabled: bool = True
    retrain_interval_hours: int = 168
    lstm: LSTMConfig = Field(default_factory=LSTMConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)


# ── Backtest ──────────────────────────────────────────────────────────────────


class BacktestConfig(BaseModel):
    initial_capital: float = 10_000_000.0
    commission_per_contract: float = 0.65
    slippage_bps: int = Field(5, ge=0)
    walk_forward_train_days: int = 252
    walk_forward_test_days: int = 63
    monte_carlo_simulations: int = 10_000
    regime_aware: bool = True


# ── Dashboard ─────────────────────────────────────────────────────────────────


class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)
    ws_heartbeat_seconds: int = 15


# ── Redis ─────────────────────────────────────────────────────────────────────


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = Field(6379, ge=1, le=65535)
    db: int = Field(0, ge=0)
    password: str = ""
    ssl: bool = False


# ── Database ──────────────────────────────────────────────────────────────────


class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = Field(5432, ge=1, le=65535)
    name: str = "hedgefund"
    user: str = "postgres"
    password: str = ""
    pool_size: int = Field(10, ge=1)
    ssl: bool = False

    @property
    def dsn(self) -> str:
        """Build an asyncpg-compatible DSN."""
        scheme = "postgresql+asyncpg"
        creds = f"{self.user}:{self.password}" if self.password else self.user
        return f"{scheme}://{creds}@{self.host}:{self.port}/{self.name}"

    @property
    def asyncpg_dsn(self) -> str:
        """Plain ``postgresql://`` DSN for raw asyncpg connections."""
        creds = f"{self.user}:{self.password}" if self.password else self.user
        return f"postgresql://{creds}@{self.host}:{self.port}/{self.name}"


# ── MongoDB ──────────────────────────────────────────────────────────────────


class MongoConfig(BaseModel):
    uri: str = "mongodb://172.21.0.2:27017"
    db_name: str = "hedgefund"


# ── Auth / JWT ───────────────────────────────────────────────────────────────


class AuthConfig(BaseModel):
    jwt_secret: str = ""  # Auto-generated if empty
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(60, ge=1)
    refresh_token_expire_days: int = Field(7, ge=1)
    allow_registration: bool = True


# ── Agents ───────────────────────────────────────────────────────────────────


class AgentWeightsConfig(BaseModel):
    """Per-agent default weights for signal fusion."""
    trend_following: float = Field(0.15, ge=0.0, le=1.0)
    mean_reversion: float = Field(0.15, ge=0.0, le=1.0)
    options_volatility: float = Field(0.15, ge=0.0, le=1.0)
    gamma_scalping: float = Field(0.10, ge=0.0, le=1.0)
    news_reaction: float = Field(0.15, ge=0.0, le=1.0)
    social_sentiment: float = Field(0.10, ge=0.0, le=1.0)
    liquidity_sweep: float = Field(0.10, ge=0.0, le=1.0)
    smart_money_flow: float = Field(0.10, ge=0.0, le=1.0)


class AgentsConfig(BaseModel):
    """Multi-AI agent system configuration."""
    enabled: bool = True
    weights: AgentWeightsConfig = Field(default_factory=AgentWeightsConfig)
    min_active_agents: int = Field(2, ge=1)
    signal_timeout_seconds: float = Field(300.0, ge=1.0)
    require_live_data: bool = True


# ── Portfolio Optimizer ──────────────────────────────────────────────────────


class PortfolioOptimizerConfig(BaseModel):
    method: Literal["mean_variance", "kelly", "risk_parity"] = "risk_parity"
    rebalance_interval_hours: int = Field(24, ge=1)
    min_history_days: int = Field(30, ge=5)


# ── Data Source ──────────────────────────────────────────────────────────────


class DataSourceConfig(BaseModel):
    require_verified_source: bool = True
    stale_threshold_seconds: int = Field(60, ge=1)
    allowed_sources: List[str] = Field(
        default_factory=lambda: [
            "zerodha", "binance", "groww", "indmoney",
            "polygon", "yahoo", "news_api", "twitter", "reuters",
        ]
    )
