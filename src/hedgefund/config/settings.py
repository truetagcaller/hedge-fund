"""Root application settings combining every config section.

Usage::

    from hedgefund.config.settings import get_settings

    settings = get_settings()          # uses default config dir + env
    print(settings.risk.max_drawdown_pct)

``Settings`` is intentionally a plain Pydantic model (not ``BaseSettings``)
because the heavy lifting of env-var and YAML merging is done in
:mod:`hedgefund.config.loader`.  We use ``pydantic-settings`` only for the
thin ``AppSettings`` wrapper that reads ``HEDGEFUND_ENV``.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from hedgefund.config.loader import load_config
from hedgefund.config.schema import (
    AgentsConfig,
    AnalysisConfig,
    AuthConfig,
    BacktestConfig,
    DashboardConfig,
    DataConfig,
    DataSourceConfig,
    DatabaseConfig,
    ExecutionConfig,
    FeaturesConfig,
    LearningConfig,
    MongoConfig,
    PortfolioOptimizerConfig,
    RedisConfig,
    RiskConfig,
    SentimentConfig,
    SignalConfig,
)


# ── Thin env-aware wrapper ────────────────────────────────────────────────────


class AppSettings(BaseSettings):
    """Minimal settings read purely from environment variables."""

    name: str = "HedgeFund AI Options Agent"
    version: str = "1.0.0"
    log_level: str = "INFO"
    json_logs: bool = False
    env: str = "development"

    model_config = {"env_prefix": "HEDGEFUND_APP_"}


# ── Root settings model ──────────────────────────────────────────────────────


class Settings(BaseModel):
    """Unified, validated configuration for the entire application."""

    app: AppSettings = Field(default_factory=AppSettings)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    signals: SignalConfig = Field(default_factory=SignalConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    mongo: MongoConfig = Field(default_factory=MongoConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    portfolio_optimizer: PortfolioOptimizerConfig = Field(default_factory=PortfolioOptimizerConfig)
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)

    model_config = {"frozen": True}


def build_settings(
    config_dir: Path | str | None = None,
    env: str | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Settings:
    """Build a fully-validated :class:`Settings` instance.

    Parameters
    ----------
    config_dir:
        Path to the directory containing YAML config files.
    env:
        Environment name override.
    overrides:
        Arbitrary dict merged on top of the loaded config (useful in tests).
    """
    raw = load_config(config_dir=config_dir, env=env)
    if overrides:
        raw.update(overrides)
    return Settings.model_validate(raw)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, singleton :class:`Settings` instance."""
    return build_settings()
