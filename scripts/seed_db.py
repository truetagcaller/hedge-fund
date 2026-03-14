#!/usr/bin/env python3
"""Initialize database schemas for the hedgefund trading system."""

import asyncio
import asyncpg


SCHEMAS = [
    """
    CREATE TABLE IF NOT EXISTS trades (
        id SERIAL PRIMARY KEY,
        trade_id VARCHAR(64) UNIQUE NOT NULL,
        signal_id VARCHAR(64) NOT NULL,
        underlying VARCHAR(16) NOT NULL,
        symbol VARCHAR(64) NOT NULL,
        option_type VARCHAR(8) NOT NULL,
        strike DECIMAL(12,2) NOT NULL,
        expiration DATE NOT NULL,
        side VARCHAR(8) NOT NULL,
        entry_price DECIMAL(12,4) NOT NULL,
        exit_price DECIMAL(12,4),
        quantity INTEGER NOT NULL,
        pnl DECIMAL(14,2),
        pnl_pct DECIMAL(8,4),
        commission DECIMAL(10,2) DEFAULT 0,
        strategy_name VARCHAR(64),
        regime VARCHAR(32),
        sentiment_score DECIMAL(6,4),
        entry_time TIMESTAMPTZ NOT NULL,
        exit_time TIMESTAMPTZ,
        metadata JSONB DEFAULT '{}'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        id SERIAL PRIMARY KEY,
        signal_id VARCHAR(64) UNIQUE NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        underlying VARCHAR(16) NOT NULL,
        action VARCHAR(16) NOT NULL,
        direction VARCHAR(8) NOT NULL,
        confidence DECIMAL(6,4) NOT NULL,
        strategy_name VARCHAR(64) NOT NULL,
        entry_price DECIMAL(12,4) NOT NULL,
        stop_loss DECIMAL(12,4) NOT NULL,
        target_price DECIMAL(12,4) NOT NULL,
        risk_reward_ratio DECIMAL(6,2) NOT NULL,
        reasoning TEXT,
        outcome VARCHAR(16),
        metadata JSONB DEFAULT '{}'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        cash DECIMAL(16,2) NOT NULL,
        net_liquidation DECIMAL(16,2) NOT NULL,
        total_delta DECIMAL(12,4),
        total_gamma DECIMAL(12,4),
        total_theta DECIMAL(12,4),
        total_vega DECIMAL(12,4),
        daily_pnl DECIMAL(14,2),
        total_pnl DECIMAL(14,2),
        drawdown_pct DECIMAL(8,6),
        high_water_mark DECIMAL(16,2),
        position_count INTEGER,
        metadata JSONB DEFAULT '{}'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS market_data (
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(16) NOT NULL,
        timeframe VARCHAR(8) NOT NULL,
        open DECIMAL(12,4) NOT NULL,
        high DECIMAL(12,4) NOT NULL,
        low DECIMAL(12,4) NOT NULL,
        close DECIMAL(12,4) NOT NULL,
        volume BIGINT NOT NULL,
        PRIMARY KEY (timestamp, symbol, timeframe)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS model_metrics (
        id SERIAL PRIMARY KEY,
        model_name VARCHAR(64) NOT NULL,
        version VARCHAR(32) NOT NULL,
        training_date TIMESTAMPTZ NOT NULL,
        metric_name VARCHAR(64) NOT NULL,
        metric_value DECIMAL(12,6) NOT NULL,
        metadata JSONB DEFAULT '{}'
    );
    """,
    # Indexes
    "CREATE INDEX IF NOT EXISTS idx_trades_underlying ON trades(underlying);",
    "CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);",
    "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_name);",
    "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_signals_underlying ON signals(underlying);",
    "CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp ON portfolio_snapshots(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol, timeframe);",
]


async def seed_database(dsn: str = "postgresql://localhost:5432/hedgefund") -> None:
    conn = await asyncpg.connect(dsn)
    try:
        for schema in SCHEMAS:
            await conn.execute(schema)
        print("Database seeded successfully.")
    finally:
        await conn.close()


if __name__ == "__main__":
    import sys

    dsn = sys.argv[1] if len(sys.argv) > 1 else "postgresql://localhost:5432/hedgefund"
    asyncio.run(seed_database(dsn))
