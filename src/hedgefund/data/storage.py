"""PostgreSQL persistence layer for trades, signals, and portfolio snapshots.

Uses :pypi:`asyncpg` for raw performance where needed and :pypi:`SQLAlchemy`
(async) for schema management and complex queries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import asyncpg
import sqlalchemy as sa
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from hedgefund.config.schema import DatabaseConfig
from hedgefund.logger import get_logger

log = get_logger(__name__)

metadata = MetaData()

# ── Table definitions ─────────────────────────────────────────────────────────

trade_signals = Table(
    "trade_signals",
    metadata,
    Column("signal_id", String(32), primary_key=True),
    Column("timestamp", DateTime(timezone=True), nullable=False, index=True),
    Column("underlying", String(10), nullable=False, index=True),
    Column("action", String(20), nullable=False),
    Column("direction", String(10), nullable=False),
    Column("confidence", Float, nullable=False),
    Column("strategy_name", String(64), nullable=False),
    Column("entry_price", Float, nullable=False),
    Column("stop_loss", Float, nullable=False),
    Column("target_price", Float, nullable=False),
    Column("risk_reward_ratio", Float, nullable=False),
    Column("reasoning", Text, nullable=False, server_default=""),
    Column("metadata_json", JSON, nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=sa.func.now()),
)

trade_records = Table(
    "trade_records",
    metadata,
    Column("trade_id", String(32), primary_key=True),
    Column("signal_id", String(32), nullable=False, index=True),
    Column("underlying", String(10), nullable=False, index=True),
    Column("side", String(4), nullable=False),
    Column("entry_price", Float, nullable=False),
    Column("exit_price", Float, nullable=False),
    Column("quantity", Integer, nullable=False),
    Column("pnl", Float, nullable=False),
    Column("pnl_pct", Float, nullable=False),
    Column("entry_time", DateTime(timezone=True), nullable=False),
    Column("exit_time", DateTime(timezone=True), nullable=False),
    Column("hold_duration_minutes", Integer, nullable=False),
    Column("strategy_name", String(64), nullable=False),
    Column("regime_at_entry", String(30), nullable=True),
    Column("sentiment_at_entry", Float, nullable=True),
    Column("metadata_json", JSON, nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=sa.func.now()),
)

portfolio_snapshots = Table(
    "portfolio_snapshots",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True), nullable=False, index=True),
    Column("cash", Float, nullable=False),
    Column("net_liquidation", Float, nullable=False),
    Column("total_delta", Float, server_default="0"),
    Column("total_gamma", Float, server_default="0"),
    Column("total_theta", Float, server_default="0"),
    Column("total_vega", Float, server_default="0"),
    Column("daily_pnl", Float, server_default="0"),
    Column("total_pnl", Float, server_default="0"),
    Column("drawdown_pct", Float, server_default="0"),
    Column("high_water_mark", Float, server_default="0"),
    Column("position_count", Integer, server_default="0"),
    Column("positions_json", JSON, nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=sa.func.now()),
)


# ── Storage class ─────────────────────────────────────────────────────────────


class PostgresStorage:
    """Async PostgreSQL storage for the trading system.

    Parameters
    ----------
    config:
        :class:`~hedgefund.config.schema.DatabaseConfig` instance.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._engine: Optional[AsyncEngine] = None
        self._raw_pool: Optional[asyncpg.Pool] = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the SQLAlchemy async engine and raw asyncpg pool."""
        self._engine = create_async_engine(
            self._config.dsn,
            pool_size=self._config.pool_size,
            echo=False,
        )
        self._raw_pool = await asyncpg.create_pool(
            dsn=self._config.asyncpg_dsn,
            min_size=2,
            max_size=self._config.pool_size,
        )
        log.info("postgres_connected", host=self._config.host, db=self._config.name)

    async def disconnect(self) -> None:
        """Dispose engine and close pool."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
        if self._raw_pool is not None:
            await self._raw_pool.close()
            self._raw_pool = None
        log.info("postgres_disconnected")

    async def __aenter__(self) -> "PostgresStorage":
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.disconnect()

    async def create_tables(self) -> None:
        """Create all tables if they don't already exist."""
        if self._engine is None:
            raise RuntimeError("Not connected")
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
        log.info("tables_created")

    # ── helpers ────────────────────────────────────────────────────────────

    def _ensure_engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("PostgresStorage is not connected. Call .connect() first.")
        return self._engine

    # ── Signals ────────────────────────────────────────────────────────────

    async def save_signal(self, signal: Dict[str, Any]) -> None:
        """Persist a trade signal."""
        engine = self._ensure_engine()
        async with engine.begin() as conn:
            await conn.execute(trade_signals.insert().values(**signal))
        log.debug("signal_saved", signal_id=signal.get("signal_id"))

    async def get_signals(
        self,
        underlying: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query trade signals with optional filters."""
        engine = self._ensure_engine()
        query = trade_signals.select().order_by(trade_signals.c.timestamp.desc()).limit(limit)
        if underlying:
            query = query.where(trade_signals.c.underlying == underlying)
        if since:
            query = query.where(trade_signals.c.timestamp >= since)

        async with engine.connect() as conn:
            result = await conn.execute(query)
            return [dict(row._mapping) for row in result.fetchall()]

    # ── Trade records ─────────────────────────────────────────────────────

    async def save_trade(self, trade: Dict[str, Any]) -> None:
        """Persist a completed trade record."""
        engine = self._ensure_engine()
        async with engine.begin() as conn:
            await conn.execute(trade_records.insert().values(**trade))
        log.debug("trade_saved", trade_id=trade.get("trade_id"))

    async def get_trades(
        self,
        underlying: Optional[str] = None,
        strategy: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Query trade records with optional filters."""
        engine = self._ensure_engine()
        query = trade_records.select().order_by(trade_records.c.exit_time.desc()).limit(limit)
        if underlying:
            query = query.where(trade_records.c.underlying == underlying)
        if strategy:
            query = query.where(trade_records.c.strategy_name == strategy)
        if since:
            query = query.where(trade_records.c.entry_time >= since)

        async with engine.connect() as conn:
            result = await conn.execute(query)
            return [dict(row._mapping) for row in result.fetchall()]

    async def get_trade_stats(
        self,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute aggregate trade statistics."""
        engine = self._ensure_engine()
        cols = trade_records.c
        query = sa.select(
            sa.func.count().label("total_trades"),
            sa.func.count().filter(cols.pnl > 0).label("winning_trades"),
            sa.func.count().filter(cols.pnl <= 0).label("losing_trades"),
            sa.func.sum(cols.pnl).label("total_pnl"),
            sa.func.avg(cols.pnl).label("avg_pnl"),
            sa.func.avg(cols.pnl_pct).label("avg_pnl_pct"),
            sa.func.avg(cols.hold_duration_minutes).label("avg_hold_minutes"),
        )
        if strategy:
            query = query.where(cols.strategy_name == strategy)

        async with engine.connect() as conn:
            row = (await conn.execute(query)).fetchone()
            if row is None:
                return {}
            mapping = dict(row._mapping)
            total = mapping.get("total_trades", 0) or 0
            wins = mapping.get("winning_trades", 0) or 0
            mapping["win_rate"] = wins / total if total > 0 else 0.0
            return mapping

    # ── Portfolio snapshots ───────────────────────────────────────────────

    async def save_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Persist a portfolio snapshot."""
        engine = self._ensure_engine()
        async with engine.begin() as conn:
            await conn.execute(portfolio_snapshots.insert().values(**snapshot))
        log.debug("snapshot_saved")

    async def get_snapshots(
        self,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query portfolio snapshots."""
        engine = self._ensure_engine()
        query = (
            portfolio_snapshots.select()
            .order_by(portfolio_snapshots.c.timestamp.desc())
            .limit(limit)
        )
        if since:
            query = query.where(portfolio_snapshots.c.timestamp >= since)

        async with engine.connect() as conn:
            result = await conn.execute(query)
            return [dict(row._mapping) for row in result.fetchall()]

    async def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Return the most recent portfolio snapshot, or ``None``."""
        results = await self.get_snapshots(limit=1)
        return results[0] if results else None

    # ── Raw queries (via asyncpg for performance) ─────────────────────────

    async def execute_raw(self, query: str, *args: Any) -> List[asyncpg.Record]:
        """Execute a raw SQL query via the asyncpg pool.

        Use this for performance-critical read paths where SQLAlchemy overhead
        is unacceptable.
        """
        if self._raw_pool is None:
            raise RuntimeError("Raw pool not initialised. Call .connect() first.")
        async with self._raw_pool.acquire() as conn:
            return await conn.fetch(query, *args)
