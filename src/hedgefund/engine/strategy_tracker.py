"""Per-strategy performance tracking and metrics computation.

Records every trade outcome tagged by strategy, computes rolling metrics
over configurable time windows, and caches results in MongoDB.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from hedgefund.logger import get_logger
from hedgefund.types import StrategyMetrics

log = get_logger(__name__)

_WINDOWS = {
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90),
}


class StrategyPerformanceTracker:
    """Tracks and computes per-strategy performance metrics."""

    def __init__(self, db: Any) -> None:
        self._db = db

    # -- Recording -----------------------------------------------------------

    async def record_trade(
        self,
        user_id: str,
        strategy_name: str,
        trade_result: dict[str, Any],
    ) -> None:
        """Persist a completed trade to the strategy_trades collection."""
        doc = {
            "user_id": user_id,
            "strategy_name": strategy_name,
            "trade_id": trade_result.get("trade_id", ""),
            "signal_id": trade_result.get("signal_id", ""),
            "symbol": trade_result.get("symbol", ""),
            "action": trade_result.get("action", ""),
            "entry_price": trade_result.get("entry_price", 0.0),
            "exit_price": trade_result.get("exit_price", 0.0),
            "quantity": trade_result.get("quantity", 0),
            "pnl": trade_result.get("pnl", 0.0),
            "pnl_pct": trade_result.get("pnl_pct", 0.0),
            "hold_minutes": trade_result.get("hold_minutes", 0),
            "entry_time": trade_result.get("entry_time"),
            "exit_time": trade_result.get("exit_time", datetime.now(timezone.utc)),
            "regime": trade_result.get("regime", ""),
            "broker_id": trade_result.get("broker_id", ""),
            "source": "execution_bridge",
        }
        try:
            await self._db.strategy_trades.insert_one(doc)
            log.info(
                "strategy_trade.recorded",
                user_id=user_id,
                strategy=strategy_name,
                pnl=doc["pnl"],
            )
        except Exception:
            log.warning("strategy_trade.record_failed", exc_info=True)

    # -- Metrics computation -------------------------------------------------

    async def compute_metrics(
        self,
        user_id: str,
        strategy_name: str,
        window: str = "30d",
    ) -> StrategyMetrics:
        """Compute performance metrics for a strategy over a time window."""
        query: dict[str, Any] = {
            "user_id": user_id,
            "strategy_name": strategy_name,
        }
        if window in _WINDOWS:
            cutoff = datetime.now(timezone.utc) - _WINDOWS[window]
            query["exit_time"] = {"$gte": cutoff}

        cursor = self._db.strategy_trades.find(query).sort("exit_time", -1)
        trades: list[dict[str, Any]] = await cursor.to_list(length=10_000)

        metrics = self._calculate_metrics(user_id, strategy_name, window, trades)

        # Cache in strategy_performance
        try:
            await self._db.strategy_performance.update_one(
                {
                    "user_id": user_id,
                    "strategy_name": strategy_name,
                    "window": window,
                },
                {"$set": metrics.to_dict()},
                upsert=True,
            )
        except Exception:
            log.warning("strategy_performance.cache_failed", exc_info=True)

        return metrics

    async def compute_all_metrics(
        self,
        user_id: str,
        window: str = "30d",
    ) -> dict[str, StrategyMetrics]:
        """Compute metrics for every strategy the user has traded."""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": "$strategy_name"}},
        ]
        names: list[str] = []
        async for doc in self._db.strategy_trades.aggregate(pipeline):
            names.append(doc["_id"])

        result: dict[str, StrategyMetrics] = {}
        for name in names:
            result[name] = await self.compute_metrics(user_id, name, window)
        return result

    # -- Queries -------------------------------------------------------------

    async def get_strategy_history(
        self,
        user_id: str,
        strategy_name: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return recent trades for a strategy."""
        cursor = (
            self._db.strategy_trades.find(
                {"user_id": user_id, "strategy_name": strategy_name},
                {"_id": 0},
            )
            .sort("exit_time", -1)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    async def get_strategy_equity_curve(
        self,
        user_id: str,
        strategy_name: str,
    ) -> list[dict[str, Any]]:
        """Return cumulative P&L series for a strategy."""
        cursor = self._db.strategy_trades.find(
            {"user_id": user_id, "strategy_name": strategy_name},
            {"_id": 0, "exit_time": 1, "pnl": 1},
        ).sort("exit_time", 1)
        trades = await cursor.to_list(length=10_000)

        cumulative = 0.0
        curve: list[dict[str, Any]] = []
        for t in trades:
            cumulative += t.get("pnl", 0.0)
            curve.append(
                {
                    "timestamp": t.get("exit_time", "").isoformat()
                    if hasattr(t.get("exit_time", ""), "isoformat")
                    else str(t.get("exit_time", "")),
                    "cumulative_pnl": round(cumulative, 2),
                }
            )
        return curve

    # -- Internal ------------------------------------------------------------

    @staticmethod
    def _calculate_metrics(
        user_id: str,
        strategy_name: str,
        window: str,
        trades: list[dict[str, Any]],
    ) -> StrategyMetrics:
        """Pure computation of metrics from a list of trade docs."""
        total = len(trades)
        if total == 0:
            return StrategyMetrics(
                strategy_name=strategy_name,
                user_id=user_id,
                window=window,
            )

        pnls = [t.get("pnl", 0.0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total if total else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        if math.isinf(profit_factor):
            profit_factor = 99.99

        total_pnl = sum(pnls)
        avg_win = gross_profit / win_count if win_count else 0.0
        avg_loss = gross_loss / loss_count if loss_count else 0.0
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0.0

        # Sharpe estimate (annualised from daily-ish returns)
        if len(pnls) >= 2:
            mean_r = sum(pnls) / len(pnls)
            var_r = sum((p - mean_r) ** 2 for p in pnls) / (len(pnls) - 1)
            std_r = math.sqrt(var_r) if var_r > 0 else 1e-9
            sharpe = (mean_r / std_r) * math.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown from cumulative P&L
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cum += p
            if cum > peak:
                peak = cum
            dd = (peak - cum) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        hold_minutes = [t.get("hold_minutes", 0) for t in trades]
        avg_hold = sum(hold_minutes) / total if total else 0.0

        return StrategyMetrics(
            strategy_name=strategy_name,
            user_id=user_id,
            window=window,
            total_trades=total,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            avg_rr=avg_rr,
            total_pnl=total_pnl,
            avg_hold_minutes=avg_hold,
        )
