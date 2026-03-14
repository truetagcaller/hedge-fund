"""Capital allocation engine for distributing capital across strategies.

Supports equal-weight, manual, performance-weighted, and optimizer-based
allocation methods.  Persists allocations to MongoDB and provides
per-strategy capital queries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from hedgefund.logger import get_logger
from hedgefund.types import StrategyAllocation, StrategyMetrics

log = get_logger(__name__)

# Default strategy roster (mirrors AgentRegistry.create_default_agents)
DEFAULT_STRATEGIES: list[str] = [
    "trend_following",
    "mean_reversion",
    "options_volatility",
    "gamma_scalping",
    "news_reaction",
    "social_sentiment",
    "liquidity_sweep",
    "smart_money_flow",
]


class CapitalAllocator:
    """Distributes capital across strategies per user."""

    def __init__(self, db: Any) -> None:
        self._db = db

    # -- Core allocation -----------------------------------------------------

    async def allocate(
        self,
        user_id: str,
        total_capital: float,
        method: str = "equal",
        *,
        manual_weights: dict[str, float] | None = None,
        strategy_metrics: dict[str, StrategyMetrics] | None = None,
        strategies: list[str] | None = None,
    ) -> list[StrategyAllocation]:
        """Compute and persist capital allocations.

        Parameters
        ----------
        method : str
            ``"equal"`` – uniform distribution across active strategies.
            ``"manual"`` – use *manual_weights* dict (normalised to sum 1).
            ``"performance"`` – proportional to positive Sharpe ratios.
        """
        names = strategies or DEFAULT_STRATEGIES

        if method == "manual" and manual_weights:
            weights = self._normalise(manual_weights, names)
        elif method == "performance" and strategy_metrics:
            weights = self._performance_weights(strategy_metrics, names)
        else:
            # Default: equal weight
            n = len(names)
            weights = {s: 1.0 / n for s in names} if n else {}

        allocations: list[StrategyAllocation] = []
        now = datetime.now(timezone.utc)
        for name in names:
            pct = weights.get(name, 0.0)
            allocations.append(
                StrategyAllocation(
                    strategy_name=name,
                    user_id=user_id,
                    allocation_pct=pct,
                    allocated_capital=round(total_capital * pct, 2),
                    method=method,
                    updated_at=now,
                )
            )

        await self._save_allocations(allocations)
        log.info(
            "capital.allocated",
            user_id=user_id,
            method=method,
            n_strategies=len(allocations),
        )
        return allocations

    async def get_allocations(
        self,
        user_id: str,
    ) -> list[StrategyAllocation]:
        """Load persisted allocations for a user."""
        cursor = self._db.strategy_allocations.find({"user_id": user_id}, {"_id": 0})
        docs = await cursor.to_list(length=100)
        return [
            StrategyAllocation(
                strategy_name=d["strategy_name"],
                user_id=d["user_id"],
                allocation_pct=d.get("allocation_pct", 0.0),
                allocated_capital=d.get("allocated_capital", 0.0),
                method=d.get("method", "equal"),
                updated_at=d.get("updated_at", datetime.now(timezone.utc)),
            )
            for d in docs
        ]

    async def get_strategy_capital(
        self,
        user_id: str,
        strategy_name: str,
    ) -> float:
        """Return allocated capital for a single strategy."""
        doc = await self._db.strategy_allocations.find_one(
            {"user_id": user_id, "strategy_name": strategy_name},
            {"_id": 0, "allocated_capital": 1},
        )
        return doc["allocated_capital"] if doc else 0.0

    # -- Rebalancing ---------------------------------------------------------

    async def rebalance(
        self,
        user_id: str,
        strategy_metrics: dict[str, StrategyMetrics],
        total_capital: float,
    ) -> list[StrategyAllocation]:
        """Rebalance allocations using performance-weighted method."""
        return await self.allocate(
            user_id,
            total_capital,
            method="performance",
            strategy_metrics=strategy_metrics,
            strategies=list(strategy_metrics.keys()) or DEFAULT_STRATEGIES,
        )

    # -- Weight helpers ------------------------------------------------------

    @staticmethod
    def _normalise(
        raw: dict[str, float],
        names: list[str],
    ) -> dict[str, float]:
        """Normalise raw weights to sum to 1.0 over *names*."""
        filtered = {k: max(v, 0.0) for k, v in raw.items() if k in names}
        total = sum(filtered.values())
        if total <= 0:
            n = len(names)
            return {s: 1.0 / n for s in names} if n else {}
        return {k: v / total for k, v in filtered.items()}

    @staticmethod
    def _performance_weights(
        metrics: dict[str, StrategyMetrics],
        names: list[str],
    ) -> dict[str, float]:
        """Weight proportional to positive Sharpe; floor at min share."""
        min_share = 0.02  # 2% floor for any active strategy
        scores: dict[str, float] = {}
        for name in names:
            m = metrics.get(name)
            if m and m.total_trades >= 5 and m.sharpe_ratio > 0:
                scores[name] = m.sharpe_ratio
            else:
                scores[name] = 0.0

        total_score = sum(scores.values())
        if total_score <= 0:
            n = len(names)
            return {s: 1.0 / n for s in names} if n else {}

        weights: dict[str, float] = {}
        for name in names:
            raw = scores[name] / total_score
            weights[name] = max(raw, min_share)

        # Re-normalise after applying floor
        wsum = sum(weights.values())
        return {k: v / wsum for k, v in weights.items()}

    # -- Persistence ---------------------------------------------------------

    async def _save_allocations(
        self,
        allocations: list[StrategyAllocation],
    ) -> None:
        """Upsert each allocation to MongoDB."""
        for alloc in allocations:
            try:
                await self._db.strategy_allocations.update_one(
                    {
                        "user_id": alloc.user_id,
                        "strategy_name": alloc.strategy_name,
                    },
                    {"$set": alloc.to_dict()},
                    upsert=True,
                )
            except Exception:
                log.warning(
                    "capital.save_failed",
                    strategy=alloc.strategy_name,
                    exc_info=True,
                )
