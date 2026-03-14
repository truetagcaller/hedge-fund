"""Automatic strategy evolution engine.

Monitors per-strategy performance and auto-adjusts:
- Disables strategies below performance thresholds
- Puts underperformers on probation (reduced allocation)
- Boosts allocation to top performers
- Logs all decisions to MongoDB for audit trail
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from hedgefund.engine.capital_allocator import CapitalAllocator
from hedgefund.engine.strategy_tracker import StrategyPerformanceTracker
from hedgefund.logger import get_logger
from hedgefund.types import StrategyMetrics, StrategyState

log = get_logger(__name__)


class StrategyEvolutionEngine:
    """Evaluates strategy performance and auto-adjusts state/allocation."""

    def __init__(
        self,
        strategy_tracker: StrategyPerformanceTracker,
        capital_allocator: CapitalAllocator,
        db: Any,
        *,
        eval_interval: float = 3600.0,
        min_trades_for_eval: int = 10,
        disable_threshold_sharpe: float = -0.5,
        probation_threshold_sharpe: float = 0.0,
        probation_allocation_factor: float = 0.5,
        boost_threshold_sharpe: float = 0.5,
    ) -> None:
        self._tracker = strategy_tracker
        self._allocator = capital_allocator
        self._db = db
        self._eval_interval = eval_interval
        self._min_trades = min_trades_for_eval
        self._disable_sharpe = disable_threshold_sharpe
        self._probation_sharpe = probation_threshold_sharpe
        self._probation_factor = probation_allocation_factor
        self._boost_sharpe = boost_threshold_sharpe

        # In-memory strategy states per user
        self._strategy_states: dict[str, dict[str, StrategyState]] = {}
        self._task: asyncio.Task[None] | None = None
        self._running = False

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the background evaluation loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._eval_loop())
        log.info("strategy_evolution.started", interval=self._eval_interval)

    async def shutdown(self) -> None:
        """Stop the evaluation loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        log.info("strategy_evolution.shutdown")

    # -- Core evaluation -----------------------------------------------------

    async def evaluate_strategies(
        self,
        user_id: str,
    ) -> list[dict[str, Any]]:
        """Evaluate all strategies for a user and return actions taken."""
        all_metrics = await self._tracker.compute_all_metrics(user_id, window="30d")
        if not all_metrics:
            return []

        user_states = self._strategy_states.setdefault(user_id, {})
        actions: list[dict[str, Any]] = []

        for name, metrics in all_metrics.items():
            current = user_states.get(name, StrategyState.ENABLED)

            if metrics.total_trades < self._min_trades:
                continue  # Not enough data to evaluate

            new_state = self._decide_state(metrics, current)
            if new_state != current:
                action = await self._apply_transition(user_id, name, current, new_state, metrics)
                actions.append(action)
                user_states[name] = new_state

        return actions

    # -- Manual overrides ----------------------------------------------------

    async def force_enable(
        self,
        user_id: str,
        strategy_name: str,
    ) -> None:
        """Override: force a strategy to ENABLED."""
        user_states = self._strategy_states.setdefault(user_id, {})
        prev = user_states.get(strategy_name, StrategyState.ENABLED)
        user_states[strategy_name] = StrategyState.ENABLED
        await self._log_action(
            user_id,
            strategy_name,
            prev,
            StrategyState.ENABLED,
            "manual force_enable",
            None,
        )

    async def force_disable(
        self,
        user_id: str,
        strategy_name: str,
    ) -> None:
        """Override: force a strategy to DISABLED."""
        user_states = self._strategy_states.setdefault(user_id, {})
        prev = user_states.get(strategy_name, StrategyState.ENABLED)
        user_states[strategy_name] = StrategyState.DISABLED
        await self._log_action(
            user_id,
            strategy_name,
            prev,
            StrategyState.DISABLED,
            "manual force_disable",
            None,
        )

    # -- Status --------------------------------------------------------------

    def get_evolution_status(
        self,
        user_id: str,
    ) -> dict[str, Any]:
        """Return current strategy states for a user."""
        user_states = self._strategy_states.get(user_id, {})
        return {
            "user_id": user_id,
            "strategies": {name: state.value for name, state in user_states.items()},
            "thresholds": {
                "disable_sharpe": self._disable_sharpe,
                "probation_sharpe": self._probation_sharpe,
                "boost_sharpe": self._boost_sharpe,
                "min_trades": self._min_trades,
            },
        }

    def get_strategy_state(
        self,
        user_id: str,
        strategy_name: str,
    ) -> StrategyState:
        """Return the current state of a specific strategy."""
        return self._strategy_states.get(user_id, {}).get(strategy_name, StrategyState.ENABLED)

    # -- Internal ------------------------------------------------------------

    def _decide_state(
        self,
        metrics: StrategyMetrics,
        current: StrategyState,
    ) -> StrategyState:
        """Pure function: decide new state based on metrics."""
        sharpe = metrics.sharpe_ratio

        if sharpe < self._disable_sharpe:
            return StrategyState.DISABLED
        if sharpe < self._probation_sharpe:
            return StrategyState.PROBATION
        if sharpe >= self._boost_sharpe:
            return StrategyState.ENABLED
        # In the middle zone, stay in current state (avoid flip-flopping)
        if current == StrategyState.DISABLED:
            return StrategyState.PROBATION
        return current

    async def _apply_transition(
        self,
        user_id: str,
        strategy_name: str,
        prev: StrategyState,
        new: StrategyState,
        metrics: StrategyMetrics,
    ) -> dict[str, Any]:
        """Apply a state transition and log it."""
        action_name = f"{prev.value} -> {new.value}"
        reason = (
            f"30d sharpe={metrics.sharpe_ratio:.3f}, "
            f"trades={metrics.total_trades}, "
            f"win_rate={metrics.win_rate:.2%}"
        )
        await self._log_action(user_id, strategy_name, prev, new, reason, metrics)
        log.info(
            "strategy_evolution.transition",
            user_id=user_id,
            strategy=strategy_name,
            action=action_name,
            sharpe=metrics.sharpe_ratio,
        )
        return {
            "strategy_name": strategy_name,
            "action": action_name,
            "previous_state": prev.value,
            "new_state": new.value,
            "reason": reason,
        }

    async def _log_action(
        self,
        user_id: str,
        strategy_name: str,
        prev: StrategyState,
        new: StrategyState,
        reason: str,
        metrics: StrategyMetrics | None,
    ) -> None:
        """Persist evolution decision to MongoDB."""
        try:
            await self._db.strategy_evolution_log.insert_one(
                {
                    "user_id": user_id,
                    "strategy_name": strategy_name,
                    "action": new.value.lower(),
                    "previous_state": prev.value,
                    "new_state": new.value,
                    "reason": reason,
                    "metrics_snapshot": metrics.to_dict() if metrics else {},
                    "timestamp": datetime.now(timezone.utc),
                }
            )
        except Exception:
            log.warning("strategy_evolution.log_failed", exc_info=True)

    async def _eval_loop(self) -> None:
        """Background loop that evaluates all active user engines."""
        while self._running:
            try:
                await asyncio.sleep(self._eval_interval)
                # Evaluate all users with strategy states
                for user_id in list(self._strategy_states.keys()):
                    try:
                        await self.evaluate_strategies(user_id)
                    except Exception:
                        log.warning(
                            "strategy_evolution.eval_error",
                            user_id=user_id,
                            exc_info=True,
                        )
            except asyncio.CancelledError:
                break
            except Exception:
                log.warning("strategy_evolution.loop_error", exc_info=True)
