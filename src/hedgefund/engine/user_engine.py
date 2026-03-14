"""Per-user isolated execution engine and manager.

Each user gets a ``UserExecutionEngine`` with its own signal queue,
drawdown monitor, strategy configuration, and execution context.
The ``UserEngineManager`` is the factory/lifecycle manager.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from hedgefund.engine.capital_allocator import CapitalAllocator
from hedgefund.engine.strategy_tracker import StrategyPerformanceTracker
from hedgefund.logger import get_logger
from hedgefund.risk.drawdown import DrawdownMonitor, TradingState
from hedgefund.types import (
    StrategyState,
    TradingExecutionContext,
    UserEngineState,
)

log = get_logger(__name__)


class UserExecutionEngine:
    """Per-user isolated execution engine.

    Owns:
    - execution_context (TradingExecutionContext)
    - drawdown_monitor (per-user risk state)
    - signal_queue (asyncio.Queue for per-user signals)
    - strategy_states (which strategies are enabled/disabled)
    """

    def __init__(
        self,
        user_id: str,
        *,
        event_bus: Any,
        broker_manager: Any,
        broker_router: Any,
        execution_bridge: Any | None = None,
        session_manager: Any | None = None,
        db: Any,
        capital_allocator: CapitalAllocator,
        strategy_tracker: StrategyPerformanceTracker,
        max_drawdown_pct: float = 0.10,
    ) -> None:
        self.user_id = user_id
        self._event_bus = event_bus
        self._broker_manager = broker_manager
        self._broker_router = broker_router
        self._execution_bridge = execution_bridge
        self._session_manager = session_manager
        self._db = db
        self._capital_allocator = capital_allocator
        self._strategy_tracker = strategy_tracker

        self._state = UserEngineState.INITIALIZING
        self._context: TradingExecutionContext | None = None
        self._signal_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self._drawdown = DrawdownMonitor(max_drawdown_pct=max_drawdown_pct)
        self._strategy_states: dict[str, StrategyState] = {}
        self._consumer_task: asyncio.Task[None] | None = None
        self._signals_processed: int = 0
        self._signals_rejected: int = 0
        self._started_at: datetime | None = None

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the engine and begin processing signals."""
        if self._state == UserEngineState.ACTIVE:
            return

        # Load execution context from session manager if available
        if self._session_manager:
            try:
                ctx_dict = await self._session_manager.get_execution_context(self.user_id)
                if ctx_dict:
                    from hedgefund.types import AssetClass, TradingMode

                    self._context = TradingExecutionContext(
                        user_id=ctx_dict["user_id"],
                        active_broker=ctx_dict.get("active_broker", "paper"),
                        asset_class=AssetClass(ctx_dict.get("asset_class", "OPTIONS")),
                        trading_mode=TradingMode(ctx_dict.get("trading_mode", "PAPER")),
                        broker_account_id=ctx_dict.get("broker_account_id", ""),
                        market_segment=ctx_dict.get("market_segment", ""),
                    )
            except Exception:
                log.warning(
                    "user_engine.context_load_failed",
                    user_id=self.user_id,
                    exc_info=True,
                )

        self._consumer_task = asyncio.create_task(self._process_signal_queue())
        self._state = UserEngineState.ACTIVE
        self._started_at = datetime.now(timezone.utc)
        log.info("user_engine.started", user_id=self.user_id)

    async def shutdown(self) -> None:
        """Stop the engine and drain signal queue."""
        self._state = UserEngineState.SHUTDOWN
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        self._consumer_task = None
        log.info("user_engine.shutdown", user_id=self.user_id)

    # -- Context management --------------------------------------------------

    async def set_execution_context(
        self,
        ctx: TradingExecutionContext,
    ) -> None:
        """Set the execution context for this engine."""
        self._context = ctx
        if self._session_manager:
            try:
                await self._session_manager.save_execution_context(self.user_id, ctx.to_dict())
            except Exception:
                log.warning("user_engine.context_save_failed", exc_info=True)
        # Also propagate to the execution bridge
        if self._execution_bridge:
            await self._execution_bridge.set_execution_context(ctx)

    async def get_execution_context(
        self,
    ) -> TradingExecutionContext | None:
        """Return current execution context."""
        return self._context

    # -- Strategy management -------------------------------------------------

    def enable_strategy(self, strategy_name: str) -> None:
        """Enable a strategy for signal processing."""
        self._strategy_states[strategy_name] = StrategyState.ENABLED

    def disable_strategy(self, strategy_name: str) -> None:
        """Disable a strategy — its signals will be skipped."""
        self._strategy_states[strategy_name] = StrategyState.DISABLED

    def set_strategy_state(
        self,
        strategy_name: str,
        state: StrategyState,
    ) -> None:
        """Set an arbitrary strategy state."""
        self._strategy_states[strategy_name] = state

    def get_active_strategies(self) -> list[str]:
        """Return names of strategies that are not DISABLED."""
        return [
            name for name, state in self._strategy_states.items() if state != StrategyState.DISABLED
        ]

    def get_strategy_states(self) -> dict[str, str]:
        """Return all strategy states as a dict."""
        return {k: v.value for k, v in self._strategy_states.items()}

    # -- Signal processing ---------------------------------------------------

    async def enqueue_signal(self, signal_doc: dict[str, Any]) -> None:
        """Add a signal to this engine's processing queue."""
        if self._state != UserEngineState.ACTIVE:
            return
        try:
            self._signal_queue.put_nowait(signal_doc)
        except asyncio.QueueFull:
            self._signals_rejected += 1
            log.warning(
                "user_engine.signal_queue_full",
                user_id=self.user_id,
                rejected=self._signals_rejected,
            )

    async def _process_signal_queue(self) -> None:
        """Consumer loop: process signals from the queue."""
        while self._state == UserEngineState.ACTIVE:
            try:
                signal_doc = await asyncio.wait_for(self._signal_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._handle_signal(signal_doc)
            except Exception:
                log.warning(
                    "user_engine.signal_processing_error",
                    user_id=self.user_id,
                    exc_info=True,
                )

    async def _handle_signal(self, signal_doc: dict[str, Any]) -> None:
        """Process a single signal through the execution pipeline."""
        # Check strategy filter
        strategy = signal_doc.get("strategy_name", "")
        if strategy:
            state = self._strategy_states.get(strategy, StrategyState.ENABLED)
            if state == StrategyState.DISABLED:
                self._signals_rejected += 1
                return

        # Check drawdown circuit breaker
        if not self._drawdown.is_trading_allowed:
            self._signals_rejected += 1
            log.info(
                "user_engine.signal_blocked_drawdown",
                user_id=self.user_id,
                state=self._drawdown.state.value,
            )
            return

        # Delegate to execution bridge
        if self._execution_bridge:
            try:
                record = await self._execution_bridge.execute_signal(signal_doc, self.user_id)
                self._signals_processed += 1

                # Record trade for strategy tracking
                if (
                    record
                    and hasattr(record, "status")
                    and record.status == "executed"
                    and strategy
                ):
                    await self._strategy_tracker.record_trade(
                        self.user_id,
                        strategy,
                        {
                            "trade_id": getattr(record, "trade_id", ""),
                            "signal_id": getattr(record, "signal_id", ""),
                            "symbol": getattr(record, "symbol", ""),
                            "action": getattr(record, "action", ""),
                            "entry_price": getattr(record, "filled_price", 0.0),
                            "quantity": getattr(record, "quantity", 0),
                            "entry_time": datetime.now(timezone.utc),
                            "broker_id": getattr(record, "broker_id", ""),
                        },
                    )
            except Exception:
                log.warning(
                    "user_engine.execution_error",
                    user_id=self.user_id,
                    exc_info=True,
                )

    # -- Portfolio -----------------------------------------------------------

    async def get_portfolio_snapshot(self) -> dict[str, Any]:
        """Get portfolio snapshot via broker router."""
        try:
            return await self._broker_router.get_portfolio(self.user_id)
        except Exception:
            return {"error": "portfolio unavailable"}

    # -- Risk ----------------------------------------------------------------

    def update_equity(self, equity: float) -> None:
        """Update the per-user drawdown monitor."""
        self._drawdown.update(equity)

    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed per drawdown state."""
        return self._drawdown.is_trading_allowed

    def get_drawdown_state(self) -> TradingState:
        """Return the current drawdown FSM state."""
        return self._drawdown.state

    # -- Status --------------------------------------------------------------

    @property
    def state(self) -> UserEngineState:
        return self._state

    def get_status(self) -> dict[str, Any]:
        """Return comprehensive engine status."""
        return {
            "user_id": self.user_id,
            "state": self._state.value,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "context": self._context.to_dict() if self._context else None,
            "signal_queue_depth": self._signal_queue.qsize(),
            "signals_processed": self._signals_processed,
            "signals_rejected": self._signals_rejected,
            "strategy_states": self.get_strategy_states(),
            "active_strategies": self.get_active_strategies(),
            "drawdown_state": self._drawdown.state.value,
            "trading_allowed": self._drawdown.is_trading_allowed,
            "max_drawdown_observed": self._drawdown.max_drawdown_observed,
        }


class UserEngineManager:
    """Factory and lifecycle manager for all UserExecutionEngines."""

    def __init__(
        self,
        event_bus: Any,
        broker_manager: Any,
        broker_router: Any,
        session_manager: Any | None,
        db: Any,
        capital_allocator: CapitalAllocator,
        strategy_tracker: StrategyPerformanceTracker,
        *,
        execution_bridge: Any | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._broker_manager = broker_manager
        self._broker_router = broker_router
        self._session_manager = session_manager
        self._db = db
        self._capital_allocator = capital_allocator
        self._strategy_tracker = strategy_tracker
        self._execution_bridge = execution_bridge

        self._engines: dict[str, UserExecutionEngine] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_engine(
        self,
        user_id: str,
    ) -> UserExecutionEngine:
        """Get existing engine or create and start a new one."""
        if user_id in self._engines:
            engine = self._engines[user_id]
            if engine.state != UserEngineState.SHUTDOWN:
                return engine

        async with self._lock:
            # Double-check after acquiring lock
            if user_id in self._engines:
                engine = self._engines[user_id]
                if engine.state != UserEngineState.SHUTDOWN:
                    return engine

            engine = UserExecutionEngine(
                user_id,
                event_bus=self._event_bus,
                broker_manager=self._broker_manager,
                broker_router=self._broker_router,
                execution_bridge=self._execution_bridge,
                session_manager=self._session_manager,
                db=self._db,
                capital_allocator=self._capital_allocator,
                strategy_tracker=self._strategy_tracker,
            )
            await engine.start()
            self._engines[user_id] = engine
            log.info(
                "engine_manager.engine_created",
                user_id=user_id,
                total_engines=len(self._engines),
            )
            return engine

    async def get_engine(
        self,
        user_id: str,
    ) -> UserExecutionEngine | None:
        """Get engine if it exists and is active."""
        engine = self._engines.get(user_id)
        if engine and engine.state != UserEngineState.SHUTDOWN:
            return engine
        return None

    async def shutdown_engine(self, user_id: str) -> None:
        """Shutdown a specific user's engine."""
        engine = self._engines.get(user_id)
        if engine:
            await engine.shutdown()
            log.info("engine_manager.engine_shutdown", user_id=user_id)

    async def shutdown_all(self) -> None:
        """Shutdown all active engines."""
        for user_id, engine in self._engines.items():
            if engine.state != UserEngineState.SHUTDOWN:
                await engine.shutdown()
        count = len(self._engines)
        self._engines.clear()
        log.info("engine_manager.all_shutdown", count=count)

    async def broadcast_signal(self, signal_doc: dict[str, Any]) -> None:
        """Push a signal to all active user engines."""
        for engine in self._engines.values():
            if engine.state == UserEngineState.ACTIVE:
                await engine.enqueue_signal(signal_doc)

    def get_active_engines(self) -> list[dict[str, Any]]:
        """Return status of all engines."""
        return [
            engine.get_status()
            for engine in self._engines.values()
            if engine.state != UserEngineState.SHUTDOWN
        ]
