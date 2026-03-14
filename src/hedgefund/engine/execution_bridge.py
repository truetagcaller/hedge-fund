"""Execution bridge — connects AI signals to live order execution.

This is the core Level-3 module that bridges the gap between signal
generation (SignalRunner/FusionEngine) and actual trade execution via
the multi-broker routing layer.

Pipeline:
    FusedSignal (MongoDB) → Validation → Risk Check → Instrument Mapping
    → Market Session Check → Broker Routing → Order Execution → Record

CRITICAL: Only executes signals with ``is_live_data=True`` from verified
data sources.  Never auto-executes synthetic or unverified signals.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import structlog

from hedgefund.exceptions import (
    BrokerCapabilityError,
    ExecutionError,
    InstrumentMappingError,
)
from hedgefund.execution.broker_manager import BrokerManager
from hedgefund.execution.broker_router import BrokerRouter
from hedgefund.execution.instrument_mapper import InstrumentMapper
from hedgefund.execution.market_session import MarketSessionManager
from hedgefund.execution.session_manager import BrokerSessionManager
from hedgefund.risk.base import RiskManager
from hedgefund.risk.position_sizer import PositionSizer
from hedgefund.streaming.event_bus import Event, EventBus, EventType
from hedgefund.types import (
    AssetClass,
    OptionContract,
    OptionType,
    Order,
    OrderStatus,
    OrderType,
    Side,
    SignalAction,
    SignalDirection,
    TradeSignal,
    TradingExecutionContext,
    TradingMode,
)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ExecutionRecord:
    """Outcome of executing a signal through the bridge."""

    record_id: str
    signal_id: str
    user_id: str
    broker_id: str
    symbol: str
    action: str
    status: str  # "executed", "rejected", "failed", "pending"
    order: Optional[Order] = None
    trade_id: Optional[str] = None
    filled_price: Optional[float] = None
    quantity: int = 0
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "signal_id": self.signal_id,
            "user_id": self.user_id,
            "broker_id": self.broker_id,
            "symbol": self.symbol,
            "action": self.action,
            "status": self.status,
            "trade_id": self.trade_id,
            "filled_price": self.filled_price,
            "quantity": self.quantity,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Execution Bridge
# ---------------------------------------------------------------------------

class ExecutionBridge:
    """Bridges AI signal generation and live order execution.

    Parameters
    ----------
    event_bus:
        Central event bus for FILL / PORTFOLIO_UPDATE events.
    broker_manager:
        Multi-broker connection manager.
    broker_router:
        Per-user broker routing with asset-class validation.
    risk_manager:
        Risk validation pipeline (optional).
    position_sizer:
        Position sizing engine.
    session_manager:
        Redis-backed broker session tracker (optional).
    db:
        MongoDB instance for reading signals and recording executions.
    auto_execute:
        If ``True``, automatically execute qualifying signals.
        If ``False``, only execute on explicit API calls.
    poll_interval:
        Seconds between auto-execution polls.
    """

    def __init__(
        self,
        event_bus: EventBus,
        broker_manager: BrokerManager,
        broker_router: BrokerRouter,
        *,
        risk_manager: Optional[RiskManager] = None,
        position_sizer: Optional[PositionSizer] = None,
        session_manager: Optional[BrokerSessionManager] = None,
        strategy_tracker: Any = None,
        db: Any = None,
        auto_execute: bool = False,
        poll_interval: float = 30.0,
    ) -> None:
        self._bus = event_bus
        self._broker_mgr = broker_manager
        self._broker_router = broker_router
        self._risk_mgr = risk_manager
        self._sizer = position_sizer or PositionSizer()
        self._session_mgr = session_manager
        self._strategy_tracker = strategy_tracker
        self._db = db
        self._auto_execute = auto_execute
        self._poll_interval = poll_interval

        self._instrument_mapper = InstrumentMapper()
        self._market_session = MarketSessionManager()

        # State
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None
        self._execution_history: list[ExecutionRecord] = []
        self._open_trades: Dict[str, Dict[str, Any]] = {}

        # Per-user execution contexts (in-memory, backed by session_manager)
        self._user_contexts: Dict[str, TradingExecutionContext] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the execution bridge (and auto-execution loop if enabled)."""
        self._running = True
        if self._auto_execute:
            self._task = asyncio.create_task(
                self._auto_execute_loop(), name="execution-bridge",
            )
        log.info(
            "execution_bridge.started",
            auto_execute=self._auto_execute,
            poll_interval=self._poll_interval,
        )

    async def shutdown(self) -> None:
        """Stop the execution bridge."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info(
            "execution_bridge.shutdown",
            executions=len(self._execution_history),
        )

    # ── Execution context management ──────────────────────────────────

    async def set_execution_context(
        self, ctx: TradingExecutionContext,
    ) -> None:
        """Set the execution context for a user."""
        self._user_contexts[ctx.user_id] = ctx

        if self._session_mgr is not None:
            await self._session_mgr.save_execution_context(
                ctx.user_id, ctx.to_dict(),
            )

        log.info(
            "execution_bridge.context_set",
            user_id=ctx.user_id,
            broker=ctx.active_broker,
            asset_class=ctx.asset_class.value,
            mode=ctx.trading_mode.value,
        )

    async def get_execution_context(
        self, user_id: str,
    ) -> TradingExecutionContext | None:
        """Get the execution context for a user."""
        ctx = self._user_contexts.get(user_id)
        if ctx is not None:
            return ctx

        if self._session_mgr is not None:
            data = await self._session_mgr.get_execution_context(user_id)
            if data:
                ctx = TradingExecutionContext(
                    user_id=data["user_id"],
                    active_broker=data["active_broker"],
                    asset_class=AssetClass(data["asset_class"]),
                    trading_mode=TradingMode(data["trading_mode"]),
                    broker_account_id=data.get("broker_account_id", ""),
                    market_segment=data.get("market_segment", ""),
                )
                self._user_contexts[user_id] = ctx
                return ctx

        return None

    # ── Core execution ────────────────────────────────────────────────

    async def execute_signal(
        self,
        signal_doc: Dict[str, Any],
        user_id: str,
        *,
        broker_id: str | None = None,
    ) -> ExecutionRecord:
        """Execute a single signal through the full pipeline.

        Steps:
        1. Resolve execution context
        2. Validate signal data integrity
        3. Validate asset class match
        4. Check market session
        5. Risk validation
        6. Position sizing
        7. Instrument mapping
        8. Build & route order
        9. Record result
        """
        signal_id = signal_doc.get("signal_id", "UNKNOWN")
        symbol = signal_doc.get("underlying", "")
        action_str = signal_doc.get("action", "NO_TRADE")

        # 1. Resolve execution context
        ctx = await self.get_execution_context(user_id)
        if ctx is None:
            # Build default context from active broker
            active = await self._broker_router.get_active_broker(user_id)
            if active is None:
                return self._fail(
                    signal_id, user_id, symbol, action_str,
                    "No execution context or active broker set.",
                )
            ctx = TradingExecutionContext(
                user_id=user_id,
                active_broker=active,
                asset_class=AssetClass.OPTIONS,
                trading_mode=TradingMode.PAPER,
            )

        target_broker = broker_id or ctx.active_broker

        # 2. Validate signal data integrity
        if not signal_doc.get("is_live_data", False):
            return self._fail(
                signal_id, user_id, symbol, action_str,
                "Signal does not have verified live data.",
                broker_id=target_broker,
            )

        if action_str == "NO_TRADE":
            return self._fail(
                signal_id, user_id, symbol, action_str,
                "Signal action is NO_TRADE.",
                broker_id=target_broker,
            )

        # 3. Validate asset class
        broker_type = self._broker_router._get_broker_type_sync(target_broker)
        if broker_type and not self._broker_router.validate_asset_class(
            broker_type, ctx.asset_class,
        ):
            # Try auto-routing to a capable broker
            alt = self._broker_router.find_broker_for_asset_class(
                user_id, ctx.asset_class,
            )
            if alt is None:
                return self._fail(
                    signal_id, user_id, symbol, action_str,
                    f"No broker supports {ctx.asset_class.value}.",
                    broker_id=target_broker,
                )
            target_broker = alt

        # 4. Check market session
        if ctx.trading_mode == TradingMode.LIVE:
            if not self._market_session.is_asset_class_tradeable(
                ctx.asset_class.value,
            ):
                return self._fail(
                    signal_id, user_id, symbol, action_str,
                    f"Market closed for {ctx.asset_class.value}.",
                    broker_id=target_broker,
                )

        # 5. Risk validation
        if self._risk_mgr is not None:
            try:
                portfolio = await self._broker_mgr.get_aggregate_portfolio()
                trade_signal = self._doc_to_signal(signal_doc)
                verdict = await self._risk_mgr.validate_signal(
                    trade_signal, portfolio,
                )
                if not verdict.approved:
                    return self._fail(
                        signal_id, user_id, symbol, action_str,
                        f"Risk rejected: {verdict.reason}",
                        broker_id=target_broker,
                    )
            except Exception as exc:
                return self._fail(
                    signal_id, user_id, symbol, action_str,
                    f"Risk check error: {exc}",
                    broker_id=target_broker,
                )

        # 6. Position sizing
        entry_price = signal_doc.get("entry_price", 0)
        stop_loss = signal_doc.get("stop_loss", 0)
        risk_per_contract = abs(entry_price - stop_loss) if stop_loss else entry_price * 0.02

        capital = await self._get_capital(target_broker)
        sizing = self._sizer.fixed_fraction(
            capital=capital,
            risk_per_contract=risk_per_contract,
        )
        quantity = sizing.quantity
        if quantity <= 0:
            return self._fail(
                signal_id, user_id, symbol, action_str,
                "Position sizing returned 0 contracts.",
                broker_id=target_broker,
            )

        # 7. Instrument mapping
        try:
            action = SignalAction(action_str)
            mapper = InstrumentMapper(broker_type or "zerodha")
            mapped = mapper.map(
                symbol=symbol,
                action=action,
                asset_class=ctx.asset_class,
                strike=entry_price if ctx.asset_class == AssetClass.OPTIONS else None,
            )
        except (InstrumentMappingError, ValueError) as exc:
            return self._fail(
                signal_id, user_id, symbol, action_str,
                f"Instrument mapping failed: {exc}",
                broker_id=target_broker,
            )

        # 8. Build & route order
        side = (
            Side.BUY
            if action in (SignalAction.BUY_CALL, SignalAction.BUY_PUT)
            else Side.SELL
        )
        option_type = (
            OptionType.CALL
            if action in (SignalAction.BUY_CALL, SignalAction.SELL_CALL)
            else OptionType.PUT
        )

        contract = mapped.contract or OptionContract(
            symbol=mapped.broker_symbol,
            underlying=symbol,
            option_type=option_type,
            strike=entry_price,
            expiration=datetime.now(timezone.utc).date(),
        )

        order = Order(
            order_id=Order.generate_id(),
            signal_id=signal_id,
            contract=contract,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            limit_price=entry_price if entry_price > 0 else None,
        )

        try:
            filled_order = await self._broker_router.route_with_context(
                ctx, order, broker_id=target_broker,
            )
        except (ExecutionError, BrokerCapabilityError) as exc:
            return self._fail(
                signal_id, user_id, symbol, action_str,
                f"Order routing failed: {exc}",
                broker_id=target_broker,
            )

        if filled_order.status not in (OrderStatus.FILLED, OrderStatus.PARTIAL, OrderStatus.SUBMITTED):
            return self._fail(
                signal_id, user_id, symbol, action_str,
                f"Order not filled: {filled_order.status.value}",
                broker_id=target_broker,
            )

        # 9. Record result
        fill_price = filled_order.filled_price or entry_price
        trade_id = f"TRD-{uuid.uuid4().hex[:12].upper()}"

        record = ExecutionRecord(
            record_id=f"EXE-{uuid.uuid4().hex[:12].upper()}",
            signal_id=signal_id,
            user_id=user_id,
            broker_id=target_broker,
            symbol=symbol,
            action=action_str,
            status="executed",
            order=filled_order,
            trade_id=trade_id,
            filled_price=fill_price,
            quantity=filled_order.filled_quantity or quantity,
        )
        self._execution_history.append(record)

        # Track open trade
        self._open_trades[trade_id] = {
            "trade_id": trade_id,
            "signal_id": signal_id,
            "user_id": user_id,
            "broker_id": target_broker,
            "symbol": symbol,
            "action": action_str,
            "entry_price": fill_price,
            "stop_loss": stop_loss,
            "target_price": signal_doc.get("target_price", 0),
            "quantity": record.quantity,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }

        # Publish fill event
        await self._bus.publish(Event(
            event_type=EventType.FILL,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            data={
                "trade_id": trade_id,
                "signal_id": signal_id,
                "action": action_str,
                "fill_price": fill_price,
                "quantity": record.quantity,
                "broker_id": target_broker,
                "source": "execution_bridge",
            },
            source="execution_bridge",
        ))

        # Update signal outcome in MongoDB
        await self._update_signal_outcome(signal_id, "executed", trade_id)

        # Level-4: record trade for strategy performance tracking
        if self._strategy_tracker is not None:
            strategy_name = signal_doc.get("strategy_name", "signal_fusion")
            try:
                await self._strategy_tracker.record_trade(
                    user_id,
                    strategy_name,
                    {
                        "trade_id": trade_id,
                        "signal_id": signal_id,
                        "symbol": symbol,
                        "action": action_str,
                        "entry_price": fill_price,
                        "quantity": record.quantity,
                        "entry_time": datetime.now(timezone.utc),
                        "broker_id": target_broker,
                    },
                )
            except Exception:
                log.debug(
                    "execution_bridge.strategy_track_failed", exc_info=True
                )

        log.info(
            "execution_bridge.executed",
            signal_id=signal_id,
            trade_id=trade_id,
            symbol=symbol,
            action=action_str,
            fill_price=fill_price,
            quantity=record.quantity,
            broker_id=target_broker,
        )

        return record

    # ── Auto-execution loop ───────────────────────────────────────────

    async def _auto_execute_loop(self) -> None:
        """Poll MongoDB for pending signals and execute them."""
        while self._running:
            try:
                await self._poll_and_execute()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("execution_bridge.auto_execute_error")
            await asyncio.sleep(self._poll_interval)

    async def _poll_and_execute(self) -> None:
        """Find and execute pending signals."""
        if self._db is None:
            return

        cursor = self._db.signals.find({
            "outcome": "active",
            "is_live_data": True,
        }).sort("timestamp", -1).limit(10)

        async for doc in cursor:
            user_id = doc.get("user_id", "")
            if not user_id:
                continue

            signal_id = doc.get("signal_id", "")
            action = doc.get("action", "NO_TRADE")
            if action == "NO_TRADE":
                continue

            # Check if already executed
            if any(
                r.signal_id == signal_id and r.status == "executed"
                for r in self._execution_history
            ):
                continue

            try:
                await self.execute_signal(doc, user_id)
            except Exception:
                log.exception(
                    "execution_bridge.signal_execution_failed",
                    signal_id=signal_id,
                )

    # ── Trade management ──────────────────────────────────────────────

    async def close_trade(
        self, trade_id: str, reason: str = "manual",
    ) -> ExecutionRecord | None:
        """Close an open trade by submitting a closing order."""
        trade = self._open_trades.pop(trade_id, None)
        if trade is None:
            return None

        user_id = trade["user_id"]
        broker_id = trade["broker_id"]
        action = trade["action"]

        # Determine close side
        close_side = (
            Side.SELL
            if action in ("BUY_CALL", "BUY_PUT")
            else Side.BUY
        )

        contract = OptionContract(
            symbol=trade["symbol"],
            underlying=trade["symbol"],
            option_type=OptionType.CALL if "CALL" in action else OptionType.PUT,
            strike=trade["entry_price"],
            expiration=datetime.now(timezone.utc).date(),
        )

        order = Order(
            order_id=Order.generate_id(),
            signal_id=trade["signal_id"],
            contract=contract,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=trade["quantity"],
        )

        try:
            broker = await self._broker_mgr.get_broker(broker_id)
            filled = await broker.submit_order(order)
            exit_price = filled.filled_price or trade["entry_price"]
        except Exception as exc:
            log.error(
                "execution_bridge.close_failed",
                trade_id=trade_id,
                error=str(exc),
            )
            exit_price = trade["entry_price"]

        record = ExecutionRecord(
            record_id=f"EXE-{uuid.uuid4().hex[:12].upper()}",
            signal_id=trade["signal_id"],
            user_id=user_id,
            broker_id=broker_id,
            symbol=trade["symbol"],
            action=f"CLOSE_{action}",
            status="executed",
            trade_id=trade_id,
            filled_price=exit_price,
            quantity=trade["quantity"],
            reason=reason,
        )
        self._execution_history.append(record)

        await self._bus.publish(Event(
            event_type=EventType.PORTFOLIO_UPDATE,
            timestamp=datetime.now(timezone.utc),
            symbol=trade["symbol"],
            data={
                "trade_id": trade_id,
                "action": "close",
                "reason": reason,
                "exit_price": exit_price,
                "source": "execution_bridge",
            },
            source="execution_bridge",
        ))

        # Level-4: record close for strategy tracking
        if self._strategy_tracker is not None:
            strategy = trade.get("strategy_name", "signal_fusion")
            entry_px = trade.get("entry_price", 0)
            qty = trade.get("quantity", 0)
            pnl = (exit_price - entry_px) * qty if entry_px else 0.0
            try:
                await self._strategy_tracker.record_trade(
                    user_id,
                    strategy,
                    {
                        "trade_id": trade_id,
                        "signal_id": trade.get("signal_id", ""),
                        "symbol": trade.get("symbol", ""),
                        "action": f"CLOSE_{action}",
                        "entry_price": entry_px,
                        "exit_price": exit_price,
                        "quantity": qty,
                        "pnl": pnl,
                        "pnl_pct": (pnl / (entry_px * qty) * 100)
                        if entry_px and qty
                        else 0.0,
                        "hold_minutes": 0,
                        "entry_time": trade.get("opened_at"),
                        "exit_time": datetime.now(timezone.utc),
                        "broker_id": broker_id,
                    },
                )
            except Exception:
                log.debug(
                    "execution_bridge.strategy_close_track_failed",
                    exc_info=True,
                )

        log.info(
            "execution_bridge.trade_closed",
            trade_id=trade_id,
            reason=reason,
            exit_price=exit_price,
        )

        return record

    # ── Queries ───────────────────────────────────────────────────────

    def get_open_trades(self) -> list[Dict[str, Any]]:
        """Return all actively tracked open trades."""
        return list(self._open_trades.values())

    def get_execution_history(self, limit: int = 100) -> list[Dict[str, Any]]:
        """Return recent execution records."""
        return [r.to_dict() for r in self._execution_history[-limit:]]

    # ── Helpers ───────────────────────────────────────────────────────

    def _fail(
        self,
        signal_id: str,
        user_id: str,
        symbol: str,
        action: str,
        reason: str,
        broker_id: str = "",
    ) -> ExecutionRecord:
        """Create a rejection/failure record."""
        record = ExecutionRecord(
            record_id=f"EXE-{uuid.uuid4().hex[:12].upper()}",
            signal_id=signal_id,
            user_id=user_id,
            broker_id=broker_id,
            symbol=symbol,
            action=action,
            status="rejected",
            reason=reason,
        )
        self._execution_history.append(record)
        log.info(
            "execution_bridge.rejected",
            signal_id=signal_id,
            reason=reason,
        )
        return record

    async def _get_capital(self, broker_id: str) -> float:
        """Fetch available capital from the broker."""
        try:
            broker = await self._broker_mgr.get_broker(broker_id)
            portfolio = await broker.get_portfolio()
            return portfolio.cash
        except Exception:
            return 100_000.0

    async def _update_signal_outcome(
        self, signal_id: str, outcome: str, trade_id: str = "",
    ) -> None:
        """Update the signal document in MongoDB with execution outcome."""
        if self._db is None:
            return
        try:
            update: Dict[str, Any] = {
                "outcome": outcome,
                "executed_at": datetime.now(timezone.utc),
            }
            if trade_id:
                update["trade_id"] = trade_id
            await self._db.signals.update_one(
                {"signal_id": signal_id},
                {"$set": update},
            )
        except Exception:
            log.debug("execution_bridge.outcome_update_failed", exc_info=True)

    @staticmethod
    def _doc_to_signal(doc: Dict[str, Any]) -> TradeSignal:
        """Convert a MongoDB signal document to a TradeSignal."""
        return TradeSignal(
            signal_id=doc.get("signal_id", ""),
            timestamp=doc.get("timestamp", datetime.now(timezone.utc)),
            underlying=doc.get("underlying", ""),
            action=SignalAction(doc.get("action", "NO_TRADE")),
            direction=SignalDirection(doc.get("direction", "NEUTRAL")),
            confidence=doc.get("confidence", 0.0),
            strategy_name=doc.get("strategy_name", "signal_fusion"),
            entry_price=doc.get("entry_price", 0.0),
            stop_loss=doc.get("stop_loss", 0.0),
            target_price=doc.get("target_price", 0.0),
            risk_reward_ratio=doc.get("risk_reward_ratio", 0.0),
            reasoning=doc.get("reasoning", ""),
        )
