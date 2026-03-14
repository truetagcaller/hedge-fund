"""Automated trade execution engine.

Takes :class:`TradeDecision` objects from the :class:`TradingDecisionEngine`,
validates them through the risk manager, builds orders, routes them to the
appropriate broker via :class:`BrokerManager`, and monitors open positions
for stop-loss and take-profit exits.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from hedgefund.engine.decision_engine import TradeAction, TradeDecision
from hedgefund.execution.broker_manager import BrokerManager
from hedgefund.risk.base import RiskManager, RiskVerdict
from hedgefund.risk.position_sizer import PositionSizer
from hedgefund.streaming.event_bus import Event, EventBus, EventType
from hedgefund.types import (
    Greeks,
    OptionContract,
    OptionType,
    Order,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    Side,
)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ExecutionResult:
    """Outcome of attempting to execute a trade decision."""

    decision_id: str
    success: bool
    order: Optional[Order] = None
    trade_id: Optional[str] = None
    reason: str = ""
    filled_price: Optional[float] = None
    quantity: int = 0
    commission: float = 0.0


@dataclass(slots=True)
class OpenTrade:
    """An actively monitored open trade."""

    trade_id: str
    decision_id: str
    symbol: str
    action: TradeAction
    broker_id: str
    order: Order
    entry_price: float
    current_price: float
    stop_loss: float
    target_price: float
    trailing_stop: float
    quantity: int
    opened_at: datetime
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    atr: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @staticmethod
    def generate_id() -> str:
        return f"TRD-{uuid.uuid4().hex[:12].upper()}"


@dataclass(slots=True)
class CompletedTrade:
    """Record of a completed (closed) trade."""

    trade_id: str
    decision_id: str
    symbol: str
    action: TradeAction
    broker_id: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    commission: float
    opened_at: datetime
    closed_at: datetime
    close_reason: str
    hold_duration_seconds: float


class TradeExecutor:
    """Automated trade execution and position monitoring engine.

    Parameters
    ----------
    event_bus:
        Central event bus for publishing fill and portfolio events.
    broker_manager:
        Multi-broker manager for routing orders.
    risk_manager:
        Risk validation pipeline (optional -- if None, all decisions pass).
    position_sizer:
        Position sizing engine (optional -- defaults to fixed fraction).
    atr_trailing_multiplier:
        ATR multiplier for trailing stop distance.
    monitor_interval_seconds:
        How often to check stops and targets.
    """

    def __init__(
        self,
        event_bus: EventBus,
        broker_manager: BrokerManager,
        *,
        risk_manager: Optional[RiskManager] = None,
        position_sizer: Optional[PositionSizer] = None,
        atr_trailing_multiplier: float = 2.0,
        monitor_interval_seconds: float = 5.0,
    ) -> None:
        self._bus = event_bus
        self._broker_mgr = broker_manager
        self._risk_mgr = risk_manager
        self._sizer = position_sizer or PositionSizer()
        self._atr_trail_mult = atr_trailing_multiplier
        self._monitor_interval = monitor_interval_seconds

        # State -- protected by lock for thread safety
        self._lock = asyncio.Lock()
        self._open_trades: Dict[str, OpenTrade] = {}
        self._trade_history: list[CompletedTrade] = []
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self._running = False

        # Subscribe to tick events for position monitoring
        self._bus.subscribe(EventType.TICK, self._on_tick)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the position monitoring loop."""
        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="trade-executor-monitor"
        )
        log.info("trade_executor.started")

    async def shutdown(self) -> None:
        """Stop monitoring and close resources."""
        self._running = False
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        log.info("trade_executor.shutdown", open_trades=len(self._open_trades))

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def execute_decision(
        self,
        decision: TradeDecision,
        broker_id: str,
    ) -> ExecutionResult:
        """Validate and execute a trading decision.

        Steps:
        1. Risk validation (if risk manager is configured)
        2. Position sizing
        3. Order construction
        4. Broker submission
        5. Position tracking setup
        """
        # 1. Risk validation
        if self._risk_mgr is not None:
            try:
                portfolio = await self._broker_mgr.get_aggregate_portfolio()
                verdict = await self._risk_mgr.validate_signal(
                    self._decision_to_signal(decision), portfolio
                )
                if not verdict.approved:
                    log.info(
                        "trade_executor.risk_rejected",
                        decision_id=decision.decision_id,
                        reason=verdict.reason,
                    )
                    return ExecutionResult(
                        decision_id=decision.decision_id,
                        success=False,
                        reason=f"Risk rejected: {verdict.reason}",
                    )
            except Exception as exc:
                log.error(
                    "trade_executor.risk_check_failed",
                    decision_id=decision.decision_id,
                    error=str(exc),
                )
                return ExecutionResult(
                    decision_id=decision.decision_id,
                    success=False,
                    reason=f"Risk check error: {exc}",
                )

        # 2. Position sizing
        atr = decision.metadata.get("atr", decision.entry_price * 0.02)
        risk_per_contract = abs(decision.entry_price - decision.stop_loss)
        sizing = self._sizer.fixed_fraction(
            capital=await self._get_capital(broker_id),
            risk_per_contract=risk_per_contract,
        )
        quantity = sizing.quantity
        if quantity <= 0:
            return ExecutionResult(
                decision_id=decision.decision_id,
                success=False,
                reason="Position sizing returned 0 contracts",
            )

        # 3. Build order
        side = Side.BUY if decision.action in (TradeAction.BUY_CALL, TradeAction.BUY_PUT) else Side.SELL
        option_type = (
            OptionType.CALL
            if decision.action in (TradeAction.BUY_CALL, TradeAction.SELL_CALL)
            else OptionType.PUT
        )

        contract = OptionContract(
            symbol=f"{decision.symbol}_OPT",
            underlying=decision.symbol,
            option_type=option_type,
            strike=round(decision.entry_price, 2),
            expiration=datetime.utcnow().date(),
        )

        order = Order(
            order_id=Order.generate_id(),
            signal_id=decision.decision_id,
            contract=contract,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            limit_price=decision.entry_price,
        )

        # 4. Submit to broker
        try:
            broker = await self._broker_mgr.get_broker(broker_id)
            filled_order = await broker.submit_order(order)
        except Exception as exc:
            log.error(
                "trade_executor.submit_failed",
                decision_id=decision.decision_id,
                broker_id=broker_id,
                error=str(exc),
            )
            return ExecutionResult(
                decision_id=decision.decision_id,
                success=False,
                reason=f"Order submission failed: {exc}",
            )

        if filled_order.status not in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            return ExecutionResult(
                decision_id=decision.decision_id,
                success=False,
                order=filled_order,
                reason=f"Order not filled: status={filled_order.status.value}",
            )

        fill_price = filled_order.filled_price or decision.entry_price

        # 5. Track the open position
        trade = OpenTrade(
            trade_id=OpenTrade.generate_id(),
            decision_id=decision.decision_id,
            symbol=decision.symbol,
            action=decision.action,
            broker_id=broker_id,
            order=filled_order,
            entry_price=fill_price,
            current_price=fill_price,
            stop_loss=decision.stop_loss,
            target_price=decision.target_price,
            trailing_stop=decision.stop_loss,
            quantity=filled_order.filled_quantity or quantity,
            opened_at=datetime.utcnow(),
            highest_price=fill_price,
            lowest_price=fill_price,
            atr=atr,
        )

        async with self._lock:
            self._open_trades[trade.trade_id] = trade

        # 6. Publish fill event
        await self._bus.publish(Event(
            event_type=EventType.FILL,
            timestamp=datetime.utcnow(),
            symbol=decision.symbol,
            data={
                "trade_id": trade.trade_id,
                "decision_id": decision.decision_id,
                "action": decision.action.value,
                "fill_price": fill_price,
                "quantity": trade.quantity,
                "broker_id": broker_id,
            },
            source="trade_executor",
        ))

        log.info(
            "trade_executor.executed",
            trade_id=trade.trade_id,
            symbol=decision.symbol,
            action=decision.action.value,
            fill_price=fill_price,
            quantity=trade.quantity,
            broker_id=broker_id,
        )

        return ExecutionResult(
            decision_id=decision.decision_id,
            success=True,
            order=filled_order,
            trade_id=trade.trade_id,
            filled_price=fill_price,
            quantity=trade.quantity,
            commission=filled_order.commission,
        )

    # ------------------------------------------------------------------
    # Position monitoring
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        """Continuously monitor open trades for stop/target hits."""
        while self._running:
            try:
                await self.monitor_positions()
                await asyncio.sleep(self._monitor_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("trade_executor.monitor_error")
                await asyncio.sleep(self._monitor_interval)

    async def monitor_positions(self) -> None:
        """Check all open trades against their stops and targets."""
        async with self._lock:
            trades = list(self._open_trades.values())

        for trade in trades:
            is_long = trade.action in (TradeAction.BUY_CALL, TradeAction.BUY_PUT)

            # Update trailing stop for long trades
            if is_long and trade.current_price > trade.highest_price:
                trade.highest_price = trade.current_price
                atr = trade.atr if trade.atr > 0 else trade.entry_price * 0.02
                new_trailing = trade.current_price - atr * self._atr_trail_mult
                if new_trailing > trade.trailing_stop:
                    trade.trailing_stop = new_trailing

            # Update trailing stop for short trades
            if not is_long and trade.current_price < trade.lowest_price:
                trade.lowest_price = trade.current_price
                atr = trade.atr if trade.atr > 0 else trade.entry_price * 0.02
                new_trailing = trade.current_price + atr * self._atr_trail_mult
                if new_trailing < trade.trailing_stop:
                    trade.trailing_stop = new_trailing

            # Check stop-loss (using trailing stop)
            effective_stop = max(trade.stop_loss, trade.trailing_stop) if is_long else min(trade.stop_loss, trade.trailing_stop)
            if is_long and trade.current_price <= effective_stop:
                await self.close_position(trade.trade_id, reason="stop_loss_hit")
            elif not is_long and trade.current_price >= effective_stop:
                await self.close_position(trade.trade_id, reason="stop_loss_hit")

            # Check target
            if is_long and trade.current_price >= trade.target_price:
                await self.close_position(trade.trade_id, reason="target_hit")
            elif not is_long and trade.current_price <= trade.target_price:
                await self.close_position(trade.trade_id, reason="target_hit")

    async def _on_tick(self, event: Event) -> None:
        """Update current prices for open trades matching the symbol."""
        price = event.data.get("price", event.data.get("close", 0.0))
        if price <= 0:
            return

        async with self._lock:
            for trade in self._open_trades.values():
                if trade.symbol == event.symbol:
                    trade.current_price = price
                    multiplier = 100
                    if trade.action in (TradeAction.BUY_CALL, TradeAction.BUY_PUT):
                        trade.unrealized_pnl = (price - trade.entry_price) * trade.quantity * multiplier
                    else:
                        trade.unrealized_pnl = (trade.entry_price - price) * trade.quantity * multiplier

    # ------------------------------------------------------------------
    # Position closing
    # ------------------------------------------------------------------

    async def close_position(self, trade_id: str, reason: str = "manual") -> None:
        """Close a specific open position."""
        async with self._lock:
            trade = self._open_trades.pop(trade_id, None)

        if trade is None:
            log.warning("trade_executor.close_not_found", trade_id=trade_id)
            return

        # Submit closing order
        try:
            close_side = Side.SELL if trade.action in (TradeAction.BUY_CALL, TradeAction.BUY_PUT) else Side.BUY
            close_order = Order(
                order_id=Order.generate_id(),
                signal_id=trade.decision_id,
                contract=trade.order.contract,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=trade.quantity,
                limit_price=trade.current_price,
            )
            broker = await self._broker_mgr.get_broker(trade.broker_id)
            filled = await broker.submit_order(close_order)
            exit_price = filled.filled_price or trade.current_price
        except Exception as exc:
            log.error(
                "trade_executor.close_order_failed",
                trade_id=trade_id,
                error=str(exc),
            )
            exit_price = trade.current_price

        # Compute P&L
        multiplier = 100
        if trade.action in (TradeAction.BUY_CALL, TradeAction.BUY_PUT):
            pnl = (exit_price - trade.entry_price) * trade.quantity * multiplier
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity * multiplier
        pnl_pct = (exit_price - trade.entry_price) / trade.entry_price if trade.entry_price > 0 else 0.0

        now = datetime.utcnow()
        completed = CompletedTrade(
            trade_id=trade.trade_id,
            decision_id=trade.decision_id,
            symbol=trade.symbol,
            action=trade.action,
            broker_id=trade.broker_id,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            quantity=trade.quantity,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 6),
            commission=trade.order.commission,
            opened_at=trade.opened_at,
            closed_at=now,
            close_reason=reason,
            hold_duration_seconds=(now - trade.opened_at).total_seconds(),
        )

        self._trade_history.append(completed)

        # Publish portfolio update
        await self._bus.publish(Event(
            event_type=EventType.PORTFOLIO_UPDATE,
            timestamp=now,
            symbol=trade.symbol,
            data={
                "trade_id": trade.trade_id,
                "action": "close",
                "reason": reason,
                "pnl": completed.pnl,
                "pnl_pct": completed.pnl_pct,
                "exit_price": exit_price,
            },
            source="trade_executor",
        ))

        log.info(
            "trade_executor.position_closed",
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            reason=reason,
            pnl=completed.pnl,
            pnl_pct=completed.pnl_pct,
        )

    async def close_all(self, reason: str = "emergency") -> None:
        """Emergency close all open positions."""
        async with self._lock:
            trade_ids = list(self._open_trades.keys())

        log.warning(
            "trade_executor.closing_all",
            count=len(trade_ids),
            reason=reason,
        )

        for trade_id in trade_ids:
            try:
                await self.close_position(trade_id, reason=reason)
            except Exception:
                log.exception("trade_executor.emergency_close_failed", trade_id=trade_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_open_trades(self) -> list[Dict[str, Any]]:
        """Return all active trades with current P&L."""
        result = []
        for t in self._open_trades.values():
            result.append({
                "trade_id": t.trade_id,
                "decision_id": t.decision_id,
                "symbol": t.symbol,
                "action": t.action.value,
                "broker_id": t.broker_id,
                "entry_price": t.entry_price,
                "current_price": t.current_price,
                "stop_loss": t.stop_loss,
                "target_price": t.target_price,
                "trailing_stop": t.trailing_stop,
                "quantity": t.quantity,
                "unrealized_pnl": round(t.unrealized_pnl, 2),
                "pnl_pct": round(t.pnl_pct, 6),
                "opened_at": t.opened_at.isoformat(),
            })
        return result

    def get_trade_history(self, limit: int = 100) -> list[Dict[str, Any]]:
        """Return completed trades."""
        trades = self._trade_history[-limit:]
        result = []
        for t in trades:
            result.append({
                "trade_id": t.trade_id,
                "decision_id": t.decision_id,
                "symbol": t.symbol,
                "action": t.action.value,
                "broker_id": t.broker_id,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "close_reason": t.close_reason,
                "opened_at": t.opened_at.isoformat(),
                "closed_at": t.closed_at.isoformat(),
                "hold_duration_seconds": t.hold_duration_seconds,
            })
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_capital(self, broker_id: str) -> float:
        """Fetch available capital from the broker."""
        try:
            broker = await self._broker_mgr.get_broker(broker_id)
            portfolio = await broker.get_portfolio()
            return portfolio.cash
        except Exception:
            return 100_000.0  # fallback

    @staticmethod
    def _decision_to_signal(decision: TradeDecision) -> Any:
        """Convert a TradeDecision into a TradeSignal for risk validation."""
        from hedgefund.types import SignalAction, SignalDirection, TradeSignal

        action_map = {
            TradeAction.BUY_CALL: SignalAction.BUY_CALL,
            TradeAction.BUY_PUT: SignalAction.BUY_PUT,
            TradeAction.SELL_CALL: SignalAction.SELL_CALL,
            TradeAction.SELL_PUT: SignalAction.SELL_PUT,
            TradeAction.NO_TRADE: SignalAction.NO_TRADE,
        }
        direction = (
            SignalDirection.LONG
            if decision.action in (TradeAction.BUY_CALL, TradeAction.SELL_PUT)
            else SignalDirection.SHORT
        )

        return TradeSignal(
            signal_id=decision.decision_id,
            timestamp=decision.timestamp,
            underlying=decision.symbol,
            action=action_map.get(decision.action, SignalAction.NO_TRADE),
            direction=direction,
            confidence=decision.confidence,
            strategy_name="decision_engine",
            entry_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            target_price=decision.target_price,
            risk_reward_ratio=decision.risk_reward_ratio,
            reasoning=decision.reasoning,
        )
