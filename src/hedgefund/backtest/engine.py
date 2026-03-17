"""Event-driven backtesting engine that reuses production components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np
import structlog

from hedgefund.types import (
    BacktestMetrics,
    OHLCV,
    Order,
    OrderStatus,
    PortfolioSnapshot,
    Position,
    Side,
    TradeRecord,
    TradeSignal,
)

log = structlog.get_logger(__name__)


# ── Protocols for pluggable production components ─────────────────────────────


class SignalGeneratorProtocol(Protocol):
    """Minimal interface expected from the production signal generator."""

    def generate(self, bar: OHLCV, features: dict[str, float]) -> list[TradeSignal]: ...


class RiskManagerProtocol(Protocol):
    """Minimal interface expected from the production risk manager."""

    def check(self, signal: TradeSignal, portfolio: PortfolioSnapshot) -> bool: ...


class FeaturePipelineProtocol(Protocol):
    """Minimal interface expected from the production feature pipeline."""

    def compute(self, bars: list[OHLCV]) -> dict[str, float]: ...


# ── Paper Broker ──────────────────────────────────────────────────────────────


class PaperBroker:
    """Simulates order execution with configurable fill assumptions.

    Fills at the mid-point between the bar's open and close (approximation),
    applies a fixed commission, and manages a simple flat position book.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_per_contract: float = 0.65,
        slippage_pct: float = 0.001,
    ) -> None:
        self.cash: float = initial_capital
        self.initial_capital = initial_capital
        self.commission = commission_per_contract
        self.slippage_pct = slippage_pct
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.fills: list[Order] = []
        self._log = log.bind(component="paper_broker")

    def submit_order(self, order: Order, bar: OHLCV) -> Order:
        """Simulate immediate fill against the current bar."""
        fill_price = (bar.open + bar.close) / 2.0
        # Apply slippage.
        if order.side == Side.BUY:
            fill_price *= 1.0 + self.slippage_pct
        else:
            fill_price *= 1.0 - self.slippage_pct

        cost = fill_price * order.quantity * order.contract.multiplier
        total_commission = self.commission * order.quantity

        if order.side == Side.BUY:
            if self.cash < cost + total_commission:
                order.status = OrderStatus.REJECTED
                self._log.warning("order_rejected_insufficient_funds", order_id=order.order_id)
                return order
            self.cash -= cost + total_commission
        else:
            self.cash += cost - total_commission

        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.filled_at = bar.timestamp
        order.commission = total_commission
        self.fills.append(order)

        self._update_position(order)
        return order

    def _update_position(self, order: Order) -> None:
        key = order.contract.osi_symbol
        if key in self.positions:
            pos = self.positions[key]
            if order.side == Side.BUY:
                total_qty = pos.quantity + order.filled_quantity
                if total_qty > 0:
                    pos.avg_entry = (
                        (pos.avg_entry * pos.quantity + order.filled_price * order.filled_quantity)
                        / total_qty
                    )
                pos.quantity = total_qty
            else:
                pos.quantity -= order.filled_quantity
                if pos.quantity <= 0:
                    pos.realized_pnl += (
                        (order.filled_price - pos.avg_entry)
                        * order.filled_quantity
                        * order.contract.multiplier
                    )
                    del self.positions[key]
                    return
        else:
            if order.side == Side.BUY:
                self.positions[key] = Position(
                    contract=order.contract,
                    quantity=order.filled_quantity,
                    avg_entry=order.filled_price,
                    current_price=order.filled_price,
                    greeks=order.contract.__class__.__mro__[0]  # placeholder
                    if False
                    else _zero_greeks(),
                    unrealized_pnl=0.0,
                )

    def portfolio_snapshot(self, timestamp: datetime) -> PortfolioSnapshot:
        positions = list(self.positions.values())
        market_value = sum(p.market_value for p in positions)
        nlv = self.cash + market_value
        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            net_liquidation=nlv,
            positions=positions,
            daily_pnl=0.0,
            total_pnl=nlv - self.initial_capital,
        )


def _zero_greeks():
    from hedgefund.types import Greeks

    return Greeks(delta=0, gamma=0, theta=0, vega=0)


# ── Backtest Engine ───────────────────────────────────────────────────────────


@dataclass(slots=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission: float = 0.65
    slippage_pct: float = 0.001


class BacktestEngine:
    """Event-driven backtesting loop.

    Processes historical bars one at a time through the same
    :class:`SignalGenerator`, :class:`RiskManager`, and
    :class:`FeaturePipeline` used in production, then executes via
    :class:`PaperBroker`.
    """

    def __init__(
        self,
        signal_generator: SignalGeneratorProtocol,
        risk_manager: RiskManagerProtocol,
        feature_pipeline: FeaturePipelineProtocol,
        config: BacktestConfig | None = None,
    ) -> None:
        self.cfg = config or BacktestConfig()
        self._signal_gen = signal_generator
        self._risk_mgr = risk_manager
        self._features = feature_pipeline
        self._broker = PaperBroker(
            initial_capital=self.cfg.initial_capital,
            commission_per_contract=self.cfg.commission,
            slippage_pct=self.cfg.slippage_pct,
        )
        self._equity_curve: list[float] = []
        self._trade_records: list[TradeRecord] = []
        self._signals: list[TradeSignal] = []
        self._bars_processed: int = 0
        self._log = log.bind(component="backtest_engine")

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self, bars: list[OHLCV]) -> BacktestMetrics:
        """Execute the backtest over a sequence of historical bars.

        Returns comprehensive performance metrics.
        """
        self._log.info("backtest_started", n_bars=len(bars))
        history: list[OHLCV] = []

        for bar in bars:
            history.append(bar)
            features = self._features.compute(history)
            signals = self._signal_gen.generate(bar, features)

            portfolio = self._broker.portfolio_snapshot(bar.timestamp)

            for signal in signals:
                if not self._risk_mgr.check(signal, portfolio):
                    self._log.debug("signal_rejected_by_risk", signal_id=signal.signal_id)
                    continue

                self._signals.append(signal)

                # Convert signal to orders and execute.
                for contract in signal.contracts:
                    side = Side.BUY if signal.direction.value == "LONG" else Side.SELL
                    order = Order(
                        order_id=Order.generate_id(),
                        signal_id=signal.signal_id,
                        contract=contract,
                        side=side,
                        order_type=signal.metadata.get("order_type", "MARKET"),
                        quantity=signal.metadata.get("quantity", 1),
                    )
                    self._broker.submit_order(order, bar)

            nlv = self._broker.portfolio_snapshot(bar.timestamp).net_liquidation
            self._equity_curve.append(nlv)
            self._bars_processed += 1

        metrics = self._compute_metrics()
        self._log.info("backtest_complete", **metrics.__dict__)
        return metrics

    # ── Results ───────────────────────────────────────────────────────

    @property
    def equity_curve(self) -> np.ndarray:
        return np.array(self._equity_curve)

    @property
    def trade_records(self) -> list[TradeRecord]:
        return list(self._trade_records)

    @property
    def signals(self) -> list[TradeSignal]:
        return list(self._signals)

    # ── Metrics computation ───────────────────────────────────────────

    def _compute_metrics(self) -> BacktestMetrics:
        from hedgefund.backtest.metrics import MetricsCalculator

        eq = np.array(self._equity_curve)
        if len(eq) < 2:
            return BacktestMetrics(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_return=0, annualized_return=0,
                sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
                profit_factor=0, avg_win=0, avg_loss=0, expectancy=0,
            )

        returns = np.diff(eq) / eq[:-1]
        fills = self._broker.fills

        wins = [f for f in fills if f.filled_price and f.side == Side.SELL and f.filled_price > 0]
        calc = MetricsCalculator(returns, eq)
        return calc.compute_all(
            total_trades=len(fills),
            winning_trades=len(wins),
        )
