"""Live portfolio management and analytics.

Aggregates positions across all connected brokers, tracks real-time P&L,
portfolio Greeks, equity curve, and trade statistics.  Subscribes to
:class:`EventBus` for fills and price updates to maintain an up-to-date
view of the trading account.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Deque, List, Optional

import structlog

from hedgefund.execution.broker_manager import BrokerManager
from hedgefund.streaming.event_bus import Event, EventBus, EventType
from hedgefund.types import Greeks, PortfolioSnapshot, Position

log = structlog.get_logger(__name__)

# 30 days of equity data at 1-second resolution would be ~2.6M points.
# We sample once per minute -> ~43,200 points for 30 days.
_EQUITY_BUFFER_SIZE = 43_200


@dataclass(slots=True)
class TradeStatistics:
    """Rolling trade performance statistics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    sharpe_estimate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 4),
            "total_pnl": round(self.total_pnl, 2),
            "sharpe_estimate": round(self.sharpe_estimate, 4),
        }


@dataclass(slots=True)
class EquityPoint:
    """Single point on the equity curve."""

    timestamp: datetime
    equity: float
    cash: float
    daily_pnl: float


class PortfolioManager:
    """Live portfolio management and analytics engine.

    Parameters
    ----------
    broker_manager:
        Multi-broker manager for fetching positions and balances.
    event_bus:
        Central event bus for subscribing to fill/price events.
    refresh_interval_seconds:
        Minimum seconds between full broker refreshes.
    """

    def __init__(
        self,
        broker_manager: BrokerManager,
        event_bus: Optional[EventBus] = None,
        *,
        refresh_interval_seconds: float = 30.0,
    ) -> None:
        self._broker_mgr = broker_manager
        self._bus = event_bus
        self._refresh_interval = refresh_interval_seconds

        # Cached snapshot
        self._snapshot: Optional[PortfolioSnapshot] = None
        self._last_refresh: float = 0.0
        self._lock = asyncio.Lock()

        # Equity curve ring buffer
        self._equity_curve: Deque[EquityPoint] = deque(maxlen=_EQUITY_BUFFER_SIZE)
        self._last_equity_sample: float = 0.0
        self._equity_sample_interval = 60.0  # sample every 60 seconds

        # High water mark tracking
        self._high_water_mark: float = 0.0
        self._max_drawdown: float = 0.0

        # Daily P&L tracking
        self._day_start_equity: float = 0.0
        self._current_day: Optional[str] = None

        # Trade statistics
        self._trade_pnls: list[float] = []
        self._stats = TradeStatistics()

        # Realized / unrealized P&L
        self._total_realized_pnl: float = 0.0

        if event_bus is not None:
            self.subscribe_to_updates(event_bus)

    # ------------------------------------------------------------------
    # EventBus integration
    # ------------------------------------------------------------------

    def subscribe_to_updates(self, event_bus: EventBus) -> None:
        """Subscribe to fill and portfolio update events."""
        event_bus.subscribe(EventType.FILL, self._on_fill)
        event_bus.subscribe(EventType.PORTFOLIO_UPDATE, self._on_portfolio_update)
        event_bus.subscribe(EventType.TICK, self._on_tick)
        log.info("portfolio_manager.subscribed")

    async def _on_fill(self, event: Event) -> None:
        """Handle fill events -- invalidate cache to trigger refresh."""
        self._last_refresh = 0.0  # Force next get_snapshot() to refresh
        log.debug(
            "portfolio_manager.fill_received",
            symbol=event.symbol,
            trade_id=event.data.get("trade_id"),
        )

    async def _on_portfolio_update(self, event: Event) -> None:
        """Handle trade close events -- update statistics."""
        pnl = event.data.get("pnl", 0.0)
        if event.data.get("action") == "close":
            self._trade_pnls.append(pnl)
            self._total_realized_pnl += pnl
            self._recompute_stats()
            self._last_refresh = 0.0
            log.debug(
                "portfolio_manager.trade_closed",
                symbol=event.symbol,
                pnl=pnl,
            )

    async def _on_tick(self, event: Event) -> None:
        """Handle tick events -- sample equity curve periodically."""
        now = time.monotonic()
        if (now - self._last_equity_sample) >= self._equity_sample_interval:
            if self._snapshot is not None:
                self._sample_equity(self._snapshot)
            self._last_equity_sample = now

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def get_snapshot(self) -> PortfolioSnapshot:
        """Return current portfolio snapshot, refreshing if stale."""
        now = time.monotonic()
        if self._snapshot is not None and (now - self._last_refresh) < self._refresh_interval:
            return self._snapshot

        return await self.refresh()

    async def refresh(self) -> PortfolioSnapshot:
        """Pull latest portfolio data from all brokers."""
        async with self._lock:
            try:
                snapshot = await self._broker_mgr.get_aggregate_portfolio()
            except Exception as exc:
                log.error("portfolio_manager.refresh_failed", error=str(exc))
                if self._snapshot is not None:
                    return self._snapshot
                # Return empty snapshot
                snapshot = PortfolioSnapshot(
                    timestamp=datetime.utcnow(),
                    cash=0.0,
                    net_liquidation=0.0,
                    positions=[],
                )

            # Update high water mark and drawdown
            nlv = snapshot.net_liquidation
            if nlv > self._high_water_mark:
                self._high_water_mark = nlv

            if self._high_water_mark > 0:
                dd = (self._high_water_mark - nlv) / self._high_water_mark
                self._max_drawdown = max(self._max_drawdown, dd)
                snapshot = PortfolioSnapshot(
                    timestamp=snapshot.timestamp,
                    cash=snapshot.cash,
                    net_liquidation=snapshot.net_liquidation,
                    positions=snapshot.positions,
                    total_delta=snapshot.total_delta,
                    total_gamma=snapshot.total_gamma,
                    total_theta=snapshot.total_theta,
                    total_vega=snapshot.total_vega,
                    daily_pnl=self._compute_daily_pnl(nlv),
                    total_pnl=snapshot.total_pnl,
                    drawdown_pct=dd,
                    high_water_mark=self._high_water_mark,
                )

            self._snapshot = snapshot
            self._last_refresh = time.monotonic()

            # Sample equity
            self._sample_equity(snapshot)

            return snapshot

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_trade_statistics(self) -> TradeStatistics:
        """Return current trade performance statistics."""
        return self._stats

    def get_equity_curve(self, limit: int = 0) -> list[Dict[str, Any]]:
        """Return equity curve points as dicts.

        Parameters
        ----------
        limit:
            Maximum points to return (0 = all).
        """
        points = list(self._equity_curve)
        if limit > 0:
            points = points[-limit:]
        return [
            {
                "timestamp": p.timestamp.isoformat(),
                "equity": round(p.equity, 2),
                "cash": round(p.cash, 2),
                "daily_pnl": round(p.daily_pnl, 2),
            }
            for p in points
        ]

    def get_portfolio_greeks(self) -> Dict[str, float]:
        """Return aggregate portfolio Greeks."""
        if self._snapshot is None:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
        return {
            "delta": round(self._snapshot.total_delta, 4),
            "gamma": round(self._snapshot.total_gamma, 4),
            "theta": round(self._snapshot.total_theta, 4),
            "vega": round(self._snapshot.total_vega, 4),
        }

    @property
    def high_water_mark(self) -> float:
        return self._high_water_mark

    @property
    def max_drawdown(self) -> float:
        return self._max_drawdown

    @property
    def realized_pnl(self) -> float:
        return self._total_realized_pnl

    @property
    def unrealized_pnl(self) -> float:
        if self._snapshot is None:
            return 0.0
        return sum(p.unrealized_pnl for p in self._snapshot.positions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_equity(self, snapshot: PortfolioSnapshot) -> None:
        """Record a point on the equity curve."""
        point = EquityPoint(
            timestamp=datetime.utcnow(),
            equity=snapshot.net_liquidation,
            cash=snapshot.cash,
            daily_pnl=snapshot.daily_pnl,
        )
        self._equity_curve.append(point)

    def _compute_daily_pnl(self, current_nlv: float) -> float:
        """Compute P&L since start of current trading day."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if self._current_day != today:
            self._current_day = today
            self._day_start_equity = current_nlv
            return 0.0
        return current_nlv - self._day_start_equity

    def _recompute_stats(self) -> None:
        """Recompute trade statistics from P&L history."""
        pnls = self._trade_pnls
        if not pnls:
            self._stats = TradeStatistics()
            return

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total = len(pnls)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total if total > 0 else 0.0
        avg_win = sum(wins) / win_count if win_count > 0 else 0.0
        avg_loss = abs(sum(losses) / loss_count) if loss_count > 0 else 0.0

        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf") if total_wins > 0 else 0.0

        # Simplified Sharpe estimate from trade returns
        sharpe = 0.0
        if len(pnls) >= 2:
            mean_pnl = sum(pnls) / len(pnls)
            variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
            std_pnl = math.sqrt(variance) if variance > 0 else 1e-10
            sharpe = (mean_pnl / std_pnl) * math.sqrt(252)  # annualized

        self._stats = TradeStatistics(
            total_trades=total,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_pnl=sum(pnls),
            sharpe_estimate=sharpe,
        )
