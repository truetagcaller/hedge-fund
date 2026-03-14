"""Drawdown monitoring and circuit-breaker logic.

Tracks the equity curve, computes current and maximum drawdown, and
implements a circuit breaker that halts new trading when drawdown exceeds
a configurable threshold (default 10 %).  Includes automatic recovery
detection with reduced position sizing.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class TradingState(enum.Enum):
    """Overall trading permission state."""

    ACTIVE = "ACTIVE"
    HALTED = "HALTED"
    RECOVERY = "RECOVERY"


@dataclass(slots=True)
class DrawdownSnapshot:
    """Point-in-time drawdown record."""

    timestamp: datetime
    equity: float
    high_water_mark: float
    drawdown_pct: float
    max_drawdown_pct: float
    state: TradingState


class DrawdownMonitor:
    """Monitors equity curve and enforces drawdown-based circuit breakers.

    Parameters
    ----------
    max_drawdown_pct:
        Threshold at which trading is halted (default 0.10 = 10 %).
    recovery_threshold_pct:
        Drawdown must fall below this level to leave HALTED and enter
        RECOVERY mode (default 0.05 = 5 %).
    recovery_size_factor:
        Position-size multiplier during RECOVERY (default 0.5 = half size).
    min_halt_duration:
        Minimum time to stay halted regardless of recovery (default 1 hour).
    """

    def __init__(
        self,
        *,
        max_drawdown_pct: float = 0.10,
        recovery_threshold_pct: float = 0.05,
        recovery_size_factor: float = 0.5,
        min_halt_duration: timedelta = timedelta(hours=1),
        initial_equity: float = 0.0,
    ) -> None:
        self._max_drawdown_pct = max_drawdown_pct
        self._recovery_threshold_pct = recovery_threshold_pct
        self._recovery_size_factor = recovery_size_factor
        self._min_halt_duration = min_halt_duration

        # State
        self._high_water_mark: float = initial_equity
        self._max_drawdown_observed: float = 0.0
        self._state: TradingState = TradingState.ACTIVE
        self._halted_at: Optional[datetime] = None
        self._equity_history: list[tuple[datetime, float]] = []

        if initial_equity > 0:
            self._equity_history.append((datetime.utcnow(), initial_equity))

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def state(self) -> TradingState:
        return self._state

    @property
    def high_water_mark(self) -> float:
        return self._high_water_mark

    @property
    def max_drawdown_observed(self) -> float:
        return self._max_drawdown_observed

    @property
    def size_factor(self) -> float:
        """Multiplier to apply to position sizes given the current state."""
        if self._state == TradingState.HALTED:
            return 0.0
        if self._state == TradingState.RECOVERY:
            return self._recovery_size_factor
        return 1.0

    @property
    def is_trading_allowed(self) -> bool:
        return self._state != TradingState.HALTED

    # ── Core API ──────────────────────────────────────────────────────────

    def update(self, equity: float, timestamp: datetime | None = None) -> DrawdownSnapshot:
        """Record a new equity observation and update the circuit breaker.

        Parameters
        ----------
        equity:
            Current portfolio net liquidation value.
        timestamp:
            Observation time (defaults to now).

        Returns
        -------
        DrawdownSnapshot with the current drawdown state.
        """
        ts = timestamp or datetime.utcnow()
        self._equity_history.append((ts, equity))

        # Update high-water mark
        if equity > self._high_water_mark:
            self._high_water_mark = equity

        # Compute drawdowns
        current_dd = self._compute_drawdown(equity)
        if current_dd > self._max_drawdown_observed:
            self._max_drawdown_observed = current_dd

        # State transitions
        self._transition(current_dd, ts)

        snapshot = DrawdownSnapshot(
            timestamp=ts,
            equity=equity,
            high_water_mark=self._high_water_mark,
            drawdown_pct=current_dd,
            max_drawdown_pct=self._max_drawdown_observed,
            state=self._state,
        )

        logger.info(
            "drawdown_monitor.update",
            equity=equity,
            hwm=self._high_water_mark,
            drawdown_pct=f"{current_dd:.4%}",
            max_drawdown_pct=f"{self._max_drawdown_observed:.4%}",
            state=self._state.value,
        )
        return snapshot

    def current_drawdown(self) -> float:
        """Return the most recent drawdown percentage (0.0 to 1.0)."""
        if not self._equity_history:
            return 0.0
        _, last_equity = self._equity_history[-1]
        return self._compute_drawdown(last_equity)

    def reset(self, new_equity: float) -> None:
        """Hard-reset the monitor (e.g. after an injection of capital)."""
        self._high_water_mark = new_equity
        self._max_drawdown_observed = 0.0
        self._state = TradingState.ACTIVE
        self._halted_at = None
        self._equity_history.clear()
        self._equity_history.append((datetime.utcnow(), new_equity))
        logger.info("drawdown_monitor.reset", equity=new_equity)

    def get_equity_curve(self) -> list[tuple[datetime, float]]:
        """Return the full equity history as (timestamp, equity) pairs."""
        return list(self._equity_history)

    # ── Internal ──────────────────────────────────────────────────────────

    def _compute_drawdown(self, equity: float) -> float:
        if self._high_water_mark <= 0:
            return 0.0
        return max(0.0, (self._high_water_mark - equity) / self._high_water_mark)

    def _transition(self, current_dd: float, ts: datetime) -> None:
        """Manage state machine transitions."""
        previous = self._state

        if self._state == TradingState.ACTIVE:
            if current_dd >= self._max_drawdown_pct:
                self._state = TradingState.HALTED
                self._halted_at = ts
                logger.warning(
                    "drawdown_monitor.circuit_breaker_tripped",
                    drawdown=f"{current_dd:.4%}",
                    threshold=f"{self._max_drawdown_pct:.4%}",
                )

        elif self._state == TradingState.HALTED:
            # Check minimum halt duration
            if self._halted_at is not None and (ts - self._halted_at) < self._min_halt_duration:
                return
            # Transition to recovery if drawdown has receded
            if current_dd <= self._recovery_threshold_pct:
                self._state = TradingState.RECOVERY
                logger.info(
                    "drawdown_monitor.entering_recovery",
                    drawdown=f"{current_dd:.4%}",
                    size_factor=self._recovery_size_factor,
                )

        elif self._state == TradingState.RECOVERY:
            if current_dd >= self._max_drawdown_pct:
                # Drawdown worsened again -- halt
                self._state = TradingState.HALTED
                self._halted_at = ts
                logger.warning("drawdown_monitor.re_halted", drawdown=f"{current_dd:.4%}")
            elif current_dd <= 0.0:
                # Fully recovered -- new high-water mark hit
                self._state = TradingState.ACTIVE
                logger.info("drawdown_monitor.fully_recovered")

        if self._state != previous:
            logger.info(
                "drawdown_monitor.state_change",
                previous=previous.value,
                new=self._state.value,
            )
