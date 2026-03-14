"""Backtest management endpoints: run, results, and listing."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from hedgefund.types import BacktestMetrics

log = structlog.get_logger(__name__)

router = APIRouter(tags=["backtest"])


# ── Request / response models ────────────────────────────────────────────────


class BacktestRunRequest(BaseModel):
    """Parameters to trigger a new backtest."""

    strategy: str = Field(..., description="Strategy name to backtest")
    symbols: List[str] = Field(..., min_length=1, description="Symbols to include")
    start_date: date
    end_date: date
    initial_capital: float = Field(10_000_000.0, gt=0)
    regime_aware: bool = True

    def model_post_init(self, __context: Any) -> None:
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")


class BacktestRunResponse(BaseModel):
    backtest_id: str
    status: str
    submitted_at: str


class BacktestSummary(BaseModel):
    backtest_id: str
    strategy: str
    symbols: List[str]
    start_date: str
    end_date: str
    status: str
    submitted_at: str
    completed_at: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/backtest/run", status_code=status.HTTP_202_ACCEPTED)
async def run_backtest(
    request: Request,
    body: BacktestRunRequest,
) -> BacktestRunResponse:
    """Submit a new backtest for asynchronous execution.

    Returns 202 with a backtest ID that can be polled for results.
    """
    backtest_engine = getattr(request.app.state, "backtest_engine", None)
    if backtest_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backtest engine not available",
        )

    backtest_id = f"BT-{uuid.uuid4().hex[:12].upper()}"
    now = datetime.now(timezone.utc)

    await backtest_engine.submit(
        backtest_id=backtest_id,
        strategy=body.strategy,
        symbols=body.symbols,
        start_date=body.start_date,
        end_date=body.end_date,
        initial_capital=body.initial_capital,
        regime_aware=body.regime_aware,
    )

    log.info(
        "backtest_submitted",
        backtest_id=backtest_id,
        strategy=body.strategy,
        symbols=body.symbols,
    )

    return BacktestRunResponse(
        backtest_id=backtest_id,
        status="submitted",
        submitted_at=now.isoformat(),
    )


@router.get("/backtest/results/{backtest_id}")
async def get_backtest_results(
    request: Request,
    backtest_id: str,
) -> Dict[str, Any]:
    """Retrieve results for a completed (or in-progress) backtest."""
    backtest_engine = getattr(request.app.state, "backtest_engine", None)
    if backtest_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backtest engine not available",
        )

    result = await backtest_engine.get_result(backtest_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest {backtest_id} not found",
        )

    response: Dict[str, Any] = {
        "backtest_id": backtest_id,
        "status": result.get("status", "unknown"),
        "submitted_at": result.get("submitted_at"),
        "completed_at": result.get("completed_at"),
        "parameters": result.get("parameters", {}),
    }

    # Include metrics when available
    metrics: Optional[BacktestMetrics] = result.get("metrics")
    if metrics is not None:
        response["metrics"] = asdict(metrics)
        response["metrics"]["loss_rate"] = metrics.loss_rate

    # Include equity curve when available
    equity_curve = result.get("equity_curve")
    if equity_curve is not None:
        response["equity_curve"] = equity_curve

    return response


@router.get("/backtest/list")
async def list_backtests(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    strategy: Optional[str] = Query(None, description="Filter by strategy name"),
) -> Dict[str, Any]:
    """List past backtests, most recent first."""
    backtest_engine = getattr(request.app.state, "backtest_engine", None)
    if backtest_engine is None:
        return {"backtests": [], "total": 0, "limit": limit, "offset": offset}

    backtests: List[Dict[str, Any]] = await backtest_engine.list_backtests(
        limit=limit,
        offset=offset,
        strategy=strategy,
    )
    total: int = await backtest_engine.count_backtests(strategy=strategy)

    return {
        "backtests": backtests,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
