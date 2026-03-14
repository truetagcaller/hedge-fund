"""Level-4 dashboard API routes.

Provides endpoints for per-user execution engines, capital allocation,
strategy performance tracking, strategy evolution, and system health.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from hedgefund.auth.middleware import get_current_user
from hedgefund.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/l4", tags=["level4"])


# ── Request models ─────────────────────────────────────────────────────────


class AllocationRequest(BaseModel):
    method: str = "equal"  # equal, manual, performance
    weights: Optional[Dict[str, float]] = None
    total_capital: Optional[float] = None


class RebalanceRequest(BaseModel):
    method: str = "performance"
    total_capital: Optional[float] = None


# ── Helpers ────────────────────────────────────────────────────────────────


def _get_engine_manager(request: Request) -> Any:
    mgr = getattr(request.app.state, "engine_manager", None)
    if mgr is None:
        return None
    return mgr


def _get_capital_allocator(request: Request) -> Any:
    return getattr(request.app.state, "capital_allocator", None)


def _get_strategy_tracker(request: Request) -> Any:
    return getattr(request.app.state, "strategy_tracker", None)


def _get_strategy_evolution(request: Request) -> Any:
    return getattr(request.app.state, "strategy_evolution", None)


# ── Engine endpoints ───────────────────────────────────────────────────────


@router.get("/engines")
async def get_active_engines(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """List all active user engines."""
    mgr = _get_engine_manager(request)
    if mgr is None:
        return {"engines": [], "message": "L4 engine manager not started"}
    return {"engines": mgr.get_active_engines()}


@router.get("/engine/status")
async def get_engine_status(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get the current user's engine status."""
    mgr = _get_engine_manager(request)
    if mgr is None:
        return {"status": None, "message": "L4 engine manager not started"}

    engine = await mgr.get_engine(user["user_id"])
    if engine is None:
        return {"status": None, "message": "No engine running for user"}
    return {"status": engine.get_status()}


@router.post("/engine/start")
async def start_user_engine(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Start an execution engine for the current user."""
    mgr = _get_engine_manager(request)
    if mgr is None:
        return {"ok": False, "message": "L4 engine manager not started"}

    engine = await mgr.get_or_create_engine(user["user_id"])
    return {"ok": True, "status": engine.get_status()}


@router.post("/engine/stop")
async def stop_user_engine(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Stop the current user's execution engine."""
    mgr = _get_engine_manager(request)
    if mgr is None:
        return {"ok": False, "message": "L4 engine manager not started"}

    await mgr.shutdown_engine(user["user_id"])
    return {"ok": True, "message": "Engine stopped"}


# ── Capital allocation endpoints ───────────────────────────────────────────


@router.get("/capital-allocation")
async def get_capital_allocation(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get per-strategy capital allocation for the current user."""
    allocator = _get_capital_allocator(request)
    if allocator is None:
        return {"allocations": [], "message": "Capital allocator not started"}

    allocs = await allocator.get_allocations(user["user_id"])
    return {"allocations": [a.to_dict() for a in allocs]}


@router.post("/capital-allocation")
async def set_capital_allocation(
    request: Request,
    body: AllocationRequest,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Set capital allocation (manual weights or method)."""
    allocator = _get_capital_allocator(request)
    if allocator is None:
        return {"ok": False, "message": "Capital allocator not started"}

    total = body.total_capital or 100_000.0
    allocs = await allocator.allocate(
        user["user_id"],
        total,
        method=body.method,
        manual_weights=body.weights,
    )
    return {
        "ok": True,
        "allocations": [a.to_dict() for a in allocs],
    }


@router.post("/rebalance")
async def trigger_rebalance(
    request: Request,
    body: RebalanceRequest,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Trigger a capital rebalance based on strategy performance."""
    allocator = _get_capital_allocator(request)
    tracker = _get_strategy_tracker(request)
    if allocator is None or tracker is None:
        return {"ok": False, "message": "L4 components not started"}

    metrics = await tracker.compute_all_metrics(user["user_id"], "30d")
    total = body.total_capital or 100_000.0
    allocs = await allocator.rebalance(user["user_id"], metrics, total)
    return {
        "ok": True,
        "allocations": [a.to_dict() for a in allocs],
    }


# ── Strategy performance endpoints ────────────────────────────────────────


@router.get("/strategy-performance")
async def get_strategy_performance(
    request: Request,
    user: dict = Depends(get_current_user),
    window: str = Query("30d", pattern="^(7d|30d|90d|all)$"),
) -> Dict[str, Any]:
    """Get performance metrics for all strategies."""
    tracker = _get_strategy_tracker(request)
    if tracker is None:
        return {"metrics": {}, "message": "Strategy tracker not started"}

    metrics = await tracker.compute_all_metrics(user["user_id"], window)
    return {
        "metrics": {k: v.to_dict() for k, v in metrics.items()},
        "window": window,
    }


@router.get("/strategy-performance/{strategy_name}")
async def get_strategy_detail(
    request: Request,
    strategy_name: str,
    user: dict = Depends(get_current_user),
    window: str = Query("30d", pattern="^(7d|30d|90d|all)$"),
) -> Dict[str, Any]:
    """Get detailed metrics for a single strategy."""
    tracker = _get_strategy_tracker(request)
    if tracker is None:
        return {"metrics": None, "message": "Strategy tracker not started"}

    metrics = await tracker.compute_metrics(user["user_id"], strategy_name, window)
    return {"metrics": metrics.to_dict()}


@router.get("/strategy-performance/{strategy_name}/history")
async def get_strategy_trade_history(
    request: Request,
    strategy_name: str,
    user: dict = Depends(get_current_user),
    limit: int = Query(100, ge=1, le=1000),
) -> Dict[str, Any]:
    """Get trade history for a strategy."""
    tracker = _get_strategy_tracker(request)
    if tracker is None:
        return {"trades": [], "message": "Strategy tracker not started"}

    trades = await tracker.get_strategy_history(user["user_id"], strategy_name, limit)
    return {"trades": trades, "count": len(trades)}


@router.get("/strategy-performance/{strategy_name}/equity-curve")
async def get_strategy_equity_curve(
    request: Request,
    strategy_name: str,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get cumulative P&L curve for a strategy."""
    tracker = _get_strategy_tracker(request)
    if tracker is None:
        return {"curve": [], "message": "Strategy tracker not started"}

    curve = await tracker.get_strategy_equity_curve(user["user_id"], strategy_name)
    return {"curve": curve, "strategy_name": strategy_name}


# ── Strategy evolution endpoints ──────────────────────────────────────────


@router.get("/strategy-evolution")
async def get_evolution_status(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get current strategy evolution states."""
    evolution = _get_strategy_evolution(request)
    if evolution is None:
        return {"status": {}, "message": "Strategy evolution not started"}
    return evolution.get_evolution_status(user["user_id"])


@router.post("/strategy-evolution/evaluate")
async def trigger_evaluation(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Manually trigger strategy evaluation."""
    evolution = _get_strategy_evolution(request)
    if evolution is None:
        return {"ok": False, "message": "Strategy evolution not started"}

    actions = await evolution.evaluate_strategies(user["user_id"])
    return {"ok": True, "actions": actions}


@router.post("/strategy-evolution/{strategy_name}/enable")
async def force_enable_strategy(
    request: Request,
    strategy_name: str,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Force-enable a strategy."""
    evolution = _get_strategy_evolution(request)
    if evolution is None:
        return {"ok": False, "message": "Strategy evolution not started"}

    await evolution.force_enable(user["user_id"], strategy_name)

    # Also enable in user engine if running
    mgr = _get_engine_manager(request)
    if mgr:
        engine = await mgr.get_engine(user["user_id"])
        if engine:
            engine.enable_strategy(strategy_name)

    return {"ok": True, "strategy": strategy_name, "state": "ENABLED"}


@router.post("/strategy-evolution/{strategy_name}/disable")
async def force_disable_strategy(
    request: Request,
    strategy_name: str,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Force-disable a strategy."""
    evolution = _get_strategy_evolution(request)
    if evolution is None:
        return {"ok": False, "message": "Strategy evolution not started"}

    await evolution.force_disable(user["user_id"], strategy_name)

    # Also disable in user engine if running
    mgr = _get_engine_manager(request)
    if mgr:
        engine = await mgr.get_engine(user["user_id"])
        if engine:
            engine.disable_strategy(strategy_name)

    return {"ok": True, "strategy": strategy_name, "state": "DISABLED"}


# ── System health endpoint ────────────────────────────────────────────────


@router.get("/system-health")
async def get_system_health(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Comprehensive Level-4 system health check."""
    health: Dict[str, Any] = {"level": 4}

    # Engine manager
    mgr = _get_engine_manager(request)
    if mgr:
        engines = mgr.get_active_engines()
        health["engines"] = {
            "active": len(engines),
            "details": engines,
        }
    else:
        health["engines"] = {"active": 0, "status": "not_started"}

    # Capital allocator
    allocator = _get_capital_allocator(request)
    health["capital_allocator"] = "running" if allocator else "not_started"

    # Strategy tracker
    tracker = _get_strategy_tracker(request)
    health["strategy_tracker"] = "running" if tracker else "not_started"

    # Strategy evolution
    evolution = _get_strategy_evolution(request)
    health["strategy_evolution"] = "running" if evolution else "not_started"

    # Event bus stats
    dsm = getattr(request.app.state, "data_source_manager", None)
    if dsm and hasattr(dsm, "event_bus"):
        try:
            health["event_bus"] = dsm.event_bus.get_stats()
        except Exception:
            health["event_bus"] = "error"

    # Broker connections
    bm = getattr(request.app.state, "broker_manager", None)
    if bm:
        try:
            health["broker_connections"] = len(bm.get_connected_broker_ids())
        except Exception:
            health["broker_connections"] = 0

    return health
