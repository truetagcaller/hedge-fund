"""Execution management endpoints — Level-3 trade execution from AI signals.

Provides endpoints to:
- Execute a specific signal
- Configure execution context (asset class, trading mode, broker)
- View open trades and execution history
- Close trades manually
- Get market session status
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from hedgefund.auth.middleware import get_current_user
from hedgefund.types import AssetClass, TradingExecutionContext, TradingMode

log = structlog.get_logger(__name__)

router = APIRouter(tags=["execution"])


# ── Request / response models ─────────────────────────────────────────────


class ExecuteSignalRequest(BaseModel):
    """Request to execute a specific signal."""

    signal_id: str = Field(..., description="Signal ID from MongoDB")
    broker_id: Optional[str] = Field(
        None, description="Override broker (uses active broker if omitted)",
    )


class ExecutionContextRequest(BaseModel):
    """Request to set execution context."""

    active_broker: str = Field(..., description="Broker ID to use")
    asset_class: str = Field(
        "OPTIONS", description="EQUITY, FUTURES, OPTIONS, COMMODITY, CRYPTO",
    )
    trading_mode: str = Field("PAPER", description="PAPER, LIVE, BACKTEST")
    market_segment: str = Field("", description="e.g. NFO, SPOT, MCX")


class CloseTradeRequest(BaseModel):
    """Request to close an open trade."""

    reason: str = Field("manual", description="Close reason")


# ── Helpers ───────────────────────────────────────────────────────────────


def _get_bridge(request: Request):
    """Get ExecutionBridge from app state."""
    bridge = getattr(request.app.state, "execution_bridge", None)
    if bridge is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Execution bridge not available.",
        )
    return bridge


def _get_db(request: Request):
    return getattr(request.app.state, "db", None)


# ── Execution context endpoints ───────────────────────────────────────────


@router.get("/execution/context")
async def get_execution_context(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get the current user's execution context."""
    bridge = _get_bridge(request)
    ctx = await bridge.get_execution_context(user["user_id"])
    if ctx is None:
        return {
            "context": None,
            "message": "No execution context set. Configure one to start trading.",
        }
    return {"context": ctx.to_dict()}


@router.post("/execution/context")
async def set_execution_context(
    request: Request,
    body: ExecutionContextRequest,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Set the execution context for the current user."""
    bridge = _get_bridge(request)

    try:
        asset_class = AssetClass(body.asset_class.upper())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid asset class: {body.asset_class}. "
            f"Valid: {[a.value for a in AssetClass]}",
        )

    try:
        trading_mode = TradingMode(body.trading_mode.upper())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid trading mode: {body.trading_mode}. "
            f"Valid: {[m.value for m in TradingMode]}",
        )

    ctx = TradingExecutionContext(
        user_id=user["user_id"],
        active_broker=body.active_broker,
        asset_class=asset_class,
        trading_mode=trading_mode,
        market_segment=body.market_segment,
    )

    await bridge.set_execution_context(ctx)

    return {
        "context": ctx.to_dict(),
        "message": "Execution context updated.",
    }


# ── Signal execution endpoints ────────────────────────────────────────────


@router.post("/execution/execute")
async def execute_signal(
    request: Request,
    body: ExecuteSignalRequest,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Execute a specific signal from the signal store."""
    bridge = _get_bridge(request)
    db = _get_db(request)

    if db is None:
        raise HTTPException(503, "Database not available.")

    # Fetch signal from MongoDB
    signal_doc = await db.signals.find_one({"signal_id": body.signal_id})
    if signal_doc is None:
        raise HTTPException(404, f"Signal {body.signal_id} not found.")

    signal_doc.pop("_id", None)

    # Verify ownership
    if signal_doc.get("user_id") and signal_doc["user_id"] != user["user_id"]:
        raise HTTPException(403, "Signal belongs to another user.")

    # Execute
    record = await bridge.execute_signal(
        signal_doc, user["user_id"], broker_id=body.broker_id,
    )

    return {
        "execution": record.to_dict(),
        "success": record.status == "executed",
    }


@router.post("/execution/execute-latest")
async def execute_latest_signal(
    request: Request,
    user: dict = Depends(get_current_user),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
) -> Dict[str, Any]:
    """Execute the most recent active signal for the user."""
    bridge = _get_bridge(request)
    db = _get_db(request)

    if db is None:
        raise HTTPException(503, "Database not available.")

    query: Dict[str, Any] = {
        "user_id": user["user_id"],
        "outcome": {"$in": ["active", "pending"]},
        "is_live_data": True,
    }
    if symbol:
        query["underlying"] = symbol.upper()

    signal_doc = await db.signals.find_one(
        query, sort=[("timestamp", -1)],
    )
    if signal_doc is None:
        raise HTTPException(404, "No active signal found.")

    signal_doc.pop("_id", None)

    record = await bridge.execute_signal(signal_doc, user["user_id"])

    return {
        "execution": record.to_dict(),
        "success": record.status == "executed",
    }


# ── Trade management endpoints ────────────────────────────────────────────


@router.get("/execution/trades")
async def get_open_trades(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return all open trades for the user."""
    bridge = _get_bridge(request)
    trades = bridge.get_open_trades()
    user_trades = [t for t in trades if t.get("user_id") == user["user_id"]]
    return {"trades": user_trades, "count": len(user_trades)}


@router.get("/execution/history")
async def get_execution_history(
    request: Request,
    user: dict = Depends(get_current_user),
    limit: int = Query(100, ge=1, le=1000),
) -> Dict[str, Any]:
    """Return execution history for the user."""
    bridge = _get_bridge(request)
    history = bridge.get_execution_history(limit=limit)
    user_history = [h for h in history if h.get("user_id") == user["user_id"]]
    return {"executions": user_history, "count": len(user_history)}


@router.post("/execution/close/{trade_id}")
async def close_trade(
    request: Request,
    trade_id: str,
    body: CloseTradeRequest,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Close a specific open trade."""
    bridge = _get_bridge(request)

    # Verify ownership
    trades = bridge.get_open_trades()
    trade = next((t for t in trades if t["trade_id"] == trade_id), None)
    if trade is None:
        raise HTTPException(404, f"Trade {trade_id} not found.")
    if trade.get("user_id") != user["user_id"]:
        raise HTTPException(403, "Trade belongs to another user.")

    record = await bridge.close_trade(trade_id, reason=body.reason)
    if record is None:
        raise HTTPException(404, "Trade already closed.")

    return {
        "execution": record.to_dict(),
        "success": record.status == "executed",
    }


@router.post("/execution/close-all")
async def close_all_trades(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Emergency close all open trades for the user."""
    bridge = _get_bridge(request)
    trades = bridge.get_open_trades()
    user_trades = [t for t in trades if t.get("user_id") == user["user_id"]]

    results = []
    for trade in user_trades:
        record = await bridge.close_trade(
            trade["trade_id"], reason="emergency_close_all",
        )
        if record:
            results.append(record.to_dict())

    return {
        "closed": len(results),
        "results": results,
    }


# ── Market session endpoints ──────────────────────────────────────────────


@router.get("/execution/market-sessions")
async def get_market_sessions(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return all market sessions with current open/closed status."""
    from hedgefund.execution.market_session import MarketSessionManager
    mgr = MarketSessionManager()
    return {"sessions": mgr.get_all_sessions()}


@router.get("/execution/asset-classes")
async def get_asset_classes(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return available asset classes with broker support info."""
    from hedgefund.execution.capabilities import BROKER_CAPABILITIES
    from hedgefund.execution.market_session import MarketSessionManager

    mgr = MarketSessionManager()

    classes = []
    for ac in AssetClass:
        brokers_supporting = []
        for bt, caps in BROKER_CAPABILITIES.items():
            mapping = {
                "EQUITY": caps.equity_trading,
                "FUTURES": caps.futures_trading,
                "OPTIONS": caps.options_trading,
                "COMMODITY": caps.futures_trading,
                "CRYPTO": caps.crypto_trading,
            }
            if mapping.get(ac.value, False):
                brokers_supporting.append(caps.display_name)

        classes.append({
            "asset_class": ac.value,
            "is_tradeable_now": mgr.is_asset_class_tradeable(ac.value),
            "supported_brokers": brokers_supporting,
        })

    return {"asset_classes": classes}


# ── Execution settings ────────────────────────────────────────────────────


@router.get("/execution/settings")
async def get_execution_settings(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return the user's current execution configuration."""
    bridge = _get_bridge(request)
    ctx = await bridge.get_execution_context(user["user_id"])

    broker_router = getattr(request.app.state, "broker_router", None)
    active_broker = None
    if broker_router:
        active_broker = await broker_router.get_active_broker(user["user_id"])

    return {
        "context": ctx.to_dict() if ctx else None,
        "active_broker": active_broker,
        "auto_execute": bridge._auto_execute,
        "asset_classes": [ac.value for ac in AssetClass],
        "trading_modes": [tm.value for tm in TradingMode],
    }
