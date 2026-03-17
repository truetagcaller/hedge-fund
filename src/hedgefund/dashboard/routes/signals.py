"""Signal management endpoints with multi-user data isolation.

Active signals, history, and manual overrides -- all scoped to the authenticated user.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, Field

from hedgefund.auth.middleware import get_current_user
from hedgefund.types import SignalAction, SignalDirection, TradeSignal

log = structlog.get_logger(__name__)

router = APIRouter(tags=["signals"])


# ── Request / response models ────────────────────────────────────────────────


class SignalOverrideRequest(BaseModel):
    """Payload for manually overriding or injecting a signal."""

    signal_id: Optional[str] = Field(None, description="Existing signal ID to override, or None to create new")
    underlying: str = Field(..., description="Underlying symbol")
    action: SignalAction
    direction: SignalDirection
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., min_length=1, max_length=500)


class SignalOverrideResponse(BaseModel):
    signal_id: str
    status: str
    timestamp: str


def _get_db(request: Request):
    """Extract MongoDB instance from app state, or None."""
    return getattr(request.app.state, "db", None)


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/signals")
async def get_active_signals(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return all currently active (non-expired, non-executed) signals for the user."""
    user_id = user["user_id"]
    db = _get_db(request)

    # Try MongoDB first
    if db is not None:
        query = {
            "user_id": user_id,
            "outcome": {"$in": [None, "pending", "active"]},
        }
        signals: List[Dict[str, Any]] = []
        async for doc in db.signals.find(query).sort("timestamp", -1):
            doc.pop("_id", None)
            doc.pop("user_id", None)
            if isinstance(doc.get("timestamp"), datetime):
                doc["timestamp"] = doc["timestamp"].isoformat()
            signals.append(doc)

        if signals:
            return {"signals": signals, "count": len(signals)}

    # Fall back to in-memory signal store
    signal_store = getattr(request.app.state, "signal_store", None)
    if signal_store is None:
        return {"signals": [], "count": 0}

    active: List[TradeSignal] = await signal_store.get_active()
    return {
        "signals": [asdict(s) for s in active],
        "count": len(active),
    }


@router.get("/signals/history")
async def get_signal_history(
    request: Request,
    user: dict = Depends(get_current_user),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    symbol: Optional[str] = Query(None, description="Filter by underlying"),
    strategy: Optional[str] = Query(None, description="Filter by strategy name"),
) -> Dict[str, Any]:
    """Return historical signals with their outcomes for the user."""
    user_id = user["user_id"]
    db = _get_db(request)

    # Try MongoDB first
    if db is not None:
        query: Dict[str, Any] = {"user_id": user_id}
        if symbol:
            query["underlying"] = symbol.upper()
        if strategy:
            query["strategy_name"] = strategy

        total = await db.signals.count_documents(query)
        cursor = db.signals.find(query).sort("timestamp", -1).skip(offset).limit(limit)
        signals: List[Dict[str, Any]] = []
        async for doc in cursor:
            doc.pop("_id", None)
            doc.pop("user_id", None)
            if isinstance(doc.get("timestamp"), datetime):
                doc["timestamp"] = doc["timestamp"].isoformat()
            signals.append(doc)

        return {
            "signals": signals,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    # Fall back to in-memory signal store
    signal_store = getattr(request.app.state, "signal_store", None)
    if signal_store is None:
        return {"signals": [], "total": 0, "limit": limit, "offset": offset}

    history: List[Dict[str, Any]] = await signal_store.get_history(
        limit=limit,
        offset=offset,
        symbol=symbol,
        strategy=strategy,
    )
    total_count: int = await signal_store.count_history(symbol=symbol, strategy=strategy)

    return {
        "signals": history,
        "total": total_count,
        "limit": limit,
        "offset": offset,
    }


@router.post("/signals/override", status_code=status.HTTP_201_CREATED)
async def override_signal(
    request: Request,
    body: SignalOverrideRequest,
    user: dict = Depends(get_current_user),
) -> SignalOverrideResponse:
    """Manually override or inject a trading signal for the current user.

    The signal still passes through risk validation before execution.
    """
    user_id = user["user_id"]
    db = _get_db(request)

    now = datetime.now(timezone.utc)
    signal_id = body.signal_id or TradeSignal.generate_id()

    signal = TradeSignal(
        signal_id=signal_id,
        timestamp=now,
        underlying=body.underlying,
        action=body.action,
        direction=body.direction,
        confidence=body.confidence,
        strategy_name="manual_override",
        entry_price=0.0,
        stop_loss=0.0,
        target_price=0.0,
        risk_reward_ratio=0.0,
        reasoning=body.reason,
        metadata={"source": "dashboard_override", "operator_override": True},
    )

    # Store in MongoDB with user_id
    if db is not None:
        signal_doc = asdict(signal)
        signal_doc["user_id"] = user_id
        signal_doc["outcome"] = "pending"
        await db.signals.insert_one(signal_doc)

    # Also put in in-memory store if available
    signal_store = getattr(request.app.state, "signal_store", None)
    if signal_store is not None:
        await signal_store.put(signal)

    log.info(
        "signal_override_applied",
        signal_id=signal_id,
        user_id=user_id,
        underlying=body.underlying,
        action=body.action.value,
    )

    return SignalOverrideResponse(
        signal_id=signal_id,
        status="accepted",
        timestamp=now.isoformat(),
    )
