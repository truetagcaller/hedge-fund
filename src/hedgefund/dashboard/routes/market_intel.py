"""Market intelligence, AI engine, and trading statistics endpoints.

CRITICAL: This file NEVER generates synthetic, mock, or random data.
When real data is unavailable, endpoints return explicit
"DATA SOURCE NOT CONNECTED" messages with empty data arrays.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from hedgefund.auth.middleware import get_current_user

log = structlog.get_logger(__name__)

router = APIRouter(tags=["market_intel"])

_NOT_CONNECTED = "DATA SOURCE NOT CONNECTED"


# ── Pydantic models ──────────────────────────────────────────────────────────


class EngineModeRequest(BaseModel):
    mode: str = Field(..., description="Trading mode: 'live' or 'paper'")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_engine(request: Request) -> Any:
    return getattr(request.app.state, "decision_engine", None)


def _get_market_data(request: Request) -> Any:
    return getattr(request.app.state, "market_data", None)


def _get_db(request: Request):
    return getattr(request.app.state, "db", None)


def _get_data_source_manager(request: Request) -> Any:
    return getattr(request.app.state, "data_source_manager", None)


def _empty_response(message: str) -> Dict[str, Any]:
    """Standard empty response with NOT CONNECTED message."""
    return {
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Order Book ───────────────────────────────────────────────────────────────


@router.get("/market/order-book/{symbol}")
async def get_order_book(request: Request, symbol: str) -> Dict[str, Any]:
    """Return live order book for a symbol."""
    market_data = _get_market_data(request)
    if market_data is not None:
        try:
            return await market_data.get_order_book(symbol)
        except Exception:
            log.warning("order_book.fetch_failed", symbol=symbol, exc_info=True)

    return {
        "symbol": symbol.upper(),
        "bids": [],
        "asks": [],
        "spread": 0,
        "imbalance": 0,
        "large_orders": [],
        "pressure": "unknown",
        "message": _NOT_CONNECTED,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Smart Money ──────────────────────────────────────────────────────────────


@router.get("/market/smart-money/{symbol}")
async def get_smart_money(request: Request, symbol: str) -> Dict[str, Any]:
    """Return smart money signals for a symbol."""
    market_data = _get_market_data(request)
    if market_data is not None:
        try:
            return await market_data.get_smart_money_signals(symbol)
        except Exception:
            log.warning("smart_money.fetch_failed", symbol=symbol, exc_info=True)

    return {
        "symbol": symbol.upper(),
        "block_trades": [],
        "volume_spike": False,
        "vwap_deviation": 0,
        "accumulation_distribution": 0,
        "score": 0,
        "message": _NOT_CONNECTED,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Options Flow ─────────────────────────────────────────────────────────────


@router.get("/market/options-flow/{symbol}")
async def get_options_flow(request: Request, symbol: str) -> Dict[str, Any]:
    """Return options flow intelligence for a symbol."""
    market_data = _get_market_data(request)
    if market_data is not None:
        try:
            return await market_data.get_options_flow(symbol)
        except Exception:
            log.warning("options_flow.fetch_failed", symbol=symbol, exc_info=True)

    return {
        "symbol": symbol.upper(),
        "unusual_activity": [],
        "gex": 0,
        "dealer_positioning": "unknown",
        "oi_buildup": "unknown",
        "pcr": 0,
        "max_pain": 0,
        "iv_rank": 0,
        "skew": 0,
        "message": _NOT_CONNECTED,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Liquidity Sweeps ─────────────────────────────────────────────────────────


@router.get("/market/liquidity-sweeps")
async def get_liquidity_sweeps(request: Request) -> Dict[str, Any]:
    """Return recent liquidity sweep detections."""
    market_data = _get_market_data(request)
    if market_data is not None:
        try:
            return await market_data.get_liquidity_sweeps()
        except Exception:
            log.warning("liquidity_sweeps.fetch_failed", exc_info=True)

    return {
        "sweeps": [],
        "message": _NOT_CONNECTED,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── AI Engine Status ─────────────────────────────────────────────────────────


@router.get("/engine/status")
async def get_engine_status(request: Request) -> Dict[str, Any]:
    """Return current AI engine status."""
    engine = _get_engine(request)
    if engine is not None:
        try:
            return await engine.get_status()
        except Exception:
            log.warning("engine_status.fetch_failed", exc_info=True)

    # Check agent registry
    registry = getattr(request.app.state, "agent_registry", None)
    agent_status = {}
    if registry is not None:
        agent_status = registry.get_status()

    return {
        "mode": getattr(request.app.state, "engine_mode", "paper"),
        "components": {
            "event_bus": {"status": "idle"},
            "decision_engine": {"status": "idle", "signals_generated": 0},
            "executor": {"status": "idle", "open_trades": 0},
        },
        "agents": agent_status,
        "message": "Engine idle — waiting for live market data connection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── AI Engine Decisions ──────────────────────────────────────────────────────


@router.get("/engine/decisions")
async def get_engine_decisions(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return recent AI-generated decisions."""
    user_id = user["user_id"]
    db = _get_db(request)

    # Check MongoDB for real decisions
    if db is not None:
        cursor = db.signals.find(
            {"user_id": user_id},
        ).sort("timestamp", -1).limit(20)

        decisions: List[Dict[str, Any]] = []
        async for doc in cursor:
            decisions.append({
                "symbol": doc.get("underlying", ""),
                "action": doc.get("action", ""),
                "confidence": doc.get("confidence", 0.0),
                "reasoning": doc.get("reasoning", ""),
                "signal_breakdown": doc.get("metadata", {}).get(
                    "signal_breakdown", {},
                ),
                "timestamp": (
                    doc["timestamp"].isoformat()
                    if isinstance(doc.get("timestamp"), datetime)
                    else str(doc.get("timestamp", ""))
                ),
                "outcome": doc.get("outcome"),
            })
        if decisions:
            return {"decisions": decisions}

    engine = _get_engine(request)
    if engine is not None:
        try:
            decisions = await engine.get_recent_decisions()
            return {"decisions": decisions}
        except Exception:
            log.warning("engine_decisions.fetch_failed", exc_info=True)

    return {
        "decisions": [],
        "message": "No trading decisions yet — AI agents waiting for live data",
    }


# ── Engine Mode Control ──────────────────────────────────────────────────────


@router.post("/engine/mode")
async def set_engine_mode(
    request: Request,
    body: EngineModeRequest,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Switch between LIVE and PAPER trading mode."""
    mode = body.mode.lower()
    if mode not in ("live", "paper"):
        raise HTTPException(
            status_code=400, detail="Mode must be 'live' or 'paper'",
        )

    engine = _get_engine(request)
    if engine is not None:
        try:
            await engine.set_mode(mode)
        except Exception:
            log.warning("engine_mode.set_failed", mode=mode, exc_info=True)

    request.app.state.engine_mode = mode
    log.info("engine_mode_changed", mode=mode, user_id=user["user_id"])

    return {
        "mode": mode,
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Trading Statistics ───────────────────────────────────────────────────────


@router.get("/trades/stats")
async def get_trade_stats(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return aggregate trading statistics from real trades only."""
    user_id = user["user_id"]
    db = _get_db(request)

    # Compute from real MongoDB trade data
    if db is not None:
        pipeline = [
            {"$match": {"user_id": user_id}},
            {
                "$group": {
                    "_id": None,
                    "total_trades": {"$sum": 1},
                    "wins": {
                        "$sum": {"$cond": [{"$gt": ["$pnl", 0]}, 1, 0]}
                    },
                    "total_pnl": {"$sum": "$pnl"},
                    "max_pnl": {"$max": "$pnl"},
                    "min_pnl": {"$min": "$pnl"},
                    "avg_win": {
                        "$avg": {
                            "$cond": [{"$gt": ["$pnl", 0]}, "$pnl", None]
                        }
                    },
                    "avg_loss": {
                        "$avg": {
                            "$cond": [{"$lte": ["$pnl", 0]}, "$pnl", None]
                        }
                    },
                }
            },
        ]
        results = await db.trades.aggregate(pipeline).to_list(1)
        if results and results[0].get("total_trades", 0) > 0:
            agg = results[0]
            total = agg["total_trades"]
            wins = agg.get("wins", 0)
            avg_win = abs(agg.get("avg_win") or 0)
            avg_loss = abs(agg.get("avg_loss") or 0)
            gross_profit = wins * avg_win
            gross_loss = (total - wins) * avg_loss

            return {
                "total_trades": total,
                "win_rate": round(wins / max(total, 1), 4),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(
                    gross_profit / max(gross_loss, 1), 2,
                ),
                "best_trade": round(agg.get("max_pnl", 0), 2),
                "worst_trade": round(agg.get("min_pnl", 0), 2),
                "total_pnl": round(agg.get("total_pnl", 0), 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    # No trades — return zeros, NOT fake data
    return {
        "total_trades": 0,
        "win_rate": 0,
        "avg_win": 0,
        "avg_loss": 0,
        "profit_factor": 0,
        "sharpe": 0,
        "sortino": 0,
        "max_drawdown": 0,
        "best_trade": 0,
        "worst_trade": 0,
        "message": "No trades executed yet",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Data Sources (user-scoped) ───────────────────────────────────────────────


@router.get("/data-sources")
async def get_data_sources(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return list of connected data sources."""
    user_id = user["user_id"]
    db = _get_db(request)

    # Check user-scoped MongoDB sources
    if db is not None:
        sources: List[Dict[str, Any]] = []
        async for doc in db.data_sources.find(
            {"user_id": user_id},
            {"credentials": 0},
        ):
            sources.append({
                "id": doc.get("source_id", str(doc["_id"])),
                "type": doc.get("source_type", ""),
                "name": doc.get("name", ""),
                "status": doc.get("status", "unknown"),
            })
        if sources:
            return {"sources": sources}

    # Fall back to global DataSourceManager
    manager = _get_data_source_manager(request)
    if manager is not None:
        try:
            return {"sources": manager.list_sources()}
        except Exception:
            log.warning("data_sources.list_failed", exc_info=True)

    return {"sources": []}


@router.post("/data-sources")
async def add_data_source(
    request: Request,
    body: Dict[str, Any],
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Add a new data source."""
    manager = _get_data_source_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Data source manager unavailable")

    source_type = body.get("source_type", "")
    try:
        source_id = manager.add_source(source_type, body)
        return {"source_id": source_id, "status": "added"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/data-sources/{source_id}")
async def remove_data_source(
    request: Request,
    source_id: str,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Remove a data source."""
    manager = _get_data_source_manager(request)
    if manager is not None:
        try:
            manager.remove_source(source_id)
        except Exception:
            log.warning("data_sources.remove_failed", exc_info=True)

    db = _get_db(request)
    if db is not None:
        await db.data_sources.delete_one({"source_id": source_id})

    return {"status": "removed"}
