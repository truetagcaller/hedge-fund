"""Broker switching, capabilities, and multi-broker management API.

Works in both full TradingApplication mode and dashboard-only mode by
checking MongoDB + credential store directly when BrokerRouter is absent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from hedgefund.auth.middleware import get_current_user
from hedgefund.execution.capabilities import (
    BROKER_CAPABILITIES,
    get_capabilities,
)

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/broker", tags=["broker-switch"])


class SwitchBrokerRequest(BaseModel):
    broker_id: str = Field(..., description="Broker ID to switch to")


def _broker_type_from_id(broker_id: str) -> str:
    """Extract the broker type from a broker_id.

    Handles formats like ``zerodha_main``, ``groww_1710432000000``, etc.
    """
    for known in BROKER_CAPABILITIES:
        if broker_id == known or broker_id.startswith(known + "_"):
            return known
    # Last-resort: strip common suffixes
    return (
        broker_id.replace("_main", "")
        .replace("_default", "")
        .split("_")[0]
    )


def _caps_to_dict(caps: Any) -> Dict[str, Any]:
    if caps is None:
        return {}
    if hasattr(caps, "to_dict"):
        return caps.to_dict()
    return {
        "options_trading": getattr(caps, "options_trading", False),
        "futures_trading": getattr(caps, "futures_trading", False),
        "crypto_trading": getattr(caps, "crypto_trading", False),
        "equity_trading": getattr(caps, "equity_trading", True),
        "mutual_funds": getattr(caps, "mutual_funds", False),
        "us_stocks": getattr(caps, "us_stocks", False),
        "market_data": getattr(caps, "market_data", False),
        "websocket_feed": getattr(caps, "websocket_feed", False),
        "order_placement": getattr(caps, "order_placement", False),
        "paper_trading": getattr(caps, "paper_trading", False),
        "supported_exchanges": list(
            getattr(caps, "supported_exchanges", [])
        ),
        "supported_order_types": list(
            getattr(caps, "supported_order_types", [])
        ),
    }


# ── Active broker ────────────────────────────────────────────────────────


@router.get("/active")
async def get_active_broker(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return the user's currently active broker."""
    user_id = user["user_id"]
    db = getattr(request.app.state, "db", None)

    active_id = ""

    # Check BrokerRouter in-memory (always up-to-date after switch)
    br = getattr(request.app.state, "broker_router", None)
    if br:
        try:
            active_id = await br.get_active_broker(user_id) or ""
        except Exception:
            pass

    # Fall back to MongoDB preference
    if not active_id and db is not None:
        try:
            pref = await db.user_preferences.find_one({"user_id": user_id})
            if pref:
                active_id = pref.get("active_broker", "")
        except Exception:
            pass

    # Fall back to first connected broker
    if not active_id and db is not None:
        try:
            doc = await db.broker_connections.find_one(
                {"is_active": True}, sort=[("connected_at", -1)],
            )
            if doc:
                active_id = f"{doc.get('broker', '')}_main"
        except Exception:
            pass

    broker_type = _broker_type_from_id(active_id) if active_id else ""
    try:
        caps = get_capabilities(broker_type) if broker_type else None
    except KeyError:
        caps = None

    return {
        "active_broker": active_id,
        "broker_type": broker_type,
        "display_name": (
            getattr(caps, "display_name", broker_type) if caps else ""
        ),
        "capabilities": _caps_to_dict(caps),
    }


# ── Switch broker ────────────────────────────────────────────────────────


@router.post("/switch")
async def switch_broker(
    request: Request,
    body: SwitchBrokerRequest,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Switch the user's active broker. Persists to MongoDB."""
    user_id = user["user_id"]
    broker_id = body.broker_id
    db = getattr(request.app.state, "db", None)

    # Persist preference
    if db is not None:
        try:
            await db.user_preferences.update_one(
                {"user_id": user_id},
                {"$set": {
                    "user_id": user_id,
                    "active_broker": broker_id,
                    "updated_at": datetime.now(timezone.utc),
                }},
                upsert=True,
            )
        except Exception:
            log.warning("broker_switch.db_failed", exc_info=True)

    # Update in-memory router if available
    br = getattr(request.app.state, "broker_router", None)
    if br:
        try:
            await br.set_active_broker(user_id, broker_id)
        except Exception:
            # In-memory update is best-effort; DB preference already saved
            log.debug("broker_switch.router_update_failed", exc_info=True)

    broker_type = _broker_type_from_id(broker_id)
    try:
        caps = get_capabilities(broker_type)
    except KeyError:
        caps = None

    log.info(
        "broker_switch.switched",
        user_id=user_id,
        broker_id=broker_id,
        broker_type=broker_type,
    )

    return {
        "status": "ok",
        "active_broker": broker_id,
        "broker_type": broker_type,
        "display_name": (
            getattr(caps, "display_name", broker_type) if caps else broker_type
        ),
        "capabilities": _caps_to_dict(caps),
    }


# ── Capabilities ─────────────────────────────────────────────────────────


@router.get("/capabilities/{broker_type}")
async def get_broker_capabilities_endpoint(
    broker_type: str,
) -> Dict[str, Any]:
    """Return capabilities for a specific broker type."""
    caps = get_capabilities(broker_type)
    if caps is None:
        raise HTTPException(404, f"Unknown broker type: {broker_type}")
    return {
        "broker_type": broker_type,
        "capabilities": _caps_to_dict(caps),
    }


@router.get("/capabilities")
async def get_all_capabilities() -> Dict[str, Any]:
    """Return capabilities for all known broker types."""
    return {
        "brokers": {
            bt: _caps_to_dict(caps)
            for bt, caps in BROKER_CAPABILITIES.items()
        },
    }


# ── Connected brokers ────────────────────────────────────────────────────


@router.get("/connected")
async def get_connected_brokers(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return all brokers connected for this user."""
    user_id = user["user_id"]
    db = getattr(request.app.state, "db", None)
    brokers: List[Dict[str, Any]] = []
    seen_types: set[str] = set()

    # Get active broker — check in-memory router first, then MongoDB
    active_id = ""
    br = getattr(request.app.state, "broker_router", None)
    if br:
        try:
            active_id = await br.get_active_broker(user_id) or ""
        except Exception:
            pass
    if not active_id and db is not None:
        try:
            pref = await db.user_preferences.find_one({"user_id": user_id})
            if pref:
                active_id = pref.get("active_broker", "")
        except Exception:
            pass

    # Check MongoDB broker_connections
    if db is not None:
        try:
            async for doc in db.broker_connections.find({"is_active": True}):
                bt = doc.get("broker", "")
                bid = f"{bt}_main"
                try:
                    caps = get_capabilities(bt)
                except KeyError:
                    caps = None
                seen_types.add(bt)
                brokers.append({
                    "broker_id": bid,
                    "broker_type": bt,
                    "display_name": (
                        getattr(caps, "display_name", bt) if caps else bt
                    ),
                    "status": "connected",
                    "user_name": doc.get("user_name", ""),
                    "connected_at": (
                        doc["connected_at"].isoformat()
                        if hasattr(doc.get("connected_at"), "isoformat")
                        else str(doc.get("connected_at", ""))
                    ),
                    "capabilities": _caps_to_dict(caps),
                    "is_active": bid == active_id,
                })
        except Exception:
            log.debug("broker_connected.db_error", exc_info=True)

    # Also check broker_accounts (used by /api/brokers/connect)
    if db is not None:
        try:
            query = {"user_id": user_id, "status": {"$ne": "disconnected"}}
            async for doc in db.broker_accounts.find(query):
                bid = doc.get("broker_id", "")
                bt = doc.get("broker_type", "")
                if bid in {b["broker_id"] for b in brokers}:
                    continue  # already listed
                try:
                    caps = get_capabilities(bt)
                except KeyError:
                    caps = None
                seen_types.add(bt)
                brokers.append({
                    "broker_id": bid,
                    "broker_type": bt,
                    "display_name": (
                        getattr(caps, "display_name", bt) if caps else bt
                    ),
                    "status": doc.get("status", "connected"),
                    "user_name": "",
                    "connected_at": (
                        doc["connected_at"].isoformat()
                        if hasattr(doc.get("connected_at"), "isoformat")
                        else str(doc.get("connected_at", ""))
                    ),
                    "capabilities": _caps_to_dict(caps),
                    "is_active": bid == active_id,
                })
        except Exception:
            log.debug("broker_connected.accounts_error", exc_info=True)

    # Check credential store
    try:
        from hedgefund.security.credential_store import CredentialStore
        store = CredentialStore()
        for ns in store.list_namespaces():
            if ns in ("twitter",) or ns in seen_types:
                continue
            if ns in BROKER_CAPABILITIES and store.retrieve(ns, "api_key"):
                caps = get_capabilities(ns)
                bid = f"{ns}_main"
                seen_types.add(ns)
                brokers.append({
                    "broker_id": bid,
                    "broker_type": ns,
                    "display_name": (
                        getattr(caps, "display_name", ns) if caps else ns
                    ),
                    "status": "configured",
                    "user_name": "",
                    "connected_at": "",
                    "capabilities": _caps_to_dict(caps),
                    "is_active": bid == active_id,
                })
    except Exception:
        pass

    # Auto-set first broker as active if none selected
    if not active_id and brokers:
        brokers[0]["is_active"] = True
        active_id = brokers[0]["broker_id"]

    return {
        "brokers": brokers,
        "active_broker": active_id,
        "total": len(brokers),
    }
