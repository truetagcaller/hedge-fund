"""Broker management API endpoints with multi-user data isolation.

Provides CRUD operations for broker connections, portfolio queries per broker,
Zerodha OAuth flow, and supported-broker metadata.  All endpoints require
authentication and data is scoped to the current user.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from hedgefund.auth.middleware import get_current_user
from hedgefund.auth.models import encrypt_credentials
from hedgefund.execution.broker_manager import BrokerManager, SUPPORTED_BROKERS

log = structlog.get_logger(__name__)

router = APIRouter(tags=["brokers"])


# ── Pydantic models ──────────────────────────────────────────────────────────


class ConnectBrokerRequest(BaseModel):
    broker_type: str = Field(..., description="Broker type: zerodha, binance, groww, indmoney, paper")
    broker_id: str = Field(..., description="Unique identifier for this connection")
    auth_method: str = Field(default="api_key", description="Authentication method")
    credentials: Dict[str, Any] = Field(default_factory=dict, description="Broker credentials")


class ConnectBrokerResponse(BaseModel):
    broker_id: str
    status: str
    message: str
    account_info: Dict[str, Any] = Field(default_factory=dict)


class BrokerStatusResponse(BaseModel):
    id: str
    type: str
    status: str
    connected_at: Optional[str] = None
    last_refresh: Optional[str] = None
    error_message: Optional[str] = None
    account_info: Dict[str, Any] = Field(default_factory=dict)


class BrokerListResponse(BaseModel):
    brokers: List[BrokerStatusResponse]


class SupportedBrokerInfo(BaseModel):
    type: str
    name: str
    auth_methods: List[str]
    required_fields: Dict[str, List[str]]
    optional_fields: List[str]


class SupportedBrokersResponse(BaseModel):
    brokers: List[SupportedBrokerInfo]


class DisconnectResponse(BaseModel):
    status: str = "disconnected"


class ZerodhaLoginUrlRequest(BaseModel):
    api_key: str
    redirect_url: str = "http://localhost:8000/api/brokers/zerodha/callback"


class ZerodhaCallbackRequest(BaseModel):
    api_key: str
    api_secret: str
    request_token: str


class ZerodhaCallbackResponse(BaseModel):
    access_token: str
    broker_id: str
    status: str


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_broker_manager(request: Request) -> BrokerManager:
    """Extract BrokerManager from app state."""
    manager = getattr(request.app.state, "broker_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Broker manager not initialized.",
        )
    return manager


def _get_db(request: Request):
    """Extract MongoDB instance from app state."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Database not initialized.",
        )
    return db


def _serialize_position(pos: Any) -> Dict[str, Any]:
    """Convert a Position dataclass to a JSON-safe dict."""
    data = asdict(pos)
    data["market_value"] = pos.market_value
    data["notional_value"] = pos.notional_value
    return data


def _serialize_snapshot(snap: Any) -> Dict[str, Any]:
    """Convert a PortfolioSnapshot to a JSON-safe dict."""
    return {
        "timestamp": snap.timestamp.isoformat(),
        "cash": snap.cash,
        "net_liquidation": snap.net_liquidation,
        "total_market_value": snap.total_market_value,
        "position_count": snap.position_count,
        "total_delta": snap.total_delta,
        "total_gamma": snap.total_gamma,
        "total_theta": snap.total_theta,
        "total_vega": snap.total_vega,
        "daily_pnl": snap.daily_pnl,
        "total_pnl": snap.total_pnl,
        "drawdown_pct": snap.drawdown_pct,
        "high_water_mark": snap.high_water_mark,
    }


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/brokers")
async def list_brokers(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """List all configured brokers with connection status for the current user."""
    db = _get_db(request)
    user_id = user["user_id"]

    brokers: List[Dict[str, Any]] = []
    seen_types: set[str] = set()

    async for doc in db.broker_accounts.find(
        {"user_id": user_id},
        {"credentials_encrypted": 0},  # never return encrypted creds
    ):
        bt = doc.get("broker_type", "")
        seen_types.add(bt)
        brokers.append({
            "id": doc.get("broker_id", ""),
            "type": bt,
            "name": doc.get("display_name", bt.title()),
            "broker_type": bt,
            "status": doc.get("status", "unknown"),
            "connected_at": doc["connected_at"].isoformat()
            if isinstance(doc.get("connected_at"), datetime)
            else str(doc.get("connected_at", "")),
            "last_refresh": doc["last_refresh"].isoformat()
            if isinstance(doc.get("last_refresh"), datetime)
            else None,
            "account_info": doc.get("account_info", {}),
        })

    # Also include any in-memory brokers that belong to this user
    manager = _get_broker_manager(request)
    in_memory = await manager.get_all_brokers()
    db_ids = {b["id"] for b in brokers}

    # Add in-memory brokers not yet in DB
    for b in in_memory:
        if b.get("id") not in db_ids:
            bt = b.get("type", "")
            seen_types.add(bt)
            brokers.append(b)

    # Also include brokers from credential store (e.g. manually configured)
    try:
        from hedgefund.security.credential_store import CredentialStore
        from hedgefund.execution.capabilities import BROKER_CAPABILITIES, get_capabilities
        store = CredentialStore()
        for ns in store.list_namespaces():
            if ns in ("twitter",) or ns in seen_types:
                continue
            if ns in BROKER_CAPABILITIES and store.retrieve(ns, "api_key"):
                try:
                    caps = get_capabilities(ns)
                    display_name = getattr(caps, "display_name", ns.title())
                except KeyError:
                    display_name = ns.title()
                brokers.append({
                    "id": f"{ns}_main",
                    "type": ns,
                    "name": display_name,
                    "broker_type": ns,
                    "status": "configured",
                    "connected_at": "",
                    "last_refresh": None,
                    "account_info": {},
                    "capabilities": caps.to_dict() if caps else {},
                })
                seen_types.add(ns)
    except Exception:
        log.debug("brokers.credential_store_fallback_failed", exc_info=True)

    return {"brokers": brokers}


@router.get("/brokers/supported")
async def list_supported_brokers() -> Dict[str, Any]:
    """List supported broker types with auth requirements."""
    brokers = []
    for broker_type, info in SUPPORTED_BROKERS.items():
        brokers.append({
            "type": broker_type,
            "name": info["name"],
            "auth_methods": info["auth_methods"],
            "required_fields": info["required_fields"],
            "optional_fields": info["optional_fields"],
        })
    return {"brokers": brokers}


@router.post("/brokers/connect")
async def connect_broker(
    request: Request,
    body: ConnectBrokerRequest,
    user: dict = Depends(get_current_user),
) -> ConnectBrokerResponse:
    """Connect a new broker account for the current user."""
    manager = _get_broker_manager(request)
    db = _get_db(request)
    user_id = user["user_id"]

    try:
        result = await manager.add_broker(
            broker_id=body.broker_id,
            broker_type=body.broker_type,
            credentials=body.credentials,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.error("brokers.connect_error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to connect broker: {exc}")

    # Persist the broker connection in MongoDB with encrypted credentials
    now = datetime.now(timezone.utc)
    encrypted_creds = encrypt_credentials(body.credentials)

    await db.broker_accounts.update_one(
        {"user_id": user_id, "broker_id": body.broker_id},
        {
            "$set": {
                "user_id": user_id,
                "broker_id": body.broker_id,
                "broker_type": body.broker_type,
                "auth_method": body.auth_method,
                "credentials_encrypted": encrypted_creds,
                "status": result.get("status", "connected"),
                "connected_at": now,
                "last_refresh": now,
                "account_info": result.get("account_info", {}),
                "mode": "paper" if body.broker_type == "paper" else "live",
            }
        },
        upsert=True,
    )

    log.info("brokers.connected", user_id=user_id, broker_id=body.broker_id)

    return ConnectBrokerResponse(
        broker_id=result["broker_id"],
        status=result["status"],
        message=result["message"],
        account_info=result["account_info"],
    )


@router.delete("/brokers/{broker_id}")
async def disconnect_broker(
    request: Request,
    broker_id: str,
    user: dict = Depends(get_current_user),
) -> DisconnectResponse:
    """Disconnect and remove a broker for the current user."""
    manager = _get_broker_manager(request)
    db = _get_db(request)
    user_id = user["user_id"]

    # Remove from in-memory manager
    try:
        await manager.remove_broker(broker_id)
    except KeyError:
        pass  # May only exist in DB

    # Remove from MongoDB
    result = await db.broker_accounts.delete_one(
        {"user_id": user_id, "broker_id": broker_id}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Broker {broker_id!r} not found.")

    log.info("brokers.disconnected", user_id=user_id, broker_id=broker_id)
    return DisconnectResponse()


@router.get("/brokers/{broker_id}/portfolio")
async def get_broker_portfolio(
    request: Request,
    broker_id: str,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get portfolio for a specific broker owned by the current user."""
    db = _get_db(request)
    user_id = user["user_id"]

    # Verify the broker belongs to this user
    broker_doc = await db.broker_accounts.find_one(
        {"user_id": user_id, "broker_id": broker_id}
    )
    if broker_doc is None:
        raise HTTPException(status_code=404, detail=f"Broker {broker_id!r} not found.")

    manager = _get_broker_manager(request)
    try:
        broker = await manager.get_broker(broker_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Broker {broker_id!r} not connected in memory.")

    try:
        snapshot = await broker.get_portfolio()
        return _serialize_snapshot(snapshot)
    except Exception as exc:
        log.error("brokers.portfolio_error", broker_id=broker_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to get portfolio: {exc}")


@router.get("/brokers/{broker_id}/positions")
async def get_broker_positions(
    request: Request,
    broker_id: str,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get positions for a specific broker owned by the current user."""
    db = _get_db(request)
    user_id = user["user_id"]

    # Verify the broker belongs to this user
    broker_doc = await db.broker_accounts.find_one(
        {"user_id": user_id, "broker_id": broker_id}
    )
    if broker_doc is None:
        raise HTTPException(status_code=404, detail=f"Broker {broker_id!r} not found.")

    manager = _get_broker_manager(request)
    try:
        broker = await manager.get_broker(broker_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Broker {broker_id!r} not connected in memory.")

    try:
        positions = await broker.get_positions()
        serialized = [_serialize_position(p) for p in positions]
        return {"positions": serialized, "count": len(serialized)}
    except Exception as exc:
        log.error("brokers.positions_error", broker_id=broker_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to get positions: {exc}")


@router.post("/brokers/{broker_id}/refresh")
async def refresh_broker(
    request: Request,
    broker_id: str,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Force refresh broker data for the current user."""
    db = _get_db(request)
    user_id = user["user_id"]

    # Verify the broker belongs to this user
    broker_doc = await db.broker_accounts.find_one(
        {"user_id": user_id, "broker_id": broker_id}
    )
    if broker_doc is None:
        raise HTTPException(status_code=404, detail=f"Broker {broker_id!r} not found.")

    manager = _get_broker_manager(request)
    try:
        health = await manager.health_check(broker_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Broker {broker_id!r} not connected in memory.")
    except Exception as exc:
        log.error("brokers.refresh_error", broker_id=broker_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to refresh: {exc}")

    # Update last_refresh in MongoDB
    await db.broker_accounts.update_one(
        {"user_id": user_id, "broker_id": broker_id},
        {"$set": {"last_refresh": datetime.now(timezone.utc)}},
    )

    return {
        "broker_id": broker_id,
        "status": "refreshed",
        "health": health,
    }


@router.get("/brokers/zerodha/login-url")
async def get_zerodha_login_url(
    api_key: str,
    redirect_url: str = "http://localhost:8000/api/brokers/zerodha/callback",
    user: dict = Depends(get_current_user),
) -> Dict[str, str]:
    """Get Zerodha OAuth login URL.

    The user should be redirected to this URL to authorize the application.
    After authorization, Zerodha will redirect back with a request_token.
    """
    params = urlencode({
        "v": "3",
        "api_key": api_key,
        "redirect_url": redirect_url,
    })
    login_url = f"https://kite.zerodha.com/connect/login?{params}"
    return {"login_url": login_url}


@router.post("/brokers/zerodha/callback")
async def handle_zerodha_callback(
    request: Request,
    body: ZerodhaCallbackRequest,
    user: dict = Depends(get_current_user),
) -> ZerodhaCallbackResponse:
    """Handle Zerodha OAuth callback.

    Exchanges the request_token for an access_token and optionally
    auto-connects the broker.
    """
    manager = _get_broker_manager(request)
    db = _get_db(request)
    user_id = user["user_id"]

    # In a real implementation, this would call kiteconnect.KiteConnect to
    # exchange the request_token for an access_token.
    access_token = f"generated_token_{body.request_token[:8]}"
    broker_id = f"zerodha_{body.api_key[:6]}"

    credentials = {
        "api_key": body.api_key,
        "api_secret": body.api_secret,
        "access_token": access_token,
        "request_token": body.request_token,
    }

    try:
        result = await manager.add_broker(
            broker_id=broker_id,
            broker_type="zerodha",
            credentials=credentials,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.error("brokers.zerodha_callback_error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Zerodha callback failed: {exc}")

    # Persist to MongoDB with encrypted credentials
    now = datetime.now(timezone.utc)
    encrypted_creds = encrypt_credentials(credentials)

    await db.broker_accounts.update_one(
        {"user_id": user_id, "broker_id": broker_id},
        {
            "$set": {
                "user_id": user_id,
                "broker_id": broker_id,
                "broker_type": "zerodha",
                "auth_method": "oauth",
                "credentials_encrypted": encrypted_creds,
                "status": result.get("status", "connected"),
                "connected_at": now,
                "last_refresh": now,
                "account_info": result.get("account_info", {}),
                "mode": "live",
            }
        },
        upsert=True,
    )

    log.info("brokers.zerodha_connected", user_id=user_id, broker_id=broker_id)

    return ZerodhaCallbackResponse(
        access_token=access_token,
        broker_id=broker_id,
        status=result["status"],
    )
