"""Zerodha Kite Connect OAuth login and postback handlers.

Zerodha OAuth flow:
1. User clicks "Connect Zerodha" in dashboard.
2. ``GET /api/auth/zerodha/login`` redirects to Kite login page.
3. User logs in on Zerodha. Kite redirects back to
   ``GET /api/auth/zerodha/callback?request_token=...&status=success``
4. We exchange the request_token for an access_token using the
   Kite session API (api_key + api_secret + SHA256 checksum).
5. Store the access_token and redirect to dashboard.

Postback URL:
   ``POST /api/broker/zerodha/postback`` receives order update webhooks.

Redirect URL: https://hedgefund.viewfir.com/api/auth/zerodha/callback
Postback URL: https://hedgefund.viewfir.com/api/broker/zerodha/postback
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import structlog
from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse

log = structlog.get_logger(__name__)

router = APIRouter(tags=["zerodha-oauth"])

_KITE_LOGIN_URL = "https://kite.zerodha.com/connect/login"
_KITE_API_URL = "https://api.kite.trade"


def _get_zerodha_config() -> Dict[str, str]:
    """Retrieve Zerodha Kite credentials from credential store or env."""
    api_key = os.environ.get("ZERODHA_API_KEY", "")
    api_secret = os.environ.get("ZERODHA_API_SECRET", "")

    if not api_key:
        try:
            from hedgefund.security.credential_store import CredentialStore
            store = CredentialStore()
            stored_key = store.retrieve("zerodha", "api_key")
            stored_secret = store.retrieve("zerodha", "api_secret")
            if stored_key:
                api_key = stored_key
            if stored_secret:
                api_secret = stored_secret
        except Exception:
            pass

    return {
        "api_key": api_key,
        "api_secret": api_secret,
    }


# ---------------------------------------------------------------------------
# Step 1: Redirect to Kite login
# ---------------------------------------------------------------------------

@router.get("/auth/zerodha/login")
async def zerodha_login(request: Request) -> RedirectResponse:
    """Redirect user to Zerodha Kite login page."""
    config = _get_zerodha_config()
    if not config["api_key"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Zerodha API key not configured. "
                   "Set ZERODHA_API_KEY environment variable or store via "
                   "credential store under 'zerodha/api_key'.",
        )

    login_url = f"{_KITE_LOGIN_URL}?v=3&api_key={config['api_key']}"
    log.info("zerodha_oauth.login_redirect", api_key=config["api_key"][:6] + "...")
    return RedirectResponse(url=login_url)


# ---------------------------------------------------------------------------
# Step 2: Handle callback from Kite
# ---------------------------------------------------------------------------

@router.get("/auth/zerodha/callback")
async def zerodha_callback(
    request: Request,
    request_token: str = Query(None, description="Request token from Kite"),
    status_param: str = Query(None, alias="status", description="Login status"),
    action: str = Query(None, description="Action (login)"),
) -> RedirectResponse:
    """Handle Kite OAuth callback — exchange request_token for access_token.

    Kite redirects here with:
    ``?request_token=<token>&action=login&status=success``
    """
    if status_param != "success" or not request_token:
        log.warning(
            "zerodha_oauth.callback_failed",
            status=status_param,
            has_token=bool(request_token),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Zerodha login failed or was cancelled. Please try again.",
        )

    config = _get_zerodha_config()
    if not config["api_key"] or not config["api_secret"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Zerodha API key/secret not configured.",
        )

    # Generate checksum: SHA256(api_key + request_token + api_secret)
    checksum_input = config["api_key"] + request_token + config["api_secret"]
    checksum = hashlib.sha256(checksum_input.encode()).hexdigest()

    # Exchange for access_token
    import httpx

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{_KITE_API_URL}/session/token",
                data={
                    "api_key": config["api_key"],
                    "request_token": request_token,
                    "checksum": checksum,
                },
            )
    except httpx.RequestError as exc:
        log.error("zerodha_oauth.token_exchange_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to reach Zerodha API.",
        )

    if resp.status_code != 200:
        log.error(
            "zerodha_oauth.token_error",
            status=resp.status_code,
            body=resp.text[:500],
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Zerodha rejected the request token. Please try again.",
        )

    data = resp.json().get("data", {})
    access_token = data.get("access_token", "")
    user_id = data.get("user_id", "")
    user_name = data.get("user_name", "")
    email = data.get("email", "")

    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No access token received from Zerodha.",
        )

    log.info(
        "zerodha_oauth.authenticated",
        user_id=user_id,
        user_name=user_name,
    )

    # Store access token in credential store
    try:
        from hedgefund.security.credential_store import CredentialStore
        store = CredentialStore()
        store.store("zerodha", "access_token", access_token)
        store.store("zerodha", "user_id", user_id)
        log.info("zerodha_oauth.credentials_stored")
    except Exception:
        log.warning("zerodha_oauth.credential_store_failed", exc_info=True)

    # Add broker to BrokerManager if available
    broker_manager = getattr(request.app.state, "broker_manager", None)
    if broker_manager:
        try:
            await broker_manager.add_broker(
                broker_id="zerodha_main",
                broker_type="zerodha",
                credentials={
                    "api_key": config["api_key"],
                    "access_token": access_token,
                },
            )
            log.info("zerodha_oauth.broker_added")
        except Exception:
            log.warning("zerodha_oauth.broker_add_failed", exc_info=True)

    # Update data source validator
    dsv = getattr(request.app.state, "data_source_validator", None)
    if dsv:
        from hedgefund.engine.data_source_validator import DataSourceStatus
        dsv.update_broker_status(DataSourceStatus.CONNECTED, "zerodha")
        dsv.update_market_feed_status(DataSourceStatus.CONNECTED, "Zerodha Kite WebSocket")

    # Store in MongoDB
    db = getattr(request.app.state, "db", None)
    if db:
        try:
            from hedgefund.auth.models import encrypt_credentials
            encrypted = encrypt_credentials({
                "access_token": access_token,
                "api_key": config["api_key"],
            })
            await db.broker_connections.update_one(
                {"broker": "zerodha", "user_id": user_id},
                {
                    "$set": {
                        "broker": "zerodha",
                        "user_id": user_id,
                        "user_name": user_name,
                        "email": email,
                        "credentials_encrypted": encrypted,
                        "connected_at": datetime.now(timezone.utc),
                        "is_active": True,
                    }
                },
                upsert=True,
            )
        except Exception:
            log.debug("zerodha_oauth.db_store_failed", exc_info=True)

    # Redirect to dashboard
    return RedirectResponse(url="/")


# ---------------------------------------------------------------------------
# Postback: Order update webhook from Zerodha
# ---------------------------------------------------------------------------

@router.post("/broker/zerodha/postback")
async def zerodha_postback(request: Request) -> Dict[str, str]:
    """Receive order update postbacks from Zerodha.

    Zerodha sends POST with order update JSON whenever an order status
    changes (placed, completed, cancelled, rejected, etc.).

    Payload fields:
    - order_id, exchange_order_id, status, tradingsymbol, quantity,
      average_price, filled_quantity, transaction_type, etc.
    """
    try:
        payload = await request.json()
    except Exception:
        log.warning("zerodha_postback.invalid_payload")
        return {"status": "error"}

    order_id = payload.get("order_id", "")
    order_status = payload.get("status", "")
    symbol = payload.get("tradingsymbol", "")
    filled_qty = payload.get("filled_quantity", 0)
    avg_price = payload.get("average_price", 0)

    log.info(
        "zerodha_postback.order_update",
        order_id=order_id,
        status=order_status,
        symbol=symbol,
        filled_qty=filled_qty,
        avg_price=avg_price,
        source="Zerodha Postback",
    )

    # Publish to EventBus as FILL event
    dsm = getattr(request.app.state, "data_source_manager", None)
    if dsm:
        event_bus = getattr(dsm, "_event_bus", None)
        if event_bus:
            from hedgefund.streaming.event_bus import Event, EventType
            event = Event(
                event_type=EventType.FILL,
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                data={
                    "order_id": order_id,
                    "status": order_status,
                    "filled_quantity": filled_qty,
                    "average_price": avg_price,
                    "tradingsymbol": symbol,
                    "transaction_type": payload.get("transaction_type", ""),
                    "exchange": payload.get("exchange", ""),
                    "source": "zerodha",
                    "is_live": True,
                },
                source="Zerodha Postback",
            )
            await event_bus.publish(event)

    # Store in MongoDB
    db = getattr(request.app.state, "db", None)
    if db:
        try:
            from hedgefund.data.write_guard import WriteGuard
            doc = {**payload, "source": "zerodha", "received_at": datetime.now(timezone.utc)}
            WriteGuard.validate_trade(doc)
            await db.order_updates.insert_one(doc)
        except Exception:
            log.debug("zerodha_postback.db_store_failed", exc_info=True)

    return {"status": "ok"}
