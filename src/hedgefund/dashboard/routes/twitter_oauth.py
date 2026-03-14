"""Twitter OAuth 2.0 callback handler.

Handles the OAuth 2.0 Authorization Code Flow with PKCE for X (Twitter).

Flow:
1. User clicks "Connect X" in dashboard.
2. Frontend redirects to ``GET /api/auth/twitter/login`` which builds the
   Twitter authorize URL and redirects the user there.
3. User authorizes the app on Twitter.
4. Twitter redirects back to ``GET /api/auth/twitter/callback`` with
   ``code`` and ``state`` params.
5. We exchange the code for tokens, fetch the user profile, store the
   credentials, and redirect back to the dashboard.

Callback URL registered with Twitter:
    https://hedgefund.viewfir.com/api/auth/twitter/callback
"""

from __future__ import annotations

import hashlib
import base64
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict
from urllib.parse import urlencode

import structlog
from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/auth/twitter", tags=["twitter-oauth"])

# In-memory state store (per-process; for production use Redis)
_pending_states: Dict[str, Dict[str, str]] = {}

# Twitter OAuth 2.0 endpoints
_TWITTER_AUTHORIZE_URL = "https://twitter.com/i/oauth2/authorize"
_TWITTER_TOKEN_URL = "https://api.twitter.com/2/oauth2/token"
_TWITTER_USER_URL = "https://api.twitter.com/2/users/me"


def _get_twitter_config(request: Request) -> Dict[str, str]:
    """Retrieve Twitter app credentials from settings / env."""
    client_id = os.environ.get("TWITTER_CLIENT_ID", "")
    client_secret = os.environ.get("TWITTER_CLIENT_SECRET", "")
    callback_url = os.environ.get(
        "TWITTER_CALLBACK_URL",
        "https://hedgefund.viewfir.com/api/auth/twitter/callback",
    )

    # Also try credential store
    if not client_id:
        try:
            from hedgefund.security.credential_store import CredentialStore
            store = CredentialStore()
            stored_id = store.retrieve("twitter", "client_id")
            stored_secret = store.retrieve("twitter", "client_secret")
            if stored_id:
                client_id = stored_id
            if stored_secret:
                client_secret = stored_secret
        except Exception:
            pass

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "callback_url": callback_url,
    }


# ---------------------------------------------------------------------------
# Step 1: Redirect user to Twitter authorization
# ---------------------------------------------------------------------------

@router.get("/login")
async def twitter_login(request: Request) -> RedirectResponse:
    """Initiate Twitter OAuth 2.0 login.

    Generates a PKCE code challenge and redirects the user to Twitter's
    authorization page.
    """
    config = _get_twitter_config(request)
    if not config["client_id"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Twitter client_id not configured. "
                   "Set TWITTER_CLIENT_ID environment variable.",
        )

    # Generate PKCE code verifier and challenge
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b"=").decode()

    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)
    _pending_states[state] = {
        "code_verifier": code_verifier,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    params = {
        "response_type": "code",
        "client_id": config["client_id"],
        "redirect_uri": config["callback_url"],
        "scope": "tweet.read users.read offline.access",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }

    authorize_url = f"{_TWITTER_AUTHORIZE_URL}?{urlencode(params)}"
    log.info("twitter_oauth.login_redirect", client_id=config["client_id"][:8] + "...")
    return RedirectResponse(url=authorize_url)


# ---------------------------------------------------------------------------
# Step 2: Handle callback from Twitter
# ---------------------------------------------------------------------------

@router.get("/callback")
async def twitter_callback(
    request: Request,
    code: str = Query(..., description="Authorization code from Twitter"),
    state: str = Query(..., description="State for CSRF verification"),
) -> RedirectResponse:
    """Handle OAuth 2.0 callback from Twitter.

    Exchanges the authorization code for access/refresh tokens, fetches the
    user profile, stores credentials, and redirects to the dashboard.
    """
    # Verify state
    pending = _pending_states.pop(state, None)
    if pending is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state. Please try logging in again.",
        )

    code_verifier = pending["code_verifier"]
    config = _get_twitter_config(request)

    # Exchange code for tokens
    import httpx

    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": config["callback_url"],
        "code_verifier": code_verifier,
    }

    # Twitter OAuth 2.0 with confidential client: use Basic Auth
    # (client_id:client_secret) and do NOT include client_id in body
    if config["client_secret"]:
        auth = (config["client_id"], config["client_secret"])
    else:
        # Public client: include client_id in body, no Basic Auth
        token_data["client_id"] = config["client_id"]
        auth = None

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            token_resp = await client.post(
                _TWITTER_TOKEN_URL,
                data=token_data,
                auth=auth,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
    except httpx.RequestError as exc:
        log.error("twitter_oauth.token_exchange_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to reach Twitter token endpoint",
        )

    if token_resp.status_code != 200:
        log.error(
            "twitter_oauth.token_error",
            status=token_resp.status_code,
            body=token_resp.text[:500],
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Twitter rejected the authorization code. Please try again.",
        )

    tokens = token_resp.json()
    access_token = tokens.get("access_token", "")
    refresh_token = tokens.get("refresh_token", "")

    # Fetch user profile
    user_info: Dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            user_resp = await client.get(
                _TWITTER_USER_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                params={"user.fields": "id,name,username,profile_image_url"},
            )
        if user_resp.status_code == 200:
            user_info = user_resp.json().get("data", {})
    except Exception as exc:
        log.warning("twitter_oauth.user_fetch_failed", error=str(exc))

    x_user_id = user_info.get("id", "")
    username = user_info.get("username", "")
    display_name = user_info.get("name", "")

    log.info(
        "twitter_oauth.authenticated",
        x_user_id=x_user_id,
        username=username,
    )

    # Store credentials in the social stream manager
    social_manager = getattr(request.app.state, "social_stream_manager", None)
    if social_manager:
        try:
            social_manager.add_account(
                platform="twitter",
                credentials={
                    "bearer_token": access_token,
                    "refresh_token": refresh_token,
                },
            )
            log.info("twitter_oauth.account_added_to_social_manager")
        except Exception:
            log.warning("twitter_oauth.social_manager_add_failed", exc_info=True)

    # Store in MongoDB for the data source manager
    db = getattr(request.app.state, "db", None)
    if db:
        try:
            from hedgefund.auth.models import encrypt_credentials
            encrypted = encrypt_credentials({
                "access_token": access_token,
                "refresh_token": refresh_token,
            })
            # Get the logged-in user_id from request state (set by AuthMiddleware)
            app_user_id = getattr(request.state, "user_id", None) or ""

            await db.x_accounts.update_one(
                {"x_user_id": x_user_id},
                {
                    "$set": {
                        "x_user_id": x_user_id,
                        "user_id": app_user_id,
                        "username": username,
                        "display_name": display_name,
                        "auth_method": "oauth2",
                        "credentials_encrypted": encrypted,
                        "connected_at": datetime.now(timezone.utc),
                        "is_active": True,
                    }
                },
                upsert=True,
            )
            log.info("twitter_oauth.stored_in_db", username=username)
        except Exception:
            log.warning("twitter_oauth.db_store_failed", exc_info=True)

    # Update data source validator
    dsv = getattr(request.app.state, "data_source_validator", None)
    if dsv:
        from hedgefund.engine.data_source_validator import DataSourceStatus
        dsv.update_x_status(DataSourceStatus.CONNECTED, "twitter")

    # Redirect back to dashboard
    return RedirectResponse(url="/data_sources.html")
