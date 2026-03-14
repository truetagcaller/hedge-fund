"""FastAPI authentication middleware and dependency functions.

Provides:
- ``AuthMiddleware`` -- ASGI middleware that extracts JWT claims from the
  ``Authorization`` header or ``access_token`` cookie and stores them on
  ``request.state``.
- Dependency functions (``get_current_user``, ``require_admin``, etc.) for
  per-route protection.
"""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from hedgefund.auth.jwt_handler import JWTHandler, _resolve_secret
from hedgefund.logger import get_logger

log = get_logger(__name__)

# Module-level JWTHandler (lazily initialised so imports don't fail without
# env vars set).
_jwt: Optional[JWTHandler] = None


def _get_jwt() -> JWTHandler:
    global _jwt
    if _jwt is None:
        _jwt = JWTHandler()
    return _jwt


def _extract_token(request: Request) -> Optional[str]:
    """Try to extract a JWT from the request.

    Checks:
    1. ``Authorization: Bearer <token>`` header
    2. ``access_token`` cookie
    """
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]

    return request.cookies.get("access_token")


# ---------------------------------------------------------------------------
# ASGI Middleware
# ---------------------------------------------------------------------------


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that populates ``request.state`` with user claims.

    If a valid token is found, ``request.state.user_id`` and
    ``request.state.user_role`` are set.  If the token is missing or
    invalid the request is still allowed through -- individual routes use
    the dependency functions below to enforce authentication.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request.state.user_id = None
        request.state.user_role = None

        token = _extract_token(request)
        if token:
            handler = _get_jwt()
            payload = handler.decode_token(token)
            if payload and payload.get("type") == "access":
                request.state.user_id = payload.get("sub")
                request.state.user_role = payload.get("role")

        response = await call_next(request)
        return response


# ---------------------------------------------------------------------------
# Dependency functions
# ---------------------------------------------------------------------------


async def get_current_user(request: Request) -> dict:
    """FastAPI dependency: extract the authenticated user from the request.

    Raises ``HTTPException(401)`` when no valid token is present.
    """
    token = _extract_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    handler = _get_jwt()
    payload = handler.decode_token(token)
    if payload is None or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "user_id": payload["sub"],
        "email": payload.get("email"),
        "role": payload.get("role"),
    }


async def get_current_user_optional(request: Request) -> Optional[dict]:
    """Like ``get_current_user`` but returns ``None`` instead of raising."""
    try:
        return await get_current_user(request)
    except HTTPException:
        return None


async def require_admin(request: Request) -> dict:
    """FastAPI dependency: require an authenticated admin user.

    Raises ``HTTPException(401)`` if not authenticated, ``HTTPException(403)``
    if authenticated but not an admin.
    """
    user = await get_current_user(request)
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


async def get_user_id(request: Request) -> str:
    """Shortcut dependency that returns just the ``user_id`` string."""
    user = await get_current_user(request)
    return user["user_id"]
