"""Authentication REST endpoints.

Provides registration, login, token refresh, logout, profile management,
and admin user listing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field

from hedgefund.auth.middleware import get_current_user, require_admin
from hedgefund.auth.service import AuthService

log = structlog.get_logger(__name__)

router = APIRouter(tags=["auth"])


# ── Pydantic request / response models ────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class RefreshRequest(BaseModel):
    refresh_token: Optional[str] = None


class RefreshResponse(BaseModel):
    access_token: str


class LogoutResponse(BaseModel):
    status: str = "logged_out"


class UpdatePreferencesRequest(BaseModel):
    preferences: Dict[str, Any]


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)


class ChangePasswordResponse(BaseModel):
    status: str = "changed"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_auth_service(request: Request) -> AuthService:
    """Extract AuthService from app state."""
    svc = getattr(request.app.state, "auth_service", None)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth service not initialized",
        )
    return svc


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(request: Request, body: RegisterRequest) -> Dict[str, Any]:
    """Register a new user account."""
    auth_service = _get_auth_service(request)

    try:
        result = await auth_service.register(
            email=body.email,
            username=body.username,
            password=body.password,
        )
    except ValueError as exc:
        detail = str(exc)
        if "already" in detail.lower():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    log.info("auth.registered", email=body.email, username=body.username)

    # Map the returned user doc to the expected response shape
    return {
        "user_id": result.get("_id", ""),
        "email": result.get("email", ""),
        "username": result.get("username", ""),
        "role": result.get("role", "user"),
        "created_at": result["created_at"].isoformat()
        if hasattr(result.get("created_at"), "isoformat")
        else str(result.get("created_at", "")),
    }


@router.post("/auth/login")
async def login(request: Request, response: Response, body: LoginRequest) -> Dict[str, Any]:
    """Authenticate user and return access + refresh tokens."""
    auth_service = _get_auth_service(request)

    try:
        result = await auth_service.login(email=body.email, password=body.password)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Set HttpOnly cookie with the access token
    response.set_cookie(
        key="access_token",
        value=result["access_token"],
        httponly=True,
        samesite="lax",
        secure=False,  # Set True in production with HTTPS
        max_age=3600,
    )

    user_doc = result["user"]
    return {
        "access_token": result["access_token"],
        "refresh_token": result["refresh_token"],
        "user": {
            "user_id": user_doc.get("_id", ""),
            "email": user_doc.get("email", ""),
            "username": user_doc.get("username", ""),
            "role": user_doc.get("role", "user"),
        },
    }


@router.post("/auth/refresh")
async def refresh(request: Request, body: RefreshRequest) -> RefreshResponse:
    """Refresh the access token using a valid refresh token."""
    auth_service = _get_auth_service(request)

    # Try body first, then cookie
    refresh_token = body.refresh_token
    if not refresh_token:
        refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="refresh_token is required",
        )

    try:
        result = await auth_service.refresh(refresh_token)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    return RefreshResponse(access_token=result["access_token"])


@router.post("/auth/logout")
async def logout(response: Response) -> LogoutResponse:
    """Clear the access_token cookie."""
    response.delete_cookie(key="access_token", httponly=True, samesite="lax")
    return LogoutResponse()


@router.get("/auth/me")
async def get_me(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return the current user's profile."""
    auth_service = _get_auth_service(request)
    user_id = user["user_id"]

    profile = await auth_service.get_user(user_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return {
        "user_id": profile.get("_id", user_id),
        "email": profile.get("email", ""),
        "username": profile.get("username", ""),
        "role": profile.get("role", "user"),
        "preferences": profile.get("preferences", {}),
        "created_at": profile["created_at"].isoformat()
        if hasattr(profile.get("created_at"), "isoformat")
        else str(profile.get("created_at", "")),
        "last_login": profile["last_login"].isoformat()
        if hasattr(profile.get("last_login"), "isoformat")
        else profile.get("last_login"),
    }


@router.put("/auth/me/preferences")
async def update_preferences(
    request: Request,
    body: UpdatePreferencesRequest,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Update current user's preferences."""
    auth_service = _get_auth_service(request)
    user_id = user["user_id"]

    try:
        updated = await auth_service.update_preferences(user_id, body.preferences)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return {
        "user_id": updated.get("_id", user_id),
        "email": updated.get("email", ""),
        "username": updated.get("username", ""),
        "role": updated.get("role", "user"),
        "preferences": updated.get("preferences", {}),
        "created_at": updated["created_at"].isoformat()
        if hasattr(updated.get("created_at"), "isoformat")
        else str(updated.get("created_at", "")),
        "last_login": updated["last_login"].isoformat()
        if hasattr(updated.get("last_login"), "isoformat")
        else updated.get("last_login"),
    }


@router.put("/auth/me/password")
async def change_password(
    request: Request,
    body: ChangePasswordRequest,
    user: dict = Depends(get_current_user),
) -> ChangePasswordResponse:
    """Change the current user's password."""
    auth_service = _get_auth_service(request)
    user_id = user["user_id"]

    try:
        await auth_service.change_password(
            user_id=user_id,
            old_password=body.old_password,
            new_password=body.new_password,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return ChangePasswordResponse()


@router.get("/admin/users")
async def list_users(
    request: Request,
    admin: dict = Depends(require_admin),
) -> Dict[str, Any]:
    """List all users (admin only)."""
    auth_service = _get_auth_service(request)
    users = await auth_service.list_users()
    return {"users": users}
