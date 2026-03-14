"""JWT token creation and verification for the hedge fund auth system.

Supports both access tokens (short-lived, carry user claims) and refresh
tokens (longer-lived, used to obtain new access tokens).
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import jwt as pyjwt

from hedgefund.logger import get_logger

log = get_logger(__name__)

_HEDGEFUND_DIR = Path.home() / ".hedgefund"
_JWT_SECRET_FILE = _HEDGEFUND_DIR / ".jwt_secret"


def _resolve_secret() -> str:
    """Resolve the JWT signing secret.

    Order of precedence:
    1. ``HEDGEFUND_JWT_SECRET`` environment variable.
    2. Persisted file at ``~/.hedgefund/.jwt_secret``.
    3. Generate a random secret, persist it, and return.
    """
    env_secret = os.environ.get("HEDGEFUND_JWT_SECRET")
    if env_secret:
        return env_secret

    if _JWT_SECRET_FILE.exists():
        return _JWT_SECRET_FILE.read_text().strip()

    # Generate and persist
    _HEDGEFUND_DIR.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_urlsafe(64)
    _JWT_SECRET_FILE.write_text(secret)
    os.chmod(_JWT_SECRET_FILE, 0o600)
    log.info("jwt_handler.secret_generated", path=str(_JWT_SECRET_FILE))
    return secret


class JWTHandler:
    """Create and verify JWT access / refresh tokens."""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_expire_minutes: int = 60,
        refresh_expire_days: int = 7,
    ) -> None:
        self._secret = secret_key or _resolve_secret()
        self._algorithm = algorithm
        self._access_expire_minutes = access_expire_minutes
        self._refresh_expire_days = refresh_expire_days

    # -- token creation ------------------------------------------------------

    def create_access_token(
        self,
        user_id: str,
        email: str,
        role: str,
    ) -> str:
        """Create a short-lived access token carrying user claims."""
        now = datetime.now(timezone.utc)
        payload: Dict[str, Any] = {
            "sub": user_id,
            "email": email,
            "role": role,
            "exp": now + timedelta(minutes=self._access_expire_minutes),
            "iat": now,
            "type": "access",
        }
        token = pyjwt.encode(payload, self._secret, algorithm=self._algorithm)
        log.info("jwt_handler.access_token_created", user_id=user_id)
        return token

    def create_refresh_token(self, user_id: str) -> str:
        """Create a longer-lived refresh token."""
        now = datetime.now(timezone.utc)
        payload: Dict[str, Any] = {
            "sub": user_id,
            "exp": now + timedelta(days=self._refresh_expire_days),
            "iat": now,
            "type": "refresh",
        }
        token = pyjwt.encode(payload, self._secret, algorithm=self._algorithm)
        log.info("jwt_handler.refresh_token_created", user_id=user_id)
        return token

    # -- token verification --------------------------------------------------

    def verify_token(self, token: str) -> dict:
        """Verify and decode a JWT token.

        Raises ``pyjwt.ExpiredSignatureError`` or ``pyjwt.InvalidTokenError``
        on failure.
        """
        payload = pyjwt.decode(
            token,
            self._secret,
            algorithms=[self._algorithm],
        )
        return payload

    def decode_token(self, token: str) -> Optional[dict]:
        """Decode a token without raising on errors.

        Returns the payload dict on success, ``None`` on any failure
        (expired, invalid signature, malformed, etc.).
        """
        try:
            return self.verify_token(token)
        except (pyjwt.ExpiredSignatureError, pyjwt.InvalidTokenError):
            return None
