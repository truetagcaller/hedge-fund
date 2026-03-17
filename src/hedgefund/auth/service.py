"""Authentication service -- business logic for user management.

All database interaction goes through the ``MongoDB`` wrapper and all token
operations through ``JWTHandler``.  Passwords and tokens are **never**
logged.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Optional

from bson import ObjectId

from hedgefund.auth.database import MongoDB
from hedgefund.auth.jwt_handler import JWTHandler
from hedgefund.auth.models import create_user_document, verify_password
from hedgefund.logger import get_logger

log = get_logger(__name__)

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
_MIN_PASSWORD_LENGTH = 8


def _strip_password(doc: dict) -> dict:
    """Return a copy of a user document without the password hash."""
    if doc is None:
        return doc  # type: ignore[return-value]
    out = dict(doc)
    out.pop("password_hash", None)
    # Convert ObjectId to str for JSON serialisation
    if "_id" in out:
        out["_id"] = str(out["_id"])
    return out


class AuthService:
    """High-level authentication and user management operations."""

    def __init__(self, db: MongoDB, jwt: JWTHandler) -> None:
        self._db = db
        self._jwt = jwt

    # -- registration --------------------------------------------------------

    async def register(
        self,
        email: str,
        username: str,
        password: str,
    ) -> dict:
        """Register a new user account.

        Validates email format and password strength, checks uniqueness,
        creates the user document and returns it (minus the password hash).
        """
        email = email.lower().strip()
        username = username.strip()

        # Validation
        if not _EMAIL_RE.match(email):
            raise ValueError("Invalid email format")

        if len(password) < _MIN_PASSWORD_LENGTH:
            raise ValueError(
                f"Password must be at least {_MIN_PASSWORD_LENGTH} characters"
            )

        # Uniqueness checks
        existing = await self._db.users.find_one(
            {"$or": [{"email": email}, {"username": username}]}
        )
        if existing is not None:
            if existing["email"] == email:
                raise ValueError("Email already registered")
            raise ValueError("Username already taken")

        doc = create_user_document(email, username, password)
        result = await self._db.users.insert_one(doc)
        doc["_id"] = result.inserted_id

        log.info("auth.user_registered", user_id=str(doc["_id"]), email=email)
        return _strip_password(doc)

    # -- login / token management -------------------------------------------

    async def login(self, email: str, password: str) -> dict:
        """Authenticate a user and return tokens plus user info.

        Returns a dict with keys ``access_token``, ``refresh_token``, and
        ``user``.
        """
        email = email.lower().strip()
        user = await self._db.users.find_one({"email": email})

        if user is None or not verify_password(password, user["password_hash"]):
            log.warning("auth.login_failed", reason="invalid_credentials")
            raise ValueError("Invalid email or password")

        if not user.get("is_active", True):
            log.warning(
                "auth.login_failed",
                reason="inactive_account",
                user_id=str(user["_id"]),
            )
            raise ValueError("Account is deactivated")

        user_id = str(user["_id"])

        # Update last_login
        await self._db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.now(timezone.utc)}},
        )

        access_token = self._jwt.create_access_token(
            user_id=user_id,
            email=user["email"],
            role=user["role"],
        )
        refresh_token = self._jwt.create_refresh_token(user_id=user_id)

        log.info("auth.login_success", user_id=user_id)
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user": _strip_password(user),
        }

    async def refresh(self, refresh_token: str) -> dict:
        """Exchange a valid refresh token for a new access token."""
        payload = self._jwt.decode_token(refresh_token)

        if payload is None or payload.get("type") != "refresh":
            raise ValueError("Invalid refresh token")

        user_id = payload["sub"]
        user = await self._db.users.find_one({"_id": ObjectId(user_id)})
        if user is None or not user.get("is_active", True):
            raise ValueError("User not found or inactive")

        access_token = self._jwt.create_access_token(
            user_id=user_id,
            email=user["email"],
            role=user["role"],
        )
        log.info("auth.token_refreshed", user_id=user_id)
        return {"access_token": access_token}

    # -- user queries --------------------------------------------------------

    async def get_user(self, user_id: str) -> Optional[dict]:
        """Retrieve a user document by id (without password hash)."""
        try:
            doc = await self._db.users.find_one({"_id": ObjectId(user_id)})
        except Exception:
            return None
        if doc is None:
            return None
        return _strip_password(doc)

    async def update_preferences(
        self,
        user_id: str,
        preferences: dict,
    ) -> dict:
        """Merge new preferences into the user's existing preferences."""
        result = await self._db.users.find_one_and_update(
            {"_id": ObjectId(user_id)},
            {"$set": {f"preferences.{k}": v for k, v in preferences.items()}},
            return_document=True,
        )
        if result is None:
            raise ValueError("User not found")
        log.info("auth.preferences_updated", user_id=user_id)
        return _strip_password(result)

    async def change_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str,
    ) -> None:
        """Change a user's password after verifying the old one."""
        if len(new_password) < _MIN_PASSWORD_LENGTH:
            raise ValueError(
                f"New password must be at least {_MIN_PASSWORD_LENGTH} characters"
            )

        user = await self._db.users.find_one({"_id": ObjectId(user_id)})
        if user is None:
            raise ValueError("User not found")

        if not verify_password(old_password, user["password_hash"]):
            raise ValueError("Current password is incorrect")

        import bcrypt

        salt = bcrypt.gensalt()
        new_hash = bcrypt.hashpw(new_password.encode(), salt).decode()

        await self._db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"password_hash": new_hash}},
        )
        log.info("auth.password_changed", user_id=user_id)

    # -- admin operations ----------------------------------------------------

    async def list_users(self) -> List[dict]:
        """List all users (admin only). Password hashes are excluded."""
        cursor = self._db.users.find({}, {"password_hash": 0})
        users = await cursor.to_list(length=None)
        return [_strip_password(u) for u in users]

    async def deactivate_user(self, user_id: str) -> dict:
        """Mark a user account as inactive."""
        result = await self._db.users.find_one_and_update(
            {"_id": ObjectId(user_id)},
            {"$set": {"is_active": False}},
            return_document=True,
        )
        if result is None:
            raise ValueError("User not found")
        log.info("auth.user_deactivated", user_id=user_id)
        return _strip_password(result)
