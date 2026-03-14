"""Broker session manager — persistent session tracking via Redis.

Maintains broker connection state in Redis so that sessions survive
server restarts and can be shared across workers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from hedgefund.logger import get_logger

log = get_logger(__name__)

_SESSION_PREFIX = "hedgefund:broker_session:"
_CONTEXT_PREFIX = "hedgefund:exec_context:"
_SESSION_TTL = 86400  # 24 hours


class BrokerSessionManager:
    """Track active broker sessions and execution contexts in Redis.

    Parameters
    ----------
    redis:
        An ``aioredis``-compatible async Redis client.  If ``None``,
        the manager operates in memory-only mode (no persistence).
    """

    def __init__(self, redis: Any | None = None) -> None:
        self._redis = redis
        self._local_sessions: dict[str, dict[str, Any]] = {}
        self._local_contexts: dict[str, dict[str, Any]] = {}

    # ── Session management ─────────────────────────────────────────────

    async def register_session(
        self,
        user_id: str,
        broker_id: str,
        broker_type: str,
        *,
        account_info: dict[str, Any] | None = None,
    ) -> None:
        """Register an active broker session."""
        session = {
            "user_id": user_id,
            "broker_id": broker_id,
            "broker_type": broker_type,
            "status": "active",
            "connected_at": datetime.now(timezone.utc).isoformat(),
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            "account_info": account_info or {},
        }

        key = f"{_SESSION_PREFIX}{user_id}:{broker_id}"
        self._local_sessions[f"{user_id}:{broker_id}"] = session

        if self._redis is not None:
            await self._redis.setex(
                key, _SESSION_TTL, json.dumps(session),
            )

        log.info(
            "session_manager.registered",
            user_id=user_id,
            broker_id=broker_id,
        )

    async def heartbeat(self, user_id: str, broker_id: str) -> None:
        """Update the heartbeat timestamp for a session."""
        local_key = f"{user_id}:{broker_id}"
        session = self._local_sessions.get(local_key)
        if session:
            session["last_heartbeat"] = datetime.now(timezone.utc).isoformat()

        if self._redis is not None:
            key = f"{_SESSION_PREFIX}{user_id}:{broker_id}"
            raw = await self._redis.get(key)
            if raw:
                data = json.loads(raw)
                data["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
                await self._redis.setex(key, _SESSION_TTL, json.dumps(data))

    async def remove_session(self, user_id: str, broker_id: str) -> None:
        """Remove a broker session."""
        self._local_sessions.pop(f"{user_id}:{broker_id}", None)

        if self._redis is not None:
            key = f"{_SESSION_PREFIX}{user_id}:{broker_id}"
            await self._redis.delete(key)

        log.info(
            "session_manager.removed",
            user_id=user_id,
            broker_id=broker_id,
        )

    async def get_session(
        self, user_id: str, broker_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve a session."""
        local_key = f"{user_id}:{broker_id}"
        local = self._local_sessions.get(local_key)
        if local is not None:
            return local

        if self._redis is not None:
            key = f"{_SESSION_PREFIX}{user_id}:{broker_id}"
            raw = await self._redis.get(key)
            if raw:
                data = json.loads(raw)
                self._local_sessions[local_key] = data
                return data

        return None

    async def get_user_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """Return all active sessions for a user."""
        sessions = [
            s for key, s in self._local_sessions.items()
            if key.startswith(f"{user_id}:")
        ]

        if self._redis is not None and not sessions:
            pattern = f"{_SESSION_PREFIX}{user_id}:*"
            keys = []
            async for key in self._redis.scan_iter(pattern):
                keys.append(key)
            for key in keys:
                raw = await self._redis.get(key)
                if raw:
                    sessions.append(json.loads(raw))

        return sessions

    # ── Execution context persistence ─────────────────────────────────

    async def save_execution_context(
        self, user_id: str, context: dict[str, Any],
    ) -> None:
        """Persist the user's current execution context."""
        self._local_contexts[user_id] = context

        if self._redis is not None:
            key = f"{_CONTEXT_PREFIX}{user_id}"
            await self._redis.setex(key, _SESSION_TTL, json.dumps(context))

    async def get_execution_context(
        self, user_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve the user's execution context."""
        local = self._local_contexts.get(user_id)
        if local is not None:
            return local

        if self._redis is not None:
            key = f"{_CONTEXT_PREFIX}{user_id}"
            raw = await self._redis.get(key)
            if raw:
                data = json.loads(raw)
                self._local_contexts[user_id] = data
                return data

        return None
