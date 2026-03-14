"""Redis-backed cache for market data with TTL support.

All values are serialised with :pypi:`orjson` for speed.  The cache is fully
async and safe for concurrent access from multiple coroutines.
"""

from __future__ import annotations

from typing import Any, Optional

import orjson
import redis.asyncio as aioredis

from hedgefund.config.schema import RedisConfig
from hedgefund.logger import get_logger

log = get_logger(__name__)

_KEY_PREFIX = "hf:"


class RedisCache:
    """Async Redis cache with automatic key-prefixing and JSON ser/de.

    Parameters
    ----------
    config:
        :class:`~hedgefund.config.schema.RedisConfig` instance.
    key_prefix:
        Namespace prefix applied to all keys (default ``hf:``).
    default_ttl:
        Default time-to-live in seconds when none is specified per-call.
    """

    def __init__(
        self,
        config: RedisConfig,
        key_prefix: str = _KEY_PREFIX,
        default_ttl: int = 30,
    ) -> None:
        self._config = config
        self._prefix = key_prefix
        self._default_ttl = default_ttl
        self._pool: Optional[aioredis.Redis] = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the Redis connection pool."""
        self._pool = aioredis.Redis(
            host=self._config.host,
            port=self._config.port,
            db=self._config.db,
            password=self._config.password or None,
            ssl=self._config.ssl,
            decode_responses=False,
        )
        # Verify connectivity.
        await self._pool.ping()
        log.info(
            "redis_connected",
            host=self._config.host,
            port=self._config.port,
            db=self._config.db,
        )

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.aclose()  # type: ignore[attr-defined]
            self._pool = None
            log.info("redis_disconnected")

    async def __aenter__(self) -> "RedisCache":
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.disconnect()

    # ── internal helpers ───────────────────────────────────────────────────

    def _fqkey(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def _ensure_pool(self) -> aioredis.Redis:
        if self._pool is None:
            raise RuntimeError("RedisCache is not connected. Call .connect() first.")
        return self._pool

    # ── public API ─────────────────────────────────────────────────────────

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve and deserialise a cached value, or ``None`` on miss."""
        pool = self._ensure_pool()
        raw: Optional[bytes] = await pool.get(self._fqkey(key))
        if raw is None:
            return None
        try:
            return orjson.loads(raw)
        except orjson.JSONDecodeError:
            log.warning("cache_decode_error", key=key)
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Serialise and store *value* with an optional TTL (seconds)."""
        pool = self._ensure_pool()
        ttl = ttl if ttl is not None else self._default_ttl
        data = orjson.dumps(value)
        await pool.set(self._fqkey(key), data, ex=ttl)

    async def delete(self, key: str) -> bool:
        """Remove a key. Returns ``True`` if the key existed."""
        pool = self._ensure_pool()
        removed: int = await pool.delete(self._fqkey(key))
        return removed > 0

    async def exists(self, key: str) -> bool:
        """Check whether *key* is present in the cache."""
        pool = self._ensure_pool()
        return bool(await pool.exists(self._fqkey(key)))

    async def get_or_set(
        self,
        key: str,
        factory: Any,
        ttl: Optional[int] = None,
    ) -> Any:
        """Return cached value or compute via *factory*, cache, and return.

        *factory* should be an async callable (``async def``) returning the
        value to cache.
        """
        cached = await self.get(key)
        if cached is not None:
            return cached
        value = await factory()
        await self.set(key, value, ttl=ttl)
        return value

    async def flush_prefix(self, prefix: str = "") -> int:
        """Delete all keys matching ``{self._prefix}{prefix}*``.

        Returns the number of keys deleted.
        """
        pool = self._ensure_pool()
        pattern = f"{self._prefix}{prefix}*"
        count = 0
        async for key in pool.scan_iter(match=pattern, count=500):
            await pool.delete(key)
            count += 1
        log.info("cache_flushed", pattern=pattern, deleted=count)
        return count
