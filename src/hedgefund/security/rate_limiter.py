"""Token-bucket rate limiter for external API calls.

Provides per-service rate limiting with pre-configured defaults for
Zerodha, Binance, and Twitter (X) API v2.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict

from hedgefund.logger import get_logger

log = get_logger(__name__)


@dataclass
class _Bucket:
    """Token bucket state for a single rate-limit window."""

    max_tokens: int
    window_seconds: float
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = float(self.max_tokens)
        self.last_refill = time.monotonic()

    @property
    def refill_rate(self) -> float:
        """Tokens added per second."""
        return self.max_tokens / self.window_seconds

    def refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def try_acquire(self) -> bool:
        self.refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def time_until_available(self) -> float:
        """Seconds until at least one token is available."""
        self.refill()
        if self.tokens >= 1.0:
            return 0.0
        deficit = 1.0 - self.tokens
        return deficit / self.refill_rate


@dataclass
class _RateLimitConfig:
    """Holds the list of buckets for a named service.

    A service can have multiple windows (e.g. 3 req/sec AND 200 req/min).
    """

    name: str
    buckets: list[_Bucket] = field(default_factory=list)


class RateLimiter:
    """Async-friendly token-bucket rate limiter.

    Supports multiple overlapping windows per service (e.g. per-second
    *and* per-minute limits).

    Pre-configured services:

    * **zerodha** -- 3 req/sec, 200 req/min
    * **binance** -- 10 req/sec, 1200 req/min
    * **twitter** -- 15 req/15 min (API v2 app-level rate limit)

    Example::

        limiter = RateLimiter()
        await limiter.acquire("zerodha")  # blocks until a token is available
    """

    def __init__(self) -> None:
        self._services: Dict[str, _RateLimitConfig] = {}
        self._lock = asyncio.Lock()
        self._setup_defaults()

    # -- configuration ------------------------------------------------------

    def _setup_defaults(self) -> None:
        self.configure("zerodha", max_requests=3, window_seconds=1)
        self.configure("zerodha", max_requests=200, window_seconds=60)

        self.configure("binance", max_requests=10, window_seconds=1)
        self.configure("binance", max_requests=1200, window_seconds=60)

        self.configure("twitter", max_requests=15, window_seconds=900)

    def configure(self, name: str, *, max_requests: int, window_seconds: int) -> None:
        """Add a rate-limit window for *name*.

        Can be called multiple times for the same service to layer windows
        (e.g. per-second + per-minute).
        """
        bucket = _Bucket(max_tokens=max_requests, window_seconds=float(window_seconds))
        if name not in self._services:
            self._services[name] = _RateLimitConfig(name=name)
        self._services[name].buckets.append(bucket)
        log.info(
            "rate_limiter.configured",
            service=name,
            max_requests=max_requests,
            window_seconds=window_seconds,
        )

    # -- public API ---------------------------------------------------------

    async def acquire(self, name: str) -> None:
        """Wait until a request token is available for *name*.

        Blocks (via ``asyncio.sleep``) if the service is currently
        rate-limited, retrying until all windows have capacity.
        """
        if name not in self._services:
            raise KeyError(f"Rate limiter not configured for service: {name!r}")

        while True:
            async with self._lock:
                config = self._services[name]
                # Check ALL buckets -- we need a token from every window.
                all_ok = all(b.try_acquire() for b in config.buckets)
                if all_ok:
                    return

                # Find the longest wait across buckets.
                # Re-refill first since try_acquire consumed partial tokens.
                max_wait = max(b.time_until_available() for b in config.buckets)

            log.debug("rate_limiter.waiting", service=name, wait_seconds=round(max_wait, 3))
            await asyncio.sleep(max_wait)

    def is_limited(self, name: str) -> bool:
        """Return ``True`` if *name* is currently rate-limited (no tokens)."""
        config = self._services.get(name)
        if config is None:
            raise KeyError(f"Rate limiter not configured for service: {name!r}")
        for bucket in config.buckets:
            bucket.refill()
            if bucket.tokens < 1.0:
                return True
        return False

    def get_remaining(self, name: str) -> int:
        """Return the minimum available tokens across all windows for *name*."""
        config = self._services.get(name)
        if config is None:
            raise KeyError(f"Rate limiter not configured for service: {name!r}")
        remaining = float("inf")
        for bucket in config.buckets:
            bucket.refill()
            remaining = min(remaining, bucket.tokens)
        return int(remaining)
