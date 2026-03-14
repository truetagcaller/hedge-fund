"""Health check endpoints for liveness and readiness probes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import structlog
from fastapi import APIRouter, Request, Response, status

log = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def liveness() -> Dict[str, Any]:
    """Liveness probe -- confirms the process is running."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ready")
async def readiness(request: Request, response: Response) -> Dict[str, Any]:
    """Readiness probe -- checks critical dependencies.

    Returns 200 when all subsystems are reachable, 503 otherwise.
    """
    checks: Dict[str, Dict[str, Any]] = {}
    all_healthy = True

    # ── Redis check ──────────────────────────────────────────────────────
    checks["redis"] = await _check_redis(request)
    if checks["redis"]["status"] != "ok":
        all_healthy = False

    # ── Broker connection check ──────────────────────────────────────────
    checks["broker"] = await _check_broker(request)
    if checks["broker"]["status"] != "ok":
        all_healthy = False

    # ── Data feed check ──────────────────────────────────────────────────
    checks["data_feed"] = await _check_data_feed(request)
    if checks["data_feed"]["status"] != "ok":
        all_healthy = False

    if not all_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if all_healthy else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }


async def _check_redis(request: Request) -> Dict[str, Any]:
    """Ping Redis and report connectivity."""
    try:
        redis_client = getattr(request.app.state, "redis", None)
        if redis_client is None:
            return {"status": "ok", "detail": "redis not configured (optional)"}
        await redis_client.ping()
        return {"status": "ok"}
    except Exception as exc:
        log.warning("redis_health_check_failed", error=str(exc))
        return {"status": "error", "detail": str(exc)}


async def _check_broker(request: Request) -> Dict[str, Any]:
    """Verify broker adapter is connected."""
    try:
        broker = getattr(request.app.state, "broker", None)
        if broker is None:
            return {"status": "ok", "detail": "broker not attached"}
        connected = await broker.is_connected()
        return {"status": "ok" if connected else "error"}
    except Exception as exc:
        log.warning("broker_health_check_failed", error=str(exc))
        return {"status": "error", "detail": str(exc)}


async def _check_data_feed(request: Request) -> Dict[str, Any]:
    """Verify data feed is delivering fresh data."""
    try:
        data_feed = getattr(request.app.state, "data_feed", None)
        if data_feed is None:
            return {"status": "ok", "detail": "data feed not attached"}
        is_fresh = await data_feed.is_healthy()
        return {"status": "ok" if is_fresh else "error", "detail": "stale data" if not is_fresh else None}
    except Exception as exc:
        log.warning("data_feed_health_check_failed", error=str(exc))
        return {"status": "error", "detail": str(exc)}
