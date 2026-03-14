"""FastAPI routes for managing news and social data sources with multi-user isolation.

Provides endpoints for configuring data sources, viewing live news feeds,
querying real-time sentiment, and tracking trending tickers.
All data source configuration is scoped to the authenticated user.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from hedgefund.auth.middleware import get_current_user
from hedgefund.streaming.data_source_manager import DataSourceManager
from hedgefund.streaming.news_stream import NewsStreamManager
from hedgefund.streaming.social_stream import SocialStreamManager

log = structlog.get_logger(__name__)

router = APIRouter(tags=["data-sources"])


# ── Pydantic models ──────────────────────────────────────────────────────────


class DataSourceConfig(BaseModel):
    name: str = Field(..., description="Human-readable name for the source")
    url: Optional[str] = Field(None, description="Source URL (for RSS, API, calendar)")
    base_url: Optional[str] = Field(None, description="Base URL (for news APIs)")
    poll_interval: Optional[int] = Field(None, description="Poll interval in seconds")
    tickers: Optional[List[str]] = Field(None, description="Tickers to track (Twitter)")
    credentials: Optional[Dict[str, str]] = Field(None, description="Source credentials")


class ConnectSourceRequest(BaseModel):
    source_type: str = Field(
        ...,
        description="Source type: rss, news_api, twitter, economic_calendar, earnings",
    )
    name: str = Field(..., description="Display name for this source")
    credentials: Optional[Dict[str, str]] = Field(
        default=None,
        description="Credentials (bearer_token, api_key, etc.)",
    )
    url: Optional[str] = Field(None, description="Source URL")
    base_url: Optional[str] = Field(None, description="Base API URL")
    poll_interval: Optional[int] = Field(None, description="Poll interval in seconds")
    tickers: Optional[List[str]] = Field(None, description="Tickers to track")


class ConnectSourceResponse(BaseModel):
    source_id: str
    status: str
    message: str


class SourceInfo(BaseModel):
    id: str
    type: str
    name: str
    status: str
    last_update: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class SourceListResponse(BaseModel):
    sources: List[SourceInfo]


class SupportedSourceType(BaseModel):
    type: str
    name: str
    auth_methods: List[str]
    fields: List[str]


class SupportedSourcesResponse(BaseModel):
    types: List[SupportedSourceType]


class NewsItemResponse(BaseModel):
    title: str
    summary: str
    source: str
    tickers: List[str]
    sentiment: float
    impact: str
    timestamp: str
    url: str = ""


class NewsFeedResponse(BaseModel):
    news: List[NewsItemResponse]
    count: int


class TickerSentimentResponse(BaseModel):
    score: float
    magnitude: float
    sources_count: int
    trending: bool
    score_1h: float = 0.0
    score_4h: float = 0.0
    score_24h: float = 0.0
    mention_count_1h: int = 0


class LiveSentimentResponse(BaseModel):
    sentiments: Dict[str, TickerSentimentResponse]


class TrendingItem(BaseModel):
    symbol: str
    mentions: int
    sentiment: float
    change_pct: float


class TrendingResponse(BaseModel):
    trending: List[TrendingItem]


class FeedItemResponse(BaseModel):
    items: List[Dict[str, Any]]
    count: int


# ── Helpers ───────────────────────────────────────────────────────────────────

_SUPPORTED_SOURCES: List[Dict[str, Any]] = [
    {
        "type": "rss",
        "name": "RSS Feed",
        "auth_methods": ["none"],
        "fields": ["name", "url", "poll_interval"],
    },
    {
        "type": "news_api",
        "name": "News API",
        "auth_methods": ["api_key", "bearer_token"],
        "fields": ["name", "base_url", "poll_interval", "credentials"],
    },
    {
        "type": "twitter",
        "name": "X (Twitter)",
        "auth_methods": ["bearer_token", "oauth2_user", "session_token"],
        "fields": ["name", "credentials", "tickers"],
    },
    {
        "type": "economic_calendar",
        "name": "Economic Calendar",
        "auth_methods": ["none", "api_key"],
        "fields": ["name", "url", "poll_interval"],
    },
    {
        "type": "earnings",
        "name": "Earnings Announcements",
        "auth_methods": ["none", "api_key"],
        "fields": ["name", "url", "poll_interval", "credentials"],
    },
]


def _get_data_source_manager(request: Request) -> DataSourceManager:
    """Extract DataSourceManager from app state."""
    manager = getattr(request.app.state, "data_source_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Data source manager not initialized.",
        )
    return manager


def _get_news_stream_manager(request: Request) -> NewsStreamManager:
    """Extract NewsStreamManager from app state."""
    manager = getattr(request.app.state, "news_stream_manager", None)
    if manager is None:
        dsm = getattr(request.app.state, "data_source_manager", None)
        if dsm is not None:
            return dsm.news_manager
        raise HTTPException(
            status_code=503,
            detail="News stream manager not initialized.",
        )
    return manager


def _get_social_stream_manager(request: Request) -> SocialStreamManager:
    """Extract SocialStreamManager from app state."""
    manager = getattr(request.app.state, "social_stream_manager", None)
    if manager is None:
        dsm = getattr(request.app.state, "data_source_manager", None)
        if dsm is not None:
            return dsm.social_manager
        raise HTTPException(
            status_code=503,
            detail="Social stream manager not initialized.",
        )
    return manager


def _get_db(request: Request):
    """Extract MongoDB instance from app state, or None."""
    return getattr(request.app.state, "db", None)


# ── Endpoints: Data Sources ───────────────────────────────────────────────────


@router.get("/data-sources")
async def list_data_sources(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """List all configured data sources for the current user."""
    user_id = user["user_id"]
    db = _get_db(request)

    # Try MongoDB first for user-scoped sources
    if db is not None:
        sources: List[Dict[str, Any]] = []
        async for doc in db.data_sources.find(
            {"user_id": user_id},
            {"credentials": 0},  # never expose raw credentials
        ):
            sources.append({
                "id": doc.get("source_id", str(doc["_id"])),
                "type": doc.get("source_type", ""),
                "name": doc.get("name", ""),
                "status": doc.get("status", "unknown"),
                "last_update": doc["last_update"].isoformat()
                if isinstance(doc.get("last_update"), datetime)
                else None,
                "config": doc.get("config", {}),
            })
        if sources:
            return {"sources": sources}

    # Fall back to global data source manager
    manager = _get_data_source_manager(request)
    global_sources = manager.list_sources()
    return {"sources": global_sources}


@router.get("/data-sources/supported")
async def list_supported_sources() -> Dict[str, Any]:
    """List supported data source types with auth methods and fields."""
    return {"types": _SUPPORTED_SOURCES}


@router.post("/data-sources/connect")
async def connect_data_source(
    request: Request,
    body: ConnectSourceRequest,
    user: dict = Depends(get_current_user),
) -> ConnectSourceResponse:
    """Add and connect a new data source for the current user."""
    user_id = user["user_id"]
    db = _get_db(request)
    manager = _get_data_source_manager(request)

    config: Dict[str, Any] = {"name": body.name}
    if body.url:
        config["url"] = body.url
    if body.base_url:
        config["base_url"] = body.base_url
    if body.poll_interval is not None:
        config["poll_interval"] = body.poll_interval
    if body.credentials:
        config["credentials"] = body.credentials
    if body.tickers:
        config["tickers"] = body.tickers

    try:
        source_id = manager.add_source(body.source_type, config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.error("data_source_connect_error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to connect source: {exc}")

    # Persist in MongoDB with user_id
    if db is not None:
        now = datetime.now(timezone.utc)
        await db.data_sources.update_one(
            {"user_id": user_id, "source_id": source_id},
            {
                "$set": {
                    "user_id": user_id,
                    "source_id": source_id,
                    "source_type": body.source_type,
                    "name": body.name,
                    "status": "connected",
                    "config": {k: v for k, v in config.items() if k != "credentials"},
                    "connected_at": now,
                    "last_update": now,
                }
            },
            upsert=True,
        )

    log.info("data_source.connected", user_id=user_id, source_id=source_id)

    return ConnectSourceResponse(
        source_id=source_id,
        status="connected",
        message=f"Successfully connected {body.source_type} source '{body.name}'.",
    )


@router.delete("/data-sources/{source_id}")
async def remove_data_source(
    request: Request,
    source_id: str,
    user: dict = Depends(get_current_user),
) -> Dict[str, str]:
    """Remove a data source for the current user."""
    user_id = user["user_id"]
    db = _get_db(request)
    manager = _get_data_source_manager(request)

    try:
        manager.remove_source(source_id)
    except KeyError:
        pass  # May only exist in DB
    except Exception as exc:
        log.error("data_source_remove_error", source_id=source_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to remove source: {exc}")

    # Remove from MongoDB
    if db is not None:
        result = await db.data_sources.delete_one(
            {"user_id": user_id, "source_id": source_id}
        )
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Source {source_id!r} not found.")

    log.info("data_source.removed", user_id=user_id, source_id=source_id)
    return {"status": "removed", "source_id": source_id}


@router.get("/data-sources/{source_id}/feed")
async def get_source_feed(
    request: Request,
    source_id: str,
    user: dict = Depends(get_current_user),
    limit: int = Query(default=50, ge=1, le=200),
) -> FeedItemResponse:
    """Get recent items from a specific data source."""
    manager = _get_data_source_manager(request)
    sources = {s["id"]: s for s in manager.list_sources()}

    if source_id not in sources:
        raise HTTPException(status_code=404, detail=f"Source {source_id!r} not found.")

    source_info = sources[source_id]
    stype = source_info["type"]

    if stype in ("rss", "news_api", "economic_calendar", "earnings"):
        news_mgr = manager.news_manager
        items = [it.to_dict() for it in news_mgr.get_recent(limit=limit)]
        items = [it for it in items if it.get("source", "") == source_info.get("name", "")]
        if not items:
            items = [it.to_dict() for it in news_mgr.get_recent(limit=limit)]
    elif stype == "twitter":
        social_mgr = manager.social_manager
        items = social_mgr.get_influencer_feed(limit=limit)
    else:
        items = []

    return FeedItemResponse(items=items[:limit], count=len(items[:limit]))


# ── Endpoints: News Feed ─────────────────────────────────────────────────────


@router.get("/news/feed")
async def get_news_feed(
    request: Request,
    user: dict = Depends(get_current_user),
    limit: int = Query(default=50, ge=1, le=200),
    symbol: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Get aggregated news feed from all sources."""
    news_mgr = _get_news_stream_manager(request)
    items = news_mgr.get_recent(limit=limit, symbol=symbol)

    news = []
    for item in items:
        news.append({
            "title": item.title,
            "summary": item.summary,
            "source": item.source,
            "tickers": item.tickers,
            "sentiment": item.sentiment_score,
            "impact": item.impact_level.value,
            "timestamp": item.published_at.isoformat(),
            "url": item.url,
        })

    return {"news": news, "count": len(news)}


# ── Endpoints: Sentiment ─────────────────────────────────────────────────────


@router.get("/sentiment/live")
async def get_live_sentiment(
    request: Request,
    user: dict = Depends(get_current_user),
    symbols: str = Query(..., description="Comma-separated symbols (e.g. SPY,QQQ,NIFTY)"),
) -> Dict[str, Any]:
    """Get live sentiment for specified symbols."""
    social_mgr = _get_social_stream_manager(request)
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    sentiments: Dict[str, Any] = {}
    for sym in symbol_list:
        ts = social_mgr.get_sentiment(sym)
        sentiments[sym] = {
            "score": ts.score_1h,
            "magnitude": ts.magnitude,
            "sources_count": ts.sources_count,
            "trending": ts.trending,
            "score_1h": ts.score_1h,
            "score_4h": ts.score_4h,
            "score_24h": ts.score_24h,
            "mention_count_1h": ts.mention_count_1h,
        }

    return {"sentiments": sentiments}


@router.get("/sentiment/trending")
async def get_trending_tickers(
    request: Request,
    user: dict = Depends(get_current_user),
    limit: int = Query(default=20, ge=1, le=100),
) -> Dict[str, Any]:
    """Get trending tickers from social sentiment analysis."""
    social_mgr = _get_social_stream_manager(request)
    trending = social_mgr.get_trending(limit=limit)
    return {"trending": trending}
