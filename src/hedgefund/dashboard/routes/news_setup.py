"""Auto-setup free news sources — RSS feeds and open APIs.

Registers curated financial RSS feeds and free news APIs with the
DataSourceManager on demand. No API keys required.
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog
from fastapi import APIRouter, Request

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/news", tags=["news-setup"])

# ── Free RSS feeds (no API key needed) ────────────────────────────────────

FREE_RSS_FEEDS: List[Dict[str, Any]] = [
    {
        "name": "Yahoo Finance - Market News",
        "url": "https://finance.yahoo.com/news/rssindex",
        "poll_interval": 120,
    },
    {
        "name": "Yahoo Finance - Top Stories",
        "url": "https://finance.yahoo.com/rss/topstories",
        "poll_interval": 120,
    },
    {
        "name": "Google News - Business",
        "url": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
        "poll_interval": 180,
    },
    {
        "name": "Reuters - Business",
        "url": "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best",
        "poll_interval": 180,
    },
    {
        "name": "CNBC - Top News",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "poll_interval": 180,
    },
    {
        "name": "CNBC - Finance",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
        "poll_interval": 180,
    },
    {
        "name": "Investing.com - Market News",
        "url": "https://www.investing.com/rss/news.rss",
        "poll_interval": 180,
    },
    {
        "name": "MarketWatch - Top Stories",
        "url": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "poll_interval": 180,
    },
    {
        "name": "MarketWatch - Market Pulse",
        "url": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
        "poll_interval": 180,
    },
    {
        "name": "Moneycontrol - Market News",
        "url": "https://www.moneycontrol.com/rss/marketreports.xml",
        "poll_interval": 120,
    },
    {
        "name": "Moneycontrol - Business News",
        "url": "https://www.moneycontrol.com/rss/business.xml",
        "poll_interval": 120,
    },
    {
        "name": "Economic Times - Markets",
        "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "poll_interval": 120,
    },
    {
        "name": "Seeking Alpha - Market News",
        "url": "https://seekingalpha.com/market_currents.xml",
        "poll_interval": 180,
    },
]

# ── Free news APIs (no key required) ─────────────────────────────────────

FREE_NEWS_APIS: List[Dict[str, Any]] = [
    {
        "name": "GNews - Business",
        "base_url": "https://gnews.io/api/v4/top-headlines?category=business&lang=en&max=50&apikey=free",
        "poll_interval": 300,
    },
]


@router.post("/setup-free")
async def setup_free_news_sources(request: Request) -> Dict[str, Any]:
    """Register all free RSS feeds and news APIs.

    Adds curated financial news RSS feeds that require no API keys.
    Safe to call multiple times — duplicates are skipped.
    """
    dsm = getattr(request.app.state, "data_source_manager", None)
    if dsm is None:
        return {"status": "error", "message": "Data source manager not available."}

    added: List[str] = []
    skipped: List[str] = []
    errors: List[str] = []

    # Get existing source names to avoid duplicates
    existing_names = set()
    try:
        for src in dsm.list_sources():
            existing_names.add(src.get("name", ""))
    except Exception:  # noqa: S110
            log.debug("unexpected_error", exc_info=True)

    # Add RSS feeds
    for feed in FREE_RSS_FEEDS:
        name = feed["name"]
        if name in existing_names:
            skipped.append(name)
            continue
        try:
            source_id = dsm.add_source("rss", {
                "name": name,
                "url": feed["url"],
                "poll_interval": feed["poll_interval"],
            })
            added.append(name)
            log.info("news_setup.rss_added", name=name, source_id=source_id)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            log.warning("news_setup.rss_error", name=name, error=str(exc))

    # Add free APIs
    for api in FREE_NEWS_APIS:
        name = api["name"]
        if name in existing_names:
            skipped.append(name)
            continue
        try:
            source_id = dsm.add_source("news_api", {
                "name": name,
                "base_url": api["base_url"],
                "poll_interval": api["poll_interval"],
                "credentials": {},
            })
            added.append(name)
            log.info("news_setup.api_added", name=name, source_id=source_id)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            log.warning("news_setup.api_error", name=name, error=str(exc))

    # Update data source validator if available
    dsv = getattr(request.app.state, "data_source_validator", None)
    if dsv and added:
        from hedgefund.engine.data_source_validator import DataSourceStatus
        dsv.update_news_status(DataSourceStatus.CONNECTED, "RSS Feeds")

    return {
        "status": "ok",
        "added": added,
        "skipped": skipped,
        "errors": errors,
        "total_sources": len(added) + len(skipped),
    }


@router.get("/sources")
async def list_news_sources(request: Request) -> Dict[str, Any]:
    """List all configured news sources."""
    dsm = getattr(request.app.state, "data_source_manager", None)
    if dsm is None:
        return {"sources": [], "message": "Data source manager not available."}

    try:
        all_sources = dsm.list_sources()
        news_types = {"rss", "news_api", "economic_calendar", "earnings"}
        news_sources = [s for s in all_sources if s.get("type") in news_types]
        return {"sources": news_sources, "total": len(news_sources)}
    except Exception as exc:
        return {"sources": [], "error": str(exc)}


@router.get("/available-feeds")
async def list_available_feeds(request: Request) -> Dict[str, Any]:
    """List all available free feeds that can be added."""
    return {
        "rss_feeds": [
            {"name": f["name"], "url": f["url"]}
            for f in FREE_RSS_FEEDS
        ],
        "apis": [
            {"name": a["name"], "url": a["base_url"]}
            for a in FREE_NEWS_APIS
        ],
        "total": len(FREE_RSS_FEEDS) + len(FREE_NEWS_APIS),
    }


@router.get("/recent")
async def get_recent_news(
    request: Request,
    limit: int = 50,
    symbol: str = "",
) -> Dict[str, Any]:
    """Return recent news items from all feeds.

    This is the REST fallback for the News & Sentiment dashboard tab
    when WebSocket push isn't available (dashboard-only mode).
    """
    dsm = getattr(request.app.state, "data_source_manager", None)
    if dsm is None:
        return {"news": [], "message": "Data source manager not available."}

    news_mgr = dsm.news_manager
    try:
        items = news_mgr.get_recent(limit=limit, symbol=symbol or None)
        return {
            "news": [
                {
                    "headline": getattr(it, "title", getattr(it, "headline", "")),
                    "title": getattr(it, "title", ""),
                    "source": getattr(it, "source_name", getattr(it, "source", "")),
                    "url": getattr(it, "url", ""),
                    "timestamp": (
                        it.published_at.isoformat()
                        if hasattr(it, "published_at") and it.published_at
                        else getattr(it, "timestamp", "")
                    ),
                    "symbols": getattr(it, "tickers", []),
                    "sentiment": (
                        "bullish" if getattr(it, "sentiment_score", 0) > 0.1
                        else "bearish" if getattr(it, "sentiment_score", 0) < -0.1
                        else "neutral"
                    ),
                    "sentiment_score": getattr(it, "sentiment_score", 0.0),
                    "impact": getattr(it, "impact", ""),
                }
                for it in items
            ],
            "total": len(items),
            "source_count": len(news_mgr._sources) if hasattr(news_mgr, "_sources") else 0,
        }
    except Exception as exc:
        log.warning("news_recent.error", error=str(exc))
        return {"news": [], "error": str(exc)}


@router.get("/x-feed")
async def get_x_sentiment_feed(request: Request, limit: int = 50) -> Dict[str, Any]:
    """Return recent X/Twitter sentiment data."""
    # Try SocialStreamManager first
    social_mgr = getattr(request.app.state, "social_stream_manager", None)
    social_items: List[Dict[str, Any]] = []

    if social_mgr is not None:
        try:
            recent = getattr(social_mgr, "_recent_tweets", None)
            if recent:
                for tweet in list(recent)[:limit]:
                    social_items.append({
                        "text": getattr(tweet, "text", ""),
                        "author": getattr(tweet, "author_name", getattr(tweet, "author_id", "")),
                        "sentiment": (
                            "bullish" if getattr(tweet, "sentiment_score", 0) > 0.1
                            else "bearish" if getattr(tweet, "sentiment_score", 0) < -0.1
                            else "neutral"
                        ),
                        "sentiment_score": getattr(tweet, "sentiment_score", 0.0),
                        "tickers": getattr(tweet, "tickers", []),
                        "timestamp": (
                            tweet.created_at.isoformat()
                            if hasattr(tweet, "created_at") and tweet.created_at
                            else ""
                        ),
                        "source": "twitter",
                    })
        except Exception:  # noqa: S110
                log.debug("unexpected_error", exc_info=True)

    # Also check MongoDB for webhook-delivered tweets
    db = getattr(request.app.state, "db", None)
    if db is not None and len(social_items) < limit:
        try:
            remaining = limit - len(social_items)
            cursor = db.x_posts.find({}).sort("timestamp", -1).limit(remaining)
            async for doc in cursor:
                social_items.append({
                    "text": doc.get("text", ""),
                    "author": doc.get("author", ""),
                    "sentiment": doc.get("sentiment_label", "neutral"),
                    "sentiment_score": doc.get("sentiment_score", 0.0),
                    "tickers": doc.get("ticker_mentions", []),
                    "timestamp": (
                        doc["timestamp"].isoformat()
                        if hasattr(doc.get("timestamp"), "isoformat")
                        else str(doc.get("timestamp", ""))
                    ),
                    "source": "twitter",
                })
        except Exception:  # noqa: S110
                log.debug("unexpected_error", exc_info=True)

    return {"social": social_items, "total": len(social_items)}


@router.get("/sentiment-summary")
async def get_sentiment_summary(request: Request) -> Dict[str, Any]:
    """Return sentiment heatmap and trending tickers from news + social data."""
    dsm = getattr(request.app.state, "data_source_manager", None)
    heatmap: Dict[str, float] = {}
    trending: List[Dict[str, Any]] = []

    # Build heatmap from recent news items
    if dsm is not None:
        news_mgr = dsm.news_manager
        try:
            items = news_mgr.get_recent(limit=200)
            ticker_scores: Dict[str, List[float]] = {}
            ticker_counts: Dict[str, int] = {}
            for it in items:
                tickers = getattr(it, "tickers", [])
                score = getattr(it, "sentiment_score", 0.0)
                for t in tickers:
                    ticker_scores.setdefault(t, []).append(score)
                    ticker_counts[t] = ticker_counts.get(t, 0) + 1

            for sym, scores in ticker_scores.items():
                heatmap[sym] = round(sum(scores) / len(scores), 4) if scores else 0.0

            # Build trending by mention count
            sorted_tickers = sorted(
                ticker_counts.items(), key=lambda x: x[1], reverse=True,
            )
            for sym, count in sorted_tickers[:20]:
                avg_score = heatmap.get(sym, 0.0)
                trending.append({
                    "symbol": sym,
                    "mentions": count,
                    "sentiment": round(avg_score, 4),
                })
        except Exception as exc:
            log.debug("sentiment_summary.news_error", error=str(exc))

    # Also check social stream for trending
    social_mgr = getattr(request.app.state, "social_stream_manager", None)
    if social_mgr is not None:
        try:
            social_trending = social_mgr.get_trending(limit=20)
            for t in social_trending:
                sym = t.get("symbol", "")
                if sym and sym not in heatmap:
                    heatmap[sym] = t.get("sentiment", 0.0)
                    trending.append({
                        "symbol": sym,
                        "mentions": t.get("mentions", 0),
                        "sentiment": t.get("sentiment", 0.0),
                    })
        except Exception:  # noqa: S110
                log.debug("unexpected_error", exc_info=True)

    return {
        "heatmap": heatmap,
        "trending": trending,
    }
