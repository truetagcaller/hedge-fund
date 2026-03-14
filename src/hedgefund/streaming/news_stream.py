"""Real-time news ingestion pipeline with multi-source support.

Aggregates news from RSS feeds, financial news APIs, economic calendars, and
earnings announcements into a unified stream.  Each item is enriched with
ticker extraction, sentiment scoring, and impact classification before being
published to the event bus.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from xml.etree import ElementTree

import httpx
import structlog

from hedgefund.sentiment.news_sentiment import NewsSentimentScorer
from hedgefund.types import SentimentResult

log = structlog.get_logger(__name__)


# ── Enums & Constants ─────────────────────────────────────────────────────────


class ImpactLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class NewsSourceType(str, Enum):
    RSS = "rss"
    NEWS_API = "news_api"
    ECONOMIC_CALENDAR = "economic_calendar"
    EARNINGS = "earnings"


_HIGH_IMPACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(earnings|revenue)\s+(beat|miss|surprise)",
        r"\bFDA\b.{0,30}\b(approv|reject|delay)",
        r"\b(merger|acquisition|buyout|takeover)\b",
        r"\b(rate\s+hike|rate\s+cut|interest\s+rate)\b",
        r"\b(bankruptcy|default|insolvency)\b",
        r"\b(IPO|stock\s+split|spin[- ]?off)\b",
        r"\b(guidance\s+(raise|lower|cut))\b",
        r"\b(SEC|DOJ|antitrust)\s+(investigat|charg|su)",
        r"\b(war|sanction|embargo|tariff)\b",
        r"\b(CPI|inflation|GDP|payroll|unemployment)\b",
    )
]

_MEDIUM_IMPACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(upgrade[sd]?|downgrade[sd]?)\b",
        r"\b(analyst|price\s+target)\b",
        r"\b(dividend|buyback|repurchase)\b",
        r"\b(insider\s+(buy|sell|trading))\b",
        r"\b(contract|partnership|deal)\b",
        r"\b(recall|lawsuit|settlement)\b",
        r"\b(CEO|CFO|executive)\s+(resign|hire|appoint|fired)",
    )
]

# Ticker extraction: $AAPL, $NIFTY, NASDAQ:AAPL, etc.
_TICKER_PATTERN = re.compile(
    r"""
    (?:\$([A-Z]{1,6}))              # cashtag: $AAPL
    |(?:\b([A-Z]{2,5}:[A-Z]{1,6})) # exchange-prefixed: NASDAQ:AAPL
    """,
    re.VERBOSE,
)

# Broad pattern to catch tickers in parentheses: (AAPL), (ticker: AAPL)
_PAREN_TICKER = re.compile(r"\((?:ticker:\s*)?([A-Z]{1,6})\)")


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class NewsItem:
    """A single enriched news item."""

    id: str
    title: str
    summary: str
    source: str
    url: str
    published_at: datetime
    tickers: list[str]
    sentiment_score: float
    impact_level: ImpactLevel
    source_type: NewsSourceType
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "tickers": self.tickers,
            "sentiment_score": self.sentiment_score,
            "impact_level": self.impact_level.value,
            "source_type": self.source_type.value,
        }


@dataclass(slots=True)
class NewsSource:
    """Configuration for a single news source."""

    name: str
    source_type: NewsSourceType
    url: str
    poll_interval: int  # seconds
    api_key: str = ""
    enabled: bool = True
    last_poll: Optional[datetime] = None
    error_count: int = 0


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_tickers(text: str) -> list[str]:
    """Extract ticker symbols from text using regex patterns."""
    tickers: set[str] = set()

    for match in _TICKER_PATTERN.finditer(text):
        cashtag, exchange_prefixed = match.groups()
        if cashtag:
            tickers.add(cashtag)
        elif exchange_prefixed:
            # Take the part after the colon
            tickers.add(exchange_prefixed.split(":")[-1])

    for match in _PAREN_TICKER.finditer(text):
        tickers.add(match.group(1))

    return sorted(tickers)


def _classify_impact(title: str, summary: str) -> ImpactLevel:
    """Classify news impact based on keyword patterns."""
    combined = f"{title} {summary}"

    for pattern in _HIGH_IMPACT_PATTERNS:
        if pattern.search(combined):
            return ImpactLevel.HIGH

    for pattern in _MEDIUM_IMPACT_PATTERNS:
        if pattern.search(combined):
            return ImpactLevel.MEDIUM

    return ImpactLevel.LOW


def _url_hash(url: str) -> str:
    """Generate a short hash for deduplication."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _parse_rss_xml(xml_text: str) -> list[dict[str, str]]:
    """Parse RSS/Atom XML into a list of article dicts."""
    items: list[dict[str, str]] = []
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        log.warning("rss_parse_error")
        return items

    # RSS 2.0
    for item in root.iter("item"):
        entry: dict[str, str] = {}
        title_el = item.find("title")
        entry["title"] = title_el.text or "" if title_el is not None else ""
        desc_el = item.find("description")
        entry["summary"] = desc_el.text or "" if desc_el is not None else ""
        link_el = item.find("link")
        entry["url"] = link_el.text or "" if link_el is not None else ""
        pub_el = item.find("pubDate")
        entry["published"] = pub_el.text or "" if pub_el is not None else ""
        items.append(entry)

    # Atom fallback
    if not items:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry_el in root.findall(".//atom:entry", ns):
            entry = {}
            title_el = entry_el.find("atom:title", ns)
            entry["title"] = title_el.text or "" if title_el is not None else ""
            summary_el = entry_el.find("atom:summary", ns)
            entry["summary"] = summary_el.text or "" if summary_el is not None else ""
            link_el = entry_el.find("atom:link", ns)
            entry["url"] = link_el.get("href", "") if link_el is not None else ""
            updated_el = entry_el.find("atom:updated", ns)
            entry["published"] = updated_el.text or "" if updated_el is not None else ""
            items.append(entry)

    return items


# ── Event Bus (lightweight pub/sub) ──────────────────────────────────────────


class EventBus:
    """Simple async event bus for decoupled component communication."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[..., Any]]] = {}

    def subscribe(self, event_type: str, handler: Callable[..., Any]) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler: Callable[..., Any]) -> None:
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                h for h in self._subscribers[event_type] if h is not handler
            ]

    async def publish(self, event_type: str, data: Any) -> None:
        for handler in self._subscribers.get(event_type, []):
            try:
                result = handler(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.error("event_handler_error", event_type=event_type, exc_info=True)


# ── News Stream Manager ──────────────────────────────────────────────────────


class NewsStreamManager:
    """Real-time news ingestion pipeline with multi-source support.

    Aggregates news from RSS feeds, financial news APIs, economic calendars,
    and earnings announcements.  Each item is enriched with ticker extraction,
    sentiment scoring, and impact classification.

    Parameters
    ----------
    event_bus:
        Optional event bus instance for publishing NEWS events.
    sentiment_scorer:
        Optional NewsSentimentScorer; a default is created if omitted.
    max_recent:
        Maximum number of recent items to keep in memory.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        sentiment_scorer: NewsSentimentScorer | None = None,
        max_recent: int = 500,
    ) -> None:
        self._event_bus = event_bus or EventBus()
        self._scorer = sentiment_scorer or NewsSentimentScorer()
        self._sources: dict[str, NewsSource] = {}
        self._recent: deque[NewsItem] = deque(maxlen=max_recent)
        self._seen_hashes: set[str] = set()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._running = False
        self._client: Optional[httpx.AsyncClient] = None

    # ── Source management ─────────────────────────────────────────────────

    def add_rss_source(
        self,
        name: str,
        url: str,
        poll_interval: int = 60,
    ) -> str:
        """Register an RSS feed source.

        Returns the source id.
        """
        source_id = f"rss_{uuid.uuid4().hex[:8]}"
        self._sources[source_id] = NewsSource(
            name=name,
            source_type=NewsSourceType.RSS,
            url=url,
            poll_interval=poll_interval,
        )
        log.info("news_source_added", source_id=source_id, name=name, type="rss")
        if self._running:
            self._tasks[source_id] = asyncio.create_task(
                self._poll_loop(source_id),
                name=f"news_poll_{source_id}",
            )
        return source_id

    def add_api_source(
        self,
        name: str,
        base_url: str,
        api_key: str = "",
        poll_interval: int = 30,
    ) -> str:
        """Register a financial news API source.

        Returns the source id.
        """
        source_id = f"api_{uuid.uuid4().hex[:8]}"
        self._sources[source_id] = NewsSource(
            name=name,
            source_type=NewsSourceType.NEWS_API,
            url=base_url,
            poll_interval=poll_interval,
            api_key=api_key,
        )
        log.info("news_source_added", source_id=source_id, name=name, type="news_api")
        if self._running:
            self._tasks[source_id] = asyncio.create_task(
                self._poll_loop(source_id),
                name=f"news_poll_{source_id}",
            )
        return source_id

    def add_calendar_source(
        self,
        name: str,
        url: str,
        poll_interval: int = 300,
    ) -> str:
        """Register an economic calendar source."""
        source_id = f"cal_{uuid.uuid4().hex[:8]}"
        self._sources[source_id] = NewsSource(
            name=name,
            source_type=NewsSourceType.ECONOMIC_CALENDAR,
            url=url,
            poll_interval=poll_interval,
        )
        log.info("news_source_added", source_id=source_id, name=name, type="economic_calendar")
        if self._running:
            self._tasks[source_id] = asyncio.create_task(
                self._poll_loop(source_id),
                name=f"news_poll_{source_id}",
            )
        return source_id

    def add_earnings_source(
        self,
        name: str,
        url: str,
        api_key: str = "",
        poll_interval: int = 300,
    ) -> str:
        """Register an earnings announcement source."""
        source_id = f"earn_{uuid.uuid4().hex[:8]}"
        self._sources[source_id] = NewsSource(
            name=name,
            source_type=NewsSourceType.EARNINGS,
            url=url,
            poll_interval=poll_interval,
            api_key=api_key,
        )
        log.info("news_source_added", source_id=source_id, name=name, type="earnings")
        if self._running:
            self._tasks[source_id] = asyncio.create_task(
                self._poll_loop(source_id),
                name=f"news_poll_{source_id}",
            )
        return source_id

    def remove_source(self, source_id: str) -> None:
        """Remove a news source and cancel its polling task."""
        if source_id in self._tasks:
            self._tasks[source_id].cancel()
            del self._tasks[source_id]
        self._sources.pop(source_id, None)
        log.info("news_source_removed", source_id=source_id)

    def list_sources(self) -> list[dict[str, Any]]:
        """Return metadata for all registered sources."""
        results: list[dict[str, Any]] = []
        for sid, src in self._sources.items():
            results.append({
                "id": sid,
                "name": src.name,
                "type": src.source_type.value,
                "url": src.url,
                "poll_interval": src.poll_interval,
                "enabled": src.enabled,
                "last_poll": src.last_poll.isoformat() if src.last_poll else None,
                "error_count": src.error_count,
            })
        return results

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start polling all registered sources."""
        if self._running:
            return
        self._running = True
        self._client = httpx.AsyncClient(timeout=20.0)

        for source_id in self._sources:
            self._tasks[source_id] = asyncio.create_task(
                self._poll_loop(source_id),
                name=f"news_poll_{source_id}",
            )
        log.info("news_stream_started", source_count=len(self._sources))

    async def stop(self) -> None:
        """Stop all polling tasks and clean up."""
        self._running = False
        for task in self._tasks.values():
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

        log.info("news_stream_stopped")

    # ── Query ─────────────────────────────────────────────────────────────

    def get_recent(self, limit: int = 50, symbol: Optional[str] = None) -> list[NewsItem]:
        """Return recent news items, optionally filtered by ticker symbol."""
        items = list(self._recent)
        if symbol:
            items = [it for it in items if symbol.upper() in it.tickers]
        return items[:limit]

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    # ── Internal polling ──────────────────────────────────────────────────

    async def _poll_loop(self, source_id: str) -> None:
        """Continuously poll a single source at its configured interval."""
        while self._running:
            source = self._sources.get(source_id)
            if source is None or not source.enabled:
                return

            try:
                if source.source_type == NewsSourceType.RSS:
                    await self._poll_rss(source_id, source)
                elif source.source_type == NewsSourceType.NEWS_API:
                    await self._poll_api(source_id, source)
                elif source.source_type in (
                    NewsSourceType.ECONOMIC_CALENDAR,
                    NewsSourceType.EARNINGS,
                ):
                    await self._poll_api(source_id, source)

                source.last_poll = datetime.now(timezone.utc)
                source.error_count = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                source.error_count += 1
                log.error(
                    "news_poll_error",
                    source_id=source_id,
                    error_count=source.error_count,
                    exc_info=True,
                )

            await asyncio.sleep(source.poll_interval)

    async def _poll_rss(self, source_id: str, source: NewsSource) -> None:
        """Fetch and parse an RSS feed."""
        if self._client is None:
            return

        resp = await self._client.get(source.url)
        resp.raise_for_status()

        articles = _parse_rss_xml(resp.text)
        new_count = 0

        for article in articles:
            url = article.get("url", "")
            if not url:
                continue

            url_h = _url_hash(url)
            if url_h in self._seen_hashes:
                continue
            self._seen_hashes.add(url_h)

            item = await self._enrich_article(
                title=article.get("title", ""),
                summary=article.get("summary", ""),
                url=url,
                source_name=source.name,
                source_type=source.source_type,
            )
            self._recent.appendleft(item)
            await self._event_bus.publish("NEWS", item.to_dict())
            new_count += 1

        if new_count:
            log.info("news_rss_polled", source=source.name, new_items=new_count)

    async def _poll_api(self, source_id: str, source: NewsSource) -> None:
        """Fetch news from a JSON API endpoint."""
        if self._client is None:
            return

        headers: dict[str, str] = {"Accept": "application/json"}
        if source.api_key:
            headers["Authorization"] = f"Bearer {source.api_key}"

        resp = await self._client.get(source.url, headers=headers)
        resp.raise_for_status()
        payload = resp.json()

        # Accept common response shapes
        articles: list[dict[str, Any]] = []
        for key in ("articles", "results", "data", "news", "items", "events"):
            if key in payload and isinstance(payload[key], list):
                articles = payload[key]
                break
        if not articles and isinstance(payload, list):
            articles = payload

        new_count = 0
        for article in articles:
            url = article.get("url", article.get("link", ""))
            title = article.get("title", article.get("headline", ""))
            if not url and not title:
                continue

            url_h = _url_hash(url or title)
            if url_h in self._seen_hashes:
                continue
            self._seen_hashes.add(url_h)

            item = await self._enrich_article(
                title=title,
                summary=article.get("description", article.get("summary", "")),
                url=url,
                source_name=source.name,
                source_type=source.source_type,
                raw_metadata=article,
            )
            self._recent.appendleft(item)
            await self._event_bus.publish("NEWS", item.to_dict())
            new_count += 1

        if new_count:
            log.info("news_api_polled", source=source.name, new_items=new_count)

    # ── Enrichment ────────────────────────────────────────────────────────

    async def _enrich_article(
        self,
        title: str,
        summary: str,
        url: str,
        source_name: str,
        source_type: NewsSourceType,
        raw_metadata: dict[str, Any] | None = None,
    ) -> NewsItem:
        """Extract tickers, compute sentiment, and classify impact."""
        combined = f"{title} {summary}"
        tickers = _extract_tickers(combined)

        # Sentiment scoring via the existing scorer
        sentiment_score = 0.0
        try:
            if tickers:
                # Score based on the headline text
                result: SentimentResult = await self._scorer.score(tickers[0])
                sentiment_score = result.score
            else:
                # Use rule-based scoring on the headline directly
                from hedgefund.sentiment.news_sentiment import _rule_based_score

                score, _ = _rule_based_score(combined)
                sentiment_score = score
        except Exception:
            log.debug("sentiment_score_fallback", title=title[:80], exc_info=True)

        impact = _classify_impact(title, summary)

        return NewsItem(
            id=f"news_{uuid.uuid4().hex[:12]}",
            title=title,
            summary=summary[:500],
            source=source_name,
            url=url,
            published_at=datetime.now(timezone.utc),
            tickers=tickers,
            sentiment_score=sentiment_score,
            impact_level=impact,
            source_type=source_type,
            raw_metadata=raw_metadata or {},
        )
