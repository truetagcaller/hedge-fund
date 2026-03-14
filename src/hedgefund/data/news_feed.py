"""Pluggable financial news feed provider.

The default implementation uses :pypi:`httpx` and supports any HTTP JSON API
via a configurable base URL and header template.  Drop-in replacements (e.g.
Polygon, Benzinga, Alpha Vantage) just need to subclass
:class:`~hedgefund.data.base.NewsProvider`.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import httpx

from hedgefund.data.base import NewsProvider
from hedgefund.logger import get_logger

log = get_logger(__name__)


# ── Response parser protocol ──────────────────────────────────────────────────

ResponseParser = Callable[[Dict[str, Any]], List[Dict[str, Any]]]


def _default_parser(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse a generic ``{articles: [...]}`` or ``{results: [...]}`` response."""
    for key in ("articles", "results", "data", "news", "items"):
        if key in payload and isinstance(payload[key], list):
            return payload[key]  # type: ignore[return-value]
    if isinstance(payload, list):
        return payload  # type: ignore[return-value]
    return []


# ── Default HTTP implementation ───────────────────────────────────────────────


class NewsFeedProvider(NewsProvider):
    """HTTP-based news provider with pluggable response parsing.

    Parameters
    ----------
    base_url:
        Root URL of the news API (e.g. ``https://newsapi.org/v2``).
    api_key:
        Bearer / query-param API key.
    headers:
        Extra headers merged into every request.
    parser:
        Callable that extracts a list of article dicts from the raw JSON
        response.  Defaults to :func:`_default_parser`.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        headers: Optional[Dict[str, str]] = None,
        parser: Optional[ResponseParser] = None,
        timeout: float = 15.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._headers = headers or {}
        self._parser = parser or _default_parser
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            default_headers: Dict[str, str] = {"Accept": "application/json"}
            if self._api_key:
                default_headers["Authorization"] = f"Bearer {self._api_key}"
            default_headers.update(self._headers)
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=default_headers,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        """Shut down the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ── NewsProvider interface ─────────────────────────────────────────────

    async def fetch_news(
        self,
        symbols: List[str],
        max_items: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch recent news for the given symbols.

        The method builds a query string from *symbols* and issues a single
        GET request.  It then normalises each article into a dict with at
        least: ``title``, ``url``, ``published_at``, ``source``, ``symbols``.
        """
        client = await self._get_client()
        query = " OR ".join(symbols)

        params: Dict[str, Any] = {
            "q": query,
            "pageSize": max_items,
            "sortBy": "publishedAt",
        }
        if self._api_key and "Authorization" not in self._headers:
            params["apiKey"] = self._api_key

        log.debug("news_fetch", symbols=symbols, max_items=max_items)

        try:
            resp = await client.get("/everything", params=params)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPStatusError as exc:
            log.error("news_fetch_http_error", status=exc.response.status_code)
            return []
        except Exception:
            log.exception("news_fetch_error")
            return []

        raw_articles = self._parser(payload)

        articles: List[Dict[str, Any]] = []
        for art in raw_articles[:max_items]:
            articles.append(
                {
                    "title": art.get("title", ""),
                    "url": art.get("url", ""),
                    "published_at": art.get("publishedAt", art.get("published_at", "")),
                    "source": (
                        art.get("source", {}).get("name", "")
                        if isinstance(art.get("source"), dict)
                        else str(art.get("source", ""))
                    ),
                    "description": art.get("description", ""),
                    "symbols": symbols,
                }
            )

        log.info("news_fetched", count=len(articles))
        return articles

    async def stream_news(
        self,
        symbols: List[str],
    ) -> AsyncIterator[Dict[str, Any]]:
        """Poll the news API periodically and yield new articles.

        This is a convenience wrapper that deduplicates by URL.
        """
        import asyncio

        seen_urls: set[str] = set()
        poll_interval = 120  # seconds

        while True:
            articles = await self.fetch_news(symbols, max_items=20)
            for article in articles:
                url = article.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    yield article
            await asyncio.sleep(poll_interval)
