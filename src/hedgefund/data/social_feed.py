"""Social media (Twitter/X) sentiment data collection.

Implements :class:`~hedgefund.data.base.SocialFeedProvider` with a pluggable
HTTP back-end.  The default adapter targets the X/Twitter API v2 but can be
swapped for any social-data vendor that returns JSON.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from hedgefund.data.base import SocialFeedProvider
from hedgefund.logger import get_logger
from hedgefund.types import SentimentResult

log = get_logger(__name__)

# Simple lexicon-based sentiment as a zero-dependency fallback.
_BULLISH_WORDS = frozenset(
    {
        "bull",
        "bullish",
        "buy",
        "long",
        "calls",
        "moon",
        "rocket",
        "breakout",
        "upgrade",
        "beat",
        "surge",
        "rally",
        "soar",
        "rip",
        "pump",
    }
)
_BEARISH_WORDS = frozenset(
    {
        "bear",
        "bearish",
        "sell",
        "short",
        "puts",
        "crash",
        "dump",
        "downgrade",
        "miss",
        "plunge",
        "tank",
        "drop",
        "fade",
        "red",
    }
)


def _lexicon_sentiment(text: str) -> tuple[float, float]:
    """Return ``(score, magnitude)`` using naive word matching.

    ``score`` ranges from -1 (bearish) to +1 (bullish).
    ``magnitude`` is the raw fraction of sentiment-carrying words.
    """
    words = set(text.lower().split())
    bull_count = len(words & _BULLISH_WORDS)
    bear_count = len(words & _BEARISH_WORDS)
    total = bull_count + bear_count
    if total == 0:
        return 0.0, 0.0
    score = (bull_count - bear_count) / total
    magnitude = min(total / max(len(words), 1), 1.0)
    return score, magnitude


class TwitterSocialFeedProvider(SocialFeedProvider):
    """Collect sentiment from Twitter/X API v2.

    Parameters
    ----------
    bearer_token:
        Twitter API v2 bearer token.  If empty, the provider degrades
        gracefully (returns empty results rather than raising).
    base_url:
        Override for testing or proxy usage.
    sentiment_fn:
        Optional callable ``(text) -> (score, magnitude)`` to replace the
        built-in lexicon scorer with an ML model.
    """

    def __init__(
        self,
        bearer_token: str = "",
        base_url: str = "https://api.twitter.com/2",
        sentiment_fn: Any = None,
        timeout: float = 15.0,
    ) -> None:
        self._bearer_token = bearer_token
        self._base_url = base_url.rstrip("/")
        self._sentiment_fn = sentiment_fn or _lexicon_sentiment
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: Dict[str, str] = {"Accept": "application/json"}
            if self._bearer_token:
                headers["Authorization"] = f"Bearer {self._bearer_token}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        """Shut down the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ── SocialFeedProvider interface ───────────────────────────────────────

    async def fetch_posts(
        self,
        symbols: List[str],
        max_items: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch recent tweets mentioning *symbols* (cashtags).

        Returns a list of dicts each containing ``text``, ``created_at``,
        ``author_id``, ``id``, and ``symbols``.
        """
        if not self._bearer_token:
            log.warning("twitter_no_bearer_token")
            return []

        client = await self._get_client()
        query = " OR ".join(f"${s}" for s in symbols)

        params: Dict[str, Any] = {
            "query": query,
            "max_results": min(max_items, 100),
            "tweet.fields": "created_at,author_id,text",
        }

        log.debug("twitter_fetch_posts", symbols=symbols, max_items=max_items)

        try:
            resp = await client.get("/tweets/search/recent", params=params)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPStatusError as exc:
            log.error("twitter_http_error", status=exc.response.status_code)
            return []
        except Exception:
            log.exception("twitter_fetch_error")
            return []

        raw_tweets = payload.get("data", [])
        posts: List[Dict[str, Any]] = []
        for tweet in raw_tweets:
            posts.append(
                {
                    "id": tweet.get("id", ""),
                    "text": tweet.get("text", ""),
                    "created_at": tweet.get("created_at", ""),
                    "author_id": tweet.get("author_id", ""),
                    "symbols": symbols,
                }
            )

        log.info("twitter_posts_fetched", count=len(posts))
        return posts

    async def get_sentiment(self, symbol: str) -> SentimentResult:
        """Aggregate sentiment for *symbol* from recent tweets."""
        posts = await self.fetch_posts([symbol], max_items=100)

        if not posts:
            return SentimentResult(
                symbol=symbol,
                score=0.0,
                magnitude=0.0,
                source="twitter",
                headline="No recent tweets",
                timestamp=datetime.now(timezone.utc),
            )

        scores: List[float] = []
        magnitudes: List[float] = []

        for post in posts:
            score, mag = self._sentiment_fn(post.get("text", ""))
            scores.append(score)
            magnitudes.append(mag)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_magnitude = sum(magnitudes) / len(magnitudes) if magnitudes else 0.0

        # Pick the most impactful headline for context.
        best_idx = max(range(len(magnitudes)), key=lambda i: magnitudes[i])
        headline = posts[best_idx].get("text", "")[:200]

        return SentimentResult(
            symbol=symbol,
            score=round(avg_score, 4),
            magnitude=round(avg_magnitude, 4),
            source="twitter",
            headline=headline,
            timestamp=datetime.now(timezone.utc),
        )
