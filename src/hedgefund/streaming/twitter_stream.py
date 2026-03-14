"""Twitter v2 tweet polling — periodic search for financial tweets.

Uses Twitter API v2 Recent Search endpoint with the app-level bearer token.
Falls back gracefully when credits are depleted (free tier monthly limit).
Publishes SENTIMENT events to the EventBus and stores tweets in MongoDB.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

import httpx
import structlog

from hedgefund.streaming.event_bus import Event, EventBus, EventType

log = structlog.get_logger(__name__)

_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
_TICKER_PATTERN = re.compile(r"\$([A-Z]{1,6})\b")

# Queries to rotate through (free tier: 1 req/15s, 10 tweets/req)
_SEARCH_QUERIES = [
    "$SPY OR $QQQ OR $AAPL OR $TSLA OR $NVDA",
    "$AMZN OR $MSFT OR $NIFTY OR $BANKNIFTY",
    "#stockmarket #trading -is:retweet lang:en",
    "#options OR #optionstrading -is:retweet lang:en",
    "$SENSEX OR #nifty OR #banknifty lang:en",
]


class TwitterPollStream:
    """Polls Twitter Recent Search API for financial tweets.

    Rotates through multiple search queries to cover different tickers
    and topics. Deduplicates by tweet ID.

    Parameters
    ----------
    bearer_token:
        App-level bearer token for Twitter API v2.
    event_bus:
        EventBus for publishing SENTIMENT events.
    db:
        MongoDB instance for storing tweets (optional).
    poll_interval:
        Seconds between poll cycles (default 60).
    """

    def __init__(
        self,
        bearer_token: str,
        event_bus: EventBus,
        db: Any = None,
        poll_interval: float = 60.0,
    ) -> None:
        self._bearer = bearer_token
        self._event_bus = event_bus
        self._db = db
        self._poll_interval = poll_interval
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None
        self._seen_ids: Set[str] = set()
        self._tweets_processed = 0
        self._query_index = 0
        self._credits_depleted = False
        self._last_error: str = ""

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def tweets_processed(self) -> int:
        return self._tweets_processed

    @property
    def status(self) -> str:
        if self._credits_depleted:
            return "credits_depleted"
        if self._running:
            return "running"
        return "stopped"

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._poll_loop(), name="twitter-poll-stream",
        )
        log.info("twitter_poll_stream.started", interval=self._poll_interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info(
            "twitter_poll_stream.stopped",
            tweets_processed=self._tweets_processed,
        )

    async def _poll_loop(self) -> None:
        """Periodically search Twitter for financial tweets."""
        while self._running:
            try:
                if not self._credits_depleted:
                    await self._search_and_process()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("twitter_poll_stream.error")

            await asyncio.sleep(self._poll_interval)

    async def _search_and_process(self) -> None:
        """Execute one search query and process results."""
        query = _SEARCH_QUERIES[self._query_index % len(_SEARCH_QUERIES)]
        self._query_index += 1

        headers = {"Authorization": f"Bearer {self._bearer}"}
        params = {
            "query": query,
            "max_results": 10,
            "tweet.fields": "created_at,author_id,public_metrics,text",
            "user.fields": "name,username,public_metrics",
            "expansions": "author_id",
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    _SEARCH_URL, headers=headers, params=params,
                )
        except httpx.RequestError as exc:
            log.warning("twitter_poll_stream.request_error", error=str(exc))
            return

        if resp.status_code == 402:
            # Credits depleted — stop polling, wait for reset
            self._credits_depleted = True
            self._last_error = "Twitter API credits depleted (free tier monthly limit)"
            log.warning("twitter_poll_stream.credits_depleted")
            return

        if resp.status_code == 429:
            log.warning("twitter_poll_stream.rate_limited")
            await asyncio.sleep(15)  # Wait 15 seconds
            return

        if resp.status_code != 200:
            log.warning(
                "twitter_poll_stream.http_error",
                status=resp.status_code,
                body=resp.text[:200],
            )
            return

        payload = resp.json()
        tweets = payload.get("data", [])

        # Build author lookup
        includes = payload.get("includes", {})
        authors = {u["id"]: u for u in includes.get("users", [])}

        new_count = 0
        for tweet in tweets:
            tweet_id = tweet.get("id", "")
            if tweet_id in self._seen_ids:
                continue
            self._seen_ids.add(tweet_id)

            # Keep seen set bounded
            if len(self._seen_ids) > 10000:
                self._seen_ids = set(list(self._seen_ids)[-5000:])

            author = authors.get(tweet.get("author_id", ""), {})
            await self._process_tweet(tweet, author)
            new_count += 1

        if new_count > 0:
            log.info(
                "twitter_poll_stream.batch",
                query=query[:40],
                new=new_count,
                total=self._tweets_processed,
            )

    async def _process_tweet(
        self, tweet: Dict[str, Any], author: Dict[str, Any],
    ) -> None:
        """Process a single tweet."""
        text = tweet.get("text", "")
        tweet_id = tweet.get("id", "")
        username = author.get("username", "")
        author_name = author.get("name", "")
        followers = author.get("public_metrics", {}).get("followers_count", 0)
        created_at = tweet.get("created_at", "")

        tickers = [m.group(1) for m in _TICKER_PATTERN.finditer(text)]
        sentiment_score, magnitude = self._score_sentiment(text)

        self._tweets_processed += 1

        log.debug(
            "twitter_poll_stream.tweet",
            author=f"@{username}",
            tickers=tickers,
            sentiment=round(sentiment_score, 3),
            source="X (Twitter) Search API",
        )

        # Publish to EventBus
        for ticker in tickers:
            event = Event(
                event_type=EventType.SENTIMENT,
                timestamp=datetime.now(timezone.utc),
                symbol=ticker,
                data={
                    "score": sentiment_score,
                    "sentiment_score": sentiment_score,
                    "magnitude": magnitude,
                    "source": "twitter",
                    "tweet_id": tweet_id,
                    "author": username,
                    "author_name": author_name,
                    "followers": followers,
                    "text": text[:500],
                    "tickers": tickers,
                    "is_live": True,
                },
                source="X (Twitter) Search API",
            )
            await self._event_bus.publish(event)

        # Store in MongoDB
        if self._db is not None:
            try:
                doc = {
                    "tweet_id": tweet_id,
                    "text": text[:500],
                    "author": username,
                    "author_name": author_name,
                    "followers": followers,
                    "tickers": tickers,
                    "sentiment_score": sentiment_score,
                    "sentiment_label": (
                        "bullish" if sentiment_score > 0.1
                        else "bearish" if sentiment_score < -0.1
                        else "neutral"
                    ),
                    "ticker_mentions": tickers,
                    "source": "twitter",
                    "timestamp": datetime.now(timezone.utc),
                    "is_live": True,
                    "created_at": created_at,
                }
                await self._db.x_posts.insert_one(doc)
            except Exception:
                log.debug("twitter_poll_stream.db_error", exc_info=True)

    @staticmethod
    def _score_sentiment(text: str) -> tuple[float, float]:
        """Lexicon-based sentiment scoring."""
        try:
            from hedgefund.data.social_feed import _lexicon_sentiment
            return _lexicon_sentiment(text)
        except ImportError:
            pass

        text_lower = text.lower()
        positive = {
            "bullish", "buy", "long", "moon", "pump", "calls", "green",
            "breakout", "surge", "rally", "up", "profit", "gain", "bull",
            "squeeze", "rip", "rocket", "soar", "boom",
        }
        negative = {
            "bearish", "sell", "short", "crash", "dump", "puts", "red",
            "breakdown", "plunge", "drop", "down", "loss", "bear",
            "tank", "drill", "fade", "collapse", "bust",
        }
        words = set(text_lower.split())
        pos = len(words & positive)
        neg = len(words & negative)
        total = pos + neg
        if total == 0:
            return 0.0, 0.0
        return (pos - neg) / total, min(1.0, total / 5.0)
