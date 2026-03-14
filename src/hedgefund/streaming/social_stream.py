"""X (Twitter) social sentiment pipeline with influence weighting and spike detection.

Tracks tweets mentioning specific tickers, financial influencer accounts, and
trending hashtags.  Processes tweets through sentiment analysis, influence
weighting, and chatter spike detection.  Publishes SENTIMENT events to the
event bus.
"""

from __future__ import annotations

import asyncio
import math
import re
import statistics
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
import structlog

from hedgefund.sentiment.social_sentiment import (
    SocialSentimentScorer,
    SocialPost,
    _influence_weight,
)
from hedgefund.streaming.news_stream import EventBus

log = structlog.get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

_TICKER_PATTERN = re.compile(r"\$([A-Z]{1,6})\b")

_DEFAULT_HASHTAGS = [
    "#options", "#trading", "#stocks", "#earnings", "#investing",
    "#wallstreet", "#forex", "#crypto", "#nifty", "#banknifty",
    "#optionstrading", "#stockmarket",
]


class AuthMethod(str, Enum):
    BEARER_TOKEN = "bearer_token"
    OAUTH2_USER = "oauth2_user"
    SESSION_TOKEN = "session_token"


class AccountStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class SocialAccount:
    """A connected social media account with credentials."""

    id: str
    platform: str
    auth_method: AuthMethod
    credentials: dict[str, str]
    status: AccountStatus = AccountStatus.INACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "platform": self.platform,
            "auth_method": self.auth_method.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "error_message": self.error_message,
        }


@dataclass(slots=True)
class TweetItem:
    """A processed tweet with metadata."""

    id: str
    text: str
    author_id: str
    author_name: str
    created_at: datetime
    tickers: list[str]
    sentiment_score: float
    influence_score: float
    followers: int
    likes: int
    retweets: int
    is_influencer: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "created_at": self.created_at.isoformat(),
            "tickers": self.tickers,
            "sentiment_score": self.sentiment_score,
            "influence_score": self.influence_score,
            "followers": self.followers,
            "likes": self.likes,
            "retweets": self.retweets,
            "is_influencer": self.is_influencer,
        }


@dataclass(slots=True)
class TickerSentiment:
    """Aggregated sentiment for a ticker across rolling windows."""

    symbol: str
    score_1h: float = 0.0
    score_4h: float = 0.0
    score_24h: float = 0.0
    magnitude: float = 0.0
    mention_count_1h: int = 0
    mention_count_4h: int = 0
    mention_count_24h: int = 0
    sources_count: int = 0
    trending: bool = False
    spike_z_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": self.score_1h,
            "score_1h": self.score_1h,
            "score_4h": self.score_4h,
            "score_24h": self.score_24h,
            "magnitude": self.magnitude,
            "mention_count_1h": self.mention_count_1h,
            "mention_count_4h": self.mention_count_4h,
            "mention_count_24h": self.mention_count_24h,
            "sources_count": self.sources_count,
            "trending": self.trending,
            "spike_z_score": self.spike_z_score,
        }


# ── Chatter spike detection ──────────────────────────────────────────────────


class _RollingChatterTracker:
    """Track mention counts per ticker and detect spikes via z-score."""

    def __init__(self, window_size: int = 24) -> None:
        self._window_size = window_size
        self._history: dict[str, deque[int]] = {}

    def update(self, symbol: str, count: int) -> float:
        """Record count and return z-score. Returns 0.0 with insufficient data."""
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self._window_size)

        history = self._history[symbol]
        history.append(count)

        if len(history) < 3:
            return 0.0

        mean = statistics.mean(history)
        stdev = statistics.stdev(history)
        if stdev == 0:
            return 0.0
        return (count - mean) / stdev


# ── Rolling sentiment window ─────────────────────────────────────────────────


@dataclass(slots=True)
class _ScoredMention:
    """A single mention with score and timestamp."""

    symbol: str
    score: float
    weight: float
    timestamp: datetime


class _RollingSentimentAggregator:
    """Aggregate sentiment across rolling time windows."""

    def __init__(self) -> None:
        # Keep up to 24h of mentions per ticker
        self._mentions: dict[str, deque[_ScoredMention]] = {}

    def add(self, symbol: str, score: float, weight: float) -> None:
        if symbol not in self._mentions:
            self._mentions[symbol] = deque(maxlen=10_000)
        self._mentions[symbol].append(
            _ScoredMention(
                symbol=symbol,
                score=score,
                weight=weight,
                timestamp=datetime.now(timezone.utc),
            )
        )

    def get_aggregate(self, symbol: str) -> dict[str, Any]:
        """Compute weighted averages over 1h, 4h, 24h windows."""
        now = datetime.now(timezone.utc)
        mentions = self._mentions.get(symbol, deque())

        windows = {
            "1h": 3600,
            "4h": 14400,
            "24h": 86400,
        }
        result: dict[str, Any] = {}

        for label, seconds in windows.items():
            cutoff = now.timestamp() - seconds
            window_mentions = [
                m for m in mentions if m.timestamp.timestamp() >= cutoff
            ]
            if window_mentions:
                total_weight = sum(m.weight for m in window_mentions)
                weighted_score = (
                    sum(m.score * m.weight for m in window_mentions) / total_weight
                    if total_weight > 0
                    else 0.0
                )
                result[f"score_{label}"] = round(max(-1.0, min(1.0, weighted_score)), 4)
                result[f"count_{label}"] = len(window_mentions)
            else:
                result[f"score_{label}"] = 0.0
                result[f"count_{label}"] = 0

        return result

    def all_symbols(self) -> list[str]:
        return list(self._mentions.keys())


# ── Social Stream Manager ────────────────────────────────────────────────────


class SocialStreamManager:
    """X (Twitter) social sentiment pipeline.

    Connects to the Twitter API v2 to track tweets mentioning tickers and
    financial influencer accounts.  Processes each tweet through sentiment
    analysis, influence weighting, and chatter spike detection.

    Parameters
    ----------
    event_bus:
        Optional event bus for publishing SENTIMENT events.
    sentiment_scorer:
        Optional SocialSentimentScorer instance.
    spike_z_threshold:
        Z-score above which a chatter spike is flagged.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        sentiment_scorer: SocialSentimentScorer | None = None,
        spike_z_threshold: float = 2.0,
    ) -> None:
        self._event_bus = event_bus or EventBus()
        self._scorer = sentiment_scorer or SocialSentimentScorer()
        self._spike_z_threshold = spike_z_threshold

        # Account management
        self._accounts: dict[str, SocialAccount] = {}

        # Tracking configuration
        self._tracked_tickers: set[str] = set()
        self._influencer_accounts: set[str] = set()
        self._tracked_hashtags: list[str] = list(_DEFAULT_HASHTAGS)

        # Processing state
        self._chatter_tracker = _RollingChatterTracker()
        self._sentiment_agg = _RollingSentimentAggregator()
        self._recent_tweets: deque[TweetItem] = deque(maxlen=1000)
        self._influencer_tweets: deque[TweetItem] = deque(maxlen=500)

        # Runtime
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False
        self._client: Optional[httpx.AsyncClient] = None
        self._poll_interval = 60  # seconds between poll cycles

    # ── Account management ────────────────────────────────────────────────

    def add_account(
        self,
        platform: str,
        credentials: dict[str, str],
    ) -> str:
        """Add a social media account.

        Parameters
        ----------
        platform:
            Platform name (e.g. ``"twitter"``).
        credentials:
            Dict with keys like ``bearer_token``, ``api_key``, ``api_secret``,
            ``access_token``, ``access_token_secret``.

        Returns
        -------
        str
            Account id.
        """
        account_id = f"social_{uuid.uuid4().hex[:8]}"

        # Determine auth method from provided credentials
        if "bearer_token" in credentials:
            auth_method = AuthMethod.BEARER_TOKEN
        elif "access_token" in credentials and "api_key" in credentials:
            auth_method = AuthMethod.OAUTH2_USER
        else:
            auth_method = AuthMethod.SESSION_TOKEN

        account = SocialAccount(
            id=account_id,
            platform=platform,
            auth_method=auth_method,
            credentials=credentials,
            status=AccountStatus.ACTIVE,
        )
        self._accounts[account_id] = account
        log.info("social_account_added", account_id=account_id, platform=platform)
        return account_id

    def remove_account(self, account_id: str) -> None:
        """Remove a social media account."""
        if account_id not in self._accounts:
            raise KeyError(f"Account {account_id!r} not found")
        del self._accounts[account_id]
        log.info("social_account_removed", account_id=account_id)

    def list_accounts(self) -> list[dict[str, Any]]:
        """List all connected accounts with status."""
        return [acc.to_dict() for acc in self._accounts.values()]

    # ── Tracking configuration ────────────────────────────────────────────

    def track_tickers(self, tickers: list[str]) -> None:
        """Add tickers to track (e.g. ['SPY', 'QQQ', 'NIFTY'])."""
        self._tracked_tickers.update(t.upper() for t in tickers)
        log.info("social_tickers_updated", tickers=sorted(self._tracked_tickers))

    def track_influencers(self, account_ids: list[str]) -> None:
        """Add influencer account IDs to track."""
        self._influencer_accounts.update(account_ids)
        log.info("social_influencers_updated", count=len(self._influencer_accounts))

    def track_hashtags(self, hashtags: list[str]) -> None:
        """Add hashtags to track."""
        for tag in hashtags:
            normalized = tag if tag.startswith("#") else f"#{tag}"
            if normalized not in self._tracked_hashtags:
                self._tracked_hashtags.append(normalized)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the social sentiment stream."""
        if self._running:
            return
        self._running = True
        self._client = httpx.AsyncClient(timeout=30.0)

        # Start polling task
        self._tasks.append(
            asyncio.create_task(
                self._poll_loop(),
                name="social_poll_loop",
            )
        )
        # Start aggregation task
        self._tasks.append(
            asyncio.create_task(
                self._aggregation_loop(),
                name="social_aggregation_loop",
            )
        )
        log.info(
            "social_stream_started",
            tracked_tickers=len(self._tracked_tickers),
            accounts=len(self._accounts),
        )

    async def stop(self) -> None:
        """Stop the social sentiment stream."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

        log.info("social_stream_stopped")

    # ── Public queries ────────────────────────────────────────────────────

    def get_sentiment(self, symbol: str) -> TickerSentiment:
        """Return current aggregate sentiment for a symbol."""
        agg = self._sentiment_agg.get_aggregate(symbol.upper())
        z_score = self._chatter_tracker.update(symbol.upper(), agg.get("count_1h", 0))

        return TickerSentiment(
            symbol=symbol.upper(),
            score_1h=agg.get("score_1h", 0.0),
            score_4h=agg.get("score_4h", 0.0),
            score_24h=agg.get("score_24h", 0.0),
            magnitude=abs(agg.get("score_1h", 0.0)),
            mention_count_1h=agg.get("count_1h", 0),
            mention_count_4h=agg.get("count_4h", 0),
            mention_count_24h=agg.get("count_24h", 0),
            sources_count=len(self._accounts),
            trending=z_score >= self._spike_z_threshold,
            spike_z_score=round(z_score, 2),
        )

    def get_trending(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return trending tickers ranked by mention count and sentiment shift."""
        results: list[dict[str, Any]] = []

        for symbol in self._sentiment_agg.all_symbols():
            agg = self._sentiment_agg.get_aggregate(symbol)
            count_1h = agg.get("count_1h", 0)
            count_4h = agg.get("count_4h", 0)

            # Change percentage: 1h count vs 4h average hourly rate
            avg_hourly = count_4h / 4 if count_4h > 0 else 0
            change_pct = (
                ((count_1h - avg_hourly) / avg_hourly * 100)
                if avg_hourly > 0
                else 0.0
            )

            results.append({
                "symbol": symbol,
                "mentions": count_1h,
                "sentiment": agg.get("score_1h", 0.0),
                "change_pct": round(change_pct, 1),
            })

        # Sort by mentions descending
        results.sort(key=lambda x: x["mentions"], reverse=True)
        return results[:limit]

    def get_influencer_feed(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent influencer tweets."""
        return [t.to_dict() for t in list(self._influencer_tweets)[:limit]]

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    # ── Internal polling ──────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Poll Twitter API for recent tweets matching tracked criteria."""
        while self._running:
            try:
                await self._fetch_and_process()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.error("social_poll_error", exc_info=True)

            await asyncio.sleep(self._poll_interval)

    async def _aggregation_loop(self) -> None:
        """Periodically publish aggregate sentiment events."""
        while self._running:
            try:
                for symbol in list(self._tracked_tickers):
                    sentiment = self.get_sentiment(symbol)
                    if sentiment.mention_count_1h > 0:
                        await self._event_bus.publish("SENTIMENT", sentiment.to_dict())
            except asyncio.CancelledError:
                raise
            except Exception:
                log.error("social_aggregation_error", exc_info=True)

            await asyncio.sleep(300)  # every 5 minutes

    async def _fetch_and_process(self) -> None:
        """Fetch tweets from all active accounts and process them."""
        if not self._tracked_tickers and not self._influencer_accounts:
            return

        for account in self._accounts.values():
            if account.status != AccountStatus.ACTIVE:
                continue

            try:
                tweets = await self._fetch_tweets(account)
                for tweet_data in tweets:
                    item = self._process_tweet(tweet_data, account)
                    if item:
                        self._recent_tweets.appendleft(item)
                        if item.is_influencer:
                            self._influencer_tweets.appendleft(item)

                        # Update aggregator for each mentioned ticker
                        for ticker in item.tickers:
                            self._sentiment_agg.add(
                                ticker,
                                item.sentiment_score,
                                item.influence_score,
                            )

                account.last_used = datetime.now(timezone.utc)
                account.error_message = ""
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    account.status = AccountStatus.RATE_LIMITED
                    account.error_message = "Rate limited"
                    log.warning("social_rate_limited", account_id=account.id)
                else:
                    account.status = AccountStatus.ERROR
                    account.error_message = f"HTTP {exc.response.status_code}"
                    log.error("social_fetch_error", account_id=account.id, status=exc.response.status_code)
            except Exception as exc:
                account.status = AccountStatus.ERROR
                account.error_message = str(exc)[:200]
                log.error("social_fetch_error", account_id=account.id, exc_info=True)

    async def _fetch_tweets(
        self,
        account: SocialAccount,
    ) -> list[dict[str, Any]]:
        """Fetch recent tweets from the Twitter API v2."""
        if self._client is None:
            return []

        bearer = account.credentials.get(
            "bearer_token",
            account.credentials.get("access_token", ""),
        )
        if not bearer:
            return []

        # Build query: tracked tickers + hashtags
        parts: list[str] = []
        for ticker in self._tracked_tickers:
            parts.append(f"${ticker}")
        for tag in self._tracked_hashtags[:5]:  # limit to avoid overly long query
            parts.append(tag)

        if not parts:
            return []

        query = " OR ".join(parts)
        # Twitter limits query length to 512 chars
        if len(query) > 512:
            query = query[:512]

        headers = {
            "Authorization": f"Bearer {bearer}",
            "Accept": "application/json",
        }
        params: dict[str, Any] = {
            "query": query,
            "max_results": 100,
            "tweet.fields": "created_at,author_id,public_metrics",
            "user.fields": "name,public_metrics",
            "expansions": "author_id",
        }

        resp = await self._client.get(
            "https://api.twitter.com/2/tweets/search/recent",
            headers=headers,
            params=params,
        )
        resp.raise_for_status()
        payload = resp.json()

        # Build author lookup from includes
        authors: dict[str, dict[str, Any]] = {}
        for user in payload.get("includes", {}).get("users", []):
            authors[user["id"]] = user

        tweets = payload.get("data", [])
        # Attach author info
        for tweet in tweets:
            author = authors.get(tweet.get("author_id", ""), {})
            tweet["_author_name"] = author.get("name", "")
            tweet["_author_followers"] = (
                author.get("public_metrics", {}).get("followers_count", 0)
            )

        return tweets

    def _process_tweet(
        self,
        tweet_data: dict[str, Any],
        account: SocialAccount,
    ) -> Optional[TweetItem]:
        """Process a raw tweet into an enriched TweetItem."""
        text = tweet_data.get("text", "")
        if not text:
            return None

        # Extract tickers
        tickers = [m.group(1) for m in _TICKER_PATTERN.finditer(text)]
        if not tickers:
            # If no cashtag found, skip unless from an influencer
            author_id = tweet_data.get("author_id", "")
            if author_id not in self._influencer_accounts:
                return None

        # Sentiment scoring using lexicon
        from hedgefund.data.social_feed import _lexicon_sentiment

        score, magnitude = _lexicon_sentiment(text)

        # Influence weighting
        followers = tweet_data.get("_author_followers", 0)
        metrics = tweet_data.get("public_metrics", {})
        likes = metrics.get("like_count", 0)
        retweets = metrics.get("retweet_count", 0)

        post = SocialPost(
            text=text,
            author_id=tweet_data.get("author_id", ""),
            timestamp=datetime.now(timezone.utc),
            platform="twitter",
            followers=followers,
            likes=likes,
            reposts=retweets,
            sentiment_score=score,
        )
        influence = _influence_weight(post)

        # Parse timestamp
        created_str = tweet_data.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            created_at = datetime.now(timezone.utc)

        author_id = tweet_data.get("author_id", "")
        is_influencer = author_id in self._influencer_accounts

        return TweetItem(
            id=tweet_data.get("id", uuid.uuid4().hex[:12]),
            text=text[:500],
            author_id=author_id,
            author_name=tweet_data.get("_author_name", ""),
            created_at=created_at,
            tickers=tickers,
            sentiment_score=score,
            influence_score=round(influence, 4),
            followers=followers,
            likes=likes,
            retweets=retweets,
            is_influencer=is_influencer,
        )
