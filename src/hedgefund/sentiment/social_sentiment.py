"""Social media sentiment scorer with influence weighting and spike detection."""

from __future__ import annotations

import asyncio
import math
import statistics
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import structlog

from hedgefund.sentiment.base import SentimentScorer
from hedgefund.types import SentimentResult

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SocialPost:
    """A single social-media post with metadata."""

    text: str
    author_id: str
    timestamp: datetime
    platform: str  # e.g. "twitter", "reddit", "stocktwits"
    followers: int = 0
    likes: int = 0
    reposts: int = 0
    sentiment_score: float | None = None  # pre-scored if available


class SocialDataProvider(Protocol):
    """Async callable returning recent posts for a symbol."""

    async def __call__(self, symbol: str) -> list[SocialPost]: ...


# ---------------------------------------------------------------------------
# Influence weighting
# ---------------------------------------------------------------------------

def _influence_weight(post: SocialPost) -> float:
    """Compute an influence weight in (0, 1] for a post.

    Higher-follower, higher-engagement posts receive more weight.
    """
    follower_w = math.log1p(post.followers) / math.log1p(1_000_000)
    engagement = post.likes + post.reposts * 2
    engage_w = math.log1p(engagement) / math.log1p(100_000)
    raw = 0.5 * min(follower_w, 1.0) + 0.5 * min(engage_w, 1.0)
    # Ensure a minimum floor so low-engagement posts still contribute.
    return max(raw, 0.05)


# ---------------------------------------------------------------------------
# Chatter spike detection
# ---------------------------------------------------------------------------

class _ChatterTracker:
    """Rolling window tracker for post volume to detect spikes."""

    def __init__(self, window_size: int = 24) -> None:
        self._window_size = window_size
        self._history: dict[str, deque[int]] = {}

    def update(self, symbol: str, count: int) -> float:
        """Record *count* posts and return the spike z-score.

        Returns 0.0 when not enough history is available.
        """
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


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class SocialSentimentScorer(SentimentScorer):
    """Aggregate social-media sentiment with influence weighting.

    Parameters:
        data_provider: Async callable ``(symbol) -> list[SocialPost]``.
        spike_z_threshold: Z-score above which a chatter spike is flagged.
            The magnitude is boosted when a spike is detected.
        window_size: Number of periods for the rolling chatter tracker.
    """

    def __init__(
        self,
        *,
        data_provider: SocialDataProvider | None = None,
        spike_z_threshold: float = 2.0,
        window_size: int = 24,
    ) -> None:
        self._provider = data_provider
        self._spike_z = spike_z_threshold
        self._chatter = _ChatterTracker(window_size=window_size)

    async def score(self, symbol: str) -> SentimentResult:
        posts = await self._fetch_posts(symbol)
        if not posts:
            return self._neutral(symbol)

        # -- influence-weighted score --------------------------------------
        weighted_sum = 0.0
        weight_total = 0.0
        for post in posts:
            raw_score = post.sentiment_score if post.sentiment_score is not None else 0.0
            w = _influence_weight(post)
            weighted_sum += raw_score * w
            weight_total += w

        score = weighted_sum / weight_total if weight_total > 0 else 0.0

        # -- chatter spike detection ---------------------------------------
        z_score = self._chatter.update(symbol, len(posts))
        spike_detected = z_score >= self._spike_z

        # Magnitude reflects both conviction spread and spike presence.
        raw_mag = min(abs(score), 1.0)
        if spike_detected:
            raw_mag = min(raw_mag + 0.2, 1.0)
            log.info(
                "chatter_spike_detected",
                symbol=symbol,
                z_score=round(z_score, 2),
                post_count=len(posts),
            )

        return SentimentResult(
            symbol=symbol,
            score=max(-1.0, min(1.0, score)),
            magnitude=raw_mag,
            source="social",
            headline=f"Aggregated {len(posts)} posts (spike={spike_detected})",
            timestamp=datetime.now(timezone.utc),
        )

    async def score_batch(self, symbols: list[str]) -> list[SentimentResult]:
        return list(await asyncio.gather(*(self.score(s) for s in symbols)))

    # -- helpers -----------------------------------------------------------

    async def _fetch_posts(self, symbol: str) -> list[SocialPost]:
        if self._provider is None:
            return []
        try:
            return await self._provider(symbol)
        except Exception:
            log.error("social_fetch_failed", symbol=symbol, exc_info=True)
            return []

    @staticmethod
    def _neutral(symbol: str) -> SentimentResult:
        return SentimentResult(
            symbol=symbol,
            score=0.0,
            magnitude=0.0,
            source="social",
            headline="",
            timestamp=datetime.now(timezone.utc),
        )
