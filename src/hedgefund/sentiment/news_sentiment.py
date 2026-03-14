"""News headline sentiment scorer using transformer models.

Uses FinBERT when available, falling back to distilbert-base-uncased-finetuned-
sst-2-english, and finally to a deterministic rule-based scorer if no model can
be loaded.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any

import structlog

from hedgefund.sentiment.base import SentimentScorer
from hedgefund.types import SentimentResult

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

_BULLISH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(beat[s]?|surpass|exceed|top[s]?)\b.{0,30}\b(estimate|expectation|forecast)",
        r"\b(upgrade[sd]?|raise[sd]?|boost[sd]?|surge[sd]?|rally|rallies|soar)",
        r"\b(record\s+(high|revenue|profit|earnings))",
        r"\b(strong|robust|solid|stellar|blowout)\b.{0,20}\b(quarter|earnings|result|growth)",
        r"\b(buy\s*back|buyback|dividend\s+(hike|increase|raise))",
        r"\b(breakout|new\s+high|all[- ]time\s+high)",
    )
]

_BEARISH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(miss|fall\s+short|disappoint|below)\b.{0,30}\b(estimate|expectation|forecast)",
        r"\b(downgrade[sd]?|cut[s]?|slash|plunge[sd]?|tumble[sd]?|crash|sell[- ]?off)",
        r"\b(warn|warning|guidance\s+(lower|cut|reduce))",
        r"\b(weak|poor|dismal|terrible)\b.{0,20}\b(quarter|earnings|result|outlook)",
        r"\b(layoff|restructur|recall|lawsuit|investigat|fraud|default)",
        r"\b(new\s+low|52[- ]week\s+low|breakdown)",
    )
]


def _rule_based_score(headline: str) -> tuple[float, float]:
    """Return ``(score, magnitude)`` using keyword heuristics."""
    bull = sum(1 for p in _BULLISH_PATTERNS if p.search(headline))
    bear = sum(1 for p in _BEARISH_PATTERNS if p.search(headline))
    total = bull + bear
    if total == 0:
        return 0.0, 0.0
    score = (bull - bear) / total
    magnitude = min(total / len(_BULLISH_PATTERNS), 1.0)
    return score, magnitude


# ---------------------------------------------------------------------------
# LRU cache for scored headlines
# ---------------------------------------------------------------------------

class _LRUCache:
    """Thread-safe bounded LRU cache."""

    def __init__(self, maxsize: int = 2048) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> tuple[float, float] | None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    async def put(self, key: str, value: tuple[float, float]) -> None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
            self._cache[key] = value


# ---------------------------------------------------------------------------
# Transformer-backed scorer
# ---------------------------------------------------------------------------

_MODEL_CANDIDATES = [
    "ProsusAI/finbert",
    "distilbert-base-uncased-finetuned-sst-2-english",
]

# Mapping from model-specific labels to a normalised float.
_LABEL_MAP: dict[str, float] = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
    "POSITIVE": 1.0,
    "NEGATIVE": -1.0,
    "NEUTRAL": 0.0,
    "LABEL_0": -1.0,  # distilbert: NEGATIVE
    "LABEL_1": 1.0,   # distilbert: POSITIVE
}


class NewsSentimentScorer(SentimentScorer):
    """Score financial headlines with a transformer model.

    Initialisation lazily loads the first available model from
    ``_MODEL_CANDIDATES``.  If ``transformers`` is not installed the scorer
    falls back to a fast rule-based heuristic.

    Parameters:
        max_batch: Maximum number of headlines per model forward pass.
        cache_size: Maximum entries in the result cache.
        headlines_provider: Optional async callable ``(symbol) -> list[str]``
            that fetches recent headlines.  When *None* the scorer returns a
            neutral result (useful in dry-run / backtest modes).
    """

    def __init__(
        self,
        *,
        max_batch: int = 32,
        cache_size: int = 2048,
        headlines_provider: Any | None = None,
    ) -> None:
        self._max_batch = max_batch
        self._cache = _LRUCache(maxsize=cache_size)
        self._headlines_provider = headlines_provider
        self._pipeline: Any | None = None
        self._model_name: str = "rule_based"
        self._initialised = False

    # -- lazy init ----------------------------------------------------------

    def _ensure_model(self) -> None:
        """Try to load the first available transformer pipeline."""
        if self._initialised:
            return
        self._initialised = True
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]
        except ImportError:
            log.warning("transformers not installed; using rule-based fallback")
            return

        for model_name in _MODEL_CANDIDATES:
            try:
                self._pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    truncation=True,
                    max_length=512,
                )
                self._model_name = model_name
                log.info("loaded_sentiment_model", model=model_name)
                return
            except Exception:
                log.debug("model_load_failed", model=model_name, exc_info=True)

        log.warning("no_transformer_model_available; using rule-based fallback")

    # -- transformer inference ----------------------------------------------

    def _score_headlines_transformer(
        self, headlines: list[str]
    ) -> list[tuple[float, float]]:
        """Run the transformer pipeline on *headlines* in batches."""
        results: list[tuple[float, float]] = []
        for i in range(0, len(headlines), self._max_batch):
            batch = headlines[i : i + self._max_batch]
            preds = self._pipeline(batch)  # type: ignore[misc]
            for pred in preds:
                label: str = pred["label"]
                prob: float = pred["score"]
                direction = _LABEL_MAP.get(label, 0.0)
                results.append((direction * prob, prob))
        return results

    # -- public API ---------------------------------------------------------

    async def score(self, symbol: str) -> SentimentResult:
        """Score the latest headlines for *symbol*."""
        self._ensure_model()

        headlines = await self._fetch_headlines(symbol)
        if not headlines:
            return SentimentResult(
                symbol=symbol,
                score=0.0,
                magnitude=0.0,
                source=f"news:{self._model_name}",
                headline="",
                timestamp=datetime.utcnow(),
            )

        scores = await self._score_many(headlines)
        avg_score = sum(s for s, _ in scores) / len(scores)
        avg_mag = sum(m for _, m in scores) / len(scores)

        return SentimentResult(
            symbol=symbol,
            score=max(-1.0, min(1.0, avg_score)),
            magnitude=max(0.0, min(1.0, avg_mag)),
            source=f"news:{self._model_name}",
            headline=headlines[0],
            timestamp=datetime.utcnow(),
        )

    async def score_batch(self, symbols: list[str]) -> list[SentimentResult]:
        """Score multiple symbols concurrently."""
        return list(await asyncio.gather(*(self.score(s) for s in symbols)))

    # -- internals ----------------------------------------------------------

    async def _fetch_headlines(self, symbol: str) -> list[str]:
        if self._headlines_provider is not None:
            return await self._headlines_provider(symbol)  # type: ignore[no-any-return]
        return []

    async def _score_many(
        self, headlines: list[str]
    ) -> list[tuple[float, float]]:
        """Score headlines, checking cache first."""
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        results: list[tuple[float, float] | None] = [None] * len(headlines)

        for idx, hl in enumerate(headlines):
            cached = await self._cache.get(hl)
            if cached is not None:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(hl)

        if uncached_texts:
            if self._pipeline is not None:
                new_scores = await asyncio.to_thread(
                    self._score_headlines_transformer, uncached_texts
                )
            else:
                new_scores = [_rule_based_score(t) for t in uncached_texts]

            for pos, idx in enumerate(uncached_indices):
                results[idx] = new_scores[pos]
                await self._cache.put(uncached_texts[pos], new_scores[pos])

        return [r for r in results if r is not None]
