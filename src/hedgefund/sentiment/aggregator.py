"""Weighted ensemble aggregator for multiple sentiment sources."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import structlog

from hedgefund.sentiment.base import SentimentScorer
from hedgefund.types import SentimentResult

log = structlog.get_logger(__name__)


class SentimentAggregator(SentimentScorer):
    """Combine multiple :class:`SentimentScorer` instances into one score.

    Parameters:
        scorers: Mapping of ``name -> (scorer, weight)``.
        magnitude_threshold: Minimum magnitude for a component to be included
            in the final blend.  Components below the threshold are treated as
            neutral (score=0, magnitude=0) so that low-confidence noise does
            not dilute the signal.
        min_sources: Minimum number of above-threshold sources required.  If
            fewer pass, the aggregator returns a neutral result.
    """

    def __init__(
        self,
        scorers: dict[str, tuple[SentimentScorer, float]],
        *,
        magnitude_threshold: float = 0.10,
        min_sources: int = 1,
    ) -> None:
        if not scorers:
            raise ValueError("At least one scorer must be provided")

        self._scorers = scorers
        self._magnitude_threshold = magnitude_threshold
        self._min_sources = min_sources

        # Normalise weights to sum to 1.
        total_w = sum(w for _, w in scorers.values())
        if total_w <= 0:
            raise ValueError("Total weight must be positive")
        self._weights: dict[str, float] = {
            name: w / total_w for name, (_, w) in scorers.items()
        }

    # -- public API ---------------------------------------------------------

    async def score(self, symbol: str) -> SentimentResult:
        """Score *symbol* across all sources and blend."""
        component_results = await self._gather_components(symbol)
        return self._blend(symbol, component_results)

    async def score_batch(self, symbols: list[str]) -> list[SentimentResult]:
        return list(await asyncio.gather(*(self.score(s) for s in symbols)))

    # -- internals ----------------------------------------------------------

    async def _gather_components(
        self, symbol: str
    ) -> dict[str, SentimentResult]:
        async def _safe_score(
            name: str, scorer: SentimentScorer
        ) -> tuple[str, SentimentResult | None]:
            try:
                return name, await scorer.score(symbol)
            except Exception:
                log.error("scorer_failed", scorer=name, symbol=symbol, exc_info=True)
                return name, None

        tasks = [
            _safe_score(name, scorer)
            for name, (scorer, _) in self._scorers.items()
        ]
        raw = await asyncio.gather(*tasks)
        return {name: res for name, res in raw if res is not None}

    def _blend(
        self, symbol: str, components: dict[str, SentimentResult]
    ) -> SentimentResult:
        """Weighted blend with magnitude gating."""
        weighted_score = 0.0
        weighted_mag = 0.0
        effective_weight = 0.0
        active_sources = 0
        details: list[str] = []

        for name, result in components.items():
            w = self._weights.get(name, 0.0)
            if result.magnitude < self._magnitude_threshold:
                log.debug(
                    "scorer_below_threshold",
                    scorer=name,
                    magnitude=result.magnitude,
                    threshold=self._magnitude_threshold,
                )
                continue

            active_sources += 1
            weighted_score += result.score * w
            weighted_mag += result.magnitude * w
            effective_weight += w
            details.append(f"{name}={result.score:+.2f}")

        if active_sources < self._min_sources or effective_weight == 0:
            return SentimentResult(
                symbol=symbol,
                score=0.0,
                magnitude=0.0,
                source="aggregator",
                headline=f"Insufficient sources ({active_sources}/{self._min_sources})",
                timestamp=datetime.utcnow(),
            )

        final_score = weighted_score / effective_weight
        final_mag = weighted_mag / effective_weight

        return SentimentResult(
            symbol=symbol,
            score=max(-1.0, min(1.0, final_score)),
            magnitude=max(0.0, min(1.0, final_mag)),
            source="aggregator",
            headline=" | ".join(details),
            timestamp=datetime.utcnow(),
        )
