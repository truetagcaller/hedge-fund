"""Signal ensemble: combine multiple generators with conflict resolution."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any

import pandas as pd
import structlog

from hedgefund.signals.base import SignalGenerator
from hedgefund.types import (
    MarketRegime,
    SentimentResult,
    SignalAction,
    SignalDirection,
    TradeSignal,
)

log = structlog.get_logger(__name__)


class SignalEnsemble(SignalGenerator):
    """Weighted ensemble of signal generators with conflict resolution.

    Parameters:
        generators: Mapping of ``name -> (generator, weight)``.
        min_confidence: Signals below this threshold are discarded.
        min_rr: Signals below this risk/reward ratio are discarded.
        conflict_policy: How to handle opposing signals for the same underlying.
            ``"highest_confidence"`` keeps only the signal with the highest
            weighted confidence.  ``"drop"`` discards all conflicting signals.
    """

    def __init__(
        self,
        generators: dict[str, tuple[SignalGenerator, float]],
        *,
        min_confidence: float = 0.50,
        min_rr: float = 2.0,
        conflict_policy: str = "highest_confidence",
    ) -> None:
        if not generators:
            raise ValueError("At least one generator is required")

        self._generators = generators
        self._min_conf = min_confidence
        self._min_rr = min_rr
        self._conflict_policy = conflict_policy

        # Normalise weights.
        total = sum(w for _, w in generators.values())
        if total <= 0:
            raise ValueError("Total generator weight must be positive")
        self._weights: dict[str, float] = {
            name: w / total for name, (_, w) in generators.items()
        }

    async def generate(
        self,
        features_df: pd.DataFrame,
        regime: MarketRegime,
        sentiment: SentimentResult,
    ) -> list[TradeSignal]:
        # Gather signals from all generators concurrently.
        raw_signals = await self._gather(features_df, regime, sentiment)

        # Apply weight to confidence.
        weighted: list[TradeSignal] = []
        for name, signals in raw_signals.items():
            w = self._weights[name]
            for sig in signals:
                sig.confidence = sig.confidence * w
                sig.metadata["ensemble_source"] = name
                sig.metadata["ensemble_weight"] = round(w, 4)
                weighted.append(sig)

        # Filter by minimum confidence and RR.
        filtered = [
            s
            for s in weighted
            if s.confidence >= self._min_conf and s.risk_reward_ratio >= self._min_rr
        ]

        # Resolve conflicts.
        resolved = self._resolve_conflicts(filtered)

        log.info(
            "ensemble_result",
            raw=sum(len(v) for v in raw_signals.values()),
            filtered=len(filtered),
            resolved=len(resolved),
        )
        return resolved

    # -- gathering ----------------------------------------------------------

    async def _gather(
        self,
        features_df: pd.DataFrame,
        regime: MarketRegime,
        sentiment: SentimentResult,
    ) -> dict[str, list[TradeSignal]]:
        async def _safe_generate(
            name: str, gen: SignalGenerator
        ) -> tuple[str, list[TradeSignal]]:
            try:
                signals = await gen.generate(features_df, regime, sentiment)
                return name, signals
            except Exception:
                log.error("generator_failed", generator=name, exc_info=True)
                return name, []

        tasks = [
            _safe_generate(name, gen)
            for name, (gen, _) in self._generators.items()
        ]
        results = await asyncio.gather(*tasks)
        return {name: sigs for name, sigs in results}

    # -- conflict resolution -----------------------------------------------

    def _resolve_conflicts(self, signals: list[TradeSignal]) -> list[TradeSignal]:
        """Resolve opposing signals on the same underlying."""
        if self._conflict_policy == "drop":
            return self._drop_conflicts(signals)
        return self._keep_highest_confidence(signals)

    @staticmethod
    def _keep_highest_confidence(signals: list[TradeSignal]) -> list[TradeSignal]:
        """Group by underlying; when both LONG and SHORT exist, keep the one
        with the highest confidence."""
        by_underlying: dict[str, list[TradeSignal]] = defaultdict(list)
        for sig in signals:
            by_underlying[sig.underlying].append(sig)

        result: list[TradeSignal] = []
        for underlying, group in by_underlying.items():
            directions = {s.direction for s in group}
            if SignalDirection.LONG in directions and SignalDirection.SHORT in directions:
                # Conflict: keep only the highest-confidence direction.
                best = max(group, key=lambda s: s.confidence)
                same_dir = [s for s in group if s.direction == best.direction]
                result.extend(same_dir)
                log.debug(
                    "conflict_resolved",
                    underlying=underlying,
                    kept=best.direction.value,
                    dropped_count=len(group) - len(same_dir),
                )
            else:
                result.extend(group)

        return result

    @staticmethod
    def _drop_conflicts(signals: list[TradeSignal]) -> list[TradeSignal]:
        """Drop all signals for an underlying if opposing directions exist."""
        by_underlying: dict[str, list[TradeSignal]] = defaultdict(list)
        for sig in signals:
            by_underlying[sig.underlying].append(sig)

        result: list[TradeSignal] = []
        for underlying, group in by_underlying.items():
            directions = {s.direction for s in group}
            if SignalDirection.LONG in directions and SignalDirection.SHORT in directions:
                log.debug(
                    "conflict_dropped",
                    underlying=underlying,
                    count=len(group),
                )
                continue
            result.extend(group)
        return result
