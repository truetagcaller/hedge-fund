"""Options-flow based sentiment scorer.

Derives directional sentiment from observable options-market signals:
  * Put/Call ratio (volume and open-interest)
  * Large / unusual trades (size > threshold, far-OTM, short-dated)
  * Implied-volatility changes (IV crush / expansion)
"""

from __future__ import annotations

import asyncio
import math
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
class OptionsSnapshot:
    """Aggregated options statistics for one underlying at a point in time."""

    symbol: str
    timestamp: datetime

    # Volume
    call_volume: int = 0
    put_volume: int = 0

    # Open interest
    call_oi: int = 0
    put_oi: int = 0

    # Implied volatility (30-day composite or ATM average)
    current_iv: float = 0.0
    previous_iv: float = 0.0  # prior session

    # Unusual activity
    large_call_trades: int = 0  # trades > size threshold
    large_put_trades: int = 0
    large_call_premium: float = 0.0
    large_put_premium: float = 0.0


class OptionsDataProvider(Protocol):
    """Async callable returning an options snapshot."""

    async def __call__(self, symbol: str) -> OptionsSnapshot | None: ...


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class OptionsSentimentScorer(SentimentScorer):
    """Derive sentiment from options-market flow data.

    The final score is a weighted blend of three components:

    1. **Put/Call ratio signal** -- low P/C is bullish, high P/C bearish.
    2. **Unusual-trade signal** -- net premium direction of large trades.
    3. **IV-change signal** -- rising IV is bearish (fear), falling is bullish.

    Parameters:
        data_provider: Async callable returning :class:`OptionsSnapshot`.
        pc_ratio_mean: Historical mean P/C ratio used to centre the signal.
        pc_weight: Blend weight for the put/call component.
        unusual_weight: Blend weight for unusual-trade component.
        iv_weight: Blend weight for IV-change component.
        large_trade_threshold: Minimum contracts for a trade to be "large".
    """

    def __init__(
        self,
        *,
        data_provider: OptionsDataProvider | None = None,
        pc_ratio_mean: float = 0.7,
        pc_weight: float = 0.40,
        unusual_weight: float = 0.35,
        iv_weight: float = 0.25,
        large_trade_threshold: int = 100,
    ) -> None:
        self._provider = data_provider
        self._pc_mean = pc_ratio_mean
        self._w_pc = pc_weight
        self._w_unusual = unusual_weight
        self._w_iv = iv_weight
        self._large_threshold = large_trade_threshold

        total = self._w_pc + self._w_unusual + self._w_iv
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Component weights must sum to 1.0, got {total}")

    # -- public API ---------------------------------------------------------

    async def score(self, symbol: str) -> SentimentResult:
        snap = await self._fetch(symbol)
        if snap is None:
            return self._neutral(symbol)

        pc_score, pc_mag = self._put_call_signal(snap)
        ut_score, ut_mag = self._unusual_trade_signal(snap)
        iv_score, iv_mag = self._iv_change_signal(snap)

        score = (
            self._w_pc * pc_score
            + self._w_unusual * ut_score
            + self._w_iv * iv_score
        )
        magnitude = (
            self._w_pc * pc_mag
            + self._w_unusual * ut_mag
            + self._w_iv * iv_mag
        )

        log.debug(
            "options_sentiment",
            symbol=symbol,
            pc=(round(pc_score, 3), round(pc_mag, 3)),
            unusual=(round(ut_score, 3), round(ut_mag, 3)),
            iv=(round(iv_score, 3), round(iv_mag, 3)),
            final_score=round(score, 3),
        )

        return SentimentResult(
            symbol=symbol,
            score=max(-1.0, min(1.0, score)),
            magnitude=max(0.0, min(1.0, magnitude)),
            source="options_flow",
            headline=self._build_headline(snap, score),
            timestamp=snap.timestamp,
        )

    async def score_batch(self, symbols: list[str]) -> list[SentimentResult]:
        return list(await asyncio.gather(*(self.score(s) for s in symbols)))

    # -- component signals --------------------------------------------------

    def _put_call_signal(self, snap: OptionsSnapshot) -> tuple[float, float]:
        """Lower P/C -> bullish (+1), higher P/C -> bearish (-1)."""
        total_vol = snap.call_volume + snap.put_volume
        if total_vol == 0:
            return 0.0, 0.0
        pc_ratio = snap.put_volume / max(snap.call_volume, 1)
        # Normalise around historical mean; clip to [-1, 1].
        deviation = self._pc_mean - pc_ratio  # positive when calls dominate
        score = max(-1.0, min(1.0, deviation / self._pc_mean))
        magnitude = min(abs(deviation) / self._pc_mean, 1.0)
        return score, magnitude

    def _unusual_trade_signal(self, snap: OptionsSnapshot) -> tuple[float, float]:
        """Net premium direction of large/unusual trades."""
        total_premium = snap.large_call_premium + snap.large_put_premium
        if total_premium == 0:
            return 0.0, 0.0
        net = snap.large_call_premium - snap.large_put_premium
        score = net / total_premium  # already in [-1, 1]
        total_trades = snap.large_call_trades + snap.large_put_trades
        magnitude = min(math.log1p(total_trades) / math.log1p(50), 1.0)
        return max(-1.0, min(1.0, score)), magnitude

    def _iv_change_signal(self, snap: OptionsSnapshot) -> tuple[float, float]:
        """Rising IV -> bearish, falling IV -> bullish."""
        if snap.previous_iv == 0:
            return 0.0, 0.0
        pct_change = (snap.current_iv - snap.previous_iv) / snap.previous_iv
        # Invert: IV increase is fear (bearish), decrease is complacency (bullish).
        score = max(-1.0, min(1.0, -pct_change * 5.0))
        magnitude = min(abs(pct_change) * 10.0, 1.0)
        return score, magnitude

    # -- helpers -----------------------------------------------------------

    async def _fetch(self, symbol: str) -> OptionsSnapshot | None:
        if self._provider is None:
            return None
        try:
            return await self._provider(symbol)
        except Exception:
            log.error("options_fetch_failed", symbol=symbol, exc_info=True)
            return None

    @staticmethod
    def _neutral(symbol: str) -> SentimentResult:
        return SentimentResult(
            symbol=symbol,
            score=0.0,
            magnitude=0.0,
            source="options_flow",
            headline="",
            timestamp=datetime.now(timezone.utc),
        )

    @staticmethod
    def _build_headline(snap: OptionsSnapshot, score: float) -> str:
        pc = snap.put_volume / max(snap.call_volume, 1)
        direction = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
        return (
            f"P/C={pc:.2f} | "
            f"Large calls={snap.large_call_trades} puts={snap.large_put_trades} | "
            f"IV {snap.current_iv:.1%}->{snap.previous_iv:.1%} | {direction}"
        )
