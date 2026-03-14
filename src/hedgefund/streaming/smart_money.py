"""Smart money detection algorithms.

Identifies institutional activity through block trades, volume spikes,
VWAP deviations, hidden liquidity, accumulation/distribution patterns,
liquidity sweeps, and stop hunts. All methods operate on numpy arrays
for performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hedgefund.logger import get_logger
from hedgefund.streaming.order_book import OrderBook
from hedgefund.types import Side

log = get_logger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass(slots=True)
class BlockTrade:
    """A detected block trade (unusually large single trade)."""

    price: float
    volume: float
    average_volume: float
    multiplier: float
    side: Side
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class VolumeSpikeResult:
    """Result of volume spike detection."""

    is_spike: bool
    magnitude: float  # how many standard deviations above mean
    current_volume: float
    average_volume: float
    threshold: float


@dataclass(slots=True)
class LiquiditySweep:
    """A detected liquidity sweep event.

    A sweep is when price breaks above resistance or below support,
    then reverses within N bars.
    """

    direction: Side  # BUY = swept above resistance, SELL = swept below support
    sweep_price: float
    reversal_price: float
    level_price: float  # the support/resistance level that was swept
    bar_index: int
    magnitude: float  # distance of sweep beyond the level


@dataclass(slots=True)
class StopHunt:
    """A detected stop-hunt event."""

    direction: Side  # BUY = hunted above resistance, SELL = hunted below support
    trigger_price: float
    reversal_price: float
    level: float
    penetration_pct: float
    bar_index: int


@dataclass(slots=True)
class AccDistSignal:
    """Accumulation/distribution signal."""

    value: float  # current A/D line value
    trend: str  # "accumulation", "distribution", or "neutral"
    divergence: bool  # True if A/D diverges from price
    strength: float  # 0-1 signal strength


class SmartMoneyDetector:
    """Detects institutional / smart-money activity from market data.

    All detection methods accept numpy arrays and return structured results.

    Usage::

        detector = SmartMoneyDetector()
        blocks = detector.detect_block_trades(trades)
        spike = detector.detect_volume_spike(volumes)
        score = detector.aggregate_smart_money_score(all_signals)
    """

    # ── Block trade detection ─────────────────────────────────────────────

    def detect_block_trades(
        self,
        trades: List[Dict[str, Any]],
        threshold_multiplier: float = 10.0,
    ) -> List[BlockTrade]:
        """Detect trades whose volume significantly exceeds the average.

        Args:
            trades: List of trade dicts with ``price``, ``volume``, and
                optionally ``side`` keys.
            threshold_multiplier: A trade is flagged as a block if its volume
                exceeds the average by this factor.

        Returns:
            List of detected :class:`BlockTrade` instances.
        """
        if not trades:
            return []

        volumes = np.array([float(t.get("volume", t.get("quantity", 0))) for t in trades])
        avg_vol = float(np.mean(volumes))

        if avg_vol <= 0:
            return []

        threshold = avg_vol * threshold_multiplier
        results: List[BlockTrade] = []

        for i, trade in enumerate(trades):
            vol = float(volumes[i])
            if vol >= threshold:
                side_raw = trade.get("side", "").upper()
                side = Side.BUY if side_raw in ("BUY", "B") else Side.SELL
                results.append(
                    BlockTrade(
                        price=float(trade.get("price", trade.get("last_price", 0))),
                        volume=vol,
                        average_volume=avg_vol,
                        multiplier=vol / avg_vol,
                        side=side,
                    )
                )

        if results:
            log.info("block_trades_detected", count=len(results))

        return results

    # ── Volume spike detection ────────────────────────────────────────────

    def detect_volume_spike(
        self,
        volumes: np.ndarray,
        window: int = 20,
        threshold: float = 3.0,
    ) -> VolumeSpikeResult:
        """Detect if the latest volume is a spike relative to the rolling window.

        Args:
            volumes: Array of volume values (most recent last).
            window: Lookback window for computing mean and std.
            threshold: Number of standard deviations above mean to flag a spike.

        Returns:
            :class:`VolumeSpikeResult` with spike detection and magnitude.
        """
        volumes = np.asarray(volumes, dtype=np.float64)

        if len(volumes) < window + 1:
            return VolumeSpikeResult(
                is_spike=False,
                magnitude=0.0,
                current_volume=float(volumes[-1]) if len(volumes) > 0 else 0.0,
                average_volume=float(np.mean(volumes)) if len(volumes) > 0 else 0.0,
                threshold=threshold,
            )

        lookback = volumes[-(window + 1) : -1]
        current = float(volumes[-1])
        mean = float(np.mean(lookback))
        std = float(np.std(lookback))

        if std == 0:
            magnitude = 0.0
        else:
            magnitude = (current - mean) / std

        return VolumeSpikeResult(
            is_spike=magnitude >= threshold,
            magnitude=round(magnitude, 4),
            current_volume=current,
            average_volume=round(mean, 4),
            threshold=threshold,
        )

    # ── VWAP deviation ────────────────────────────────────────────────────

    def detect_vwap_deviation(
        self,
        price: float,
        vwap: float,
        threshold: float = 0.02,
    ) -> float:
        """Compute deviation of *price* from *vwap* as a fraction.

        Args:
            price: Current price.
            vwap: Volume-weighted average price.
            threshold: Minimum deviation fraction to consider significant.

        Returns:
            Signed deviation score. Positive = price above VWAP.
            Zero if deviation is below *threshold*.
        """
        if vwap == 0:
            return 0.0

        deviation = (price - vwap) / vwap
        if abs(deviation) < threshold:
            return 0.0
        return round(deviation, 6)

    # ── Hidden liquidity detection ────────────────────────────────────────

    def detect_hidden_liquidity(
        self,
        order_book: OrderBook,
        trades: List[Dict[str, Any]],
    ) -> float:
        """Estimate hidden (iceberg) liquidity from trade-vs-book analysis.

        If the traded volume at a price level consistently exceeds the
        visible book quantity, hidden orders are likely present.

        Args:
            order_book: Current order book state.
            trades: Recent trades.

        Returns:
            Score in [0, 1]. Higher values indicate more hidden liquidity.
        """
        if not trades:
            return 0.0

        snapshot = order_book.get_snapshot(levels=10)
        visible_bid_qty = {
            lv["price"]: lv["quantity"] for lv in snapshot.get("bids", [])
        }
        visible_ask_qty = {
            lv["price"]: lv["quantity"] for lv in snapshot.get("asks", [])
        }

        hidden_signals = 0
        total_checks = 0

        for trade in trades:
            price = float(trade.get("price", trade.get("last_price", 0)))
            vol = float(trade.get("volume", trade.get("quantity", 0)))

            # Check against visible levels
            for visible in (visible_bid_qty, visible_ask_qty):
                for book_price, book_qty in visible.items():
                    if abs(price - book_price) / max(price, 1e-9) < 0.001:
                        total_checks += 1
                        if vol > book_qty * 1.5:
                            hidden_signals += 1

        if total_checks == 0:
            return 0.0

        score = hidden_signals / total_checks
        return round(min(score, 1.0), 4)

    # ── Accumulation / Distribution ───────────────────────────────────────

    def detect_accumulation_distribution(
        self,
        ohlcv_data: np.ndarray,
        window: int = 20,
    ) -> AccDistSignal:
        """Compute the Accumulation/Distribution line and detect divergence.

        Args:
            ohlcv_data: Array of shape ``(N, 5)`` with columns
                ``[open, high, low, close, volume]``.
            window: Lookback window for trend detection.

        Returns:
            :class:`AccDistSignal` with A/D value, trend, and divergence flag.
        """
        ohlcv = np.asarray(ohlcv_data, dtype=np.float64)

        if ohlcv.ndim != 2 or ohlcv.shape[1] < 5 or ohlcv.shape[0] < window:
            return AccDistSignal(value=0.0, trend="neutral", divergence=False, strength=0.0)

        high = ohlcv[:, 1]
        low = ohlcv[:, 2]
        close = ohlcv[:, 3]
        volume = ohlcv[:, 4]

        # Money Flow Multiplier
        hl_range = high - low
        # Avoid division by zero
        hl_range = np.where(hl_range == 0, 1e-9, hl_range)
        mfm = ((close - low) - (high - close)) / hl_range

        # Money Flow Volume
        mfv = mfm * volume

        # Cumulative A/D line
        ad_line = np.cumsum(mfv)

        current_ad = float(ad_line[-1])

        # Trend: compare recent A/D slope with price slope
        ad_recent = ad_line[-window:]
        price_recent = close[-window:]

        ad_slope = float(np.polyfit(np.arange(window), ad_recent, 1)[0])
        price_slope = float(np.polyfit(np.arange(window), price_recent, 1)[0])

        if ad_slope > 0:
            trend = "accumulation"
        elif ad_slope < 0:
            trend = "distribution"
        else:
            trend = "neutral"

        # Divergence: A/D and price moving in opposite directions
        divergence = (ad_slope > 0 and price_slope < 0) or (ad_slope < 0 and price_slope > 0)

        # Strength: normalized magnitude of A/D slope
        ad_std = float(np.std(ad_recent))
        strength = min(abs(ad_slope) / max(ad_std, 1e-9), 1.0)

        return AccDistSignal(
            value=round(current_ad, 4),
            trend=trend,
            divergence=divergence,
            strength=round(strength, 4),
        )

    # ── Liquidity sweep detection ─────────────────────────────────────────

    def detect_liquidity_sweep(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        window: int = 20,
    ) -> List[LiquiditySweep]:
        """Detect liquidity sweeps: price breaks a level then reverses.

        A sweep above resistance: price exceeds the recent high, then
        closes back below it within a few bars. Vice versa for support.

        Args:
            highs: Array of high prices.
            lows: Array of low prices.
            closes: Array of close prices.
            window: Lookback window for identifying support/resistance.

        Returns:
            List of :class:`LiquiditySweep` events.
        """
        highs = np.asarray(highs, dtype=np.float64)
        lows = np.asarray(lows, dtype=np.float64)
        closes = np.asarray(closes, dtype=np.float64)
        n = len(highs)

        if n < window + 3:
            return []

        sweeps: List[LiquiditySweep] = []
        reversal_window = min(5, n - window)

        for i in range(window, n - 1):
            lookback_high = float(np.max(highs[i - window : i]))
            lookback_low = float(np.min(lows[i - window : i]))

            # Sweep above resistance (bearish)
            if float(highs[i]) > lookback_high:
                # Check for reversal: close comes back below the level
                end = min(i + reversal_window, n)
                for j in range(i, end):
                    if float(closes[j]) < lookback_high:
                        sweeps.append(
                            LiquiditySweep(
                                direction=Side.BUY,
                                sweep_price=float(highs[i]),
                                reversal_price=float(closes[j]),
                                level_price=lookback_high,
                                bar_index=i,
                                magnitude=float(highs[i]) - lookback_high,
                            )
                        )
                        break

            # Sweep below support (bullish)
            if float(lows[i]) < lookback_low:
                end = min(i + reversal_window, n)
                for j in range(i, end):
                    if float(closes[j]) > lookback_low:
                        sweeps.append(
                            LiquiditySweep(
                                direction=Side.SELL,
                                sweep_price=float(lows[i]),
                                reversal_price=float(closes[j]),
                                level_price=lookback_low,
                                bar_index=i,
                                magnitude=lookback_low - float(lows[i]),
                            )
                        )
                        break

        return sweeps

    # ── Stop hunt detection ───────────────────────────────────────────────

    def detect_stop_hunt(
        self,
        price_data: np.ndarray,
        support_levels: List[float],
        resistance_levels: List[float],
        penetration_threshold: float = 0.005,
    ) -> List[StopHunt]:
        """Detect stop-hunt patterns around known support/resistance levels.

        A stop hunt occurs when price briefly penetrates a key level
        (triggering stops) then reverses.

        Args:
            price_data: Array of shape ``(N, 4)`` with ``[open, high, low, close]``.
            support_levels: Known support price levels.
            resistance_levels: Known resistance price levels.
            penetration_threshold: Minimum penetration as fraction of the level price.

        Returns:
            List of :class:`StopHunt` events.
        """
        price_data = np.asarray(price_data, dtype=np.float64)
        if price_data.ndim != 2 or price_data.shape[1] < 4 or price_data.shape[0] < 3:
            return []

        highs = price_data[:, 1]
        lows = price_data[:, 2]
        closes = price_data[:, 3]
        n = len(highs)
        hunts: List[StopHunt] = []

        for i in range(1, n - 1):
            # Hunt below support
            for level in support_levels:
                if level <= 0:
                    continue
                low_val = float(lows[i])
                if low_val < level:
                    penetration = (level - low_val) / level
                    if penetration >= penetration_threshold:
                        # Check reversal: next bar closes above level
                        if float(closes[i + 1]) > level:
                            hunts.append(
                                StopHunt(
                                    direction=Side.SELL,
                                    trigger_price=low_val,
                                    reversal_price=float(closes[i + 1]),
                                    level=level,
                                    penetration_pct=round(penetration * 100, 4),
                                    bar_index=i,
                                )
                            )

            # Hunt above resistance
            for level in resistance_levels:
                if level <= 0:
                    continue
                high_val = float(highs[i])
                if high_val > level:
                    penetration = (high_val - level) / level
                    if penetration >= penetration_threshold:
                        if float(closes[i + 1]) < level:
                            hunts.append(
                                StopHunt(
                                    direction=Side.BUY,
                                    trigger_price=high_val,
                                    reversal_price=float(closes[i + 1]),
                                    level=level,
                                    penetration_pct=round(penetration * 100, 4),
                                    bar_index=i,
                                )
                            )

        return hunts

    # ── Aggregate score ───────────────────────────────────────────────────

    def aggregate_smart_money_score(
        self,
        all_signals: Dict[str, Any],
    ) -> float:
        """Combine all smart-money signals into a single score.

        Args:
            all_signals: Dict with optional keys:

                - ``block_trades``: List of :class:`BlockTrade`.
                - ``volume_spike``: :class:`VolumeSpikeResult`.
                - ``vwap_deviation``: float score from :meth:`detect_vwap_deviation`.
                - ``hidden_liquidity``: float score.
                - ``acc_dist``: :class:`AccDistSignal`.
                - ``liquidity_sweeps``: List of :class:`LiquiditySweep`.
                - ``stop_hunts``: List of :class:`StopHunt`.

        Returns:
            Score in [-1.0, +1.0]. Positive = bullish smart-money activity,
            negative = bearish.
        """
        score = 0.0
        weight_total = 0.0

        # Block trades: more buys -> bullish
        block_trades: List[BlockTrade] = all_signals.get("block_trades", [])
        if block_trades:
            buy_vol = sum(bt.volume for bt in block_trades if bt.side == Side.BUY)
            sell_vol = sum(bt.volume for bt in block_trades if bt.side == Side.SELL)
            total = buy_vol + sell_vol
            if total > 0:
                block_score = (buy_vol - sell_vol) / total
                score += block_score * 0.25
                weight_total += 0.25

        # Volume spike
        vs: Optional[VolumeSpikeResult] = all_signals.get("volume_spike")
        if vs is not None and vs.is_spike:
            # A spike is directional based on other context; give neutral weight
            spike_weight = min(vs.magnitude / 10.0, 0.15)
            weight_total += spike_weight
            # Direction inferred from VWAP deviation if available
            vwap_dev = all_signals.get("vwap_deviation", 0.0)
            score += (1.0 if vwap_dev > 0 else -1.0 if vwap_dev < 0 else 0.0) * spike_weight

        # VWAP deviation
        vwap_deviation: float = all_signals.get("vwap_deviation", 0.0)
        if vwap_deviation != 0.0:
            # Clamp to [-1, 1]
            vwap_signal = max(-1.0, min(1.0, vwap_deviation * 10.0))
            score += vwap_signal * 0.15
            weight_total += 0.15

        # Hidden liquidity (generally bullish if present on bid side)
        hidden: float = all_signals.get("hidden_liquidity", 0.0)
        if hidden > 0:
            score += hidden * 0.1
            weight_total += 0.1

        # Accumulation / Distribution
        acc_dist: Optional[AccDistSignal] = all_signals.get("acc_dist")
        if acc_dist is not None:
            if acc_dist.trend == "accumulation":
                ad_signal = acc_dist.strength
            elif acc_dist.trend == "distribution":
                ad_signal = -acc_dist.strength
            else:
                ad_signal = 0.0
            score += ad_signal * 0.2
            weight_total += 0.2

        # Liquidity sweeps
        sweeps: List[LiquiditySweep] = all_signals.get("liquidity_sweeps", [])
        if sweeps:
            buy_sweeps = sum(1 for s in sweeps if s.direction == Side.SELL)  # sweep below = bullish
            sell_sweeps = sum(1 for s in sweeps if s.direction == Side.BUY)  # sweep above = bearish
            total_sweeps = buy_sweeps + sell_sweeps
            if total_sweeps > 0:
                sweep_score = (buy_sweeps - sell_sweeps) / total_sweeps
                score += sweep_score * 0.1
                weight_total += 0.1

        # Stop hunts
        hunts: List[StopHunt] = all_signals.get("stop_hunts", [])
        if hunts:
            bull_hunts = sum(1 for h in hunts if h.direction == Side.SELL)  # hunt below = bullish
            bear_hunts = sum(1 for h in hunts if h.direction == Side.BUY)
            total_hunts = bull_hunts + bear_hunts
            if total_hunts > 0:
                hunt_score = (bull_hunts - bear_hunts) / total_hunts
                score += hunt_score * 0.05
                weight_total += 0.05

        # Normalize
        if weight_total > 0:
            score = score / weight_total

        return round(max(-1.0, min(1.0, score)), 4)
