"""Real-time order book maintenance and analysis.

Maintains a live order book from WebSocket updates and provides analytical
tools for detecting liquidity imbalances, large orders, spoofing, and
calculating VWAP from the book.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from hedgefund.logger import get_logger
from hedgefund.types import Side

log = get_logger(__name__)


@dataclass(slots=True)
class BookLevel:
    """A single price level in the order book."""

    price: float
    quantity: float
    order_count: int = 1


@dataclass(slots=True)
class LargeOrder:
    """A detected large order in the book."""

    price: float
    quantity: float
    side: Side
    average_size: float
    multiplier: float
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class SpoofingPattern:
    """A suspected spoofing event: a large order that appeared and vanished."""

    price: float
    peak_quantity: float
    side: Side
    appeared_at: float
    disappeared_at: float
    duration_seconds: float
    score: float  # 0-1, higher = more suspicious


class OrderBook:
    """Real-time order book with bid/ask level maintenance and metrics.

    Usage::

        book = OrderBook(symbol="AAPL", max_depth=20)
        book.update(bids=[...], asks=[...])
        snapshot = book.get_snapshot(levels=5)
        spread = book.bid_ask_spread
    """

    def __init__(self, symbol: str, max_depth: int = 20) -> None:
        self.symbol = symbol
        self._max_depth = max_depth
        self._bids: List[BookLevel] = []
        self._asks: List[BookLevel] = []
        self._last_update: float = 0.0
        self._update_count: int = 0

        # History for spoofing detection
        self._bid_history: Deque[Tuple[float, List[BookLevel]]] = deque(maxlen=120)
        self._ask_history: Deque[Tuple[float, List[BookLevel]]] = deque(maxlen=120)

    # ── Core operations ───────────────────────────────────────────────────

    def update(
        self,
        bids: List[Dict[str, Any]],
        asks: List[Dict[str, Any]],
    ) -> None:
        """Replace the book with new bid/ask levels.

        Each entry in *bids* / *asks* must have ``price`` and ``quantity``
        keys; ``order_count`` is optional.
        """
        now = time.time()

        self._bids = sorted(
            [
                BookLevel(
                    price=float(b["price"]),
                    quantity=float(b["quantity"]),
                    order_count=int(b.get("order_count", b.get("orders", 1))),
                )
                for b in bids
                if float(b.get("quantity", 0)) > 0
            ],
            key=lambda lv: lv.price,
            reverse=True,  # highest bid first
        )[: self._max_depth]

        self._asks = sorted(
            [
                BookLevel(
                    price=float(a["price"]),
                    quantity=float(a["quantity"]),
                    order_count=int(a.get("order_count", a.get("orders", 1))),
                )
                for a in asks
                if float(a.get("quantity", 0)) > 0
            ],
            key=lambda lv: lv.price,
        )[: self._max_depth]

        # Store history snapshots for spoofing detection
        self._bid_history.append((now, list(self._bids)))
        self._ask_history.append((now, list(self._asks)))

        self._last_update = now
        self._update_count += 1

    def get_snapshot(self, levels: int = 5) -> Dict[str, Any]:
        """Return the top *levels* of the book as a dict."""
        return {
            "symbol": self.symbol,
            "bids": [
                {"price": lv.price, "quantity": lv.quantity, "orders": lv.order_count}
                for lv in self._bids[:levels]
            ],
            "asks": [
                {"price": lv.price, "quantity": lv.quantity, "orders": lv.order_count}
                for lv in self._asks[:levels]
            ],
            "timestamp": self._last_update,
            "update_count": self._update_count,
        }

    # ── Metrics ───────────────────────────────────────────────────────────

    @property
    def best_bid(self) -> Optional[float]:
        return self._bids[0].price if self._bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self._asks[0].price if self._asks else None

    @property
    def bid_ask_spread(self) -> float:
        """Absolute spread between best bid and best ask."""
        if not self._bids or not self._asks:
            return 0.0
        return self._asks[0].price - self._bids[0].price

    @property
    def bid_ask_imbalance(self) -> float:
        """Ratio of total bid volume to total ask volume.

        Values > 1 indicate more buying pressure; < 1 indicates more selling.
        """
        total_bid = sum(lv.quantity for lv in self._bids) if self._bids else 0.0
        total_ask = sum(lv.quantity for lv in self._asks) if self._asks else 0.0
        if total_ask == 0.0:
            return float("inf") if total_bid > 0 else 1.0
        return total_bid / total_ask

    @property
    def weighted_mid_price(self) -> float:
        """Volume-weighted mid price from the best bid and ask."""
        if not self._bids or not self._asks:
            return 0.0
        bb = self._bids[0]
        ba = self._asks[0]
        total = bb.quantity + ba.quantity
        if total == 0:
            return (bb.price + ba.price) / 2.0
        return (bb.price * ba.quantity + ba.price * bb.quantity) / total

    @property
    def total_bid_liquidity(self) -> float:
        """Total notional value on the bid side."""
        return sum(lv.price * lv.quantity for lv in self._bids)

    @property
    def total_ask_liquidity(self) -> float:
        """Total notional value on the ask side."""
        return sum(lv.price * lv.quantity for lv in self._asks)

    @property
    def buy_pressure(self) -> float:
        """Buy pressure score in [0, 1] based on volume imbalance."""
        total_bid = sum(lv.quantity for lv in self._bids)
        total_ask = sum(lv.quantity for lv in self._asks)
        total = total_bid + total_ask
        return total_bid / total if total > 0 else 0.5

    @property
    def sell_pressure(self) -> float:
        """Sell pressure score in [0, 1]."""
        return 1.0 - self.buy_pressure

    def large_order_detection(self, multiplier: float = 5.0) -> List[LargeOrder]:
        """Detect orders whose size exceeds *multiplier* x average.

        Returns:
            List of :class:`LargeOrder` entries from both sides.
        """
        all_quantities = [lv.quantity for lv in self._bids] + [
            lv.quantity for lv in self._asks
        ]
        if not all_quantities:
            return []

        avg_size = float(np.mean(all_quantities))
        threshold = avg_size * multiplier
        results: List[LargeOrder] = []

        for lv in self._bids:
            if lv.quantity >= threshold:
                results.append(
                    LargeOrder(
                        price=lv.price,
                        quantity=lv.quantity,
                        side=Side.BUY,
                        average_size=avg_size,
                        multiplier=lv.quantity / avg_size if avg_size > 0 else 0,
                    )
                )
        for lv in self._asks:
            if lv.quantity >= threshold:
                results.append(
                    LargeOrder(
                        price=lv.price,
                        quantity=lv.quantity,
                        side=Side.SELL,
                        average_size=avg_size,
                        multiplier=lv.quantity / avg_size if avg_size > 0 else 0,
                    )
                )
        return results

    @property
    def spoofing_score(self) -> float:
        """Aggregate spoofing suspicion score in [0, 1].

        Computed by analyzing how frequently large orders appear and
        disappear from the book history.
        """
        patterns = self._detect_vanishing_orders(window_seconds=30)
        if not patterns:
            return 0.0
        # Normalize: cap at 10 patterns for score of 1.0
        return min(len(patterns) / 10.0, 1.0)


class OrderBookAnalyzer:
    """Higher-level analytics on an :class:`OrderBook`.

    Usage::

        analyzer = OrderBookAnalyzer(book)
        side = analyzer.detect_liquidity_imbalance(threshold=0.6)
        large = analyzer.detect_large_orders(multiplier=5.0)
        spoof = analyzer.detect_spoofing(window_seconds=30)
        vwap = analyzer.calculate_vwap_from_book()
    """

    def __init__(self, order_book: OrderBook) -> None:
        self._book = order_book

    def detect_liquidity_imbalance(self, threshold: float = 0.6) -> Optional[Side]:
        """Detect if one side of the book dominates.

        Args:
            threshold: Minimum buy-pressure ratio to declare imbalance.

        Returns:
            :attr:`Side.BUY` if buy pressure exceeds *threshold*,
            :attr:`Side.SELL` if sell pressure exceeds it, or ``None``.
        """
        bp = self._book.buy_pressure
        if bp >= threshold:
            return Side.BUY
        if (1.0 - bp) >= threshold:
            return Side.SELL
        return None

    def detect_large_orders(self, multiplier: float = 5.0) -> List[LargeOrder]:
        """Return orders whose size is at least *multiplier* x the average."""
        return self._book.large_order_detection(multiplier)

    def detect_spoofing(self, window_seconds: float = 30) -> List[SpoofingPattern]:
        """Detect large orders that appear and disappear within *window_seconds*."""
        return self._book._detect_vanishing_orders(window_seconds)

    def calculate_vwap_from_book(self, levels: int = 5) -> float:
        """Calculate volume-weighted average price from the top *levels*."""
        total_volume = 0.0
        total_value = 0.0

        for lv in self._book._bids[:levels]:
            total_value += lv.price * lv.quantity
            total_volume += lv.quantity
        for lv in self._book._asks[:levels]:
            total_value += lv.price * lv.quantity
            total_volume += lv.quantity

        return total_value / total_volume if total_volume > 0 else 0.0


# ── Spoofing detection helper on OrderBook ────────────────────────────────

def _detect_vanishing_orders(
    self: OrderBook, window_seconds: float = 30
) -> List[SpoofingPattern]:
    """Scan book history for large orders that appeared then vanished.

    A spoofing pattern is: a level's quantity spikes above 5x average,
    then drops to near zero within *window_seconds*.
    """
    now = time.time()
    cutoff = now - window_seconds
    patterns: List[SpoofingPattern] = []

    for side, history, side_enum in [
        ("bid", self._bid_history, Side.BUY),
        ("ask", self._ask_history, Side.SELL),
    ]:
        # Build a time series of quantity per price level
        price_series: Dict[float, List[Tuple[float, float]]] = {}
        for ts, levels in history:
            if ts < cutoff:
                continue
            for lv in levels:
                price_series.setdefault(lv.price, []).append((ts, lv.quantity))

        # Detect spike-then-vanish
        for price, series in price_series.items():
            if len(series) < 2:
                continue

            quantities = np.array([q for _, q in series])
            timestamps = [t for t, _ in series]
            avg_qty = float(np.mean(quantities))

            if avg_qty <= 0:
                continue

            peak_idx = int(np.argmax(quantities))
            peak_qty = float(quantities[peak_idx])

            if peak_qty < avg_qty * 5.0:
                continue

            # Check if quantity drops to near zero after the peak
            after_peak = quantities[peak_idx + 1 :] if peak_idx + 1 < len(quantities) else np.array([])
            if len(after_peak) > 0 and float(np.min(after_peak)) < avg_qty * 0.5:
                min_after_idx = peak_idx + 1 + int(np.argmin(after_peak))
                duration = timestamps[min_after_idx] - timestamps[peak_idx]
                score = min(1.0, (peak_qty / avg_qty) / 20.0) * min(1.0, 10.0 / max(duration, 0.1))

                patterns.append(
                    SpoofingPattern(
                        price=price,
                        peak_quantity=peak_qty,
                        side=side_enum,
                        appeared_at=timestamps[peak_idx],
                        disappeared_at=timestamps[min_after_idx],
                        duration_seconds=duration,
                        score=min(score, 1.0),
                    )
                )

    return patterns


# Attach the helper as a method on OrderBook
OrderBook._detect_vanishing_orders = _detect_vanishing_orders  # type: ignore[attr-defined]
