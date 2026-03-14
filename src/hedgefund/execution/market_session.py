"""Market session manager — validates trading hours per exchange.

Prevents order submission outside valid market hours.  Crypto markets
are treated as 24/7.  Indian markets follow IST-based schedules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone, timedelta
from typing import Any

from hedgefund.logger import get_logger

log = get_logger(__name__)

# IST = UTC+5:30
_IST = timezone(timedelta(hours=5, minutes=30))


@dataclass(frozen=True, slots=True)
class MarketSession:
    """Trading session definition for a market."""

    market: str
    display_name: str
    open_time: time  # In the market's local timezone
    close_time: time
    tz: timezone
    days: tuple[int, ...]  # ISO weekdays (1=Mon .. 7=Sun)

    def is_open(self, now: datetime | None = None) -> bool:
        """Return ``True`` if the market is currently open."""
        now = now or datetime.now(timezone.utc)
        local = now.astimezone(self.tz)
        if local.isoweekday() not in self.days:
            return False
        local_t = local.time()
        return self.open_time <= local_t <= self.close_time


# ---------------------------------------------------------------------------
# Pre-defined market sessions
# ---------------------------------------------------------------------------

MARKET_SESSIONS: dict[str, MarketSession] = {
    "NSE": MarketSession(
        market="NSE",
        display_name="NSE Equities",
        open_time=time(9, 15),
        close_time=time(15, 30),
        tz=_IST,
        days=(1, 2, 3, 4, 5),
    ),
    "BSE": MarketSession(
        market="BSE",
        display_name="BSE Equities",
        open_time=time(9, 15),
        close_time=time(15, 30),
        tz=_IST,
        days=(1, 2, 3, 4, 5),
    ),
    "NFO": MarketSession(
        market="NFO",
        display_name="NSE F&O",
        open_time=time(9, 15),
        close_time=time(15, 30),
        tz=_IST,
        days=(1, 2, 3, 4, 5),
    ),
    "BFO": MarketSession(
        market="BFO",
        display_name="BSE F&O",
        open_time=time(9, 15),
        close_time=time(15, 30),
        tz=_IST,
        days=(1, 2, 3, 4, 5),
    ),
    "MCX": MarketSession(
        market="MCX",
        display_name="MCX Commodities",
        open_time=time(9, 0),
        close_time=time(23, 30),
        tz=_IST,
        days=(1, 2, 3, 4, 5),
    ),
    "CDS": MarketSession(
        market="CDS",
        display_name="Currency Derivatives",
        open_time=time(9, 0),
        close_time=time(17, 0),
        tz=_IST,
        days=(1, 2, 3, 4, 5),
    ),
    "COMMODITY": MarketSession(
        market="COMMODITY",
        display_name="Commodity",
        open_time=time(9, 0),
        close_time=time(23, 30),
        tz=_IST,
        days=(1, 2, 3, 4, 5),
    ),
    # Crypto is 24/7 — all days, all hours
    "SPOT": MarketSession(
        market="SPOT",
        display_name="Crypto Spot",
        open_time=time(0, 0),
        close_time=time(23, 59, 59),
        tz=timezone.utc,
        days=(1, 2, 3, 4, 5, 6, 7),
    ),
    "USDM": MarketSession(
        market="USDM",
        display_name="Crypto USDM Futures",
        open_time=time(0, 0),
        close_time=time(23, 59, 59),
        tz=timezone.utc,
        days=(1, 2, 3, 4, 5, 6, 7),
    ),
    "COINM": MarketSession(
        market="COINM",
        display_name="Crypto COIN-M Futures",
        open_time=time(0, 0),
        close_time=time(23, 59, 59),
        tz=timezone.utc,
        days=(1, 2, 3, 4, 5, 6, 7),
    ),
    "EAPI": MarketSession(
        market="EAPI",
        display_name="Crypto Options",
        open_time=time(0, 0),
        close_time=time(23, 59, 59),
        tz=timezone.utc,
        days=(1, 2, 3, 4, 5, 6, 7),
    ),
    # Virtual/paper — always open
    "VIRTUAL": MarketSession(
        market="VIRTUAL",
        display_name="Paper Trading",
        open_time=time(0, 0),
        close_time=time(23, 59, 59),
        tz=timezone.utc,
        days=(1, 2, 3, 4, 5, 6, 7),
    ),
}

# Map asset classes to their primary exchanges
_ASSET_CLASS_EXCHANGES: dict[str, list[str]] = {
    "EQUITY": ["NSE", "BSE"],
    "FUTURES": ["NFO", "USDM", "COINM"],
    "OPTIONS": ["NFO", "BFO", "EAPI"],
    "COMMODITY": ["MCX", "COMMODITY"],
    "CRYPTO": ["SPOT", "USDM", "COINM", "EAPI"],
}


class MarketSessionManager:
    """Validates whether trading is permitted based on market hours."""

    def is_market_open(
        self, exchange: str, now: datetime | None = None,
    ) -> bool:
        """Check if a specific exchange is currently open."""
        session = MARKET_SESSIONS.get(exchange.upper())
        if session is None:
            log.warning("market_session.unknown_exchange", exchange=exchange)
            return False
        return session.is_open(now)

    def is_asset_class_tradeable(
        self, asset_class: str, now: datetime | None = None,
    ) -> bool:
        """Check if ANY exchange for the given asset class is open."""
        exchanges = _ASSET_CLASS_EXCHANGES.get(asset_class.upper(), [])
        return any(self.is_market_open(ex, now) for ex in exchanges)

    def get_exchange_for_asset_class(
        self, asset_class: str, broker_exchanges: list[str],
    ) -> str | None:
        """Find the best open exchange for an asset class among broker exchanges."""
        target_exchanges = _ASSET_CLASS_EXCHANGES.get(asset_class.upper(), [])
        for ex in broker_exchanges:
            if ex.upper() in target_exchanges and self.is_market_open(ex):
                return ex
        return None

    def get_all_sessions(self) -> list[dict[str, Any]]:
        """Return all sessions with their current open/closed status."""
        now = datetime.now(timezone.utc)
        return [
            {
                "market": s.market,
                "display_name": s.display_name,
                "is_open": s.is_open(now),
                "open_time": s.open_time.isoformat(),
                "close_time": s.close_time.isoformat(),
            }
            for s in MARKET_SESSIONS.values()
        ]
