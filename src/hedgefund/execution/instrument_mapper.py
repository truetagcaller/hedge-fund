"""Instrument mapping engine — resolves generic symbols to broker-specific contracts.

AI agents produce generic symbols like ``BANKNIFTY``, ``NIFTY``, ``RELIANCE``.
Brokers require fully qualified identifiers like ``NFO:BANKNIFTY2431520000CE``
or ``BTCUSDT``.  This module bridges that gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from hedgefund.exceptions import InstrumentMappingError
from hedgefund.logger import get_logger
from hedgefund.types import AssetClass, OptionContract, OptionType, SignalAction

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MappedInstrument:
    """Result of instrument resolution."""

    broker_symbol: str  # e.g. "NFO:NIFTY2431520000CE"
    exchange: str  # e.g. "NFO"
    underlying: str  # e.g. "NIFTY"
    asset_class: AssetClass
    contract: OptionContract | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Underlying-to-exchange mappings
# ---------------------------------------------------------------------------

_INDIAN_INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}
_INDIAN_EQUITIES = {
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HINDUNILVR",
    "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK",
    "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA", "WIPRO", "HCLTECH",
    "TATAMOTORS", "TATASTEEL", "BAJFINANCE", "ADANIENT",
}

_CRYPTO_SUFFIXES = {"USDT", "BUSD", "BTC", "ETH", "BNB", "USDC"}
_CRYPTO_PATTERN = re.compile(
    r"^[A-Z]{2,10}(?:" + "|".join(_CRYPTO_SUFFIXES) + r")$", re.IGNORECASE,
)

# MCX commodity underlyings
_COMMODITIES = {
    "GOLD", "GOLDM", "SILVER", "SILVERM", "CRUDEOIL", "NATURALGAS",
    "COPPER", "ZINC", "ALUMINIUM", "LEAD", "NICKEL",
}


class InstrumentMapper:
    """Resolves generic symbols to broker-specific contracts.

    Parameters
    ----------
    broker_type:
        Target broker (``zerodha``, ``binance``, ``groww``, ``paper``).
    """

    def __init__(self, broker_type: str = "zerodha") -> None:
        self._broker_type = broker_type.lower()

    def map(
        self,
        symbol: str,
        action: SignalAction,
        asset_class: AssetClass,
        *,
        strike: float | None = None,
        expiry: date | None = None,
        option_type: OptionType | None = None,
    ) -> MappedInstrument:
        """Resolve *symbol* to a broker-specific instrument.

        For options, if *strike*/*expiry* are not given, the mapper picks
        reasonable defaults (ATM strike, nearest weekly expiry).
        """
        symbol = symbol.upper().strip()

        if asset_class == AssetClass.CRYPTO:
            return self._map_crypto(symbol)

        if asset_class == AssetClass.COMMODITY:
            return self._map_commodity(symbol, action, strike, expiry, option_type)

        if asset_class == AssetClass.OPTIONS:
            return self._map_option(symbol, action, strike, expiry, option_type)

        if asset_class == AssetClass.FUTURES:
            return self._map_futures(symbol, expiry)

        # Equity
        return self._map_equity(symbol)

    # ── Crypto ────────────────────────────────────────────────────────

    def _map_crypto(self, symbol: str) -> MappedInstrument:
        # Already a pair like BTCUSDT
        if _CRYPTO_PATTERN.match(symbol):
            broker_sym = symbol
        else:
            broker_sym = f"{symbol}USDT"

        return MappedInstrument(
            broker_symbol=broker_sym,
            exchange="SPOT",
            underlying=symbol,
            asset_class=AssetClass.CRYPTO,
        )

    # ── Equity ────────────────────────────────────────────────────────

    def _map_equity(self, symbol: str) -> MappedInstrument:
        exchange = "NSE"
        if self._broker_type == "zerodha":
            broker_sym = f"NSE:{symbol}"
        elif self._broker_type == "groww":
            broker_sym = symbol
        else:
            broker_sym = symbol

        return MappedInstrument(
            broker_symbol=broker_sym,
            exchange=exchange,
            underlying=symbol,
            asset_class=AssetClass.EQUITY,
        )

    # ── Futures ───────────────────────────────────────────────────────

    def _map_futures(
        self, symbol: str, expiry: date | None,
    ) -> MappedInstrument:
        expiry = expiry or self._next_monthly_expiry()
        exp_str = expiry.strftime("%y%b").upper()  # e.g. "25MAR"

        if symbol in _COMMODITIES:
            exchange = "MCX"
            broker_sym = f"MCX:{symbol}{exp_str}FUT"
        elif symbol in _INDIAN_INDICES or symbol in _INDIAN_EQUITIES:
            exchange = "NFO"
            broker_sym = f"NFO:{symbol}{exp_str}FUT"
        else:
            exchange = "NFO"
            broker_sym = f"NFO:{symbol}{exp_str}FUT"

        return MappedInstrument(
            broker_symbol=broker_sym,
            exchange=exchange,
            underlying=symbol,
            asset_class=AssetClass.FUTURES,
        )

    # ── Options ───────────────────────────────────────────────────────

    def _map_option(
        self,
        symbol: str,
        action: SignalAction,
        strike: float | None,
        expiry: date | None,
        option_type: OptionType | None,
    ) -> MappedInstrument:
        # Derive option type from signal action if not specified
        if option_type is None:
            option_type = self._infer_option_type(action)

        expiry = expiry or self._next_weekly_expiry()
        exp_str = expiry.strftime("%y%m%d")  # e.g. "250327"

        if strike is None:
            raise InstrumentMappingError(
                symbol,
                "Strike price required for options. Use entry_price as ATM proxy.",
            )

        strike_int = int(strike)
        cp = "CE" if option_type == OptionType.CALL else "PE"

        if symbol in _INDIAN_INDICES or symbol in _INDIAN_EQUITIES:
            exchange = "NFO"
            broker_sym = f"NFO:{symbol}{exp_str}{strike_int}{cp}"
        else:
            exchange = "NFO"
            broker_sym = f"NFO:{symbol}{exp_str}{strike_int}{cp}"

        contract = OptionContract(
            symbol=broker_sym,
            underlying=symbol,
            option_type=option_type,
            strike=float(strike_int),
            expiration=expiry,
        )

        return MappedInstrument(
            broker_symbol=broker_sym,
            exchange=exchange,
            underlying=symbol,
            asset_class=AssetClass.OPTIONS,
            contract=contract,
        )

    # ── Commodity ────────────────────────────────────────────────────

    def _map_commodity(
        self,
        symbol: str,
        action: SignalAction,
        strike: float | None,
        expiry: date | None,
        option_type: OptionType | None,
    ) -> MappedInstrument:
        expiry = expiry or self._next_monthly_expiry()
        exp_str = expiry.strftime("%y%b").upper()

        if strike is not None:
            # Commodity option
            option_type = option_type or self._infer_option_type(action)
            cp = "CE" if option_type == OptionType.CALL else "PE"
            broker_sym = f"MCX:{symbol}{exp_str}{int(strike)}{cp}"
        else:
            broker_sym = f"MCX:{symbol}{exp_str}FUT"

        return MappedInstrument(
            broker_symbol=broker_sym,
            exchange="MCX",
            underlying=symbol,
            asset_class=AssetClass.COMMODITY,
        )

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _infer_option_type(action: SignalAction) -> OptionType:
        if action in (SignalAction.BUY_CALL, SignalAction.SELL_CALL):
            return OptionType.CALL
        return OptionType.PUT

    @staticmethod
    def _next_weekly_expiry() -> date:
        """Return the next Thursday (weekly expiry for Indian markets)."""
        today = date.today()
        days_ahead = (3 - today.weekday()) % 7  # Thursday = 3
        if days_ahead == 0:
            days_ahead = 7
        return today + timedelta(days=days_ahead)

    @staticmethod
    def _next_monthly_expiry() -> date:
        """Return the last Thursday of the current month."""
        today = date.today()
        # Find last day of current month
        if today.month == 12:
            next_month = date(today.year + 1, 1, 1)
        else:
            next_month = date(today.year, today.month + 1, 1)
        last_day = next_month - timedelta(days=1)
        # Walk back to Thursday
        while last_day.weekday() != 3:
            last_day -= timedelta(days=1)
        # If already past, go to next month
        if last_day < today:
            if today.month == 12:
                next_month = date(today.year + 1, 2, 1)
            else:
                next_month = date(today.year, today.month + 2, 1)
            last_day = next_month - timedelta(days=1)
            while last_day.weekday() != 3:
                last_day -= timedelta(days=1)
        return last_day
