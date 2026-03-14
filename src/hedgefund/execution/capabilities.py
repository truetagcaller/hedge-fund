"""Broker capability detection and registry.

Each broker adapter has a fixed set of capabilities (order placement,
market-data streaming, supported exchanges, etc.).  This module provides
a declarative registry so the routing layer can select the right broker
for a given instrument or operation without hard-coding broker names.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BrokerCapabilities:
    """What a broker can do."""

    broker_type: str
    display_name: str
    options_trading: bool = False
    futures_trading: bool = False
    crypto_trading: bool = False
    equity_trading: bool = True
    mutual_funds: bool = False
    us_stocks: bool = False
    market_data: bool = False
    websocket_feed: bool = False
    order_placement: bool = False  # Can actually place orders (not read-only)
    paper_trading: bool = False
    supported_exchanges: list[str] = field(default_factory=list)
    supported_order_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "broker_type": self.broker_type,
            "display_name": self.display_name,
            "options_trading": self.options_trading,
            "futures_trading": self.futures_trading,
            "crypto_trading": self.crypto_trading,
            "equity_trading": self.equity_trading,
            "mutual_funds": self.mutual_funds,
            "us_stocks": self.us_stocks,
            "market_data": self.market_data,
            "websocket_feed": self.websocket_feed,
            "order_placement": self.order_placement,
            "paper_trading": self.paper_trading,
            "supported_exchanges": list(self.supported_exchanges),
            "supported_order_types": list(self.supported_order_types),
        }


# ---------------------------------------------------------------------------
# Registry of known capabilities
# ---------------------------------------------------------------------------

BROKER_CAPABILITIES: dict[str, BrokerCapabilities] = {
    "zerodha": BrokerCapabilities(
        broker_type="zerodha",
        display_name="Zerodha Kite",
        options_trading=True,
        futures_trading=True,
        equity_trading=True,
        market_data=True,
        websocket_feed=True,
        order_placement=True,
        supported_exchanges=["NSE", "BSE", "NFO", "BFO", "MCX", "CDS"],
        supported_order_types=["MARKET", "LIMIT", "SL", "SL-M"],
    ),
    "binance": BrokerCapabilities(
        broker_type="binance",
        display_name="Binance",
        crypto_trading=True,
        futures_trading=True,
        options_trading=True,
        equity_trading=False,
        market_data=True,
        websocket_feed=True,
        order_placement=True,
        supported_exchanges=["SPOT", "USDM", "COINM", "EAPI"],
        supported_order_types=["MARKET", "LIMIT", "STOP_MARKET", "STOP_LIMIT"],
    ),
    "groww": BrokerCapabilities(
        broker_type="groww",
        display_name="Groww",
        options_trading=True,
        futures_trading=True,
        equity_trading=True,
        mutual_funds=True,
        market_data=True,
        order_placement=True,
        supported_exchanges=["NSE", "BSE", "NFO", "COMMODITY"],
        supported_order_types=["MARKET", "LIMIT"],
    ),
    "indmoney": BrokerCapabilities(
        broker_type="indmoney",
        display_name="INDMoney",
        equity_trading=True,
        mutual_funds=True,
        us_stocks=True,
        market_data=False,
        order_placement=False,  # read-only
        supported_exchanges=["NSE", "BSE", "US"],
        supported_order_types=[],
    ),
    "paper": BrokerCapabilities(
        broker_type="paper",
        display_name="Paper Trading",
        options_trading=True,
        futures_trading=True,
        equity_trading=True,
        crypto_trading=True,
        market_data=False,
        order_placement=True,
        paper_trading=True,
        supported_exchanges=["NSE", "NFO", "VIRTUAL"],
        supported_order_types=["MARKET", "LIMIT", "STOP", "STOP_LIMIT"],
    ),
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_capabilities(broker_type: str) -> BrokerCapabilities:
    """Return the capability descriptor for *broker_type*.

    Raises :class:`KeyError` if the broker type is not in the registry.
    """
    caps = BROKER_CAPABILITIES.get(broker_type)
    if caps is None:
        raise KeyError(
            f"Unknown broker type: {broker_type!r}. "
            f"Known types: {list(BROKER_CAPABILITIES.keys())}"
        )
    return caps


def can_trade(broker_type: str) -> bool:
    """Return ``True`` if the broker supports order placement."""
    try:
        return get_capabilities(broker_type).order_placement
    except KeyError:
        return False


def supports_instrument(broker_type: str, exchange: str) -> bool:
    """Return ``True`` if the broker lists *exchange* in its supported set."""
    try:
        caps = get_capabilities(broker_type)
    except KeyError:
        return False
    return exchange.upper() in (e.upper() for e in caps.supported_exchanges)
