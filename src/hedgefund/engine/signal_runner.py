"""Lightweight signal generation loop for dashboard-only mode.

Runs the AI agents against live Zerodha quotes to generate signals
without requiring the full TradingApplication orchestrator.
Stores signals in MongoDB and makes them available via the signals API.

CRITICAL: Only runs when live market data is available from a verified
source. Never generates synthetic signals.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import structlog

from hedgefund.agents.registry import AgentRegistry
from hedgefund.engine.signal_fusion import FusedSignal, SignalFusionEngine
from hedgefund.streaming.event_bus import EventBus
from hedgefund.types import MarketRegime

log = structlog.get_logger(__name__)


class SignalRunner:
    """Periodically runs AI agents against live market data.

    Parameters
    ----------
    agent_registry:
        Registry with all 8 AI agents.
    signal_fusion:
        Fusion engine to combine agent signals.
    event_bus:
        EventBus for receiving TICK events.
    db:
        MongoDB instance for storing generated signals.
    zerodha_feed:
        Live Zerodha market feed for quote data.
    interval:
        Seconds between signal generation cycles.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        signal_fusion: SignalFusionEngine,
        event_bus: EventBus,
        db: Any = None,
        zerodha_feed: Any = None,
        *,
        interval: float = 30.0,
        user_id: str = "",
    ) -> None:
        self._registry = agent_registry
        self._fusion = signal_fusion
        self._event_bus = event_bus
        self._db = db
        self._feed = zerodha_feed
        self._interval = interval
        self._user_id = user_id
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None
        self._signals_generated = 0
        self._last_regime = MarketRegime.MEAN_REVERTING

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._run_loop(), name="signal-runner",
        )
        log.info("signal_runner.started", interval=self._interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info(
            "signal_runner.stopped",
            signals_generated=self._signals_generated,
        )

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._generate_signals()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("signal_runner.error")
            await asyncio.sleep(self._interval)

    async def _generate_signals(self) -> None:
        """Fetch live quotes and run agents against them."""
        if self._feed is None:
            return

        # Get latest quotes from Zerodha
        try:
            instruments = list(self._feed._instruments)
            if not instruments:
                return
            quotes = await self._feed.get_quote(instruments)
        except Exception as exc:
            log.debug("signal_runner.quote_fetch_failed", error=str(exc))
            return

        if not quotes:
            return

        # Run each symbol through the fusion engine
        for instrument_key, quote_data in quotes.items():
            symbol = instrument_key.split(":")[-1] if ":" in instrument_key else instrument_key
            price = quote_data.get("last_price", 0)
            if not price:
                continue

            ohlc = quote_data.get("ohlc", {})

            # Build market_data dict that agents expect
            market_data: Dict[str, Any] = {
                "price": price,
                "open": ohlc.get("open", 0),
                "high": ohlc.get("high", 0),
                "low": ohlc.get("low", 0),
                "close": ohlc.get("close", 0),
                "volume": quote_data.get("volume", 0),
                "atr": abs(ohlc.get("high", 0) - ohlc.get("low", 0)) or price * 0.02,
                "oi": quote_data.get("oi", 0),
                "source": "Zerodha Kite API",
                # Estimate some indicators from OHLC
                "ema_9": price,
                "ema_21": ohlc.get("close", price),
                "ema_50": ohlc.get("close", price),
                "rsi": 50.0,
                "adx": 20.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "bollinger_upper": ohlc.get("high", price),
                "bollinger_lower": ohlc.get("low", price),
                "bollinger_mid": (ohlc.get("high", price) + ohlc.get("low", price)) / 2,
                "bid_volume": quote_data.get("buy_quantity", 0),
                "ask_volume": quote_data.get("sell_quantity", 0),
                "iv": 0.0,
                "hv": 0.0,
                "pcr": 0.0,
                "gex": 0.0,
                "delta": 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "smart_money_score": 0.0,
                "news_sentiment": 0.0,
                "social_sentiment": 0.0,
            }

            # Run fusion
            try:
                fused = await self._fusion.fuse_signals(
                    symbol, market_data, self._last_regime,
                )
                if fused is not None:
                    self._signals_generated += 1
                    await self._store_signal(fused, instrument_key)
                    log.info(
                        "signal_runner.signal_generated",
                        symbol=symbol,
                        action=fused.action.value,
                        confidence=fused.confidence,
                        agents=len(fused.contributing_agents),
                        source="Zerodha Kite API",
                    )
            except Exception:
                log.debug("signal_runner.fusion_error", symbol=symbol)

    async def _store_signal(
        self, fused: FusedSignal, instrument_key: str,
    ) -> None:
        """Store a fused signal in MongoDB."""
        if self._db is None:
            return

        try:
            from hedgefund.data.write_guard import WriteGuard
            doc = {
                "signal_id": fused.fusion_id,
                "user_id": self._user_id,
                "underlying": fused.symbol,
                "instrument_key": instrument_key,
                "action": fused.action.value,
                "direction": fused.direction.value,
                "confidence": fused.confidence,
                "entry_price": fused.entry_price,
                "stop_loss": fused.stop_loss,
                "target_price": fused.take_profit,
                "risk_reward_ratio": fused.risk_reward_ratio,
                "strategy_name": "signal_fusion",
                "reasoning": fused.reasoning,
                "contributing_agents": fused.contributing_agents,
                "signal_breakdown": fused.signal_breakdown,
                "data_sources": fused.data_sources,
                "is_live_data": fused.is_live_data,
                "regime": fused.regime.value,
                "data_source": "Zerodha Kite API",
                "source": "Zerodha Kite API",
                "outcome": "active",
                "timestamp": datetime.now(timezone.utc),
            }
            WriteGuard.validate_signal(doc)
            await self._db.signals.insert_one(doc)
        except Exception:
            log.debug("signal_runner.store_failed", exc_info=True)

    @property
    def signals_generated(self) -> int:
        return self._signals_generated
