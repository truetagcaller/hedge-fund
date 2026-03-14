"""Agent Registry — manages all AI trading agents.

Provides lifecycle management, signal collection, and status reporting
for the multi-agent trading system.
"""

from __future__ import annotations

from typing import Any

import structlog

from hedgefund.agents.base import AgentSignal, TradingAgent
from hedgefund.streaming.event_bus import EventBus
from hedgefund.types import MarketRegime

log = structlog.get_logger(__name__)


class AgentRegistry:
    """Central registry for all AI trading agents.

    Parameters
    ----------
    event_bus:
        Shared event bus for agent subscriptions.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._agents: dict[str, TradingAgent] = {}
        self._event_bus = event_bus

    # ── Registration ──────────────────────────────────────────────────

    def register(self, agent: TradingAgent) -> None:
        """Register an agent and subscribe it to the event bus."""
        if agent.name in self._agents:
            log.warning("agent_registry.duplicate", name=agent.name)
            return
        self._agents[agent.name] = agent
        agent.subscribe()
        log.info("agent_registry.registered", name=agent.name)

    def deregister(self, name: str) -> None:
        """Remove an agent from the registry."""
        if name in self._agents:
            self._agents[name].deactivate()
            del self._agents[name]
            log.info("agent_registry.deregistered", name=name)

    # ── Accessors ─────────────────────────────────────────────────────

    def get_agent(self, name: str) -> TradingAgent | None:
        return self._agents.get(name)

    def get_all_agents(self) -> list[TradingAgent]:
        return list(self._agents.values())

    def get_active_agents(self) -> list[TradingAgent]:
        return [a for a in self._agents.values() if a.is_active]

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    @property
    def active_count(self) -> int:
        return sum(1 for a in self._agents.values() if a.is_active)

    # ── Activation ────────────────────────────────────────────────────

    def activate_all(self, source: str = "") -> None:
        """Activate all agents whose data source has been verified.

        If *source* is provided, verify data source first.
        """
        for agent in self._agents.values():
            if source:
                agent.verify_data_source(source)
            agent.activate()
        active = self.active_count
        log.info("agent_registry.activate_all", active=active, total=len(self._agents))

    def deactivate_all(self) -> None:
        """Deactivate all agents — they will stop producing signals."""
        for agent in self._agents.values():
            agent.deactivate()
        log.info("agent_registry.deactivate_all")

    # ── Signal collection ─────────────────────────────────────────────

    async def collect_signals(
        self,
        symbol: str,
        market_data: dict[str, Any],
        regime: MarketRegime,
    ) -> list[AgentSignal]:
        """Collect signals from all active agents for *symbol*.

        Parameters
        ----------
        symbol:
            Ticker symbol to analyse.
        market_data:
            Dict with real market data (must include ``source`` key).
        regime:
            Current detected market regime.

        Returns
        -------
        list[AgentSignal]
            Signals from agents that had actionable opinions.
        """
        source = market_data.get("source", "")
        if not source:
            log.warning("agent_registry.no_data_source", symbol=symbol)
            return []

        signals: list[AgentSignal] = []
        for agent in self._agents.values():
            if not agent.is_active:
                continue
            try:
                signal = await agent.generate_signal(symbol, market_data)
                if signal is not None:
                    signals.append(signal)
            except Exception:
                log.exception(
                    "agent_registry.agent_error",
                    agent=agent.name,
                    symbol=symbol,
                )

        log.info(
            "agent_registry.signals_collected",
            symbol=symbol,
            total_agents=len(self._agents),
            active_agents=self.active_count,
            signals=len(signals),
            regime=regime.value,
        )
        return signals

    # ── Default agent factory ─────────────────────────────────────────

    def create_default_agents(self) -> None:
        """Create and register all 8 default trading agents."""
        from hedgefund.agents.gamma_scalping import GammaScalpingAgent
        from hedgefund.agents.liquidity_sweep import LiquiditySweepAgent
        from hedgefund.agents.mean_reversion import MeanReversionAgent
        from hedgefund.agents.news_reaction import NewsReactionAgent
        from hedgefund.agents.options_volatility import OptionsVolatilityAgent
        from hedgefund.agents.smart_money_flow import SmartMoneyFlowAgent
        from hedgefund.agents.social_sentiment import SocialSentimentAgent
        from hedgefund.agents.trend_following import TrendFollowingAgent

        agents: list[TradingAgent] = [
            TrendFollowingAgent(self._event_bus),
            MeanReversionAgent(self._event_bus),
            OptionsVolatilityAgent(self._event_bus),
            GammaScalpingAgent(self._event_bus),
            NewsReactionAgent(self._event_bus),
            SocialSentimentAgent(self._event_bus),
            LiquiditySweepAgent(self._event_bus),
            SmartMoneyFlowAgent(self._event_bus),
        ]

        for agent in agents:
            self.register(agent)

        log.info(
            "agent_registry.defaults_created",
            count=len(agents),
        )

    # ── Status ────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Return status of all agents."""
        agents_status = {}
        for name, agent in self._agents.items():
            agents_status[name] = agent.get_status()

        return {
            "total_agents": len(self._agents),
            "active_agents": self.active_count,
            "agents": agents_status,
        }
