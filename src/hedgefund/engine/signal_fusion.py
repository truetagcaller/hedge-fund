"""Signal Fusion Engine — combines signals from all AI agents.

Collects :class:`AgentSignal` instances from the :class:`AgentRegistry`,
applies regime-adaptive weights, and produces a single
:class:`FusedSignal` that the execution layer can act on.

CRITICAL: No signal is fused unless **every** contributing signal was
generated from verified, live market data.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from hedgefund.agents.base import AgentSignal
from hedgefund.agents.registry import AgentRegistry
from hedgefund.streaming.event_bus import EventBus
from hedgefund.types import MarketRegime, SignalAction, SignalDirection

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class FusedSignal:
    """Composite signal produced by fusing multiple agent signals."""

    fusion_id: str
    symbol: str
    timestamp: datetime
    action: SignalAction
    direction: SignalDirection
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    contributing_agents: list[str]
    signal_breakdown: dict[str, float]  # agent_name -> weighted_score
    reasoning: str
    data_sources: list[str]
    is_live_data: bool  # True only if ALL contributing signals used live data
    regime: MarketRegime = MarketRegime.MEAN_REVERTING
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_id() -> str:
        return f"FUS-{uuid.uuid4().hex[:12].upper()}"


class SignalFusionEngine:
    """Fuses signals from multiple AI agents into actionable trades.

    Parameters
    ----------
    agent_registry:
        Registry containing all active trading agents.
    event_bus:
        Central event bus.
    min_confidence:
        Minimum fused confidence to emit a signal (0-1).
    min_agents:
        Minimum number of agreeing agents required.
    min_risk_reward:
        Minimum risk-reward ratio.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        event_bus: EventBus,
        *,
        min_confidence: float = 0.65,
        min_agents: int = 2,
        min_risk_reward: float = 2.0,
    ) -> None:
        self._registry = agent_registry
        self._event_bus = event_bus
        self._min_confidence = min_confidence
        self._min_agents = min_agents
        self._min_rr = min_risk_reward
        self._fusion_history: list[FusedSignal] = []

    # ── Core fusion ───────────────────────────────────────────────────

    async def fuse_signals(
        self,
        symbol: str,
        market_data: dict[str, Any],
        regime: MarketRegime,
    ) -> FusedSignal | None:
        """Collect and fuse signals from all agents.

        Returns ``None`` if:
        - Not enough agents agree on a direction.
        - Any contributing signal uses non-live data.
        - Fused confidence is below threshold.
        - Risk/reward is insufficient.
        """
        # 1. Collect signals from all active agents
        signals = await self._registry.collect_signals(
            symbol, market_data, regime,
        )

        if len(signals) < self._min_agents:
            return None

        # 2. CRITICAL: Reject if ANY signal uses non-live data
        for sig in signals:
            if not sig.is_live_data:
                log.warning(
                    "signal_fusion.non_live_data",
                    agent=sig.agent_name,
                    symbol=symbol,
                    source=sig.data_source,
                )
                return None

        # 3. Get regime-adjusted weights for each agent
        weights = self.get_agent_weights(regime)

        # 4. Separate bullish and bearish signals
        long_signals = [s for s in signals if s.direction == SignalDirection.LONG]
        short_signals = [s for s in signals if s.direction == SignalDirection.SHORT]

        # 5. Compute directional consensus
        long_weight = sum(
            weights.get(s.agent_name, 0.1) * s.confidence
            for s in long_signals
        )
        short_weight = sum(
            weights.get(s.agent_name, 0.1) * s.confidence
            for s in short_signals
        )

        total_weight = long_weight + short_weight
        if total_weight <= 0:
            return None

        # 6. Determine direction
        if long_weight > short_weight:
            direction = SignalDirection.LONG
            action = SignalAction.BUY_CALL
            directional_signals = long_signals
            net_confidence = long_weight / total_weight
        elif short_weight > long_weight:
            direction = SignalDirection.SHORT
            action = SignalAction.BUY_PUT
            directional_signals = short_signals
            net_confidence = short_weight / total_weight
        else:
            return None  # No consensus

        if len(directional_signals) < self._min_agents:
            return None

        # 7. Compute weighted confidence
        dir_total_weight = sum(
            weights.get(s.agent_name, 0.1) for s in directional_signals
        )
        if dir_total_weight <= 0:
            return None

        weighted_confidence = sum(
            weights.get(s.agent_name, 0.1) * s.confidence
            for s in directional_signals
        ) / dir_total_weight

        # Blend with consensus ratio
        fused_confidence = weighted_confidence * 0.7 + net_confidence * 0.3
        fused_confidence = min(1.0, max(0.0, fused_confidence))

        if fused_confidence < self._min_confidence:
            return None

        # 8. Compute consensus entry/stop/target (weighted average)
        entry = self._weighted_avg(
            directional_signals, weights, lambda s: s.entry_price,
        )
        stop = self._weighted_avg(
            directional_signals, weights, lambda s: s.stop_loss,
        )
        target = self._weighted_avg(
            directional_signals, weights, lambda s: s.take_profit,
        )

        stop_dist = abs(entry - stop)
        target_dist = abs(target - entry)
        rr = target_dist / stop_dist if stop_dist > 0 else 0.0

        if rr < self._min_rr:
            return None

        # 9. Build breakdown and reasoning
        breakdown: dict[str, float] = {}
        for sig in directional_signals:
            w = weights.get(sig.agent_name, 0.1)
            breakdown[sig.agent_name] = round(w * sig.confidence, 4)

        top = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)
        reason_parts = [f"{name}={score:+.3f}" for name, score in top[:5]]
        reasoning = (
            f"Fused: confidence={fused_confidence:.3f}, "
            f"direction={direction.value}, "
            f"agents={len(directional_signals)}/{len(signals)}. "
            + "; ".join(reason_parts)
        )

        data_sources = list({
            s.data_source for s in directional_signals if s.data_source
        })

        fused = FusedSignal(
            fusion_id=FusedSignal.generate_id(),
            symbol=symbol,
            timestamp=datetime.utcnow(),
            action=action,
            direction=direction,
            confidence=round(fused_confidence, 4),
            entry_price=round(entry, 4),
            stop_loss=round(stop, 4),
            take_profit=round(target, 4),
            risk_reward_ratio=round(rr, 2),
            contributing_agents=[s.agent_name for s in directional_signals],
            signal_breakdown=breakdown,
            reasoning=reasoning,
            data_sources=data_sources,
            is_live_data=all(s.is_live_data for s in directional_signals),
            regime=regime,
        )

        self._fusion_history.append(fused)
        log.info(
            "signal_fusion.fused",
            symbol=symbol,
            direction=direction.value,
            confidence=fused.confidence,
            agents=len(directional_signals),
            rr=rr,
            sources=data_sources,
        )
        return fused

    # ── Weight computation ────────────────────────────────────────────

    def get_agent_weights(self, regime: MarketRegime) -> dict[str, float]:
        """Get regime-adjusted weights for all registered agents."""
        agents = self._registry.get_all_agents()
        if not agents:
            return {}

        raw: dict[str, float] = {}
        for agent in agents:
            raw[agent.name] = agent.get_weight(regime)

        # Normalize to sum to 1.0
        total = sum(raw.values())
        if total <= 0:
            return {name: 1.0 / len(raw) for name in raw}

        return {name: w / total for name, w in raw.items()}

    # ── History ───────────────────────────────────────────────────────

    def get_fusion_history(self, limit: int = 50) -> list[FusedSignal]:
        return list(self._fusion_history[-limit:])

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _weighted_avg(
        signals: list[AgentSignal],
        weights: dict[str, float],
        extractor: Any,
    ) -> float:
        """Weighted average of a signal attribute."""
        total_w = 0.0
        total_v = 0.0
        for sig in signals:
            w = weights.get(sig.agent_name, 0.1) * sig.confidence
            total_w += w
            total_v += w * extractor(sig)
        return total_v / total_w if total_w > 0 else 0.0
