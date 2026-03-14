"""Multi-AI trading agent system.

Provides specialized trading agents that each analyze market data from a
unique perspective and produce :class:`AgentSignal` instances.  All agents
share the :class:`TradingAgent` interface and are orchestrated by the
:class:`AgentRegistry`.
"""

from hedgefund.agents.base import AgentSignal, TradingAgent
from hedgefund.agents.registry import AgentRegistry

__all__ = [
    "AgentSignal",
    "AgentRegistry",
    "TradingAgent",
]
