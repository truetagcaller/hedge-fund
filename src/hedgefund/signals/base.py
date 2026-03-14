"""Abstract base class for trade-signal generators."""

from __future__ import annotations

import abc

import pandas as pd

from hedgefund.types import MarketRegime, SentimentResult, TradeSignal


class SignalGenerator(abc.ABC):
    """Base class for all signal generators.

    Subclasses consume a feature DataFrame (produced by the feature pipeline),
    the current market regime, and the latest sentiment, then output zero or
    more :class:`TradeSignal` instances.
    """

    @abc.abstractmethod
    async def generate(
        self,
        features_df: pd.DataFrame,
        regime: MarketRegime,
        sentiment: SentimentResult,
    ) -> list[TradeSignal]:
        """Generate trade signals from the latest feature set.

        Args:
            features_df: DataFrame with columns produced by the feature
                engineering pipeline (OHLCV, technicals, greeks, etc.).
            regime: Current detected market regime.
            sentiment: Aggregated sentiment for the underlying.

        Returns:
            A (possibly empty) list of :class:`TradeSignal`.
        """
        ...
