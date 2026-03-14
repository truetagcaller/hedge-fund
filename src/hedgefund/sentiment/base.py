"""Abstract base class for sentiment scoring."""

from __future__ import annotations

import abc

from hedgefund.types import SentimentResult


class SentimentScorer(abc.ABC):
    """Base class that all sentiment scorers must implement.

    Each scorer analyses a specific signal source (news, social, options flow)
    and returns a normalised :class:`SentimentResult` for a given symbol.
    """

    @abc.abstractmethod
    async def score(self, symbol: str) -> SentimentResult:
        """Compute a sentiment score for *symbol*.

        Args:
            symbol: Ticker symbol to score (e.g. ``"AAPL"``).

        Returns:
            A :class:`SentimentResult` with ``score`` in [-1, 1] and
            ``magnitude`` in [0, 1].
        """
        ...

    @abc.abstractmethod
    async def score_batch(self, symbols: list[str]) -> list[SentimentResult]:
        """Score multiple symbols, potentially in one batch for efficiency.

        The default implementation simply iterates, but subclasses should
        override this when the underlying model supports batching.
        """
        ...
