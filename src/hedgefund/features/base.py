"""Abstract base class for feature transformers."""

from __future__ import annotations

import abc

import pandas as pd


class FeatureTransformer(abc.ABC):
    """Base class for all feature transformers in the pipeline.

    Subclasses must implement ``name``, ``transform``, and
    ``required_columns`` so the pipeline can validate inputs and chain
    transformers together.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name used for logging and identification."""

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to *df* and return the enriched frame.

        Implementations **must not** mutate the incoming DataFrame; they should
        return a copy (or use ``df.copy()`` internally) so that upstream data
        remains untouched.
        """

    @abc.abstractmethod
    def required_columns(self) -> list[str]:
        """Return the list of column names that must be present in *df*
        before ``transform`` is called."""
