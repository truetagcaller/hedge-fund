"""Feature pipeline that chains FeatureTransformers with input validation."""

from __future__ import annotations

import time

import pandas as pd
import structlog

from hedgefund.exceptions import DataValidationError
from hedgefund.features.base import FeatureTransformer

log = structlog.get_logger(__name__)


class FeaturePipeline:
    """Chains multiple :class:`FeatureTransformer` instances and runs them in
    order, validating required columns before each step.

    Usage
    -----
    >>> pipeline = FeaturePipeline([TechnicalFeatures(), VolatilityFeatures()])
    >>> enriched = pipeline.run(raw_df)
    """

    def __init__(self, transformers: list[FeatureTransformer] | None = None) -> None:
        self._transformers: list[FeatureTransformer] = list(transformers or [])

    # ── Mutators ─────────────────────────────────────────────────────────

    def add(self, transformer: FeatureTransformer) -> "FeaturePipeline":
        """Append a transformer and return self for chaining."""
        self._transformers.append(transformer)
        return self

    def clear(self) -> None:
        self._transformers.clear()

    @property
    def stages(self) -> list[str]:
        return [t.name for t in self._transformers]

    # ── Execution ────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate inputs and execute each transformer sequentially.

        Raises
        ------
        DataValidationError
            If the DataFrame is missing columns required by any stage.
        """
        if df.empty:
            log.warning("pipeline_empty_dataframe")
            return df

        log.info("pipeline_start", stages=self.stages, rows=len(df))
        result = df

        for transformer in self._transformers:
            self._validate(result, transformer)

            t0 = time.perf_counter()
            result = transformer.transform(result)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            log.info(
                "pipeline_stage_done",
                stage=transformer.name,
                columns_added=len(result.columns) - len(df.columns),
                elapsed_ms=round(elapsed_ms, 2),
            )

        log.info(
            "pipeline_complete",
            total_columns=len(result.columns),
            rows=len(result),
        )
        return result

    # ── Validation ───────────────────────────────────────────────────────

    @staticmethod
    def _validate(df: pd.DataFrame, transformer: FeatureTransformer) -> None:
        required = set(transformer.required_columns())
        available = set(df.columns)
        missing = required - available
        if missing:
            raise DataValidationError(
                f"Stage '{transformer.name}' requires columns {sorted(missing)} "
                f"which are not present. Available: {sorted(available)}"
            )
