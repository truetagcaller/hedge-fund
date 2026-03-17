"""Abstract base class for all trading models."""

from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class ModelMetadata:
    """Metadata tracked for every trained model artifact."""

    model_name: str
    version: str
    training_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    training_rows: int = 0
    feature_columns: list[str] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    config_hash: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def compute_config_hash(self) -> str:
        """Deterministic hash of hyperparameters for deduplication."""
        canonical = json.dumps(self.hyperparameters, sort_keys=True, default=str)
        self.config_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return self.config_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "training_date": self.training_date.isoformat(),
            "training_rows": self.training_rows,
            "feature_columns": self.feature_columns,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "config_hash": self.config_hash,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        return cls(
            model_name=data["model_name"],
            version=data["version"],
            training_date=datetime.fromisoformat(data["training_date"]),
            training_rows=data.get("training_rows", 0),
            feature_columns=data.get("feature_columns", []),
            hyperparameters=data.get("hyperparameters", {}),
            metrics=data.get("metrics", {}),
            config_hash=data.get("config_hash", ""),
            tags=data.get("tags", {}),
        )


class TradingModel(abc.ABC):
    """Base class for all ML models in the trading system.

    Every concrete model must implement train, predict, save, and load.
    The base class manages metadata lifecycle and provides common utilities.
    """

    def __init__(self, name: str, version: str = "0.1.0") -> None:
        self.metadata = ModelMetadata(model_name=name, version=version)
        self._is_trained: bool = False
        self._log = log.bind(model=name, version=version)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    # ── Abstract interface ────────────────────────────────────────────

    @abc.abstractmethod
    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train the model and return a dict of training metrics.

        Implementations should populate ``self.metadata.metrics`` with at
        least the final training and validation loss.
        """

    @abc.abstractmethod
    def predict(
        self, X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """Generate predictions for the given input features.

        Returns an array of shape ``(n_samples,)`` or ``(n_samples, n_outputs)``.
        """

    @abc.abstractmethod
    def save(self, path: Path) -> Path:
        """Persist model weights and metadata to *path*.

        Returns the directory or file that was written.
        """

    @abc.abstractmethod
    def load(self, path: Path) -> None:
        """Restore model weights and metadata from *path*."""

    # ── Shared helpers ────────────────────────────────────────────────

    def _save_metadata(self, directory: Path) -> Path:
        """Write metadata JSON next to the model artifact."""
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = directory / "metadata.json"
        meta_path.write_text(
            json.dumps(self.metadata.to_dict(), indent=2, default=str)
        )
        self._log.info("metadata_saved", path=str(meta_path))
        return meta_path

    def _load_metadata(self, directory: Path) -> None:
        """Restore metadata from a previously saved JSON file."""
        meta_path = directory / "metadata.json"
        if meta_path.exists():
            data = json.loads(meta_path.read_text())
            self.metadata = ModelMetadata.from_dict(data)
            self._log.info("metadata_loaded", path=str(meta_path))
        else:
            self._log.warning("metadata_file_missing", path=str(meta_path))

    def _validate_trained(self) -> None:
        """Raise if the model has not been trained yet."""
        if not self._is_trained:
            raise RuntimeError(
                f"Model '{self.metadata.model_name}' has not been trained. "
                "Call train() before predict()."
            )
