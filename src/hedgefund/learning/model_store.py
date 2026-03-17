"""Model artifact store with versioning, metadata tracking, and promotion."""

from __future__ import annotations

import enum
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from hedgefund.learning.base import ModelMetadata, TradingModel

log = structlog.get_logger(__name__)


class ModelStage(enum.Enum):
    """Lifecycle stage of a registered model artifact."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass(slots=True)
class ModelRecord:
    """Registry entry for a single model version."""

    model_name: str
    version: str
    stage: ModelStage
    artifact_path: str
    metadata: ModelMetadata
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    promoted_at: datetime | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "stage": self.stage.value,
            "artifact_path": self.artifact_path,
            "metadata": self.metadata.to_dict(),
            "registered_at": self.registered_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelRecord:
        return cls(
            model_name=data["model_name"],
            version=data["version"],
            stage=ModelStage(data["stage"]),
            artifact_path=data["artifact_path"],
            metadata=ModelMetadata.from_dict(data["metadata"]),
            registered_at=datetime.fromisoformat(data["registered_at"]),
            promoted_at=(
                datetime.fromisoformat(data["promoted_at"])
                if data.get("promoted_at")
                else None
            ),
            description=data.get("description", ""),
        )


class ModelStore:
    """Persistent store for model artifacts with versioning and promotion.

    Directory layout::

        root_dir/
            registry.json          # global registry index
            <model_name>/
                <version>/
                    <stage>/       # symlink or copy of artifacts
                        weights.pt
                        metadata.json
                        ...
    """

    def __init__(self, root_dir: Path) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._root / "registry.json"
        self._records: list[ModelRecord] = []
        self._load_registry()
        self._log = log.bind(component="model_store", root=str(self._root))

    # ── Registry persistence ──────────────────────────────────────────

    def _load_registry(self) -> None:
        if self._registry_path.exists():
            data = json.loads(self._registry_path.read_text())
            self._records = [ModelRecord.from_dict(r) for r in data]
        else:
            self._records = []

    def _save_registry(self) -> None:
        self._registry_path.write_text(
            json.dumps([r.to_dict() for r in self._records], indent=2, default=str)
        )

    # ── Public API ────────────────────────────────────────────────────

    def register(
        self,
        model: TradingModel,
        artifact_source: Path,
        *,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        description: str = "",
    ) -> ModelRecord:
        """Copy artifacts into the store and register a new model version.

        Args:
            model: The trained model (used for metadata only).
            artifact_source: Directory containing saved model files.
            stage: Initial lifecycle stage.
            description: Free-text description.

        Returns:
            The newly created :class:`ModelRecord`.
        """
        name = model.metadata.model_name
        version = model.metadata.version
        dest = self._artifact_dir(name, version, stage)
        dest.mkdir(parents=True, exist_ok=True)

        # Copy all files from source to store.
        src = Path(artifact_source)
        for item in src.iterdir():
            if item.is_file():
                shutil.copy2(item, dest / item.name)
            elif item.is_dir():
                shutil.copytree(item, dest / item.name, dirs_exist_ok=True)

        record = ModelRecord(
            model_name=name,
            version=version,
            stage=stage,
            artifact_path=str(dest),
            metadata=model.metadata,
            description=description,
        )
        self._records.append(record)
        self._save_registry()
        self._log.info(
            "model_registered",
            model=name,
            version=version,
            stage=stage.value,
        )
        return record

    def promote(
        self,
        model_name: str,
        version: str,
        target_stage: ModelStage,
    ) -> ModelRecord:
        """Promote a model version to a new lifecycle stage.

        Copies the artifact directory to the target stage location and
        updates the registry.

        Raises:
            KeyError: If the model/version is not found.
            ValueError: If the transition is invalid (e.g. archived -> production).
        """
        record = self.get_record(model_name, version)
        if record is None:
            raise KeyError(f"Model {model_name} v{version} not found in registry.")

        _VALID_TRANSITIONS: dict[ModelStage, set[ModelStage]] = {
            ModelStage.DEVELOPMENT: {ModelStage.STAGING, ModelStage.ARCHIVED},
            ModelStage.STAGING: {ModelStage.PRODUCTION, ModelStage.ARCHIVED, ModelStage.DEVELOPMENT},
            ModelStage.PRODUCTION: {ModelStage.ARCHIVED, ModelStage.STAGING},
            ModelStage.ARCHIVED: {ModelStage.DEVELOPMENT},
        }
        if target_stage not in _VALID_TRANSITIONS.get(record.stage, set()):
            raise ValueError(
                f"Cannot promote from {record.stage.value} to {target_stage.value}."
            )

        # Demote any existing model in the target production slot.
        if target_stage == ModelStage.PRODUCTION:
            for r in self._records:
                if (
                    r.model_name == model_name
                    and r.stage == ModelStage.PRODUCTION
                    and r.version != version
                ):
                    r.stage = ModelStage.ARCHIVED
                    r.promoted_at = datetime.now(timezone.utc)
                    self._log.info(
                        "model_demoted",
                        model=r.model_name,
                        version=r.version,
                        to="archived",
                    )

        # Copy artifacts to new stage directory.
        old_dir = Path(record.artifact_path)
        new_dir = self._artifact_dir(model_name, version, target_stage)
        if new_dir != old_dir:
            new_dir.mkdir(parents=True, exist_ok=True)
            for item in old_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, new_dir / item.name)

        record.stage = target_stage
        record.promoted_at = datetime.now(timezone.utc)
        record.artifact_path = str(new_dir)
        self._save_registry()
        self._log.info(
            "model_promoted",
            model=model_name,
            version=version,
            stage=target_stage.value,
        )
        return record

    def get_record(
        self, model_name: str, version: str
    ) -> ModelRecord | None:
        """Look up a specific model version in the registry."""
        for r in reversed(self._records):
            if r.model_name == model_name and r.version == version:
                return r
        return None

    def get_production_model(self, model_name: str) -> ModelRecord | None:
        """Return the current production record for *model_name*, if any."""
        for r in reversed(self._records):
            if r.model_name == model_name and r.stage == ModelStage.PRODUCTION:
                return r
        return None

    def load_model(
        self,
        model_instance: TradingModel,
        model_name: str,
        *,
        version: str | None = None,
        stage: ModelStage = ModelStage.PRODUCTION,
    ) -> TradingModel:
        """Load model weights from the store into *model_instance*.

        If *version* is ``None`` the latest record at the requested *stage*
        is used.
        """
        if version:
            record = self.get_record(model_name, version)
        else:
            candidates = [
                r
                for r in self._records
                if r.model_name == model_name and r.stage == stage
            ]
            record = candidates[-1] if candidates else None

        if record is None:
            raise KeyError(
                f"No {stage.value} model found for '{model_name}'"
                + (f" v{version}" if version else "")
            )

        model_instance.load(Path(record.artifact_path))
        self._log.info(
            "model_loaded_from_store",
            model=model_name,
            version=record.version,
            stage=record.stage.value,
        )
        return model_instance

    def list_models(
        self, model_name: str | None = None, stage: ModelStage | None = None
    ) -> list[ModelRecord]:
        """List registered models, optionally filtered by name or stage."""
        results = self._records
        if model_name:
            results = [r for r in results if r.model_name == model_name]
        if stage:
            results = [r for r in results if r.stage == stage]
        return results

    def delete(self, model_name: str, version: str) -> bool:
        """Remove a model version from the registry and delete its artifacts.

        Returns ``True`` if the model was found and deleted.
        """
        record = self.get_record(model_name, version)
        if record is None:
            return False

        artifact_dir = Path(record.artifact_path)
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)

        self._records = [
            r
            for r in self._records
            if not (r.model_name == model_name and r.version == version)
        ]
        self._save_registry()
        self._log.info("model_deleted", model=model_name, version=version)
        return True

    # ── Internal ──────────────────────────────────────────────────────

    def _artifact_dir(
        self, model_name: str, version: str, stage: ModelStage
    ) -> Path:
        return self._root / model_name / version / stage.value
