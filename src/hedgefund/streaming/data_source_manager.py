"""Centralized data source management for news and social feeds.

Provides a single entry point to add, remove, and monitor all external data
sources (RSS, news APIs, Twitter, economic calendars).  Source configurations
are persisted to ``~/.hedgefund/data_sources.json`` with credential masking.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import structlog

from hedgefund.streaming.news_stream import EventBus, NewsStreamManager
from hedgefund.streaming.social_stream import SocialStreamManager

log = structlog.get_logger(__name__)

_CONFIG_DIR = Path.home() / ".hedgefund"
_CONFIG_FILE = _CONFIG_DIR / "data_sources.json"


class SourceType(str, Enum):
    RSS = "rss"
    NEWS_API = "news_api"
    TWITTER = "twitter"
    ECONOMIC_CALENDAR = "economic_calendar"
    EARNINGS = "earnings"


class SourceStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    CONNECTING = "connecting"


# ── Credential helpers ────────────────────────────────────────────────────────


def _mask_credential(value: str) -> str:
    """Mask all but the last 4 characters of a credential string."""
    if len(value) <= 4:
        return "****"
    return "*" * (len(value) - 4) + value[-4:]


def _mask_credentials(creds: dict[str, str]) -> dict[str, str]:
    """Return a copy of credentials with values masked for display."""
    return {k: _mask_credential(v) for k, v in creds.items()}


# ── Source record ─────────────────────────────────────────────────────────────


class _SourceRecord:
    """Internal record for a configured data source."""

    __slots__ = (
        "source_id",
        "source_type",
        "name",
        "config",
        "credentials",
        "status",
        "created_at",
        "last_update",
        "error_message",
        "_internal_id",
    )

    def __init__(
        self,
        source_id: str,
        source_type: SourceType,
        name: str,
        config: dict[str, Any],
        credentials: dict[str, str],
    ) -> None:
        self.source_id = source_id
        self.source_type = source_type
        self.name = name
        self.config = config
        self.credentials = credentials
        self.status = SourceStatus.INACTIVE
        self.created_at = datetime.now(timezone.utc)
        self.last_update: Optional[datetime] = None
        self.error_message = ""
        self._internal_id: Optional[str] = None  # id within sub-manager

    def to_dict(self, include_creds: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.source_id,
            "type": self.source_type.value,
            "name": self.name,
            "status": self.status.value,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "created_at": self.created_at.isoformat(),
            "config": {k: v for k, v in self.config.items() if k != "credentials"},
            "error_message": self.error_message,
        }
        if include_creds:
            result["credentials"] = _mask_credentials(self.credentials)
        return result

    def to_persist(self) -> dict[str, Any]:
        """Serialise for on-disk storage (credentials included)."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "name": self.name,
            "config": self.config,
            "credentials": self.credentials,
            "created_at": self.created_at.isoformat(),
        }


# ── Data Source Manager ───────────────────────────────────────────────────────


class DataSourceManager:
    """Centralized manager for all external data sources.

    Delegates to :class:`NewsStreamManager` and :class:`SocialStreamManager`
    for the actual ingestion.  Source configurations are persisted to disk.

    Parameters
    ----------
    event_bus:
        Shared event bus for NEWS and SENTIMENT events.
    news_manager:
        Optional pre-built news stream manager.
    social_manager:
        Optional pre-built social stream manager.
    persist_path:
        Override config file path (useful in tests).
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        news_manager: NewsStreamManager | None = None,
        social_manager: SocialStreamManager | None = None,
        persist_path: Path | None = None,
    ) -> None:
        self._event_bus = event_bus or EventBus()
        self._news = news_manager or NewsStreamManager(event_bus=self._event_bus)
        self._social = social_manager or SocialStreamManager(event_bus=self._event_bus)
        self._sources: dict[str, _SourceRecord] = {}
        self._persist_path = persist_path or _CONFIG_FILE
        self._load_persisted()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load_persisted(self) -> None:
        """Load previously saved sources from disk and re-register them."""
        if not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
            for entry in data.get("sources", []):
                try:
                    record = _SourceRecord(
                        source_id=entry["source_id"],
                        source_type=SourceType(entry["source_type"]),
                        name=entry["name"],
                        config=entry.get("config", {}),
                        credentials=entry.get("credentials", {}),
                    )
                    self._sources[record.source_id] = record

                    # Re-register with the sub-manager so polling resumes
                    try:
                        internal_id = self._register_with_submanager(
                            record.source_type,
                            record.config,
                            record.credentials,
                        )
                        record._internal_id = internal_id
                        record.status = SourceStatus.ACTIVE
                    except Exception as reg_exc:
                        record.status = SourceStatus.ERROR
                        record.error_message = str(reg_exc)[:200]
                        log.warning(
                            "persisted_source_register_failed",
                            source_id=record.source_id,
                            error=str(reg_exc),
                        )
                except (KeyError, ValueError):
                    log.warning("skipping_invalid_persisted_source", entry=entry)
            log.info("data_sources_loaded", count=len(self._sources))
        except Exception:
            log.error("data_sources_load_error", exc_info=True)

    def _save_persisted(self) -> None:
        """Save current sources to disk."""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "sources": [r.to_persist() for r in self._sources.values()],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._persist_path.write_text(json.dumps(data, indent=2))
        except Exception:
            log.error("data_sources_save_error", exc_info=True)

    # ── Source CRUD ───────────────────────────────────────────────────────

    def add_source(self, source_type: str, config: dict[str, Any]) -> str:
        """Add a new data source.

        Parameters
        ----------
        source_type:
            One of ``"rss"``, ``"news_api"``, ``"twitter"``,
            ``"economic_calendar"``, ``"earnings"``.
        config:
            Source configuration.  Expected keys vary by type:
            - rss: ``name``, ``url``, ``poll_interval``
            - news_api: ``name``, ``base_url``, ``credentials``
            - twitter: ``name``, ``credentials``
            - economic_calendar: ``name``, ``url``, ``poll_interval``
            - earnings: ``name``, ``url``, ``poll_interval``, ``credentials``

        Returns
        -------
        str
            The generated source id.
        """
        stype = SourceType(source_type)
        source_id = f"ds_{uuid.uuid4().hex[:8]}"
        name = config.get("name", f"{source_type}_{source_id}")
        credentials = config.get("credentials", {})

        record = _SourceRecord(
            source_id=source_id,
            source_type=stype,
            name=name,
            config=config,
            credentials=credentials,
        )

        # Delegate to the appropriate sub-manager
        try:
            internal_id = self._register_with_submanager(stype, config, credentials)
            record._internal_id = internal_id
            record.status = SourceStatus.ACTIVE
        except Exception as exc:
            record.status = SourceStatus.ERROR
            record.error_message = str(exc)[:200]
            log.error("source_add_error", source_type=source_type, error=str(exc))

        self._sources[source_id] = record
        self._save_persisted()
        log.info("data_source_added", source_id=source_id, type=source_type, name=name)
        return source_id

    def _register_with_submanager(
        self,
        stype: SourceType,
        config: dict[str, Any],
        credentials: dict[str, str],
    ) -> str | None:
        """Register the source with the appropriate streaming manager."""
        name = config.get("name", "unnamed")

        if stype == SourceType.RSS:
            return self._news.add_rss_source(
                name=name,
                url=config.get("url", ""),
                poll_interval=config.get("poll_interval", 60),
            )

        if stype == SourceType.NEWS_API:
            return self._news.add_api_source(
                name=name,
                base_url=config.get("base_url", config.get("url", "")),
                api_key=credentials.get("api_key", ""),
                poll_interval=config.get("poll_interval", 30),
            )

        if stype == SourceType.TWITTER:
            account_id = self._social.add_account(
                platform="twitter",
                credentials=credentials,
            )
            # Track configured tickers if any
            tickers = config.get("tickers", [])
            if tickers:
                self._social.track_tickers(tickers)
            return account_id

        if stype == SourceType.ECONOMIC_CALENDAR:
            return self._news.add_calendar_source(
                name=name,
                url=config.get("url", ""),
                poll_interval=config.get("poll_interval", 300),
            )

        if stype == SourceType.EARNINGS:
            return self._news.add_earnings_source(
                name=name,
                url=config.get("url", ""),
                api_key=credentials.get("api_key", ""),
                poll_interval=config.get("poll_interval", 300),
            )

        return None

    def remove_source(self, source_id: str) -> None:
        """Remove a data source by id."""
        record = self._sources.get(source_id)
        if record is None:
            raise KeyError(f"Source {source_id!r} not found")

        # Remove from sub-manager
        if record._internal_id:
            if record.source_type == SourceType.TWITTER:
                try:
                    self._social.remove_account(record._internal_id)
                except KeyError:
                    pass
            else:
                self._news.remove_source(record._internal_id)

        del self._sources[source_id]
        self._save_persisted()
        log.info("data_source_removed", source_id=source_id)

    def list_sources(self) -> list[dict[str, Any]]:
        """Return all configured sources with status."""
        return [r.to_dict() for r in self._sources.values()]

    def get_source_health(self) -> dict[str, dict[str, Any]]:
        """Return per-source health metrics."""
        health: dict[str, dict[str, Any]] = {}
        for sid, record in self._sources.items():
            health[sid] = {
                "name": record.name,
                "type": record.source_type.value,
                "status": record.status.value,
                "error_message": record.error_message,
                "last_update": (
                    record.last_update.isoformat() if record.last_update else None
                ),
            }
        return health

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start_all(self) -> None:
        """Start both news and social stream managers."""
        await self._news.start()
        await self._social.start()
        log.info("data_source_manager_started", sources=len(self._sources))

    async def stop_all(self) -> None:
        """Stop both stream managers."""
        await self._news.stop()
        await self._social.stop()
        log.info("data_source_manager_stopped")

    # ── Accessors ─────────────────────────────────────────────────────────

    @property
    def news_manager(self) -> NewsStreamManager:
        return self._news

    @property
    def social_manager(self) -> SocialStreamManager:
        return self._social

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus
