"""Broker management infrastructure for multi-broker trading.

Manages multiple broker connections simultaneously, providing a unified
interface for portfolio aggregation, credential storage, and health monitoring.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hedgefund.execution.base import Broker
from hedgefund.execution.paper import PaperBroker
from hedgefund.logger import get_logger
from hedgefund.security.api_permissions import validate_broker_permissions
from hedgefund.security.credential_store import CredentialStore
from hedgefund.security.sanitizer import Sanitizer
from hedgefund.types import Greeks, PortfolioSnapshot, Position

log = get_logger(__name__)
_sanitizer = Sanitizer()

# Supported broker types and their auth requirements
SUPPORTED_BROKERS: dict[str, dict[str, Any]] = {
    "zerodha": {
        "name": "Zerodha (Kite Connect)",
        "auth_methods": ["api_key", "login"],
        "required_fields": {
            "api_key": ["api_key", "api_secret", "access_token"],
            "login": ["api_key", "api_secret"],
        },
        "optional_fields": ["redirect_url"],
    },
    "binance": {
        "name": "Binance",
        "auth_methods": ["api_key"],
        "required_fields": {
            "api_key": ["api_key", "api_secret"],
        },
        "optional_fields": ["testnet"],
    },
    "groww": {
        "name": "Groww",
        "auth_methods": ["api_key"],
        "required_fields": {
            "api_key": ["api_key", "api_secret"],
        },
        "optional_fields": [],
    },
    "indmoney": {
        "name": "IndMoney",
        "auth_methods": ["api_key"],
        "required_fields": {
            "api_key": ["api_key", "api_secret"],
        },
        "optional_fields": [],
    },
    "paper": {
        "name": "Paper Trading (Simulated)",
        "auth_methods": ["none"],
        "required_fields": {
            "none": [],
        },
        "optional_fields": ["initial_cash", "slippage_bps", "commission_per_contract"],
    },
}

_CREDENTIALS_DIR = Path.home() / ".hedgefund"
_CREDENTIALS_FILE = _CREDENTIALS_DIR / "credentials.json"

# Shared encrypted credential store instance (lazy-initialised).
_credential_store: CredentialStore | None = None


def _get_credential_store() -> CredentialStore:
    global _credential_store
    if _credential_store is None:
        _credential_store = CredentialStore()
    return _credential_store


class _BrokerEntry:
    """Internal state for a managed broker connection."""

    __slots__ = (
        "broker_id",
        "broker_type",
        "broker",
        "credentials",
        "status",
        "connected_at",
        "last_refresh",
        "error_message",
        "account_info",
    )

    def __init__(
        self,
        broker_id: str,
        broker_type: str,
        broker: Broker,
        credentials: dict[str, Any],
    ) -> None:
        self.broker_id = broker_id
        self.broker_type = broker_type
        self.broker = broker
        self.credentials = credentials
        self.status: str = "disconnected"
        self.connected_at: Optional[datetime] = None
        self.last_refresh: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.account_info: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.broker_id,
            "type": self.broker_type,
            "status": self.status,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
            "error_message": self.error_message,
            "account_info": self.account_info,
        }


class BrokerManager:
    """Manages multiple broker connections simultaneously.

    Provides credential storage (in-memory with optional file persistence),
    unified portfolio aggregation, and per-broker health monitoring.

    Thread-safe via asyncio.Lock for all mutating operations.
    """

    def __init__(self, *, persist_credentials: bool = False) -> None:
        self._brokers: dict[str, _BrokerEntry] = {}
        self._lock = asyncio.Lock()
        self._persist_credentials = persist_credentials
        self._credential_store = _get_credential_store() if persist_credentials else None

        if persist_credentials:
            self._load_persisted_credentials()

    # ── Public API ────────────────────────────────────────────────────────

    async def add_broker(
        self,
        broker_id: str,
        broker_type: str,
        credentials: dict[str, Any],
    ) -> dict[str, Any]:
        """Add and connect a new broker.

        Parameters
        ----------
        broker_id:
            Unique identifier for this broker connection (e.g. "zerodha_main").
        broker_type:
            One of: "zerodha", "binance", "groww", "indmoney", "paper".
        credentials:
            Broker-specific credentials dict.

        Returns
        -------
        dict with broker_id, status, message, and account_info.
        """
        if broker_type not in SUPPORTED_BROKERS:
            raise ValueError(
                f"Unsupported broker type: {broker_type!r}. "
                f"Supported: {list(SUPPORTED_BROKERS.keys())}"
            )

        async with self._lock:
            if broker_id in self._brokers:
                raise ValueError(f"Broker {broker_id!r} already exists. Remove it first.")

            # Validate API permissions before connecting (non-paper brokers).
            if broker_type != "paper":
                api_key = credentials.get("api_key", "")
                perms = validate_broker_permissions(broker_type, api_key)
                log.info(
                    "broker_manager.permissions_checked",
                    broker_id=broker_id,
                    broker_type=broker_type,
                    can_read_portfolio=perms.can_read_portfolio,
                    can_place_orders=perms.can_place_orders,
                    can_cancel_orders=perms.can_cancel_orders,
                    can_withdraw=perms.can_withdraw,
                )

            # Log sanitized credentials (never expose secrets).
            log.info(
                "broker_manager.adding_broker",
                broker_id=broker_id,
                broker_type=broker_type,
                credentials=_sanitizer.sanitize_dict(credentials),
            )

            broker = self._create_broker(broker_type, credentials)
            entry = _BrokerEntry(
                broker_id=broker_id,
                broker_type=broker_type,
                broker=broker,
                credentials=credentials,
            )

            try:
                await broker.connect()
                entry.status = "connected"
                entry.connected_at = datetime.now(timezone.utc)
                entry.account_info = {"broker_type": broker_type}

                log.info(
                    "broker_manager.broker_added",
                    broker_id=broker_id,
                    broker_type=broker_type,
                )
            except Exception as exc:
                entry.status = "error"
                entry.error_message = str(exc)
                log.error(
                    "broker_manager.connection_failed",
                    broker_id=broker_id,
                    broker_type=broker_type,
                    error=str(exc),
                )

            self._brokers[broker_id] = entry

            if self._persist_credentials:
                self._save_credentials()

            return {
                "broker_id": broker_id,
                "status": entry.status,
                "message": entry.error_message or "Connected successfully",
                "account_info": entry.account_info,
            }

    async def remove_broker(self, broker_id: str) -> None:
        """Disconnect and remove a broker."""
        async with self._lock:
            entry = self._brokers.pop(broker_id, None)
            if entry is None:
                raise KeyError(f"Broker {broker_id!r} not found.")

            try:
                await entry.broker.disconnect()
            except Exception as exc:
                log.warning(
                    "broker_manager.disconnect_error",
                    broker_id=broker_id,
                    error=str(exc),
                )

            if self._persist_credentials:
                self._save_credentials()

            log.info("broker_manager.broker_removed", broker_id=broker_id)

    async def get_broker(self, broker_id: str) -> Broker:
        """Get a connected broker instance by ID."""
        async with self._lock:
            entry = self._brokers.get(broker_id)
            if entry is None:
                raise KeyError(f"Broker {broker_id!r} not found.")
            return entry.broker

    async def get_all_brokers(self) -> list[dict[str, Any]]:
        """List all configured brokers with connection status."""
        async with self._lock:
            return [entry.to_dict() for entry in self._brokers.values()]

    async def get_aggregate_portfolio(self) -> PortfolioSnapshot:
        """Aggregate portfolio across all connected brokers.

        Returns a combined PortfolioSnapshot with summed cash, positions,
        and greeks from every connected broker.
        """
        all_positions: list[Position] = []
        total_cash = 0.0
        total_nlv = 0.0
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_daily_pnl = 0.0
        total_pnl = 0.0

        async with self._lock:
            entries = list(self._brokers.values())

        for entry in entries:
            if entry.status != "connected":
                continue
            try:
                snapshot = await entry.broker.get_portfolio()
                all_positions.extend(snapshot.positions)
                total_cash += snapshot.cash
                total_nlv += snapshot.net_liquidation
                total_delta += snapshot.total_delta
                total_gamma += snapshot.total_gamma
                total_theta += snapshot.total_theta
                total_vega += snapshot.total_vega
                total_daily_pnl += snapshot.daily_pnl
                total_pnl += snapshot.total_pnl

                entry.last_refresh = datetime.now(timezone.utc)
            except Exception as exc:
                log.error(
                    "broker_manager.portfolio_fetch_error",
                    broker_id=entry.broker_id,
                    error=str(exc),
                )
                entry.error_message = f"Portfolio fetch failed: {exc}"

        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            cash=total_cash,
            net_liquidation=total_nlv,
            positions=all_positions,
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_theta=total_theta,
            total_vega=total_vega,
            daily_pnl=total_daily_pnl,
            total_pnl=total_pnl,
        )

    async def get_aggregate_positions(self) -> list[Position]:
        """Return all positions across all connected brokers."""
        all_positions: list[Position] = []

        async with self._lock:
            entries = list(self._brokers.values())

        for entry in entries:
            if entry.status != "connected":
                continue
            try:
                positions = await entry.broker.get_positions()
                all_positions.extend(positions)
                entry.last_refresh = datetime.now(timezone.utc)
            except Exception as exc:
                log.error(
                    "broker_manager.positions_fetch_error",
                    broker_id=entry.broker_id,
                    error=str(exc),
                )
                entry.error_message = f"Positions fetch failed: {exc}"

        return all_positions

    async def refresh_all(self) -> None:
        """Refresh data for all connected brokers."""
        async with self._lock:
            entries = list(self._brokers.values())

        for entry in entries:
            if entry.status != "connected":
                continue
            try:
                await entry.broker.get_portfolio()
                entry.last_refresh = datetime.now(timezone.utc)
                entry.error_message = None
                log.info("broker_manager.refreshed", broker_id=entry.broker_id)
            except Exception as exc:
                entry.error_message = f"Refresh failed: {exc}"
                log.error(
                    "broker_manager.refresh_error",
                    broker_id=entry.broker_id,
                    error=str(exc),
                )

    async def health_check(self, broker_id: str) -> dict[str, Any]:
        """Check if a broker is connected and responsive.

        Returns a dict with status, latency_ms, last_refresh, and any errors.
        """
        async with self._lock:
            entry = self._brokers.get(broker_id)
            if entry is None:
                raise KeyError(f"Broker {broker_id!r} not found.")

        result: dict[str, Any] = {
            "broker_id": broker_id,
            "status": entry.status,
            "last_refresh": entry.last_refresh.isoformat() if entry.last_refresh else None,
            "error_message": entry.error_message,
            "latency_ms": None,
        }

        if entry.status == "connected":
            start = datetime.now(timezone.utc)
            try:
                await entry.broker.get_positions()
                elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                result["latency_ms"] = round(elapsed, 2)
                result["healthy"] = True
            except Exception as exc:
                result["healthy"] = False
                result["error_message"] = str(exc)
        else:
            result["healthy"] = False

        return result

    # ── Broker factory ────────────────────────────────────────────────────

    def get_connected_broker_ids(self) -> list[str]:
        """Return IDs of all connected brokers."""
        return [
            bid for bid, entry in self._brokers.items()
            if entry.status == "connected"
        ]

    async def get_broker_status(self, broker_id: str) -> dict[str, Any]:
        """Return detailed status for a broker including capabilities."""
        async with self._lock:
            entry = self._brokers.get(broker_id)
            if entry is None:
                raise KeyError(f"Broker {broker_id!r} not found.")

        from hedgefund.execution.capabilities import get_capabilities

        status = entry.to_dict()
        try:
            caps = get_capabilities(entry.broker_type)
            status["capabilities"] = caps.to_dict()
        except KeyError:
            status["capabilities"] = {}

        return status

    @staticmethod
    def _create_broker(broker_type: str, credentials: dict[str, Any]) -> Broker:
        """Create the appropriate broker instance from type string."""
        if broker_type == "paper":
            return PaperBroker(
                initial_cash=credentials.get("initial_cash", 10_000_000.0),
                slippage_bps=credentials.get("slippage_bps", 5),
                commission_per_contract=credentials.get(
                    "commission_per_contract", 0.65,
                ),
            )

        if broker_type == "zerodha":
            from hedgefund.execution.zerodha import ZerodhaBroker, ZerodhaConfig

            config = ZerodhaConfig(
                api_key=credentials.get("api_key", ""),
                api_secret=credentials.get("api_secret", ""),
                access_token=credentials.get("access_token"),
            )
            return ZerodhaBroker(config)

        if broker_type == "binance":
            from hedgefund.execution.binance import BinanceBroker, BinanceConfig

            config = BinanceConfig(
                api_key=credentials.get("api_key", ""),
                api_secret=credentials.get("api_secret", ""),
                testnet=credentials.get("testnet", False),
            )
            return BinanceBroker(config)

        if broker_type == "groww":
            from hedgefund.execution.groww import GrowwBroker, GrowwConfig

            config = GrowwConfig(
                email=credentials.get("email", ""),
                token=credentials.get("token", credentials.get("api_key", "")),
                session_id=credentials.get("session_id", ""),
            )
            return GrowwBroker(config)

        if broker_type == "indmoney":
            from hedgefund.execution.indmoney import IndMoneyBroker, IndMoneyConfig

            config = IndMoneyConfig(
                token=credentials.get("token", credentials.get("api_key", "")),
                session_id=credentials.get("session_id", ""),
            )
            return IndMoneyBroker(config)

        raise ValueError(f"No factory for broker type: {broker_type!r}")

    # ── Credential persistence ────────────────────────────────────────────

    def _save_credentials(self) -> None:
        """Persist broker credentials to disk using encrypted CredentialStore.

        Each broker's credentials are stored under the namespace
        ``broker:<broker_id>`` with individual keys for each credential
        field, plus a ``__broker_type__`` metadata key.
        """
        try:
            store = self._credential_store or _get_credential_store()
            for broker_id, entry in self._brokers.items():
                ns = f"broker:{broker_id}"
                store.store(ns, "__broker_type__", entry.broker_type)
                for k, v in entry.credentials.items():
                    store.store(ns, k, str(v))

            log.info(
                "broker_manager.credentials_saved",
                broker_count=len(self._brokers),
            )
        except Exception as exc:
            log.error("broker_manager.credentials_save_failed", error=str(exc))

    def _load_persisted_credentials(self) -> None:
        """Load broker configs from encrypted CredentialStore (does not auto-connect).

        Credentials are loaded into the store's in-memory cache.
        Call ``add_broker()`` to establish connections.
        """
        try:
            store = self._credential_store or _get_credential_store()
            namespaces = [ns for ns in store.list_namespaces() if ns.startswith("broker:")]
            log.info(
                "broker_manager.credentials_loaded",
                broker_count=len(namespaces),
            )
        except Exception as exc:
            log.error("broker_manager.credentials_load_failed", error=str(exc))


