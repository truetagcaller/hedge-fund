"""Per-user broker router with capability-based routing and failover.

Routes trade requests to the correct broker based on:
1. Explicit per-trade broker override
2. User's currently active broker
3. Instrument-type heuristics (crypto -> Binance, options -> Zerodha, etc.)
4. Failover chain when the primary broker is unavailable
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from hedgefund.exceptions import BrokerConnectionError, ExecutionError
from hedgefund.execution.broker_manager import BrokerManager
from hedgefund.execution.capabilities import (
    BROKER_CAPABILITIES,
    can_trade,
    get_capabilities,
    supports_instrument,
)
from hedgefund.logger import get_logger
from hedgefund.types import Order

log = get_logger(__name__)

# Regex patterns used by the instrument-type heuristic.
_CRYPTO_PATTERN = re.compile(
    r"^[A-Z]{2,10}(USDT|BUSD|BTC|ETH|BNB)$", re.IGNORECASE
)
_NFO_PATTERN = re.compile(r"^NFO:", re.IGNORECASE)
_US_PATTERN = re.compile(r"^US:", re.IGNORECASE)


class BrokerRouter:
    """Routes trade requests to the correct broker based on user preference.

    Supports:
    - Per-user active broker selection
    - Per-trade broker override
    - Failover to backup broker
    - Capability-based routing (e.g., crypto -> Binance, options -> Zerodha)
    """

    def __init__(
        self,
        broker_manager: BrokerManager,
        db: Any | None = None,
    ) -> None:
        self._manager = broker_manager
        self._db = db
        # In-memory caches (authoritative source is MongoDB when available).
        self._user_active_broker: dict[str, str] = {}  # user_id -> broker_id
        self._failover_order: dict[str, list[str]] = {}  # user_id -> [broker_ids]

    # ── Active broker management ──────────────────────────────────────────

    async def set_active_broker(self, user_id: str, broker_id: str) -> None:
        """Set the active broker for a user.  Persists to MongoDB if available."""
        self._user_active_broker[user_id] = broker_id
        log.info(
            "broker_router.active_broker_set",
            user_id=user_id,
            broker_id=broker_id,
        )

        if self._db is not None:
            await self._db.user_broker_prefs.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "active_broker_id": broker_id,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
                upsert=True,
            )

    async def get_active_broker(self, user_id: str) -> str | None:
        """Get the user's active broker_id.  Checks memory then MongoDB."""
        cached = self._user_active_broker.get(user_id)
        if cached is not None:
            return cached

        if self._db is not None:
            doc = await self._db.user_broker_prefs.find_one(
                {"user_id": user_id},
                {"active_broker_id": 1},
            )
            if doc and doc.get("active_broker_id"):
                broker_id = doc["active_broker_id"]
                self._user_active_broker[user_id] = broker_id
                return broker_id

        return None

    # ── Failover order management ─────────────────────────────────────────

    async def set_failover_order(
        self, user_id: str, broker_ids: list[str],
    ) -> None:
        """Set the failover chain for a user."""
        self._failover_order[user_id] = list(broker_ids)
        log.info(
            "broker_router.failover_order_set",
            user_id=user_id,
            broker_ids=broker_ids,
        )

        if self._db is not None:
            await self._db.user_broker_prefs.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "failover_order": broker_ids,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
                upsert=True,
            )

    # ── Order routing ─────────────────────────────────────────────────────

    async def route_order(
        self,
        user_id: str,
        order: Order,
        broker_id: str | None = None,
    ) -> Order:
        """Route an order to the correct broker.

        Resolution order:
        1. Explicit *broker_id* if provided.
        2. User's active broker.
        3. Auto-select based on instrument type / symbol.
        4. On failure, attempt failover brokers.
        """
        target_broker_id = broker_id

        # Step 1/2: resolve target broker
        if target_broker_id is None:
            target_broker_id = await self.get_active_broker(user_id)

        # Step 3: auto-select by instrument if still unknown
        if target_broker_id is None:
            target_broker_id = self._select_broker_for_instrument(
                user_id, order.contract.symbol,
            )

        if target_broker_id is None:
            raise ExecutionError(
                "No broker available for order routing. "
                "Set an active broker or connect a broker that supports "
                f"the instrument {order.contract.symbol!r}."
            )

        # Validate capabilities
        broker_entry = await self._resolve_broker_entry(target_broker_id)
        broker_type = broker_entry.get("type", "")
        if not can_trade(broker_type):
            raise ExecutionError(
                f"Broker {target_broker_id!r} ({broker_type}) does not support "
                "order placement.  It is read-only."
            )

        # Submit the order
        log.info(
            "broker_router.routing_order",
            user_id=user_id,
            broker_id=target_broker_id,
            symbol=order.contract.symbol,
            side=order.side.value if hasattr(order.side, "value") else str(order.side),
        )

        try:
            broker = await self._manager.get_broker(target_broker_id)
            result = await broker.submit_order(order)
            log.info(
                "broker_router.order_submitted",
                user_id=user_id,
                broker_id=target_broker_id,
                order_id=result.order_id,
                status=result.status.value
                if hasattr(result.status, "value")
                else str(result.status),
            )
            return result
        except (ExecutionError, BrokerConnectionError, NotImplementedError) as exc:
            log.warning(
                "broker_router.order_failed",
                user_id=user_id,
                broker_id=target_broker_id,
                error=str(exc),
            )
            return await self._failover(
                user_id, order, target_broker_id, str(exc),
            )

    # ── Portfolio / position queries ──────────────────────────────────────

    async def get_balance(
        self, user_id: str, broker_id: str | None = None,
    ) -> dict[str, Any]:
        """Get balance from user's active broker (or the specified one)."""
        bid = broker_id or await self.get_active_broker(user_id)
        if bid is None:
            raise ExecutionError("No active broker set for this user.")

        broker = await self._manager.get_broker(bid)
        snapshot = await broker.get_portfolio()
        return {
            "broker_id": bid,
            "cash": snapshot.cash,
            "net_liquidation": snapshot.net_liquidation,
            "timestamp": snapshot.timestamp.isoformat(),
        }

    async def get_positions(
        self, user_id: str, broker_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get positions from user's active broker."""
        bid = broker_id or await self.get_active_broker(user_id)
        if bid is None:
            raise ExecutionError("No active broker set for this user.")

        broker = await self._manager.get_broker(bid)
        positions = await broker.get_positions()
        return [
            {
                "symbol": p.contract.symbol,
                "quantity": p.quantity,
                "avg_entry": p.avg_entry,
                "current_price": p.current_price,
                "unrealized_pnl": p.unrealized_pnl,
                "broker_id": bid,
            }
            for p in positions
        ]

    async def get_portfolio(
        self, user_id: str, broker_id: str | None = None,
    ) -> dict[str, Any]:
        """Get portfolio from active broker, or aggregate across all."""
        if broker_id is not None:
            broker = await self._manager.get_broker(broker_id)
            snapshot = await broker.get_portfolio()
            return {
                "broker_id": broker_id,
                "cash": snapshot.cash,
                "net_liquidation": snapshot.net_liquidation,
                "position_count": len(snapshot.positions),
                "total_delta": snapshot.total_delta,
                "total_gamma": snapshot.total_gamma,
                "total_theta": snapshot.total_theta,
                "total_vega": snapshot.total_vega,
                "daily_pnl": snapshot.daily_pnl,
                "total_pnl": snapshot.total_pnl,
                "timestamp": snapshot.timestamp.isoformat(),
            }

        bid = await self.get_active_broker(user_id)
        if bid is not None:
            return await self.get_portfolio(user_id, broker_id=bid)

        # No active broker -- aggregate all connected brokers.
        snapshot = await self._manager.get_aggregate_portfolio()
        return {
            "broker_id": "aggregate",
            "cash": snapshot.cash,
            "net_liquidation": snapshot.net_liquidation,
            "position_count": len(snapshot.positions),
            "total_delta": snapshot.total_delta,
            "total_gamma": snapshot.total_gamma,
            "total_theta": snapshot.total_theta,
            "total_vega": snapshot.total_vega,
            "daily_pnl": snapshot.daily_pnl,
            "total_pnl": snapshot.total_pnl,
            "timestamp": snapshot.timestamp.isoformat(),
        }

    # ── User broker listing ───────────────────────────────────────────────

    def get_user_brokers(self, user_id: str) -> list[str]:
        """List all broker_ids connected for a user.

        Currently returns all brokers in the manager.  In a multi-tenant
        deployment the manager would be scoped per user or a mapping
        would be maintained.
        """
        return self._manager.get_connected_broker_ids()

    # ── Failover ──────────────────────────────────────────────────────────

    async def _failover(
        self,
        user_id: str,
        order: Order,
        failed_broker: str,
        error: str,
    ) -> Order:
        """Try backup brokers in failover order."""
        failover_chain = self._failover_order.get(user_id, [])

        # Build candidate list: explicit failover chain, then all connected.
        candidates: list[str] = []
        for bid in failover_chain:
            if bid != failed_broker and bid not in candidates:
                candidates.append(bid)

        for bid in self._manager.get_connected_broker_ids():
            if bid != failed_broker and bid not in candidates:
                candidates.append(bid)

        for candidate_id in candidates:
            entry = await self._resolve_broker_entry(candidate_id)
            broker_type = entry.get("type", "")
            if not can_trade(broker_type):
                continue

            log.info(
                "broker_router.failover_attempt",
                user_id=user_id,
                failed_broker=failed_broker,
                candidate=candidate_id,
            )

            try:
                broker = await self._manager.get_broker(candidate_id)
                result = await broker.submit_order(order)
                log.info(
                    "broker_router.failover_success",
                    user_id=user_id,
                    broker_id=candidate_id,
                    order_id=result.order_id,
                )
                return result
            except Exception as exc:
                log.warning(
                    "broker_router.failover_failed",
                    user_id=user_id,
                    candidate=candidate_id,
                    error=str(exc),
                )
                continue

        raise ExecutionError(
            f"Order routing failed on {failed_broker!r} ({error}) "
            "and no failover broker could fulfil the order."
        )

    # ── Instrument-based broker selection ─────────────────────────────────

    def _select_broker_for_instrument(
        self, user_id: str, symbol: str,
    ) -> str | None:
        """Auto-select broker based on instrument type.

        - Crypto symbols (BTCUSDT etc) -> binance
        - Indian options (NFO:*) -> zerodha
        - US stocks (US:*) -> indmoney (read-only, so only for queries)
        - Default -> first connected broker that can trade
        """
        connected = self._manager.get_connected_broker_ids()
        if not connected:
            return None

        # Crypto heuristic
        if _CRYPTO_PATTERN.match(symbol):
            for bid in connected:
                entry = self._get_broker_type_sync(bid)
                if entry == "binance":
                    return bid
            return None

        # NFO / Indian options heuristic
        if _NFO_PATTERN.match(symbol):
            for bid in connected:
                entry = self._get_broker_type_sync(bid)
                if entry == "zerodha":
                    return bid
            return None

        # US stocks heuristic
        if _US_PATTERN.match(symbol):
            for bid in connected:
                entry = self._get_broker_type_sync(bid)
                if entry == "indmoney":
                    return bid
            return None

        # Default: first broker that can place orders
        for bid in connected:
            btype = self._get_broker_type_sync(bid)
            if can_trade(btype):
                return bid

        return connected[0] if connected else None

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_broker_type_sync(self, broker_id: str) -> str:
        """Synchronously peek at the broker type from the manager's internals.

        This avoids an ``await`` call inside synchronous selection logic by
        reading from the manager's internal ``_brokers`` dict directly.
        """
        entry = self._manager._brokers.get(broker_id)
        if entry is not None:
            return entry.broker_type
        return ""

    async def _resolve_broker_entry(self, broker_id: str) -> dict[str, Any]:
        """Return a dict with at least ``type`` for the given broker_id."""
        entry = self._manager._brokers.get(broker_id)
        if entry is not None:
            return entry.to_dict()

        # Fallback: try to look up in the manager's public API.
        all_brokers = await self._manager.get_all_brokers()
        for b in all_brokers:
            if b.get("id") == broker_id:
                return b

        return {"id": broker_id, "type": ""}
