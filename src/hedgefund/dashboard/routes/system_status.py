"""System status endpoints for data source and agent monitoring.

Checks both in-memory state (DataSourceValidator, AgentRegistry) AND
persistent state (MongoDB broker_connections, x_accounts, credential store)
to determine what is actually connected.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import structlog
from fastapi import APIRouter, Request

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/data-source-status")
async def get_data_source_status(request: Request) -> Dict[str, Any]:
    """Return current status of all data sources.

    Checks multiple sources of truth:
    1. DataSourceValidator (in-memory, if TradingApplication is running)
    2. MongoDB broker_connections and x_accounts collections
    3. Credential store for stored API keys
    4. SocialStreamManager accounts
    """
    # ── Try DataSourceValidator first (most accurate when running) ─────
    validator = getattr(request.app.state, "data_source_validator", None)
    if validator is not None:
        try:
            report = validator.get_report()
            return report.to_dict()
        except Exception as exc:
            log.warning("data_source_status.validator_error", error=str(exc))

    # ── Fall back to checking persistent state ────────────────────────
    warnings: List[str] = []

    market_feed: Dict[str, Any] = {
        "status": "NOT CONNECTED",
        "source": "",
        "last_timestamp": None,
    }
    broker: Dict[str, Any] = {
        "status": "NOT CONNECTED",
        "name": "",
    }
    news_api: Dict[str, Any] = {
        "status": "NOT CONFIGURED",
        "source": "",
    }
    x_sentiment: Dict[str, Any] = {
        "status": "NOT CONFIGURED",
        "source": "",
    }

    # ── Check credential store for stored keys ────────────────────────
    try:
        from hedgefund.security.credential_store import CredentialStore
        store = CredentialStore()
        namespaces = store.list_namespaces()

        # Zerodha
        if "zerodha" in namespaces:
            has_key = store.retrieve("zerodha", "api_key")
            has_token = store.retrieve("zerodha", "access_token")
            if has_key:
                broker["status"] = "CONNECTED" if has_token else "CONFIGURED"
                broker["name"] = "Zerodha Kite"
                # If broker connected, market feed is also available
                if has_token:
                    market_feed["status"] = "CONNECTED"
                    market_feed["source"] = "Zerodha Kite WebSocket"

        # Binance
        if "binance" in namespaces and broker["status"] != "CONNECTED":
            has_key = store.retrieve("binance", "api_key")
            if has_key:
                broker["status"] = "CONNECTED"
                broker["name"] = "Binance"
                market_feed["status"] = "CONNECTED"
                market_feed["source"] = "Binance Market Stream"

        # Twitter / X
        if "twitter" in namespaces:
            has_client = store.retrieve("twitter", "client_id")
            if has_client:
                x_sentiment["status"] = "CONFIGURED"
                x_sentiment["source"] = "X (Twitter) API"
    except Exception as exc:
        log.debug("data_source_status.credential_store_error", error=str(exc))

    # ── Check MongoDB for active connections ──────────────────────────
    db = getattr(request.app.state, "db", None)
    if db is not None:
        try:
            # Check broker connections
            broker_doc = await db.broker_connections.find_one(
                {"is_active": True},
                sort=[("connected_at", -1)],
            )
            if broker_doc:
                broker["status"] = "CONNECTED"
                broker["name"] = broker_doc.get("broker", broker["name"])
                broker_name = broker_doc.get("broker", "")
                if broker_name == "zerodha":
                    market_feed["status"] = "CONNECTED"
                    market_feed["source"] = "Zerodha Kite WebSocket"
                elif broker_name == "binance":
                    market_feed["status"] = "CONNECTED"
                    market_feed["source"] = "Binance Market Stream"
                user_name = broker_doc.get("user_name", "")
                if user_name:
                    broker["name"] += f" ({user_name})"
        except Exception as exc:
            log.debug("data_source_status.broker_db_error", error=str(exc))

        try:
            # Check X accounts
            x_doc = await db.x_accounts.find_one(
                {"is_active": True},
                sort=[("connected_at", -1)],
            )
            if x_doc:
                x_sentiment["status"] = "CONNECTED"
                username = x_doc.get("username", "")
                x_sentiment["source"] = (
                    f"@{username}" if username else "X (Twitter)"
                )
        except Exception as exc:
            log.debug("data_source_status.x_db_error", error=str(exc))

    # ── Check SocialStreamManager for active accounts ─────────────────
    social_mgr = getattr(request.app.state, "social_stream_manager", None)
    if social_mgr is not None:
        try:
            accounts = social_mgr.list_accounts()
            active = [a for a in accounts if a.get("status") == "active"]
            if active:
                x_sentiment["status"] = "CONNECTED"
                x_sentiment["source"] = "X (Twitter) Live"
        except Exception:  # noqa: S110
                log.debug("unexpected_error", exc_info=True)

    # ── Check DataSourceManager for news sources ──────────────────────
    dsm = getattr(request.app.state, "data_source_manager", None)
    if dsm is not None:
        try:
            all_sources = dsm.list_sources()
            news_types = {"rss", "news_api", "economic_calendar", "earnings"}
            news_found = [s for s in all_sources if s.get("type") in news_types]
            if news_found:
                active_news = [
                    s for s in news_found if s.get("status") == "active"
                ]
                if active_news:
                    news_api["status"] = "CONNECTED"
                    news_api["source"] = active_news[0].get("name", "News API")
                else:
                    news_api["status"] = "CONFIGURED"
                    news_api["source"] = news_found[0].get("name", "News API")
        except Exception:  # noqa: S110
                log.debug("unexpected_error", exc_info=True)

    # ── Build warnings ────────────────────────────────────────────────
    if market_feed["status"] != "CONNECTED":
        warnings.append("Market data source not connected.")
    if broker["status"] != "CONNECTED":
        warnings.append("No broker connected.")
    if news_api["status"] == "NOT CONFIGURED":
        warnings.append("News API not configured.")
    if x_sentiment["status"] == "NOT CONFIGURED":
        warnings.append("X (Twitter) sentiment not configured.")

    is_trading_ready = (
        market_feed["status"] == "CONNECTED"
        and broker["status"] == "CONNECTED"
    )

    return {
        "market_feed": market_feed,
        "broker": broker,
        "news_api": news_api,
        "x_sentiment": x_sentiment,
        "is_trading_ready": is_trading_ready,
        "is_any_source_connected": any(
            s["status"] == "CONNECTED"
            for s in [market_feed, broker, news_api, x_sentiment]
        ),
        "warnings": warnings,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/agent-status")
async def get_agent_status(request: Request) -> Dict[str, Any]:
    """Return status of all AI trading agents."""
    registry = getattr(request.app.state, "agent_registry", None)
    if registry is not None:
        try:
            return registry.get_status()
        except Exception as exc:
            log.warning("agent_status.registry_error", error=str(exc))
            return {"agents": {}, "error": str(exc)}

    # No live registry — report the 8 default agents as inactive
    default_agents = [
        "trend_following", "mean_reversion", "options_volatility",
        "gamma_scalping", "news_reaction", "social_sentiment",
        "liquidity_sweep", "smart_money_flow",
    ]
    agents = {}
    for name in default_agents:
        agents[name] = {
            "name": name,
            "active": False,
            "data_source_verified": False,
            "verified_source": "",
            "last_data_timestamp": None,
            "signals_generated": 0,
        }

    return {
        "total_agents": len(default_agents),
        "active_agents": 0,
        "agents": agents,
        "message": "Agents idle — waiting for live data source.",
    }


@router.get("/credentials-required")
async def get_credentials_required(request: Request) -> Dict[str, Any]:
    """Return what credentials / API keys are needed."""
    required: List[Dict[str, str]] = []

    # Check credential store
    has_broker = False
    has_news = False
    has_x = False

    try:
        from hedgefund.security.credential_store import CredentialStore
        store = CredentialStore()
        namespaces = store.list_namespaces()
        has_broker = "zerodha" in namespaces or "binance" in namespaces
        has_x = "twitter" in namespaces
    except Exception:  # noqa: S110
            log.debug("unexpected_error", exc_info=True)

    # Also check MongoDB
    db = getattr(request.app.state, "db", None)
    if db is not None:
        try:
            if not has_broker:
                broker_doc = await db.broker_connections.find_one({"is_active": True})
                has_broker = broker_doc is not None
        except Exception:  # noqa: S110
                log.debug("unexpected_error", exc_info=True)
        try:
            if not has_x:
                x_doc = await db.x_accounts.find_one({"is_active": True})
                has_x = x_doc is not None
        except Exception:  # noqa: S110
                log.debug("unexpected_error", exc_info=True)

    # Check news sources
    dsm = getattr(request.app.state, "data_source_manager", None)
    if dsm is not None:
        try:
            sources = dsm.list_sources()
            news_types = {"rss", "news_api", "economic_calendar", "earnings"}
            has_news = any(s.get("type") in news_types for s in sources)
        except Exception:  # noqa: S110
                log.debug("unexpected_error", exc_info=True)

    if not has_broker:
        required.append({
            "service": "broker",
            "description": (
                "Connect a broker (Zerodha, Binance, Groww, or IndMoney) "
                "for live market data and trade execution."
            ),
        })
    if not has_news:
        required.append({
            "service": "news_api",
            "description": "A News API key is recommended for fundamental analysis.",
        })
    if not has_x:
        required.append({
            "service": "x_api",
            "description": "Connect X (Twitter) for social sentiment analysis.",
        })

    return {
        "credentials_required": required,
        "all_configured": len(required) == 0,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
