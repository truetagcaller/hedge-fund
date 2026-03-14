"""FastAPI application factory for the trading dashboard with multi-user auth."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from hedgefund.auth.database import MongoDB, get_database
from hedgefund.auth.jwt_handler import JWTHandler
from hedgefund.auth.middleware import AuthMiddleware
from hedgefund.auth.service import AuthService
from hedgefund.config.settings import Settings, get_settings
from hedgefund.dashboard.routes.auth import router as auth_router
from hedgefund.dashboard.routes.backtest import router as backtest_router
from hedgefund.dashboard.routes.broker_switch import router as broker_switch_router
from hedgefund.dashboard.routes.brokers import router as brokers_router
from hedgefund.dashboard.routes.data_sources import router as data_sources_router
from hedgefund.dashboard.routes.health import router as health_router
from hedgefund.dashboard.routes.market_intel import router as market_intel_router
from hedgefund.dashboard.routes.news_setup import router as news_setup_router
from hedgefund.dashboard.routes.portfolio import router as portfolio_router
from hedgefund.dashboard.routes.signals import router as signals_router
from hedgefund.dashboard.routes.system_status import router as system_status_router
from hedgefund.dashboard.routes.twitter_oauth import router as twitter_oauth_router
from hedgefund.dashboard.routes.market_data import router as market_data_router
from hedgefund.dashboard.routes.zerodha_oauth import router as zerodha_oauth_router
from hedgefund.dashboard.routes.twitter_webhook import router as twitter_webhook_router
from hedgefund.dashboard.routes.x_accounts import router as x_accounts_router
from hedgefund.dashboard.websocket.live_feed import ConnectionManager
from hedgefund.execution.broker_manager import BrokerManager
from hedgefund.execution.broker_router import BrokerRouter
from hedgefund.streaming.data_source_manager import DataSourceManager

log = structlog.get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown resources."""
    settings: Settings = app.state.settings
    manager: ConnectionManager = app.state.ws_manager

    log.info(
        "dashboard_starting",
        host=settings.dashboard.host,
        port=settings.dashboard.port,
    )

    # ── Startup ───────────────────────────────────────────────────────────

    # 1. Connect to MongoDB
    db: MongoDB = app.state.db
    try:
        await db.connect()
        log.info("mongodb_connected")
    except Exception:
        log.warning("mongodb_connection_failed", exc_info=True)

    # 2. Create indexes
    try:
        await db.create_indexes()
        log.info("mongodb_indexes_created")
    except Exception:
        log.warning("mongodb_index_creation_failed", exc_info=True)

    # 3. Start WebSocket manager background tasks
    await manager.start()

    # 4. Start data source managers
    dsm: DataSourceManager = app.state.data_source_manager
    await dsm.start_all()

    # 5. Start Twitter Poll Stream (if bearer token is available)
    twitter_stream = None
    try:
        from hedgefund.security.credential_store import CredentialStore
        store = CredentialStore()
        bearer = store.retrieve("twitter", "app_bearer_token")
        if bearer:
            from hedgefund.streaming.twitter_stream import TwitterPollStream
            twitter_stream = TwitterPollStream(
                bearer_token=bearer,
                event_bus=dsm.event_bus,
                db=db,
                poll_interval=60.0,
            )
            await twitter_stream.start()
            app.state.twitter_stream = twitter_stream
            log.info("twitter_poll_stream_started")
        else:
            log.info("twitter_stream.skipped", reason="no bearer token")
    except Exception:
        log.warning("twitter_stream.start_failed", exc_info=True)

    # 6. Start Zerodha market feed (live quotes)
    zerodha_feed = None
    try:
        from hedgefund.security.credential_store import CredentialStore as _ZStore
        _z_store = _ZStore()
        z_access_token = _z_store.retrieve("zerodha", "access_token")
        z_api_key = _z_store.retrieve("zerodha", "api_key")
        if z_access_token and z_api_key:
            from hedgefund.streaming.zerodha_feed import ZerodhaMarketFeed
            zerodha_feed = ZerodhaMarketFeed(
                z_api_key, z_access_token, dsm.event_bus,
            )
            await zerodha_feed.start()
            app.state.zerodha_feed = zerodha_feed
            app.state.market_data_provider = zerodha_feed
            log.info("zerodha_feed.started")
        else:
            log.info("zerodha_feed.skipped", reason="no credentials")
    except Exception:
        log.warning("zerodha_feed.start_failed", exc_info=True)

    # 7. Start AI Signal Runner (agents + fusion against live quotes)
    signal_runner = None
    try:
        if zerodha_feed is not None:
            from hedgefund.agents.registry import AgentRegistry
            from hedgefund.engine.signal_fusion import SignalFusionEngine
            from hedgefund.engine.signal_runner import SignalRunner

            agent_registry = AgentRegistry(dsm.event_bus)
            agent_registry.create_default_agents()
            agent_registry.activate_all(source="Zerodha Kite API")
            app.state.agent_registry = agent_registry

            signal_fusion = SignalFusionEngine(
                agent_registry, dsm.event_bus,
                min_confidence=0.50,
                min_agents=2,
                min_risk_reward=1.5,
            )
            app.state.signal_fusion = signal_fusion

            signal_runner = SignalRunner(
                agent_registry=agent_registry,
                signal_fusion=signal_fusion,
                event_bus=dsm.event_bus,
                db=db,
                zerodha_feed=zerodha_feed,
                interval=30.0,
            )
            await signal_runner.start()
            app.state.signal_runner = signal_runner
            log.info(
                "signal_runner.started",
                agents=agent_registry.active_count,
            )
        else:
            log.info("signal_runner.skipped", reason="no zerodha feed")
    except Exception:
        log.warning("signal_runner.start_failed", exc_info=True)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────

    log.info("dashboard_shutting_down")

    # 0. Stop signal runner
    if signal_runner is not None:
        await signal_runner.stop()

    # 0a. Stop Zerodha feed
    if zerodha_feed is not None:
        await zerodha_feed.stop()

    # 0b. Stop Twitter stream
    if twitter_stream is not None:
        await twitter_stream.stop()

    # 1. Stop data source manager
    await dsm.stop_all()

    # 2. Stop WebSocket manager
    await manager.shutdown()

    # 3. Disconnect MongoDB
    try:
        await db.disconnect()
        log.info("mongodb_disconnected")
    except Exception:
        log.warning("mongodb_disconnect_failed", exc_info=True)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return a fully configured FastAPI application.

    Parameters
    ----------
    settings:
        Application settings. Falls back to the global singleton when *None*.
    """
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # ── State ────────────────────────────────────────────────────────────
    app.state.settings = settings
    app.state.ws_manager = ConnectionManager(
        heartbeat_interval=settings.dashboard.ws_heartbeat_seconds,
    )
    app.state.broker_manager = BrokerManager()

    # Data source and streaming managers
    data_source_manager = DataSourceManager()
    app.state.data_source_manager = data_source_manager
    app.state.news_stream_manager = data_source_manager.news_manager
    app.state.social_stream_manager = data_source_manager.social_manager

    # System status state (set by TradingApplication when available)
    app.state.data_source_validator = None
    app.state.agent_registry = None

    # MongoDB connection
    db = get_database()
    app.state.db = db

    # Broker router (needs both broker_manager and db)
    app.state.broker_router = BrokerRouter(
        broker_manager=app.state.broker_manager,
        db=db,
    )

    # JWT handler and auth service
    jwt_handler = JWTHandler()
    app.state.jwt_handler = jwt_handler
    app.state.auth_service = AuthService(db=db, jwt=jwt_handler)

    # ── Middleware ────────────────────────────────────────────────────────
    # Note: middleware is applied in reverse order (last added = outermost)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(AuthMiddleware)

    # ── Routes ───────────────────────────────────────────────────────────
    # Health endpoint must remain public (no auth)
    app.include_router(health_router)

    # Auth routes
    app.include_router(auth_router, prefix="/api")

    # X (Twitter) account routes
    app.include_router(x_accounts_router, prefix="/api")

    # Core trading routes
    app.include_router(portfolio_router, prefix="/api")
    app.include_router(signals_router, prefix="/api")
    app.include_router(backtest_router, prefix="/api")
    app.include_router(brokers_router, prefix="/api")
    app.include_router(broker_switch_router)  # prefix="/api/broker" on router
    app.include_router(market_intel_router, prefix="/api")
    app.include_router(data_sources_router, prefix="/api")
    app.include_router(news_setup_router)

    # Twitter OAuth + webhook (public — no auth required by Twitter)
    app.include_router(twitter_oauth_router, prefix="/api")
    app.include_router(twitter_webhook_router)

    # Zerodha OAuth + postback (public — Kite redirect needs no auth)
    app.include_router(zerodha_oauth_router, prefix="/api")

    # Market data (Zerodha Kite live quotes + historical charts)
    app.include_router(market_data_router)

    # System status (prefix already set on the router)
    app.include_router(system_status_router)

    # ── WebSocket ────────────────────────────────────────────────────────
    from hedgefund.dashboard.websocket.live_feed import websocket_endpoint

    app.add_api_websocket_route("/ws", websocket_endpoint)

    # ── Static files (MUST come LAST after all API routes) ───────────────
    if _STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    return app
