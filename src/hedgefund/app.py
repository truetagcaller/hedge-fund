"""Level-2 Hedge Fund Architecture — TradingApplication orchestrator.

Lifecycle: init -> start -> (event-driven main loop) -> shutdown

The application is entirely event-driven via :class:`EventBus`.  The
multi-AI agent system feeds the :class:`SignalFusionEngine`, which
produces fused signals for the :class:`TradeExecutor`.  The
:class:`DataSourceValidator` ensures only real market data flows through
the system — NO synthetic, mock, or random data is ever permitted.

Graceful shutdown on SIGINT / SIGTERM with component-level error isolation.
"""

from __future__ import annotations

import asyncio
import enum
import signal
import time
from datetime import datetime
from typing import Any, Dict, Optional

import structlog

from hedgefund.config.settings import Settings, get_settings
from hedgefund.dashboard.server import create_app
from hedgefund.dashboard.websocket.live_feed import ConnectionManager
from hedgefund.exceptions import (
    CircuitBreakerTrippedError,
    DataError,
    ExecutionError,
    HedgeFundError,
    RiskError,
)
from hedgefund.types import PortfolioSnapshot, TradeSignal

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Trading mode
# ---------------------------------------------------------------------------

class TradingMode(enum.Enum):
    LIVE = "LIVE"
    PAPER = "PAPER"


# ---------------------------------------------------------------------------
# Component health status
# ---------------------------------------------------------------------------

class ComponentStatus(enum.Enum):
    NOT_STARTED = "NOT_STARTED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class TradingApplication:
    """Top-level orchestrator for the live trading system.

    Initializes, wires, and manages the lifecycle of every subsystem.
    The system is event-driven: the :class:`EventBus` is the backbone
    connecting all components.

    Parameters
    ----------
    settings:
        Application configuration.  Defaults to auto-loaded settings.
    mode:
        ``LIVE`` routes orders to real brokers; ``PAPER`` uses the
        :class:`PaperBroker` for all trades.
    dashboard_broadcast_interval:
        Seconds between dashboard WebSocket broadcasts.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        mode: TradingMode = TradingMode.PAPER,
        dashboard_broadcast_interval: float = 1.0,
    ) -> None:
        self._settings = settings or get_settings()
        self._mode = mode
        self._broadcast_interval = dashboard_broadcast_interval
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._start_time: float = 0.0

        # Component health
        self._component_status: Dict[str, ComponentStatus] = {}

        # Sub-system references -- populated in start()
        self._event_bus: Any = None
        self._broker_manager: Any = None
        self._market_feed_manager: Any = None
        self._news_stream_manager: Any = None
        self._social_stream_manager: Any = None
        self._feature_pipeline: Any = None
        self._decision_engine: Any = None
        self._trade_executor: Any = None
        self._portfolio_manager: Any = None
        self._drawdown_monitor: Any = None
        self._ws_manager: Optional[ConnectionManager] = None
        self._paper_simulator: Any = None

        # Level-2 components
        self._data_source_validator: Any = None
        self._agent_registry: Any = None
        self._signal_fusion: Any = None
        self._portfolio_optimizer: Any = None

        # Background tasks
        self._broadcast_task: Optional[asyncio.Task[None]] = None
        self._decision_task: Optional[asyncio.Task[None]] = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def mode(self) -> TradingMode:
        return self._mode

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def uptime_seconds(self) -> float:
        if self._start_time <= 0:
            return 0.0
        return time.monotonic() - self._start_time

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize all subsystems and run the event-driven main loop."""
        log.info(
            "app_starting",
            env=self._settings.app.env,
            mode=self._mode.value,
        )

        self._install_signal_handlers()
        await self._init_subsystems()

        self._running = True
        self._start_time = time.monotonic()

        log.info(
            "app_started",
            mode=self._mode.value,
            broker=self._settings.execution.broker,
            components=len(self._component_status),
        )

        try:
            await self._main_loop()
        except asyncio.CancelledError:
            log.info("app_cancelled")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shut down all subsystems."""
        if not self._running:
            return
        self._running = False
        self._shutdown_event.set()

        log.info("app_shutting_down")

        # Cancel background tasks
        for task in (self._broadcast_task, self._decision_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close all open positions in paper mode on shutdown
        if self._trade_executor is not None:
            try:
                await self._trade_executor.shutdown()
                self._set_status("trade_executor", ComponentStatus.STOPPED)
            except Exception:
                log.exception("shutdown_error", component="trade_executor")

        # Teardown in reverse order of initialization
        shutdown_order = [
            ("paper_simulator", self._paper_simulator),
            ("portfolio_manager", self._portfolio_manager),
            ("decision_engine", self._decision_engine),
            ("drawdown_monitor", self._drawdown_monitor),
            ("social_stream_manager", self._social_stream_manager),
            ("news_stream_manager", self._news_stream_manager),
            ("market_feed_manager", self._market_feed_manager),
            ("feature_pipeline", self._feature_pipeline),
            ("broker_manager", self._broker_manager),
            ("ws_manager", self._ws_manager),
            ("event_bus", self._event_bus),
        ]

        for name, subsystem in shutdown_order:
            if subsystem is None:
                continue
            stop_method = getattr(subsystem, "shutdown", None) or getattr(subsystem, "stop", None)
            if stop_method is not None:
                try:
                    await stop_method()
                    self._set_status(name, ComponentStatus.STOPPED)
                    log.debug("subsystem_stopped", name=name)
                except Exception:
                    log.exception("subsystem_shutdown_error", name=name)
                    self._set_status(name, ComponentStatus.ERROR)

        log.info(
            "app_shutdown_complete",
            uptime_seconds=round(self.uptime_seconds, 1),
        )

    # ── Main loop ────────────────────────────────────────────────────────

    async def _main_loop(self) -> None:
        """Event-driven main loop.

        The EventBus drives all data flow.  This loop simply:
        1. Periodically checks for decisions from the engine
        2. Broadcasts dashboard updates
        3. Monitors component health
        """
        # Start background tasks
        self._broadcast_task = asyncio.create_task(
            self._dashboard_broadcast_loop(), name="dashboard-broadcast"
        )
        self._decision_task = asyncio.create_task(
            self._decision_loop(), name="decision-loop"
        )

        # Wait until shutdown is signalled
        await self._shutdown_event.wait()

    async def _decision_loop(self) -> None:
        """Periodically check the decision engine for actionable signals.

        Level-2 flow:
        1. Validate data source is connected (gate all trading).
        2. Use SignalFusionEngine if available (multi-agent system).
        3. Fall back to legacy DecisionEngine otherwise.
        """
        while self._running:
            try:
                # ── Gate: data source must be verified ─────────────
                if self._data_source_validator is not None:
                    if not self._data_source_validator.is_trading_allowed():
                        # Agents must remain idle when no data source
                        if self._agent_registry and self._agent_registry.active_count > 0:
                            self._agent_registry.deactivate_all()
                            log.warning(
                                "decision_loop.agents_deactivated",
                                reason="DATA SOURCE NOT CONNECTED",
                            )
                        await asyncio.sleep(5.0)
                        continue

                if self._decision_engine is not None and self._trade_executor is not None:
                    symbols = self._decision_engine.get_tracked_symbols()
                    for symbol in symbols:
                        try:
                            decision = self._decision_engine.generate_decision(symbol)
                            if decision is not None:
                                broker_id = self._get_default_broker_id()
                                result = await self._trade_executor.execute_decision(
                                    decision, broker_id
                                )
                                if result.success:
                                    log.info(
                                        "decision_executed",
                                        symbol=symbol,
                                        trade_id=result.trade_id,
                                        action=decision.action.value,
                                    )
                                else:
                                    self._decision_engine.mark_decision_complete(symbol)
                                    log.debug(
                                        "decision_skipped",
                                        symbol=symbol,
                                        reason=result.reason,
                                    )
                        except CircuitBreakerTrippedError:
                            log.critical("circuit_breaker_tripped", symbol=symbol)
                            await self._emergency_shutdown()
                            return
                        except RiskError as exc:
                            log.error("risk_violation", symbol=symbol, error=str(exc))
                        except ExecutionError as exc:
                            log.error("execution_error", symbol=symbol, error=str(exc))
                        except Exception:
                            log.exception("decision_error", symbol=symbol)

                # Check at 1-second intervals
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("decision_loop_error")
                await asyncio.sleep(5.0)

    async def _dashboard_broadcast_loop(self) -> None:
        """Broadcast portfolio state to WebSocket clients."""
        while self._running:
            try:
                await self._push_to_dashboard()
                await asyncio.sleep(self._broadcast_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("dashboard_broadcast_error")
                await asyncio.sleep(self._broadcast_interval)

    async def _push_to_dashboard(self) -> None:
        """Broadcast current state to WebSocket clients."""
        if self._ws_manager is None or self._ws_manager.connection_count == 0:
            return

        # Push portfolio snapshot
        if self._portfolio_manager is not None:
            try:
                snapshot: PortfolioSnapshot = await self._portfolio_manager.get_snapshot()
                await self._ws_manager.broadcast_portfolio(snapshot)
                await self._ws_manager.broadcast_risk(snapshot)
            except Exception:
                log.exception("dashboard_portfolio_error")

        # Push open trades
        if self._trade_executor is not None:
            try:
                trades = self._trade_executor.get_open_trades()
                if trades:
                    await self._ws_manager.broadcast({
                        "type": "trades",
                        "data": trades,
                    })
            except Exception:
                log.exception("dashboard_trades_error")

        # Push system status (includes data source panel)
        try:
            await self._ws_manager.broadcast({
                "type": "system_status",
                "data": self.get_system_status(),
            })
        except Exception:
            log.exception("dashboard_status_error")

        # Push data source status for dashboard panel
        if self._data_source_validator is not None:
            try:
                report = self._data_source_validator.get_report()
                await self._ws_manager.broadcast({
                    "type": "data_source_status",
                    "data": report.to_dict(),
                })
            except Exception:
                log.exception("dashboard_datasource_error")

        # Push agent status
        if self._agent_registry is not None:
            try:
                await self._ws_manager.broadcast({
                    "type": "agent_status",
                    "data": self._agent_registry.get_status(),
                })
            except Exception:
                log.exception("dashboard_agent_status_error")

    async def _emergency_shutdown(self) -> None:
        """Emergency shutdown: close all positions and halt."""
        log.critical("emergency_shutdown_initiated")
        if self._trade_executor is not None:
            try:
                await self._trade_executor.close_all(reason="circuit_breaker")
            except Exception:
                log.exception("emergency_close_failed")
        self._running = False
        self._shutdown_event.set()

    # ── Initialization ───────────────────────────────────────────────────

    async def _init_subsystems(self) -> None:
        """Instantiate and wire up all trading subsystems.

        Each subsystem is optional -- the pipeline gracefully handles
        components that are not configured or not available.
        """
        log.info("subsystems_initializing")

        # 1. EventBus (core infrastructure -- must succeed)
        await self._init_event_bus()

        # 2. Data Source Validator (must be early — gates all trading)
        await self._init_data_source_validator()

        # 3. BrokerManager
        await self._init_broker_manager()

        # 4. Streaming managers (optional)
        await self._init_market_feed()
        await self._init_news_stream()
        await self._init_social_stream()

        # 5. Feature pipeline (optional)
        await self._init_feature_pipeline()

        # 6. Multi-AI Agent System
        await self._init_agent_registry()

        # 7. Signal Fusion Engine
        await self._init_signal_fusion()

        # 8. Portfolio Optimizer
        await self._init_portfolio_optimizer()

        # 9. Decision engine (legacy, still used for event routing)
        await self._init_decision_engine()

        # 10. Trade executor
        await self._init_trade_executor()

        # 11. Portfolio manager
        await self._init_portfolio_manager()

        # 12. Drawdown monitor
        await self._init_drawdown_monitor()

        # 13. WebSocket connection manager
        await self._init_ws_manager()

        # 14. Paper simulator — ONLY if real market feed exists
        #     Per policy: paper trading requires real market data
        if self._mode == TradingMode.PAPER:
            if self._data_source_validator and self._data_source_validator.is_trading_allowed():
                await self._init_paper_simulator()
            else:
                log.warning(
                    "paper_simulator.skipped",
                    reason="No real market data source connected. "
                           "Paper trading requires real market prices.",
                )
                self._set_status("paper_simulator", ComponentStatus.NOT_STARTED)

        # Log data source status
        if self._data_source_validator:
            report = self._data_source_validator.get_report()
            if not report.is_trading_ready:
                log.warning(
                    "app.data_sources_not_ready",
                    warnings=report.warnings,
                    message=self._data_source_validator.get_missing_credentials_message(),
                )
                # Deactivate agents if data sources not ready
                if self._agent_registry:
                    self._agent_registry.deactivate_all()

        running = sum(1 for s in self._component_status.values() if s == ComponentStatus.RUNNING)
        log.info(
            "subsystems_initialized",
            total=len(self._component_status),
            running=running,
        )

    async def _init_event_bus(self) -> None:
        name = "event_bus"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            from hedgefund.streaming.event_bus import EventBus
            self._event_bus = EventBus()
            await self._event_bus.start()
            self._set_status(name, ComponentStatus.RUNNING)
            log.info("subsystem_ready", name=name)
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)
            raise  # EventBus is required

    async def _init_broker_manager(self) -> None:
        name = "broker_manager"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            from hedgefund.execution.broker_manager import BrokerManager
            self._broker_manager = BrokerManager()

            # In paper mode, auto-add a paper broker
            if self._mode == TradingMode.PAPER:
                await self._broker_manager.add_broker(
                    broker_id="paper_default",
                    broker_type="paper",
                    credentials={"initial_cash": 10_000_000.0},
                )
            else:
                # Add configured broker
                broker_type = self._settings.execution.broker
                if broker_type != "paper":
                    try:
                        await self._broker_manager.add_broker(
                            broker_id=f"{broker_type}_main",
                            broker_type=broker_type,
                            credentials={},
                        )
                    except Exception as exc:
                        log.warning(
                            "broker_connection_failed",
                            broker_type=broker_type,
                            error=str(exc),
                        )

            self._set_status(name, ComponentStatus.RUNNING)
            log.info("subsystem_ready", name=name, mode=self._mode.value)
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)

    async def _init_market_feed(self) -> None:
        name = "market_feed_manager"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            from hedgefund.streaming.market_feed import MarketFeedManager  # type: ignore[attr-defined]
            self._market_feed_manager = MarketFeedManager(self._event_bus, self._settings.data)
            await self._market_feed_manager.start()
            self._set_status(name, ComponentStatus.RUNNING)
            log.info("subsystem_ready", name=name)
        except (ImportError, AttributeError):
            log.warning("subsystem_unavailable", name=name)
            self._set_status(name, ComponentStatus.NOT_STARTED)
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)

    async def _init_news_stream(self) -> None:
        name = "news_stream_manager"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            from hedgefund.streaming.news_stream import NewsStreamManager  # type: ignore[attr-defined]
            self._news_stream_manager = NewsStreamManager(self._event_bus, self._settings.sentiment)
            await self._news_stream_manager.start()
            self._set_status(name, ComponentStatus.RUNNING)
            log.info("subsystem_ready", name=name)
        except (ImportError, AttributeError):
            log.warning("subsystem_unavailable", name=name)
            self._set_status(name, ComponentStatus.NOT_STARTED)
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)

    async def _init_social_stream(self) -> None:
        name = "social_stream_manager"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            from hedgefund.streaming.social_stream import SocialStreamManager  # type: ignore[attr-defined]
            self._social_stream_manager = SocialStreamManager(self._event_bus, self._settings.sentiment)
            await self._social_stream_manager.start()
            self._set_status(name, ComponentStatus.RUNNING)
            log.info("subsystem_ready", name=name)
        except (ImportError, AttributeError):
            log.warning("subsystem_unavailable", name=name)
            self._set_status(name, ComponentStatus.NOT_STARTED)
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)

    async def _init_feature_pipeline(self) -> None:
        name = "feature_pipeline"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            from hedgefund.features.pipeline import FeaturePipeline
            self._feature_pipeline = FeaturePipeline()
            self._set_status(name, ComponentStatus.RUNNING)
            log.info("subsystem_ready", name=name)
        except (ImportError, AttributeError):
            log.warning("subsystem_unavailable", name=name)
            self._set_status(name, ComponentStatus.NOT_STARTED)
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)

    async def _init_decision_engine(self) -> None:
        name = "decision_engine"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            if self._event_bus is None:
                raise RuntimeError("EventBus required for DecisionEngine")
            from hedgefund.engine.decision_engine import TradingDecisionEngine
            self._decision_engine = TradingDecisionEngine(self._event_bus)
            self._set_status(name, ComponentStatus.RUNNING)
            log.info("subsystem_ready", name=name)
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)

    async def _init_trade_executor(self) -> None:
        name = "trade_executor"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            if self._event_bus is None or self._broker_manager is None:
                raise RuntimeError("EventBus and BrokerManager required for TradeExecutor")
            from hedgefund.engine.trade_executor import TradeExecutor

            # Optionally wire up risk manager
            risk_mgr = None
            try:
                from hedgefund.risk import RiskManager as _RM  # type: ignore[attr-defined]
                risk_mgr = _RM(self._settings.risk)
            except (ImportError, AttributeError):
                pass

            self._trade_executor = TradeExecutor(
                self._event_bus,
                self._broker_manager,
                risk_manager=risk_mgr,
            )
            await self._trade_executor.start()
            self._set_status(name, ComponentStatus.RUNNING)
            log.info("subsystem_ready", name=name)
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)

    async def _init_portfolio_manager(self) -> None:
        name = "portfolio_manager"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            if self._broker_manager is None:
                raise RuntimeError("BrokerManager required for PortfolioManager")
            from hedgefund.engine.portfolio_manager import PortfolioManager
            self._portfolio_manager = PortfolioManager(
                self._broker_manager,
                self._event_bus,
            )
            self._set_status(name, ComponentStatus.RUNNING)
            log.info("subsystem_ready", name=name)
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)

    async def _init_drawdown_monitor(self) -> None:
        name = "drawdown_monitor"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            from hedgefund.risk.drawdown import DrawdownMonitor
            self._drawdown_monitor = DrawdownMonitor(
                max_drawdown_pct=self._settings.risk.max_drawdown_pct,
                initial_equity=10_000_000.0,
            )
            self._set_status(name, ComponentStatus.RUNNING)
            log.info("subsystem_ready", name=name)
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)

    async def _init_ws_manager(self) -> None:
        name = "ws_manager"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            self._ws_manager = ConnectionManager()
            await self._ws_manager.start()
            self._set_status(name, ComponentStatus.RUNNING)
            log.info("subsystem_ready", name=name)
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)

    async def _init_paper_simulator(self) -> None:
        name = "paper_simulator"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            if self._event_bus is None:
                raise RuntimeError("EventBus required for PaperSimulator")
            from hedgefund.engine.paper_simulator import PaperTradingSimulator
            self._paper_simulator = PaperTradingSimulator(
                self._event_bus,
                symbols=self._settings.data.symbols if hasattr(self._settings.data, "symbols") else ["SPY"],
            )
            await self._paper_simulator.start()
            self._set_status(name, ComponentStatus.RUNNING)
            log.info("subsystem_ready", name=name)
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)

    # ── Level-2 component initialization ─────────────────────────────────

    async def _init_data_source_validator(self) -> None:
        name = "data_source_validator"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            if self._event_bus is None:
                raise RuntimeError("EventBus required for DataSourceValidator")
            from hedgefund.engine.data_source_validator import DataSourceValidator
            self._data_source_validator = DataSourceValidator(self._event_bus)
            self._data_source_validator.subscribe()
            self._set_status(name, ComponentStatus.RUNNING)
            log.info("subsystem_ready", name=name)
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)

    async def _init_agent_registry(self) -> None:
        name = "agent_registry"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            if self._event_bus is None:
                raise RuntimeError("EventBus required for AgentRegistry")
            if not self._settings.agents.enabled:
                log.info("agent_registry.disabled_by_config")
                self._set_status(name, ComponentStatus.NOT_STARTED)
                return

            from hedgefund.agents.registry import AgentRegistry
            self._agent_registry = AgentRegistry(self._event_bus)
            self._agent_registry.create_default_agents()

            # Agents start inactive — they activate only when data
            # source is verified (done in decision loop or externally)
            self._set_status(name, ComponentStatus.RUNNING)
            log.info(
                "subsystem_ready",
                name=name,
                agents=self._agent_registry.agent_count,
            )
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)

    async def _init_signal_fusion(self) -> None:
        name = "signal_fusion"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            if self._agent_registry is None or self._event_bus is None:
                log.info("signal_fusion.skipped", reason="no agent registry")
                self._set_status(name, ComponentStatus.NOT_STARTED)
                return

            from hedgefund.engine.signal_fusion import SignalFusionEngine
            self._signal_fusion = SignalFusionEngine(
                self._agent_registry,
                self._event_bus,
                min_confidence=self._settings.signals.min_confidence,
                min_agents=self._settings.agents.min_active_agents,
                min_risk_reward=self._settings.signals.min_risk_reward,
            )
            self._set_status(name, ComponentStatus.RUNNING)
            log.info("subsystem_ready", name=name)
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)

    async def _init_portfolio_optimizer(self) -> None:
        name = "portfolio_optimizer"
        self._set_status(name, ComponentStatus.STARTING)
        try:
            from hedgefund.engine.portfolio_optimizer import PortfolioOptimizer
            self._portfolio_optimizer = PortfolioOptimizer(
                self._settings.risk,
            )
            self._set_status(name, ComponentStatus.RUNNING)
            log.info(
                "subsystem_ready",
                name=name,
                method=self._settings.portfolio_optimizer.method,
            )
        except Exception:
            log.exception("subsystem_init_failed", name=name)
            self._set_status(name, ComponentStatus.ERROR)

    # ── Signal handlers ──────────────────────────────────────────────────

    def _install_signal_handlers(self) -> None:
        """Register SIGINT and SIGTERM for graceful shutdown."""
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._handle_signal, sig)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    def _handle_signal(self, sig: signal.Signals) -> None:
        log.info("signal_received", signal=sig.name)
        self._running = False
        self._shutdown_event.set()

    # ── Status / health ──────────────────────────────────────────────────

    def _set_status(self, component: str, status: ComponentStatus) -> None:
        self._component_status[component] = status

    def _get_default_broker_id(self) -> str:
        """Return the default broker ID for order routing."""
        if self._mode == TradingMode.PAPER:
            return "paper_default"
        broker_type = self._settings.execution.broker
        return f"{broker_type}_main"

    def get_system_status(self) -> Dict[str, Any]:
        """Return comprehensive system status."""
        components = {}
        for name, status in self._component_status.items():
            components[name] = status.value

        event_bus_stats = {}
        if self._event_bus is not None:
            event_bus_stats = self._event_bus.get_stats()

        trade_stats = {}
        if self._portfolio_manager is not None:
            try:
                trade_stats = self._portfolio_manager.get_trade_statistics().to_dict()
            except Exception:
                pass

        open_trades = 0
        if self._trade_executor is not None:
            try:
                open_trades = len(self._trade_executor.get_open_trades())
            except Exception:
                pass

        # Data source status
        data_source_status = {}
        if self._data_source_validator is not None:
            report = self._data_source_validator.get_report()
            data_source_status = report.to_dict()

        # Agent status
        agent_status = {}
        if self._agent_registry is not None:
            agent_status = self._agent_registry.get_status()

        return {
            "running": self._running,
            "mode": self._mode.value,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "components": components,
            "event_bus": event_bus_stats,
            "trade_statistics": trade_stats,
            "open_trades": open_trades,
            "drawdown_state": (
                self._drawdown_monitor.state.value
                if self._drawdown_monitor is not None
                else "unknown"
            ),
            "data_source": data_source_status,
            "agents": agent_status,
        }


async def run_dashboard(settings: Settings | None = None) -> None:
    """Start only the dashboard server (no trading loop)."""
    import uvicorn

    settings = settings or get_settings()
    app = create_app(settings)

    config = uvicorn.Config(
        app,
        host=settings.dashboard.host,
        port=settings.dashboard.port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    await server.serve()
