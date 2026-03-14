# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
make install              # Production deps
make install-dev          # + pytest, mypy, ruff

# Test / Lint
make test                 # pytest with asyncio_mode=auto
make test-cov             # pytest + HTML coverage
make lint                 # ruff check src/ tests/
make format               # ruff format
make typecheck            # mypy --strict
make check                # lint + format + typecheck + test

# Run single test
PYTHONPATH=src pytest tests/unit/test_greeks.py -v
PYTHONPATH=src pytest tests/unit/test_greeks.py::TestGreeksCalculator::test_call_price_atm -v

# Run application
make run                  # Live trading (paper mode by default)
make dashboard            # Dashboard only (FastAPI on port 8000)
make backtest             # Requires STRATEGY, START, END vars
make train                # Train ML models (MODEL=lstm|transformer|rl|all)

# Docker
make docker-up            # Full stack: app + dashboard + redis + postgres
make docker-down
```

## Deployment

The dashboard runs on port 8888 and is exposed via Cloudflare Tunnel:
- **Domain**: `https://hedgefund.viewfir.com` → `http://localhost:8888`
- **Tunnel config**: `/etc/cloudflared/config.yml`
- **Start**: `PYTHONPATH=src nohup python3 -m uvicorn hedgefund.dashboard.server:create_app --factory --host 0.0.0.0 --port 8888 &`
- **Logs**: `/tmp/hedgefund-dashboard.log`

After code changes, restart the server:
```bash
lsof -ti:8888 | xargs -r kill -9; sleep 2
find src -name "__pycache__" -exec rm -rf {} + 2>/dev/null
PYTHONPATH=src nohup python3 -m uvicorn hedgefund.dashboard.server:create_app --factory --host 0.0.0.0 --port 8888 > /tmp/hedgefund-dashboard.log 2>&1 &
```

## Architecture

### Level-2 Hedge Fund Architecture

The system is a **multi-AI agent, multi-broker trading platform** built around an async `EventBus` (pub/sub with 10K queue). Supports simultaneous connections to multiple brokers with per-user broker switching, capability-based routing, and automatic failover.

### Event-Driven Pipeline

```
Zerodha/Binance feeds → EventBus → DataSourceValidator (gate)
                                  → 8 AI Agents → SignalFusionEngine → BrokerRouter → Active Broker
                                  → SignalRunner (30s cycle) → MongoDB signals
                                  → PortfolioManager → WebSocket Dashboard
```

**Event types:** TICK, ORDERBOOK, OPTIONS_CHAIN, FILL, SIGNAL, NEWS, SENTIMENT, RISK_UPDATE, PORTFOLIO_UPDATE. Handlers are async coroutines subscribed by type + optional symbol filter.

### Key Components (wired in `dashboard/server.py` lifespan + `app.py` TradingApplication)

| Component | Module | Role |
|-----------|--------|------|
| **EventBus** | `streaming/event_bus.py` | Central async pub/sub, backpressure via drop-oldest |
| **DataSourceValidator** | `engine/data_source_validator.py` | Gates all trading — verifies real data sources connected |
| **AgentRegistry** | `agents/registry.py` | Manages 8 AI trading agents, lifecycle, signal collection |
| **SignalFusionEngine** | `engine/signal_fusion.py` | Fuses multi-agent signals with regime-adaptive weights |
| **SignalRunner** | `engine/signal_runner.py` | Runs agents against live Zerodha quotes every 30s in dashboard mode |
| **PortfolioOptimizer** | `engine/portfolio_optimizer.py` | MVO, Kelly Criterion, Risk Parity allocation |
| **BrokerManager** | `execution/broker_manager.py` | Multi-broker connection manager, real adapter factory |
| **BrokerRouter** | `execution/broker_router.py` | Per-user broker selection, failover, capability-based routing |
| **BrokerCapabilities** | `execution/capabilities.py` | Declares what each broker supports (options, futures, crypto, etc.) |
| **ZerodhaMarketFeed** | `streaming/zerodha_feed.py` | Polls Kite Quote API every 2s, publishes TICK events |
| **ZerodhaHistorical** | `data/zerodha_historical.py` | Historical OHLCV candles from Kite Connect |
| **SmartOrderRouter** | `execution/router.py` | TWAP/VWAP execution, pre-trade validation, retries |
| **DecisionEngine** | `engine/decision_engine.py` | Legacy 7-signal fusion, outputs TradeDecision |
| **TradeExecutor** | `engine/trade_executor.py` | Validates via RiskManager, routes to broker, monitors stops/targets |
| **PortfolioManager** | `engine/portfolio_manager.py` | Aggregates positions across brokers, tracks equity/drawdown |
| **TwitterPollStream** | `streaming/twitter_stream.py` | Polls Twitter v2 Search API for financial tweets |
| **DrawdownMonitor** | `risk/drawdown.py` | 3-state circuit breaker (ACTIVE→HALTED→RECOVERY) |
| **WriteGuard** | `data/write_guard.py` | Validates all MongoDB writes — blocks synthetic data |
| **ConnectionManager** | `dashboard/websocket/live_feed.py` | WebSocket broadcast to dashboard clients |

### Multi-Broker System (`execution/`)

#### Broker Abstraction Layer

All brokers implement `execution/base.py::Broker` ABC:
- `connect()`, `disconnect()` — lifecycle
- `submit_order(Order)`, `cancel_order(str)` — trading
- `get_positions()`, `get_portfolio()` — queries
- `stream_fills()` — async iterator for fill updates

#### Broker Adapters

| Adapter | Module | Status | Trading | Market Data |
|---------|--------|--------|---------|-------------|
| **ZerodhaBroker** | `execution/zerodha.py` | Full | Yes (Kite Connect v3) | Yes |
| **BinanceBroker** | `execution/binance.py` | Full | Yes (Spot/Futures/Options) | Yes (WebSocket) |
| **GrowwBroker** | `execution/groww.py` | Read-only | No | No |
| **IndMoneyBroker** | `execution/indmoney.py` | Read-only | No | No |
| **PaperBroker** | `execution/paper.py` | Full | Yes (simulated) | No |

`BrokerManager._create_broker()` instantiates real adapters (ZerodhaBroker, BinanceBroker, etc.) — no stubs.

#### Broker Capabilities (`execution/capabilities.py`)

Each broker has a declared capability profile:

| Broker | Options | Futures | Crypto | Equity | MF | US Stocks | Orders | Market Data |
|--------|---------|---------|--------|--------|----|-----------|--------|-------------|
| **Zerodha** | Yes | Yes | - | Yes | - | - | Yes | Yes |
| **Binance** | Yes | Yes | Yes | - | - | - | Yes | Yes |
| **Groww** | - | - | - | Yes | Yes | - | Read-only | - |
| **INDMoney** | - | - | - | Yes | Yes | Yes | Read-only | - |
| **Paper** | Yes | Yes | Yes | Yes | - | - | Yes | - |

Functions: `get_capabilities(broker_type)`, `can_trade(broker_type)`, `supports_instrument(broker_type, exchange)`

#### Broker Router (`execution/broker_router.py`)

Routes trade requests to the correct broker:
- **Per-user active broker** — stored in MongoDB `user_preferences`, switchable via dashboard
- **Per-trade override** — `route_order(user_id, order, broker_id="binance_main")`
- **Capability-based auto-routing** — crypto → Binance, Indian options (NFO) → Zerodha, US stocks → INDMoney
- **Failover** — if active broker fails, tries backup brokers that support order placement
- **Aggregation** — `get_portfolio()` can aggregate across all connected brokers

#### Order Builder (`execution/order_builder.py`)

Constructs normalized multi-leg option orders: `single_leg()`, `vertical_spread()`, `iron_condor()`, `straddle()`, `strangle()`, `calendar_spread()`

### Multi-AI Agent System (`agents/`)

8 specialized trading agents, each implementing `TradingAgent` ABC:

| Agent | Module | Strategy | Best Regime |
|-------|--------|----------|-------------|
| **TrendFollowingAgent** | `agents/trend_following.py` | EMA alignment + ADX + MACD | TRENDING |
| **MeanReversionAgent** | `agents/mean_reversion.py` | Bollinger Bands + RSI + z-score | MEAN_REVERTING |
| **OptionsVolatilityAgent** | `agents/options_volatility.py` | IV/HV spread + PCR + GEX | HIGH_VOL_* |
| **GammaScalpingAgent** | `agents/gamma_scalping.py` | Gamma exposure + max pain | HIGH_VOL_* |
| **NewsReactionAgent** | `agents/news_reaction.py` | News sentiment + magnitude | HIGH_VOL_* |
| **SocialSentimentAgent** | `agents/social_sentiment.py` | X/Twitter + contrarian filter | HIGH_VOL_* |
| **LiquiditySweepAgent** | `agents/liquidity_sweep.py` | Order book imbalance + volume | TRENDING |
| **SmartMoneyFlowAgent** | `agents/smart_money_flow.py` | Institutional flow + divergence | TRENDING |

Each agent:
- Must have `_data_source_verified = True` before producing signals
- Returns `AgentSignal` with `data_source` and `is_live_data` fields
- Provides `get_weight(regime)` for regime-adaptive fusion
- Is managed by `AgentRegistry` which handles activation/deactivation

### Signal Generation Pipeline

In dashboard mode, `SignalRunner` (`engine/signal_runner.py`) runs every 30s:
1. Fetches live quotes from `ZerodhaMarketFeed`
2. Builds market_data dict for each instrument
3. Runs all 8 agents via `SignalFusionEngine`
4. Stores fused signals in MongoDB `signals` collection
5. Signals appear on dashboard Overview via `GET /api/signals`

Signals require: 2+ agents agreeing, 50%+ confidence, 1.5+ risk-reward ratio.

### Data Source Validation (CRITICAL)

**The system NEVER generates synthetic, mock, or random market data.**

`DataSourceValidator` (`engine/data_source_validator.py`) enforces:
1. All trading is gated — agents remain idle without verified data sources
2. Paper trading only runs with real market prices
3. Every `AgentSignal` must have `is_live_data=True`
4. `WriteGuard` blocks MongoDB writes from forbidden sources
5. All API endpoints return zeros/empty with `"DATA SOURCE NOT CONNECTED"` — NEVER random fallbacks

### External Integrations

#### Zerodha Kite Connect
- **OAuth flow**: `GET /api/auth/zerodha/login` → Kite login → `GET /api/auth/zerodha/callback`
- **Postback**: `POST /api/broker/zerodha/postback` receives order updates
- **Live quotes**: `ZerodhaMarketFeed` polls Quote API every 2s for subscribed instruments
- **Historical**: `ZerodhaHistorical` fetches OHLCV candles (minute to daily intervals)
- **Credentials**: `zerodha/` namespace — `api_key`, `api_secret`, `access_token`
- **Redirect URL**: `https://hedgefund.viewfir.com/api/auth/zerodha/callback`
- **Postback URL**: `https://hedgefund.viewfir.com/api/broker/zerodha/postback`
- **Whitelist IP**: `65.0.102.15`

#### X (Twitter)
- **OAuth 2.0 flow**: `GET /api/auth/twitter/login` → Twitter authorize → `GET /api/auth/twitter/callback`
- **Webhook**: `GET /twitter` (CRC challenge) + `POST /twitter` (Account Activity events)
- **Poll stream**: `TwitterPollStream` polls v2 Search API every 60s (auto-pauses when credits depleted)
- **Credentials**: `twitter/` namespace — `consumer_key`, `consumer_secret`, `client_id`, `client_secret`, `app_bearer_token`
- **Webhook URL**: `https://hedgefund.viewfir.com/twitter`
- **Callback URL**: `https://hedgefund.viewfir.com/api/auth/twitter/callback`

#### News Feeds (Free, no API keys)
- **Setup**: `POST /api/news/setup-free` registers 14 RSS feeds
- **Sources**: Yahoo Finance, Google News, Reuters, CNBC, Investing.com, MarketWatch, Moneycontrol, Economic Times, Seeking Alpha, GNews
- **Persistence**: `~/.hedgefund/data_sources.json`, auto-re-registers on restart
- **REST**: `GET /api/news/recent`, `GET /api/news/x-feed`, `GET /api/news/sentiment-summary`

### Dashboard

FastAPI app factory in `dashboard/server.py`. REST routes under `/api/*`, WebSocket at `/ws`. Static HTML served from `dashboard/static/`.

**Dashboard Startup Sequence** (lifespan in `server.py`):
1. MongoDB connect + indexes
2. WebSocket manager start
3. DataSourceManager start (loads persisted RSS feeds)
4. Twitter poll stream start
5. Zerodha market feed start (live quote polling)
6. AI Signal Runner start (agents + fusion, 30s cycle)

**Dashboard API Endpoints:**

| Endpoint | Source | Description |
|----------|--------|-------------|
| `GET /api/system/data-source-status` | `routes/system_status.py` | Checks credential store + MongoDB for connected sources |
| `GET /api/system/agent-status` | `routes/system_status.py` | All 8 AI agent statuses |
| `GET /api/system/credentials-required` | `routes/system_status.py` | Missing API keys |
| `GET /api/broker/connected` | `routes/broker_switch.py` | All connected brokers with capabilities |
| `GET /api/broker/active` | `routes/broker_switch.py` | User's active broker |
| `POST /api/broker/switch` | `routes/broker_switch.py` | Switch active broker |
| `GET /api/broker/capabilities` | `routes/broker_switch.py` | All broker capability profiles |
| `GET /api/market-data/quote/{symbol}` | `routes/market_data.py` | Live Zerodha quote |
| `GET /api/market-data/ltp?symbols=...` | `routes/market_data.py` | Last traded prices |
| `GET /api/market-data/historical/{token}/{interval}` | `routes/market_data.py` | Historical OHLCV candles |
| `GET /api/market-data/instruments` | `routes/market_data.py` | Instrument list by exchange |
| `GET /api/market-data/search?q=...` | `routes/market_data.py` | Instrument search |
| `GET /api/news/recent` | `routes/news_setup.py` | News feed (REST fallback) |
| `GET /api/news/x-feed` | `routes/news_setup.py` | X/Twitter tweets |
| `GET /api/news/sentiment-summary` | `routes/news_setup.py` | Heatmap + trending tickers |
| `POST /api/news/setup-free` | `routes/news_setup.py` | Register 14 free RSS feeds |
| `GET /api/trades/stats` | `routes/market_intel.py` | Real trade stats (zeros if none) |
| `GET /api/engine/decisions` | `routes/market_intel.py` | Real AI decisions (empty if none) |
| `GET /api/engine/status` | `routes/market_intel.py` | Engine status (idle/running) |
| `GET /api/auth/zerodha/login` | `routes/zerodha_oauth.py` | Zerodha OAuth redirect |
| `GET /api/auth/twitter/login` | `routes/twitter_oauth.py` | Twitter OAuth redirect |
| `GET /twitter` | `routes/twitter_webhook.py` | Twitter CRC challenge |
| `POST /twitter` | `routes/twitter_webhook.py` | Twitter webhook events |

**Dashboard UI Features:**
- Broker switcher dropdown in header
- Capability badges (Options, Futures, Crypto, Equity) per broker
- Active broker indicator
- 8 AI agent weight sliders (regime-adaptive)
- Currency display in ₹ (Indian Rupee, `en-IN` formatting)

**Important**: `market_intel.py` was purged of ALL random data generators. Every endpoint returns real data or explicit `"DATA SOURCE NOT CONNECTED"` / zeros.

App state holds: `broker_manager`, `broker_router`, `ws_manager`, `data_source_manager`, `data_source_validator`, `agent_registry`, `signal_fusion`, `signal_runner`, `twitter_stream`, `zerodha_feed`, `market_data_provider`.

### Domain Types

All shared types live in `types.py`: enums (`Side`, `OptionType`, `OrderType`, `OrderStatus`, `SignalAction`, `MarketRegime`, `DataOrigin`), dataclasses (`OHLCV`, `OptionContract`, `Greeks`, `TradeSignal`, `Order`, `Position`, `PortfolioSnapshot`, `TradeRecord`, `BacktestMetrics`). Every module communicates through these types.

### Risk Management

Chain-of-responsibility in `risk/validators.py`: CapitalAvailable → PositionLimit → DrawdownStatus → SpreadWidth → Liquidity → GreeksLimit. Position sizing: fixed_fraction (1% risk), kelly_criterion, volatility_adjusted (ATR-based). **Limits: 1% per trade, 5% daily loss, 10% max drawdown** — automatic shutdown if breached.

### Database Write Restrictions

`WriteGuard` (`data/write_guard.py`) and `GuardedMongoWriter` enforce:
- **Allowed writes**: Real market data, real broker trades, real sentiment, real backtest results (historical data)
- **Forbidden writes**: Randomly generated trades, simulated profits, placeholder signals
- Every write must include `source` field; forbidden sources are rejected with `ForbiddenWriteError`

### Credential Store

`security/credential_store.py` — Fernet-encrypted storage at `~/.hedgefund/credentials.enc`.

Currently stored namespaces:
- `zerodha/` — `api_key`, `api_secret`, `access_token`, `user_id`
- `twitter/` — `consumer_key`, `consumer_secret`, `client_id`, `client_secret`, `app_bearer_token`

API: `store.store(namespace, key, value)`, `store.retrieve(namespace, key)`, `store.list_namespaces()`

### Security

Credentials encrypted via Fernet in `security/credential_store.py`. Log sanitization via `security/sanitizer.py` masks keys matching password/secret/token/key patterns. Rate limiting per broker in `security/rate_limiter.py`.

## Conventions

- **Async-first**: All I/O modules use `async/await`. CPU-bound work (features, ML) dispatched via `asyncio.to_thread`.
- **Logging**: Use `structlog` via `hedgefund.logger.get_logger(__name__)`. Log every data ingestion with source origin.
- **Errors**: Raise from `hedgefund.exceptions` hierarchy (`DataError`, `ExecutionError`, `RiskError`, `SignalError`, `ConfigError`, `DataSourceError`, `AgentError`, `ForbiddenWriteError`).
- **No synthetic data**: Never generate random, mock, or synthetic market data anywhere in the codebase. If no data source is connected, show `"DATA SOURCE NOT CONNECTED"` and keep agents idle. API endpoints must return zeros/empty arrays, NEVER random fallback data.
- **Data origin metadata**: Every piece of data must include `source` field indicating origin (e.g. `"Zerodha Kite API"`, `"Binance Market Stream"`).
- **REST fallbacks**: Dashboard panels must have REST API fallbacks and not depend solely on WebSocket push from TradingApplication.
- **Multi-broker**: Use `BrokerRouter` for per-user broker selection. Check `BrokerCapabilities` before routing orders. Never hardcode broker types.
- **Python 3.11+**: Uses `slots=True` dataclasses, `X | Y` union syntax, `match/case` where appropriate.
- **Line length**: 100 (ruff configured in pyproject.toml).
- **Tests**: pytest with `asyncio_mode = "auto"`. Fixtures in `tests/conftest.py` provide sample OHLCV data, option contracts, portfolios, signals.
