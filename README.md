# HedgeFund AI — Level-2 Multi-Agent Trading Platform

An AI-powered options trading system with 8 specialized trading agents, multi-broker support, real-time market data streaming, and a live dashboard.

**Live Dashboard**: [hedgefund.viewfir.com](https://hedgefund.viewfir.com)

---

## Architecture

```
Zerodha/Binance Feeds ──┐
14 RSS News Feeds ──────┤
X (Twitter) Sentiment ──┤
                        ▼
                    EventBus (10K async queue)
                        │
          ┌─────────────┼─────────────────┐
          ▼             ▼                 ▼
   DataSource      8 AI Agents      News/Sentiment
   Validator       (parallel)        Processing
          │             │
          │     SignalFusionEngine
          │        (weighted)
          │             │
          └──── BrokerRouter ────► Zerodha / Binance / Paper
                    │
              PortfolioManager ──► Dashboard (WebSocket + REST)
```

## Key Features

### Multi-AI Agent System
| Agent | Strategy | Best Market Regime |
|-------|----------|-------------------|
| Trend Following | EMA alignment + ADX + MACD | Trending |
| Mean Reversion | Bollinger Bands + RSI + z-score | Mean Reverting |
| Options Volatility | IV/HV spread + PCR + GEX | High Volatility |
| Gamma Scalping | Gamma exposure + max pain | High Volatility |
| News Reaction | News sentiment + magnitude | Event Driven |
| Social Sentiment | X/Twitter + contrarian filter | High Volatility |
| Liquidity Sweep | Order book imbalance + volume | Trending |
| Smart Money Flow | Institutional flow + divergence | Trending |

Agents produce signals independently. The **Signal Fusion Engine** combines them with regime-adaptive weights, requiring 2+ agents to agree with 50%+ confidence.

### Multi-Broker Support
| Broker | Options | Futures | Crypto | Orders | Market Data |
|--------|---------|---------|--------|--------|-------------|
| Zerodha | Yes | Yes | - | Yes | Yes |
| Binance | Yes | Yes | Yes | Yes | Yes |
| Groww | - | - | - | Read-only | - |
| INDMoney | - | - | - | Read-only | - |
| Paper | Yes | Yes | Yes | Simulated | - |

- Per-user broker switching via dashboard dropdown
- Capability-based routing (crypto → Binance, options → Zerodha)
- Automatic failover if active broker goes down

### Real-Time Data Pipeline
- **Zerodha Kite API**: Live quotes (2s polling), historical OHLCV candles, instrument search
- **14 RSS News Feeds**: Yahoo Finance, CNBC, Reuters, MarketWatch, Moneycontrol, Economic Times, Seeking Alpha, and more
- **X (Twitter)**: OAuth integration + sentiment polling
- **EventBus**: Async pub/sub with 10K queue, backpressure handling

### Risk Management
- 1% max risk per trade
- 5% max daily drawdown
- 10% max portfolio drawdown — automatic trading halt
- 3-state circuit breaker: ACTIVE → HALTED → RECOVERY
- Chain-of-responsibility validators: Capital → Position Limit → Drawdown → Spread → Liquidity → Greeks

### Portfolio Optimization
- **Mean-Variance** (Markowitz efficient frontier)
- **Kelly Criterion** (fractional quarter-Kelly)
- **Risk Parity** (inverse-volatility weighting)

### Zero Synthetic Data Policy
The system **never** generates random, mock, or synthetic market data. If no data source is connected, all panels display `"DATA SOURCE NOT CONNECTED"`. The `WriteGuard` blocks any attempt to store synthetic data in MongoDB.

---

## Quick Start

### Prerequisites
- Python 3.11+
- MongoDB
- Redis (optional)

### Install
```bash
make install          # Production dependencies
make install-dev      # + pytest, mypy, ruff
```

### Configure
```bash
# Set environment
export HEDGEFUND_ENV=development

# Store broker credentials (encrypted)
python -c "
from hedgefund.security.credential_store import CredentialStore
store = CredentialStore()
store.store('zerodha', 'api_key', 'YOUR_KEY')
store.store('zerodha', 'api_secret', 'YOUR_SECRET')
"
```

### Run
```bash
# Dashboard only (recommended for getting started)
make dashboard

# Full trading application
make run

# Backtest
make backtest STRATEGY=trend_following START=2025-01-01 END=2025-12-31

# Train ML models
make train MODEL=all
```

### Docker
```bash
make docker-up        # Full stack: app + dashboard + redis + postgres + mongo
make docker-down
```

---

## Dashboard

The dashboard runs on port 8888 and provides:

### Overview
- Net liquidation value, daily P&L, drawdown
- Active trading signals from AI agents
- Recent trades

### Market Intelligence
- Live order book with buy/sell imbalance
- Smart money signals (block trades, VWAP deviation)
- Options flow (GEX, PCR, max pain, unusual activity)
- Liquidity sweep detection

### AI Signals
- Engine status (mode, agents, signals generated)
- 8 agent weight sliders (auto-adjust by market regime)
- Recent AI decisions with reasoning and signal breakdown

### Portfolio
- Equity curve
- Position details with Greeks
- Performance statistics (win rate, Sharpe, profit factor)
- Trade history

### News & Sentiment
- Live news feed from 14 RSS sources
- Social sentiment feed (X/Twitter)
- Sentiment heatmap by ticker
- Trending tickers by mention count

### Data Source Status
- Market feed connection status
- Broker connection status
- News API status
- X (Twitter) API status

---

## API Endpoints

### Market Data
| Endpoint | Description |
|----------|-------------|
| `GET /api/market-data/quote/{symbol}` | Live quote (e.g. `NSE:RELIANCE`) |
| `GET /api/market-data/ltp?symbols=...` | Last traded prices |
| `GET /api/market-data/historical/{token}/{interval}` | OHLCV candles |
| `GET /api/market-data/search?q=...` | Instrument search |

### Broker Management
| Endpoint | Description |
|----------|-------------|
| `GET /api/broker/connected` | Connected brokers with capabilities |
| `POST /api/broker/switch` | Switch active broker |
| `GET /api/broker/capabilities` | All broker capabilities |

### Trading
| Endpoint | Description |
|----------|-------------|
| `GET /api/signals` | Active trading signals |
| `GET /api/portfolio` | Portfolio snapshot (from active broker) |
| `GET /api/trades/stats` | Trading statistics |
| `GET /api/engine/status` | AI engine status |
| `GET /api/engine/decisions` | Recent AI decisions |

### News & Sentiment
| Endpoint | Description |
|----------|-------------|
| `GET /api/news/recent` | Recent news articles |
| `GET /api/news/sentiment-summary` | Heatmap + trending |
| `GET /api/news/x-feed` | X/Twitter sentiment |
| `POST /api/news/setup-free` | Register 14 free RSS feeds |

### System
| Endpoint | Description |
|----------|-------------|
| `GET /api/system/data-source-status` | All data source statuses |
| `GET /api/system/agent-status` | AI agent statuses |
| `GET /health` | Health check |

---

## Project Structure

```
src/hedgefund/
├── agents/                  # 8 AI trading agents
│   ├── base.py              # TradingAgent ABC + AgentSignal
│   ├── registry.py          # AgentRegistry lifecycle manager
│   ├── trend_following.py   # EMA + ADX + MACD
│   ├── mean_reversion.py    # Bollinger + RSI + z-score
│   ├── options_volatility.py # IV/HV + PCR + GEX
│   ├── gamma_scalping.py    # Gamma exposure + max pain
│   ├── news_reaction.py     # News sentiment
│   ├── social_sentiment.py  # X/Twitter + contrarian
│   ├── liquidity_sweep.py   # Order book imbalance
│   └── smart_money_flow.py  # Institutional flow
├── engine/                  # Core trading engine
│   ├── signal_fusion.py     # Weighted multi-agent fusion
│   ├── signal_runner.py     # 30s signal generation cycle
│   ├── decision_engine.py   # Legacy 7-signal fusion
│   ├── trade_executor.py    # Order execution + monitoring
│   ├── portfolio_manager.py # Position aggregation
│   ├── portfolio_optimizer.py # MVO / Kelly / Risk Parity
│   └── data_source_validator.py # No-synthetic-data gate
├── execution/               # Broker abstraction layer
│   ├── base.py              # Broker ABC
│   ├── broker_manager.py    # Multi-broker connection manager
│   ├── broker_router.py     # Per-user routing + failover
│   ├── capabilities.py      # Broker capability detection
│   ├── zerodha.py           # Zerodha Kite Connect v3
│   ├── binance.py           # Binance Spot/Futures/Options
│   ├── groww.py             # Groww (read-only)
│   ├── indmoney.py          # INDMoney (read-only)
│   ├── paper.py             # Paper trading simulator
│   ├── router.py            # Smart order router (TWAP/VWAP)
│   └── order_builder.py     # Multi-leg option orders
├── streaming/               # Real-time data pipeline
│   ├── event_bus.py         # Async pub/sub (10K queue)
│   ├── zerodha_feed.py      # Kite quote polling (2s)
│   ├── twitter_stream.py    # Twitter v2 search polling
│   ├── news_stream.py       # RSS/API news ingestion
│   ├── social_stream.py     # X sentiment pipeline
│   ├── market_feed.py       # WebSocket feed aggregator
│   ├── order_book.py        # Order book tracking
│   └── smart_money.py       # Institutional flow detection
├── risk/                    # Risk management
│   ├── validators.py        # Chain-of-responsibility checks
│   ├── position_sizer.py    # Fixed fraction / Kelly / ATR
│   ├── drawdown.py          # 3-state circuit breaker
│   └── portfolio_risk.py    # Portfolio-level checks
├── dashboard/               # Web dashboard
│   ├── server.py            # FastAPI app factory + lifespan
│   ├── routes/              # 13 API route modules
│   ├── static/              # HTML/CSS/JS frontend
│   └── websocket/           # WebSocket broadcast
├── data/                    # Data management
│   ├── write_guard.py       # MongoDB write validation
│   ├── zerodha_historical.py # Historical candle fetcher
│   └── ...                  # Market data, news, social feeds
├── learning/                # ML models
│   ├── lstm_model.py        # LSTM price forecaster
│   ├── transformer_model.py # Transformer-based model
│   └── rl_agent.py          # PPO reinforcement learning
├── features/                # Feature engineering
├── analysis/                # Market regime detection (HMM)
├── sentiment/               # Sentiment scoring
├── security/                # Credential encryption + rate limiting
├── config/                  # Pydantic settings + YAML loader
├── auth/                    # JWT authentication + MongoDB users
├── app.py                   # TradingApplication orchestrator
└── types.py                 # Shared domain types
```

---

## Configuration

Three-level merge (last wins):
```
config/base.yaml → config/{HEDGEFUND_ENV}.yaml → environment variables
```

Environment variables use `HEDGEFUND_` prefix with `__` nesting:
```bash
export HEDGEFUND_RISK__MAX_DRAWDOWN_PCT=0.10
export HEDGEFUND_EXECUTION__BROKER=zerodha
```

Key settings:
| Setting | Default | Description |
|---------|---------|-------------|
| `risk.risk_per_trade_pct` | 0.01 | Max 1% risk per trade |
| `risk.max_daily_loss_pct` | 0.05 | Max 5% daily drawdown |
| `risk.max_drawdown_pct` | 0.10 | Max 10% portfolio drawdown |
| `agents.enabled` | true | Enable multi-agent system |
| `agents.require_live_data` | true | Agents only run with real data |
| `portfolio_optimizer.method` | risk_parity | MVO / Kelly / Risk Parity |

---

## Security

- **Credentials**: Fernet-encrypted at `~/.hedgefund/credentials.enc`
- **Authentication**: JWT with access + refresh tokens
- **Log sanitization**: Masks passwords/secrets/tokens/keys
- **Write guard**: Blocks synthetic data from entering MongoDB
- **Rate limiting**: Per-broker API rate limit tracking

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Web Framework | FastAPI + Uvicorn |
| Database | MongoDB (Motor async driver) |
| Cache | Redis |
| ML | PyTorch, scikit-learn, Stable Baselines3 |
| Technical Analysis | ta, scipy, numpy, pandas |
| HTTP | httpx (async), websockets |
| Auth | JWT (PyJWT), bcrypt |
| Encryption | cryptography (Fernet) |
| Config | Pydantic, PyYAML |
| Logging | structlog |
| Linting | ruff |
| Testing | pytest (async) |

---

## License

Private — All rights reserved.
