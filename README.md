# Sorbot — AI-Powered BTC/USD Trading Bot

Sorbot is a **distributed algorithmic trading system** that uses machine learning to trade **BTC/USD spot** on Binance. It combines an XGBoost model trained on 122+ technical features with real-time market data, automated risk management, and continuous self-retraining — all orchestrated through a modern web dashboard.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Services Overview](#services-overview)
- [AI Engine (Python)](#ai-engine-python)
  - [API Endpoints](#ai-engine-api-endpoints)
  - [Data Pipeline](#data-pipeline)
  - [Feature Engineering (122+ Features)](#feature-engineering-122-features)
  - [Model Training (XGBoost Walk-Forward)](#model-training-xgboost-walk-forward)
  - [Prediction & Signal Generation](#prediction--signal-generation)
  - [Continuous Retraining Scheduler](#continuous-retraining-scheduler)
  - [Risk Management](#risk-management)
  - [Exchange Integration](#exchange-integration-binance-spot)
  - [Backtester](#backtester)
- [Backend (Java / Spring Boot)](#backend-java--spring-boot)
  - [API Endpoints](#backend-api-endpoints)
  - [Database Schema](#database-schema-h2)
  - [Scheduler & Trading Logic](#scheduler--trading-logic)
  - [WebSocket (Real-Time Updates)](#websocket-real-time-updates)
- [Frontend (React)](#frontend-react)
  - [Pages](#pages)
  - [Components](#components)
  - [WebSocket Client](#websocket-client)
- [Configuration Reference](#configuration-reference)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)

---

## System Architecture

```
┌──────────────┐       REST        ┌──────────────┐       REST        ┌──────────────┐
│   Frontend   │  ──────────────►  │   Backend    │  ──────────────►  │  AI Engine   │
│  React 19    │  ◄──────────────  │ Spring Boot  │  ◄──────────────  │   FastAPI    │
│  Vite 5      │    WebSocket      │  Java 17     │                   │  Python 3.11 │
│  Port 3000   │   (STOMP/SockJS)  │  Port 8081   │                   │  Port 8000   │
└──────────────┘                   └──────┬───────┘                   └──────┬───────┘
                                          │                                  │
                                          ▼                                  ▼
                                   ┌──────────────┐                   ┌──────────────┐
                                   │   H2 File DB │                   │   Binance    │
                                   │  sorbot_db   │                   │  Spot API    │
                                   └──────────────┘                   └──────────────┘
                                                                             │
                                                                      ┌──────────────┐
                                                                      │   yfinance   │
                                                                      │ (historical) │
                                                                      └──────────────┘
```

**Data flow:** Frontend → Backend (REST + WebSocket) → AI Engine (REST via WebClient) → Binance API + yfinance

---

## Services Overview

| Service | Stack | Port | Role |
|---------|-------|------|------|
| **AI Engine** | Python 3.11, FastAPI, XGBoost | 8000 | ML model training, predictions, trade execution, self-retraining |
| **Backend** | Java 17, Spring Boot 3.2.3, H2 | 8081 | REST API gateway, persistence, scheduling, WebSocket broker |
| **Frontend** | React 19, Vite 5, Nginx | 3000 | SPA dashboard for monitoring and controlling the bot |

---

## AI Engine (Python)

The core intelligence of Sorbot. Located in `ai_engine/`.

### AI Engine API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check — returns engine status, model state, retrain count |
| `POST` | `/train` | Retrain model on the latest market data; returns CV + final metrics |
| `GET` | `/predict` | Get latest BTC/USD trading signal with full market analysis |
| `POST` | `/trade` | Get prediction → if LONG + high confidence → execute on Binance |
| `GET` | `/status` | Account balance, open positions, Binance account info |
| `POST` | `/close` | Close any open BTC position at market price |
| `GET` | `/model-info` | Trained model metrics and top feature importances |
| `GET` | `/retrain-status` | Continuous retraining scheduler status + recent history |
| `POST` | `/retrain-now` | Manually trigger an immediate retrain (force mode, skips validation) |
| `GET` | `/retrain-history` | Full retraining history log (last 100 cycles) |

**Startup behavior:** On boot, the engine loads the existing model. If no model exists, it automatically fetches data and trains an initial model. The retraining scheduler starts immediately after.

---

### Data Pipeline

**File:** `ai_engine/ml_core/data_loader.py`

The data loader fetches BTC/USD OHLCV (Open, High, Low, Close, Volume) data from **yfinance** across three timeframes:

| Timeframe | Method | Coverage | Bars |
|-----------|--------|----------|------|
| **1h** (primary) | Chunked download in 59-day windows | Up to 730 days (2 years) | ~17,000 |
| **4h** (higher TF) | Resampled from 1h data | Same as 1h | ~4,250 |
| **1d** (daily context) | Single period download | Up to 5 years | ~1,800 |

**Why chunked?** yfinance limits intraday data to ~60 days per request. The loader works around this by downloading in 59-day chunks, going back up to 730 days, then stitching and deduplicating.

**Caching:** Downloaded data is cached as CSV in `ai_engine/data/`. Cache TTL is 1 hour for intraday data and 12 hours for daily data to avoid redundant API calls.

---

### Feature Engineering (122+ Features)

**File:** `ai_engine/ml_core/feature_eng.py`

Every feature is **strictly lagged** (no future data leakage). Features are organized into 10 categories:

#### Trend Features (~20)
- **EMA distances:** Price distance from EMA-9, EMA-21, EMA-50, EMA-200 (as % of price)
- **EMA crossovers:** EMA 9/21 cross, EMA 21/50 cross, EMA 50/200 cross (golden cross / death cross)
- **EMA slopes:** Rate of change of EMA-9, EMA-21, EMA-50
- **SMA distances:** Price distance from SMA-50, SMA-200
- **Golden cross:** SMA-50 above SMA-200 (binary)
- **VWAP:** Distance from VWAP, above/below VWAP flag
- **Ichimoku:** Tenkan-Kijun cross, cloud distance, above-cloud flag

#### Momentum Features (~25)
- **RSI (14):** Value, slope (rate of change), oversold (<30) / overbought (>70) flags
- **MACD (12, 26, 9):** MACD line, signal line, histogram, cross signal, histogram slope
- **Stochastic (14, 3):** %K, %D, cross signal, oversold (<20) / overbought (>80) flags
- **ADX (14):** Value, strong trend flag (>25), DI cross (DI+ vs DI–)
- **Williams %R:** Momentum oscillator
- **CCI:** Commodity Channel Index
- **MFI:** Money Flow Index (volume-weighted RSI)
- **Rate of Change:** ROC at 6, 12, 24, 48 periods
- **Divergences:** RSI divergence and MACD divergence vs price

#### Volatility Features (~15)
- **ATR (14):** Value as % of price, ATR change, ATR ratio (20-period vs 50-period)
- **Bollinger Bands (20, 2σ):** %B position, bandwidth
- **Keltner Channel:** Position within channel
- **Squeeze:** Bollinger inside Keltner (low volatility), squeeze release detection
- **Historical Volatility:** 10, 20, 50-period, volatility ratio, intrabar volatility

#### Volume Features (~10)
- **Volume ratio:** Current vs 20-period MA
- **Volume spike:** Volume > 2x average (binary)
- **Volume trend:** Linear regression slope of volume
- **OBV:** On-Balance Volume (normalized), slope
- **VWAP deviation:** Distance from volume-weighted average price
- **A/D Line:** Accumulation/Distribution line (normalized), slope
- **Volume-price confirmation:** Price and volume moving in same direction

#### Returns Features (6)
- Simple returns over 1, 2, 3, 5, 10, 20 periods

#### Candle Pattern Features (~12)
- **Body ratio:** |Close – Open| / (High – Low)
- **Shadow analysis:** Upper and lower shadow ratios
- **Patterns:** Bullish candle, doji, engulfing, hammer, shooting star
- **Consecutive candles:** Count of consecutive up/down candles
- **Structure:** Higher high + higher close, lower low + lower close

#### Structure Features (~7)
- **Pivot points:** Distance from pivot, R1, R2, S1, S2
- **Price position:** Where price sits in the 24-hour and 7-day range (0 to 1)

#### Regime Features (~5)
- **Trending:** ADX > 25 and absolute EMA-21 slope above threshold
- **Ranging:** ADX < 20 (binary)
- **Volatile:** ATR above 1.5x its 50-period moving average
- **Trend strength:** Composite score from ADX, EMA slope, and moving average alignment
- **Mean reversion:** RSI deviation from 50, scaled for ranging markets

#### Calendar Features (~8)
- **Hour of day:** Raw value + sine/cosine encoding for cyclical representation
- **Trading session:** Asia (0–8 UTC), Europe (8–14 UTC), US (14–22 UTC)
- **Day of week:** Raw value + sine/cosine encoding, weekend flag

#### Higher Timeframe Features (~8 per TF)
Applied to both 4h and 1d data, giving ~16 additional features:
- HTF RSI, EMA cross, MACD histogram, ADX, ATR %, trend direction, Bollinger %B, momentum score

#### Target Engineering
- **Lookahead:** 3 candles (3 hours on 1h timeframe)
- **Classification:** Binary — `1` if price goes up ≥ 0.3% within 3 hours, `0` if it drops ≤ –0.3%
- **Flat periods** (between –0.3% and +0.3%) are excluded from training to produce cleaner signals

---

### Model Training (XGBoost Walk-Forward)

**File:** `ai_engine/ml_core/trainer.py`

The model uses **XGBoost** (native Booster API, not sklearn wrapper) with **walk-forward cross-validation** to prevent look-ahead bias.

#### Walk-Forward Validation
```
Fold 1: [========TRAIN========][TEST]
Fold 2: [===========TRAIN===========][TEST]
Fold 3: [==============TRAIN==============][TEST]
Fold 4: [=================TRAIN=================][TEST]
Fold 5: [====================TRAIN====================][TEST]
```

- **5 folds** with an expanding training window
- **200-bar test window** per fold (200 hours)
- Minimum 500 bars required for the first training window
- **Early stopping** at 50 rounds to prevent overfitting

#### XGBoost Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_estimators` | 800 | Maximum boosting rounds |
| `max_depth` | 5 | Tree depth (prevents overfitting) |
| `learning_rate` | 0.01 | Conservative learning rate |
| `subsample` | 0.7 | Row sampling per tree |
| `colsample_bytree` | 0.7 | Feature sampling per tree |
| `min_child_weight` | 10 | Minimum leaf node weight |
| `gamma` | 0.3 | Minimum loss reduction for split |
| `reg_alpha` | 0.5 | L1 regularization |
| `reg_lambda` | 2.0 | L2 regularization |
| `scale_pos_weight` | dynamic | Adjusted per fold for class imbalance |
| `objective` | `binary:logistic` | Binary classification |
| `eval_metric` | `logloss` | Logarithmic loss |

#### Training Output
- **Model:** Saved as `ai_engine/models/btc_model.json` (native XGBoost Booster format)
- **Metadata:** Saved as `ai_engine/models/btc_meta.json` — includes training timestamp, sample counts, feature names, CV metrics, fold details, final metrics, top 20 features
- **Metrics tracked:** Accuracy, precision, recall, F1 score, AUC-ROC

---

### Prediction & Signal Generation

**File:** `ai_engine/ml_core/predictor.py`

Each prediction goes through a multi-step enrichment pipeline:

1. **Feature alignment:** Match current features to training features (fill missing with 0)
2. **Probability:** XGBoost outputs P(UP) for the latest bar
3. **Live price:** Fetches real-time BTC price from Binance public API (falls back to yfinance last close)
4. **Signal determination:**
   - P(UP) ≥ 0.65 → **LONG** (buy signal)
   - P(UP) ≤ 0.35 → **SHORT** (sell signal, blocked in spot mode)
   - 0.35 < P(UP) < 0.65 → **NO_TRADE** (uncertain zone)
5. **SL/TP calculation:** ATR-based stop-loss (1.5× ATR) and take-profit (3.0× ATR)
6. **R:R validation:** Trade rejected if reward:risk ratio < 1.8
7. **Market analysis:** Full indicator dashboard (RSI zone, MACD signal, ADX strength, squeeze state, etc.)
8. **HTF alignment:** Checks 4h and 1d bias (bullish/bearish/neutral)
9. **Conclusion:** Human-readable multi-paragraph summary of the trade rationale

---

### Continuous Retraining Scheduler

**File:** `ai_engine/ml_core/retrainer.py`

The model **automatically retrains itself** on fresh market data to adapt to changing market conditions.

#### How It Works

1. A **background daemon thread** runs alongside the FastAPI server
2. Every **6 hours** (configurable), it checks if the model needs retraining
3. If the model is stale, it:
   - **Backs up** the current model and metadata
   - **Fetches fresh data** from yfinance (force refresh, bypass cache)
   - **Builds features** (122+ indicators) and **trains a new model**
   - **Validates** the new model against the old one
   - If the new model passes validation → **hot-swaps** the predictor (zero downtime)
   - If the new model fails → **restores the backup**
4. All results are logged to `ai_engine/models/retrain_history.json`

#### Validation Gate

The new model is accepted only if its AUC-ROC and F1 scores do not drop more than **3%** below the current model's scores. This prevents replacing a good model with a worse one during market anomalies.

#### Error Handling

- **Exponential backoff:** On 3+ consecutive failures, the interval doubles (capped at 24h)
- **Graceful recovery:** Backup model is always restored on failure
- **History cap:** Last 100 retrain records are persisted

#### Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRAIN_ENABLED` | `true` | Enable/disable the scheduler |
| `RETRAIN_INTERVAL_HOURS` | `6` | Hours between retrain checks |
| `RETRAIN_FORCE_AFTER_HOURS` | `24` | Force retrain if model is older than this |
| `RETRAIN_MIN_IMPROVEMENT` | `0.03` | Max allowed metric degradation (3%) |

---

### Risk Management

**File:** `ai_engine/ml_core/risk_manager.py`

Conservative risk rules designed for a $500 spot account with no leverage:

| Rule | Value | Description |
|------|-------|-------------|
| Risk per trade | 1.5% | $7.50 per trade at $500 balance |
| Max risk per trade | 2.0% | Absolute cap of $10 |
| Max open positions | 1 | Only one trade at a time |
| Stop-loss | 1.5× ATR | Distance below/above entry |
| Take-profit | 3.0× ATR | 2:1 reward-to-risk target |
| Trailing stop | 1.0× ATR | Optional trailing stop distance |
| Minimum R:R | 1.8 | Trade rejected if R:R < 1.8 |
| Min balance | $10 | Won't trade below this |
| Min order | $10 / 0.00001 BTC | Binance minimum requirements |

**Position sizing:** Calculates BTC quantity from `(risk_amount) / (entry – SL distance)`, capped at 95% of available balance.

---

### Exchange Integration (Binance Spot)

**File:** `ai_engine/ml_core/exchange.py`

Connects to Binance via `python-binance`. Supports testnet mode.

| Method | Description |
|--------|-------------|
| `connect()` | Initialize Binance Client (spot, optional testnet) |
| `get_balance()` | Total USDT balance (free + locked) |
| `get_available_balance()` | Free USDT only |
| `get_btc_balance()` | Free BTC balance |
| `get_position()` | Open position details (qty, entry, current price, unrealized PnL) |
| `get_current_price()` | Latest BTCUSDT spot price |
| `place_order()` | Market BUY → OCO sell order (SL + TP combined); falls back to separate stop-loss if OCO fails |
| `close_position()` | Cancel all open orders → market SELL all BTC → calculate PnL |
| `cancel_all_orders()` | Cancel all open BTCUSDT orders |

**Order flow:** Market BUY at current price → immediately places an OCO (One-Cancels-Other) order with both the take-profit limit sell and stop-loss limit sell. If OCO creation fails, a standalone stop-loss limit order is placed as fallback.

---

### Backtester

**File:** `ai_engine/backtest.py`

Simulates the full trading strategy on historical data using the same logic as live trading:

- Walk-forward retraining every **168 bars** (1 week), minimum 500 bars initial training
- Same confidence gates (LONG ≥ 0.65, SHORT ≤ 0.35)
- ATR-based SL/TP with R:R validation
- Position sizing with fixed 3% risk per trade
- Trade resolution: SL hit, TP hit, or time expiry (max 12+ bars)
- Tracks: PnL, win rate, profit factor, max drawdown, equity curve
- Starting balance: $500, no leverage (spot only)

---

## Backend (Java / Spring Boot)

The middleware layer that sits between the frontend and AI engine. Located in `backend/`.

**Tech stack:** Java 17, Spring Boot 3.2.3, Spring Data JPA, H2 database, WebFlux WebClient, STOMP WebSocket, Lombok

### Backend API Endpoints

#### Dashboard (`/api`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/dashboard` | Aggregated view: settings, latest prediction, trade stats, AI health |
| `GET` | `/api/account` | Binance account status (via AI engine) |
| `GET` | `/api/model` | AI model metrics and feature importances |
| `POST` | `/api/train` | Trigger model retraining (via AI engine) |
| `GET` | `/api/health` | Backend health + AI engine connectivity |

#### Predictions (`/api/predictions`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/predictions` | Last 50 predictions |
| `GET` | `/api/predictions/latest` | Most recent prediction |
| `POST` | `/api/predictions/fetch` | Manually fetch a new prediction from AI engine |
| `POST` | `/api/predictions/{id}/accept` | Accept a pending prediction → execute trade on Binance |
| `POST` | `/api/predictions/{id}/reject` | Reject a pending prediction |

#### Trades (`/api/trades`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/trades` | Last 50 trades |
| `GET` | `/api/trades/open` | Currently open trades |
| `GET` | `/api/trades/stats` | Win count, loss count, total PnL |
| `POST` | `/api/trades/close` | Close current position (via AI engine) |

#### Settings (`/api/settings`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/settings` | Current trading settings (mode, poll interval) |
| `PUT` | `/api/settings/mode` | Switch between AUTO and MANUAL mode |

---

### Database Schema (H2)

The backend uses an H2 file-based database (`data/sorbot_db`) with auto DDL generation.

#### `predictions` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | Long (PK, auto) | Unique identifier |
| `timestamp` | LocalDateTime | When the prediction was made |
| `symbol` | String | Trading pair (BTC/USD) |
| `signal` | String | LONG, SHORT, or NO_TRADE |
| `probability_up` | Double | P(price goes up) from model |
| `probability_down` | Double | P(price goes down) |
| `confidence_pct` | Double | Max probability as percentage |
| `current_price` | Double | BTC price at prediction time |
| `atr` | Double | Average True Range value |
| `atr_pct` | Double | ATR as % of price |
| `sl_price` | Double | Stop-loss price level |
| `tp_price` | Double | Take-profit price level |
| `risk_reward` | Double | Reward-to-risk ratio |
| `reject_reason` | String | Why trade was rejected (if applicable) |
| `trend_direction` | String | Current trend (bullish/bearish/neutral) |
| `market_regime` | String | Trending, ranging, or volatile |
| `rsi` | Double | RSI value |
| `rsi_zone` | String | Oversold, overbought, or neutral |
| `adx` | Double | ADX value |
| `adx_interpretation` | String | Weak/moderate/strong trend |
| `macd_signal` | String | MACD cross signal |
| `is_squeeze` | Boolean | Bollinger/Keltner squeeze active |
| `volume_ratio` | Double | Volume vs 20-period average |
| `htf_overall_alignment` | String | Multi-timeframe alignment score |
| `htf_4h_bias` | String | 4-hour timeframe bias |
| `htf_1d_bias` | String | Daily timeframe bias |
| `conclusion` | Text | AI-generated trade summary |
| `raw_response` | Text | Full AI engine JSON response |
| `trade_status` | String | PENDING / ACCEPTED / REJECTED / AUTO_EXECUTED / SKIPPED |
| `trade_executed_at` | LocalDateTime | When trade was executed |
| `trade_mode` | String | AUTO or MANUAL |

#### `trades` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | Long (PK, auto) | Unique identifier |
| `prediction_id` | Long (FK → predictions) | Linked prediction |
| `executed_at` | LocalDateTime | Trade execution time |
| `symbol` | String | Trading pair |
| `side` | String | BUY or SELL |
| `entry_price` | Double | Entry price |
| `sl_price` | Double | Stop-loss price |
| `tp_price` | Double | Take-profit price |
| `quantity` | Double | BTC quantity traded |
| `risk_reward` | Double | R:R ratio |
| `mode` | String | AUTO or MANUAL |
| `status` | String | OPEN / CLOSED / FAILED / CANCELLED |
| `exit_price` | Double | Exit price (when closed) |
| `pnl` | Double | Profit/loss in USD |
| `pnl_pct` | Double | Profit/loss as percentage |
| `closed_at` | LocalDateTime | When trade was closed |
| `close_reason` | String | TP_HIT / SL_HIT / MANUAL_CLOSE / ERROR |
| `order_details` | Text | Raw Binance order JSON |
| `error_message` | String | Error details (if failed) |

#### `trading_settings` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | Long (PK, auto) | Unique identifier |
| `mode` | String | AUTO or MANUAL |
| `auto_trade_enabled` | Boolean | Whether auto-trading is active |
| `poll_interval_ms` | Long | Polling interval in milliseconds (default 60000) |
| `updated_at` | LocalDateTime | Last settings update |

---

### Scheduler & Trading Logic

#### Prediction Scheduler (`PredictionScheduler.java`)
- Runs on a **fixed delay** of 60 seconds (configurable via `trading.poll-interval-ms`)
- Each tick calls `TradingService.fetchNewPrediction()`

#### Trading Service (`TradingService.java`)
The core orchestrator that manages the trading lifecycle:

1. **Fetch prediction:** Calls AI engine `GET /predict` via WebClient
2. **Mode check:**
   - **AUTO mode:** If signal is LONG → calls AI engine `POST /trade` → saves trade → broadcasts update
   - **MANUAL mode:** Saves prediction as PENDING → user reviews in dashboard → Accept/Reject
3. **Accept prediction:** `POST /api/predictions/{id}/accept` → executes trade on Binance
4. **Reject prediction:** `POST /api/predictions/{id}/reject` → marks as REJECTED
5. **Trade stats:** Aggregates wins/losses/total PnL from database

#### AI Engine Client (`AiEngineClient.java`)
WebClient wrapper that communicates with the Python AI engine. Provides typed methods for each AI engine endpoint with 120-second timeout.

---

### WebSocket (Real-Time Updates)

The backend uses **STOMP over SockJS** for real-time push updates to the frontend.

| Topic | Triggered When | Payload |
|-------|---------------|---------|
| `/topic/predictions` | New prediction fetched or status changed | Prediction object |
| `/topic/trades` | Trade executed, closed, or updated | List of recent trades |
| `/topic/settings` | Trading mode changed | Settings object |

**Configuration:** Endpoint at `/ws`, simple broker on `/topic`, application prefix `/app`.

---

## Frontend (React)

A single-page application for monitoring and controlling the bot. Located in `frontend/`.

**Tech stack:** React 19, React Router DOM 7, Axios, STOMP.js, SockJS, Vite 5

**Styling:** Custom dark theme (603 lines of CSS) with `#0a0e17` background. Fixed 240px sidebar. Responsive — sidebar collapses below 768px.

### Pages

#### Dashboard (`/`)
- Real-time stats grid: total trades, win rate, wins, losses, total PnL
- Latest prediction card with full details
- Mode toggle (AUTO / MANUAL)
- Action buttons: Fetch Prediction, Accept, Reject
- Auto-refreshes every 30 seconds + live WebSocket updates

#### Predictions (`/predictions`)
- Lists last 50 predictions as cards
- Fetch New Prediction button
- Each card shows: signal, confidence, price, indicators, conclusion
- Real-time updates via WebSocket subscription to `/topic/predictions`

#### Trades (`/trades`)
- Stats cards at top (wins, losses, PnL)
- Trade history table with columns: Date, Side, Entry, SL, TP, Qty, R:R, Mode, Status, PnL
- Close Position button for open trades
- Auto-refresh every 30 seconds + WebSocket updates on `/topic/trades`

#### Settings (`/settings`)
- Trading mode toggle (AUTO ↔ MANUAL)
- AI Model info: status, accuracy, AUC-ROC, F1, feature count, training samples, retrain button
- System health: backend and AI engine connectivity status
- About section

### Components

#### PredictionCard
Renders a single prediction with:
- **Signal badge:** Color-coded (green = LONG, red = SHORT, gray = NO_TRADE)
- **Status badge:** PENDING, ACCEPTED, REJECTED, AUTO_EXECUTED, SKIPPED
- **Price info:** Current price, confidence %, P(UP), P(DOWN)
- **Trade levels:** SL, TP, R:R (only for tradeable signals)
- **Indicator chips:** Trend, Regime, RSI zone, ADX, MACD, Squeeze, Volume, HTF alignment (overall, 4H, 1D)
- **Conclusion:** Multi-paragraph AI-generated trade analysis
- **Action buttons:** Accept / Reject (only in MANUAL mode for PENDING predictions)

### WebSocket Client
- STOMP over SockJS connecting to `http://localhost:8081/ws`
- Auto-reconnect with 5-second delay
- 10-second heartbeat interval
- Manages subscriptions across reconnections

---

## Configuration Reference

### Environment Variables (`.env` at project root)

| Variable | Required | Description |
|----------|----------|-------------|
| `BINANCE_API_KEY` | Yes | Binance API key for trading |
| `BINANCE_API_SECRET` | Yes | Binance API secret |
| `BINANCE_TESTNET` | No | Set `true` for testnet (default: `false`) |
| `API_HOST` | No | AI engine host (default: `0.0.0.0`) |
| `API_PORT` | No | AI engine port (default: `8000`) |
| `AI_ENGINE_URL` | No | Backend → AI engine URL (default: `http://localhost:8000`) |
| `RETRAIN_ENABLED` | No | Enable auto-retraining (default: `true`) |
| `RETRAIN_INTERVAL_HOURS` | No | Retrain frequency (default: `6`) |
| `RETRAIN_FORCE_AFTER_HOURS` | No | Force retrain threshold (default: `24`) |
| `RETRAIN_MIN_IMPROVEMENT` | No | Max metric degradation tolerance (default: `0.03`) |

### AI Engine Config (`ai_engine/config.py`)

All feature engineering, model training, prediction, and risk management parameters are centralized in this file. See the [Feature Engineering](#feature-engineering-122-features), [Model Training](#model-training-xgboost-walk-forward), and [Risk Management](#risk-management) sections for details.

### Backend Config (`backend/src/main/resources/application.properties`)

| Property | Default | Description |
|----------|---------|-------------|
| `server.port` | 8081 | Backend server port |
| `spring.datasource.url` | `jdbc:h2:file:./data/sorbot_db` | Database path |
| `ai.engine.base-url` | `http://localhost:8000` | AI engine URL |
| `ai.engine.timeout` | 120s | AI engine request timeout |
| `trading.mode` | MANUAL | Initial trading mode |
| `trading.poll-interval-ms` | 60000 | Prediction polling interval (ms) |

---

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Binance API key and secret (or testnet credentials)

### 1. Clone the repository

```bash
git clone https://github.com/Anishalfaoui/sorbot.git
cd sorbot
```

### 2. Configure environment

Create a `.env` file at the project root:

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true
```

### 3. Start all services

**Windows (PowerShell):**
```powershell
.\start.ps1
```

**Windows (CMD):**
```cmd
start.bat
```

**Manual Docker:**
```bash
docker compose build
docker compose up -d
```

### 4. Access the dashboard

Open [http://localhost:3000](http://localhost:3000) in your browser.

### 5. Stop all services

```cmd
stop.bat
```
or
```bash
docker compose down
```

---

## Project Structure

```
sorbot/
├── .env                          # Environment variables (API keys — not tracked)
├── .gitignore                    # Root gitignore
├── docker-compose.yml            # All 3 services orchestration
├── start.bat / start.ps1         # Windows start scripts
├── stop.bat                      # Stop all containers
├── Architecture.md               # Architecture notes
│
├── ai_engine/                    # Python AI Engine (FastAPI + XGBoost)
│   ├── main.py                   # FastAPI server with all endpoints
│   ├── config.py                 # Centralized configuration
│   ├── backtest.py               # Walk-forward backtester
│   ├── run_train.py              # Standalone training script
│   ├── requirements.txt          # Python dependencies
│   ├── Dockerfile                # Python 3.11 container
│   ├── data/                     # Cached OHLCV CSVs (not tracked)
│   ├── models/                   # Trained model + metadata (not tracked)
│   │   ├── btc_model.json        # XGBoost native Booster model
│   │   ├── btc_meta.json         # Training metadata & metrics
│   │   └── retrain_history.json  # Retraining log
│   └── ml_core/                  # Machine learning modules
│       ├── data_loader.py        # yfinance data fetcher (chunked)
│       ├── feature_eng.py        # 122+ technical features
│       ├── trainer.py            # Walk-forward XGBoost training
│       ├── predictor.py          # Enriched predictions
│       ├── risk_manager.py       # Position sizing & risk rules
│       ├── exchange.py           # Binance spot API wrapper
│       └── retrainer.py          # Continuous retraining scheduler
│
├── backend/                      # Java Spring Boot Backend
│   ├── pom.xml                   # Maven dependencies
│   ├── Dockerfile                # Multi-stage Maven build
│   └── src/main/
│       ├── resources/
│       │   └── application.properties
│       └── java/com/sorbot/backend/
│           ├── SorbotBackendApplication.java
│           ├── config/
│           │   ├── CorsConfig.java
│           │   ├── WebClientConfig.java
│           │   └── WebSocketConfig.java
│           ├── controller/
│           │   ├── DashboardController.java
│           │   ├── PredictionController.java
│           │   ├── SettingsController.java
│           │   └── TradeController.java
│           ├── model/
│           │   ├── Prediction.java
│           │   ├── Trade.java
│           │   └── TradingSettings.java
│           ├── repository/
│           │   ├── PredictionRepository.java
│           │   ├── TradeRepository.java
│           │   └── TradingSettingsRepository.java
│           ├── scheduler/
│           │   └── PredictionScheduler.java
│           └── service/
│               ├── AiEngineClient.java
│               └── TradingService.java
│
└── frontend/                     # React SPA Dashboard
    ├── package.json              # NPM dependencies
    ├── vite.config.js            # Vite dev server config
    ├── index.html                # Entry HTML
    ├── nginx.conf                # Production Nginx config
    ├── Dockerfile                # Multi-stage Node build + Nginx
    └── src/
        ├── main.jsx              # React entry point
        ├── App.jsx               # Router + sidebar layout
        ├── api.js                # Axios REST client
        ├── websocket.js          # STOMP/SockJS client
        ├── index.css             # Dark theme CSS (603 lines)
        ├── components/
        │   └── PredictionCard.jsx
        └── pages/
            ├── Dashboard.jsx
            ├── Predictions.jsx
            ├── Trades.jsx
            └── Settings.jsx
```

---

## Key Data Flow

```
1. SCHEDULED POLL (every 60s)
   PredictionScheduler → TradingService → AiEngineClient → AI Engine GET /predict
   → XGBoost model predicts P(UP) on 122+ features from latest market data
   → Returns enriched signal with market analysis

2. AUTO MODE
   If signal = LONG + confidence ≥ 65%:
   TradingService → AiEngineClient → AI Engine POST /trade
   → RiskManager validates → BinanceExchange places market BUY + OCO SL/TP
   → Trade saved to H2 → WebSocket broadcast to frontend

3. MANUAL MODE
   Prediction saved as PENDING → displayed in dashboard
   → User clicks Accept → same execution flow as AUTO
   → User clicks Reject → marked REJECTED

4. CONTINUOUS RETRAINING (every 6h)
   RetrainingScheduler background thread:
   → Fetch fresh data (yfinance, force refresh)
   → Build 122+ features → Train new XGBoost model
   → Validate: new AUC-ROC & F1 within 3% of current
   → If passes: hot-swap model (zero downtime)
   → If fails: restore backup

5. REAL-TIME UPDATES
   All state changes → WebSocket STOMP broadcast
   → React frontend subscribes to /topic/predictions, /topic/trades, /topic/settings
```

---

## License

This project is for educational and personal use. Use at your own risk. Cryptocurrency trading involves significant financial risk.
