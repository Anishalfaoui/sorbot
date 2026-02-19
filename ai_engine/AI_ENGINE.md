# Sorbot AI Engine v2.0 — Complete Technical Documentation

> **Microservice 1 / 3** — The Brain  
> Python 3.11 · FastAPI · XGBoost · yfinance  
> Trading pairs: **BTC/USD** · **XAU/USD** · **EUR/USD**  
> Multi-timeframe · SL/TP · Support/Resistance · Smart Entry Timing

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [Trading Pairs Configuration](#3-trading-pairs-configuration)
4. [Installation & Setup](#4-installation--setup)
5. [Multi-Timeframe Data Pipeline](#5-multi-timeframe-data-pipeline)
6. [Feature Engineering (40+ Features)](#6-feature-engineering-40-features)
7. [Model Training](#7-model-training)
8. [Live Prediction & Trade Plan](#8-live-prediction--trade-plan)
9. [API Reference](#9-api-reference)
10. [Docker Deployment](#10-docker-deployment)
11. [Architecture Diagram](#11-architecture-diagram)
12. [How Everything Works (Step by Step)](#12-how-everything-works-step-by-step)

---

## 1. Overview

The AI Engine v2.0 is a self-contained Python microservice that:

1. **Fetches** multi-timeframe OHLCV data (1h, 4h, 1d, 1w) from Yahoo Finance.
2. **Engineers 40+ features** including trend, momentum, volatility, volume, candle anatomy, Ichimoku, ADX, EMA-200, support/resistance, and higher-timeframe confluence signals.
3. **Trains XGBoost classifiers** per trading pair with weekly confluence enrichment.
4. **Predicts** BUY / SELL / HOLD with confidence + full trade plan:
   - **Stop-Loss & Take-Profit** proposals (ATR + S/R based)
   - **Multi-timeframe confluence** analysis (1h to 1w)
   - **Best entry timing** assessment
   - **Support & resistance levels** detection
   - Risk:reward ratio validation

All exposed via a FastAPI REST API consumed by the Java backend.

---

## 2. Project Structure

```
ai_engine/
├── config.py                 # Central config: pairs, timeframes, XGB params, SL/TP
├── main.py                   # FastAPI app (endpoints, CORS, lifespan)
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container build
├── AI_ENGINE.md              # This documentation
│
├── ml_core/                  # ML pipeline
│   ├── data_loader.py        # Multi-TF OHLCV fetcher + caching
│   ├── feature_eng.py        # 40+ feature builder + S/R detection + HTF confluence
│   ├── trainer.py            # XGBoost training with multi-TF enrichment
│   └── predictor.py          # Live inference + SL/TP + confluence + timing
│
├── data/                     # Cached OHLCV CSV files (auto-created)
│   ├── BTCUSD_1d_ohlcv.csv
│   ├── BTCUSD_1w_ohlcv.csv
│   ├── BTCUSD_4h_ohlcv.csv
│   └── ...
│
└── models/                   # Trained model files (auto-created)
    ├── BTCUSD_xgb.json
    ├── BTCUSD_meta.json
    └── ...
```

---

## 3. Trading Pairs Configuration

| Symbol   | Yahoo Ticker | Category  | SL ATR Mult | TP ATR Mult |
|----------|-------------|-----------|-------------|-------------|
| BTCUSD   | BTC-USD     | Crypto    | 1.5         | 2.5         |
| XAUUSD   | GC=F        | Commodity | 1.5         | 2.0         |
| EURUSD   | EURUSD=X    | Forex     | 1.2         | 1.8         |

Each pair has custom SL/TP multipliers tuned for its volatility profile.

---

## 4. Installation & Setup

```bash
cd ai_engine
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

**Dependencies:** FastAPI, Uvicorn, XGBoost, scikit-learn, yfinance (1.2.0+), pandas, numpy, curl_cffi

### Quick Start

```bash
# 1. Train models (downloads data + trains)
python ml_core/trainer.py --refresh

# 2. Start server
python main.py
# → http://localhost:8000/docs
```

---

## 5. Multi-Timeframe Data Pipeline

### Timeframes

| Key  | Interval  | Period   | Label    | Role           |
|------|-----------|----------|----------|----------------|
| 1h   | 1 hour    | 60 days  | 1 Hour   | Entry timing   |
| 4h   | 4 hours   | 60 days  | 4 Hours  | Entry timing   |
| 1d   | 1 day     | 2 years  | 1 Day    | **Primary**    |
| 1w   | 1 week    | 5 years  | 1 Week   | HTF confluence |

- **Primary (1d):** Used for training and main signal generation
- **Entry (1h):** Used for precise entry timing assessment
- **Confluence (1w):** Higher-TF trend confirmation during training & prediction

### Data Fetching

- Uses `yfinance 1.2.0` with `curl_cffi` for reliable downloads
- Try/except fallback: `yf.download()` → `yf.Ticker().history()`
- 4h candles resampled from 1h data (Yahoo doesn't support 4h natively)
- Smart caching: intraday = 1h TTL, daily+ = 12h TTL
- Handles zero-volume forex pairs gracefully

### Key Functions

| Function                  | Description                           |
|---------------------------|---------------------------------------|
| `fetch_ohlcv(sym, tf)`    | Single symbol + timeframe download    |
| `fetch_multi_timeframe()` | All configured TFs for a symbol       |
| `fetch_all()`             | All pairs at primary TF               |

---

## 6. Feature Engineering (40+ Features)

### Base Features (per timeframe)

| Category        | Features                                               | Count |
|-----------------|--------------------------------------------------------|-------|
| **Trend**       | RSI(14), MACD (line/signal/hist), EMA cross (9/21), EMA-200 distance | 6 |
| **Volatility**  | Bollinger %B & bandwidth, ATR (normalised)             | 3     |
| **Momentum**    | ROC-5, ROC-10, ROC-20, Stochastic %K/%D, ADX           | 6     |
| **Volume**      | Volume ratio (vs 20-SMA), OBV normalised               | 2     |
| **Candle**      | Body ratio, upper shadow, lower shadow, bullish flag   | 4     |
| **Calendar**    | Day of week, month                                     | 2     |
| **Returns**     | Lagged returns: 1, 2, 3, 5, 10 periods                | 5     |
| **Ichimoku**    | Tenkan/Kijun cross, cloud distance, cloud thickness    | 3     |
| **Structure**   | Distance to nearest support/resistance, S/R level count | 3     |

### Higher-Timeframe Confluence Features

For each HTF (e.g. weekly when primary=daily):

| Feature              | Description                              |
|----------------------|------------------------------------------|
| `htf_1w_rsi`         | Weekly RSI                               |
| `htf_1w_ema_cross`   | Weekly EMA 9/21 normalised cross         |
| `htf_1w_macd_hist`   | Weekly MACD histogram                    |
| `htf_1w_adx`         | Weekly ADX (trend strength)              |
| `htf_1w_atr`         | Weekly ATR (normalised)                  |

HTF values are forward-filled to align with the primary index, so every daily row gets the latest weekly context.

### Support & Resistance Detection

- Scans last 50 bars for pivot highs/lows (5-bar window)
- Clusters nearby prices and counts "touches"
- Requires ≥3 touches to confirm a level
- Returns up to 10 nearest levels with type (support/resistance) and strength

### Target Label

```
Forward 5-day return:
  > +1%  → BUY  (2)
  < -1%  → SELL (0)
  else   → HOLD (1)
```

---

## 7. Model Training

### Pipeline

1. Fetch primary-TF OHLCV (2y daily)
2. Fetch higher-TF OHLCV (5y weekly) — only TFs higher than primary
3. Build 40 features with HTF confluence
4. Train/test split: `TimeSeriesSplit(n_splits=5)`, last fold = test
5. Fit `XGBClassifier` with tuned hyperparameters
6. Save native `xgb.json` + `_meta.json`

### XGBoost Hyperparameters

```python
{
    "n_estimators": 500,
    "max_depth": 7,
    "learning_rate": 0.03,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "min_child_weight": 5,
    "gamma": 0.2,
    "reg_alpha": 0.3,
    "reg_lambda": 1.5,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
}
```

### Latest Training Results

| Symbol | Rows | Features | Accuracy | HTF  |
|--------|------|----------|----------|------|
| BTCUSD | 681  | 40       | 35.4%    | 1w   |
| XAUUSD | 432  | 40       | 23.6%    | 1w   |
| EURUSD | 467  | 40       | 81.8%    | 1w   |

Top features across models: `ema200_dist`, `obv_norm`, `htf_1w_adx`, `rsi`, `sr_dist_support`, `htf_1w_rsi`

---

## 8. Live Prediction & Trade Plan

### Prediction Flow

1. Load cached model (native `xgb.Booster` to avoid sklearn compat issues)
2. Fetch primary + all confluence TF data
3. Build features matching training column order
4. Predict class probabilities → direction + confidence
5. Compute SL/TP from ATR + nearest S/R levels
6. Score multi-TF confluence (-100% to +100%)
7. Assess entry timing on 1h timeframe
8. Validate risk:reward ratio (minimum 1.5)

### Response Structure

```json
{
  "symbol": "BTCUSD",
  "direction": "BUY",
  "confidence": 0.7613,
  "current_price": 66281.24,
  "timestamp": "2026-02-18T23:33:10Z",

  "trade_plan": {
    "stop_loss": 60029.98,
    "take_profit": 76700.01,
    "risk_reward_ratio": 1.67,
    "risk_pct": 9.43,
    "reward_pct": 15.72,
    "viable": true,
    "note": ""
  },

  "confluence": {
    "overall_trend": "bearish",
    "confluence_score": -7,
    "confluence_pct": -77.8,
    "timeframes": {
      "4h": { "trend": "bearish", "score": -3, "rsi": 40.97 },
      "1d": { "trend": "bearish", "score": -1, "rsi": 39.34 },
      "1w": { "trend": "bearish", "score": -3, "rsi": 21.39 }
    }
  },

  "entry_timing": {
    "timing": "excellent",
    "score": 3,
    "note": "RSI oversold - ideal buy zone; Stochastic oversold"
  },

  "support_resistance": [
    { "price": 87231.57, "type": "resistance", "touches": 4, "strength": 0.08 },
    { "price": 90207.11, "type": "resistance", "touches": 9, "strength": 0.18 }
  ],

  "indicators": {
    "rsi": 39.34,
    "macd": "bullish",
    "macd_histogram": 319.64,
    "adx": 84.21,
    "adx_trend": "strong",
    "atr_abs": 4167.51,
    "bb_pctb": 0.2692,
    "ema200_dist": -0.4082,
    "stoch_k": 47.43,
    "stoch_d": 45.89
  },

  "probabilities": { "SELL": 0.04, "HOLD": 0.20, "BUY": 0.76 },
  "model_info": { "version": "2.0", "accuracy": 0.354, "n_features": 40 }
}
```

### Stop-Loss / Take-Profit Calculation

1. **ATR-based:** SL = price ± (ATR × SL_mult), TP = price ± (ATR × TP_mult)
2. **S/R-based:** SL placed beyond nearest support/resistance
3. **Final:** Uses the tighter of ATR and S/R to optimise R:R
4. **Gate:** Trade rejected if R:R < 1.5 (configurable)

### Multi-TF Confluence Scoring

Each timeframe scores -3 to +3 based on:
- RSI above/below 55/45 → ±1
- MACD histogram positive/negative → ±1
- EMA 9 vs 21 cross direction → ±1

Total score normalised to -100% (all bearish) to +100% (all bullish).

### Entry Timing Assessment

Analyses the 1h timeframe for optimal entry:
- RSI oversold/overbought alignment with direction
- Stochastic extremes
- MACD zero-line crossovers
- Returns: excellent / good / neutral / poor

---

## 9. API Reference

### Health & Config

| Method | Endpoint       | Description                                 |
|--------|---------------|---------------------------------------------|
| GET    | `/`           | Health check, version, features list        |
| GET    | `/pairs`      | All trading pairs with SL/TP config         |
| GET    | `/timeframes` | Available TFs and their roles               |

### Prediction

| Method | Endpoint                 | Description                      |
|--------|--------------------------|----------------------------------|
| GET    | `/predict?symbol=BTCUSD` | Full prediction with trade plan  |
| GET    | `/predict/all`           | Predictions for all active pairs |

### Analysis (no model required)

| Method | Endpoint            | Description                            |
|--------|---------------------|----------------------------------------|
| GET    | `/analyze?symbol=X` | Multi-TF technical analysis overview   |

### Training

| Method | Endpoint                  | Description                      |
|--------|--------------------------|----------------------------------|
| POST   | `/train`                 | Train all models                 |
| POST   | `/train?symbol=BTCUSD`   | Train single model               |
| POST   | `/train?refresh=true`    | Force fresh data download        |

### Models & Data

| Method | Endpoint            | Description                          |
|--------|---------------------|--------------------------------------|
| GET    | `/models`           | Training metadata for all symbols    |
| GET    | `/models/{symbol}`  | Training metadata for one symbol     |
| POST   | `/data/refresh`     | Force re-download all OHLCV data     |

Interactive docs: `http://localhost:8000/docs`

---

## 10. Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python ml_core/trainer.py
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t sorbot-ai .
docker run -p 8000:8000 sorbot-ai
```

---

## 11. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     SORBOT AI ENGINE v2.0                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FastAPI  (main.py)                                      │   │
│  │  /predict  /analyze  /train  /pairs  /timeframes         │   │
│  └──────────┬───────────┬───────────────────────────────────┘   │
│             │           │                                       │
│  ┌──────────▼───┐  ┌────▼────────────────────────────────┐     │
│  │  Predictor   │  │  Trainer                            │     │
│  │  ┌─────────┐ │  │  ┌──────────┐  ┌────────────────┐  │     │
│  │  │Booster  │ │  │  │XGBClass  │  │TimeSeries Split│  │     │
│  │  │(native) │ │  │  │(sklearn) │  │(5-fold)        │  │     │
│  │  └────┬────┘ │  │  └────┬─────┘  └────────────────┘  │     │
│  │       │      │  │       │                             │     │
│  │  ┌────▼────────────────▼──────────────────────────┐   │     │
│  │  │        Feature Engineering  (40+ features)      │   │     │
│  │  │  RSI│MACD│BB│ATR│ADX│Ichimoku│EMA200│S/R│HTF   │   │     │
│  │  └────────────────────┬───────────────────────────┘   │     │
│  │                       │                               │     │
│  │  ┌────────────────────▼───────────────────────────┐   │     │
│  │  │     Multi-TF Data Loader                       │   │     │
│  │  │  1h ─── 4h ─── 1d (primary) ─── 1w (HTF)     │   │     │
│  │  └────────────────────┬───────────────────────────┘   │     │
│  └───────────────────────┼───────────────────────────────┘     │
│                          │                                      │
│                  ┌───────▼───────┐                              │
│                  │  Yahoo Finance │                              │
│                  │  (yfinance)    │                              │
│                  └───────────────┘                              │
│                                                                 │
│  ┌──────────────────┐  ┌───────────────────────────────────┐   │
│  │  SL/TP Engine    │  │  Confluence Scorer                │   │
│  │  ATR + S/R based │  │  1h + 4h + 1d + 1w → -100..+100  │   │
│  │  R:R validation  │  │  Entry timing (1h)                │   │
│  └──────────────────┘  └───────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. How Everything Works (Step by Step)

### Training Pipeline

```
1. fetch_ohlcv("BTCUSD", "1d")     → 732 daily candles (2 years)
2. fetch_ohlcv("BTCUSD", "1w")     → 262 weekly candles (5 years) — HTF
3. build_base_features(daily_df)   → 35 features (RSI, MACD, BB, ATR, ADX,
                                      Ichimoku, EMA-200, S/R, candle, calendar)
4. build_htf_features(daily, {1w}) → +5 weekly confluence features (forward-filled)
5. Add target: 5-day forward return → BUY/SELL/HOLD
6. TimeSeriesSplit(5) → last fold = test set
7. XGBClassifier.fit(X_train, y_train)
8. Save model.json + meta.json (accuracy, feature importance, etc.)
```

### Live Prediction Pipeline

```
1. Load cached xgb.Booster (native API)
2. Fetch primary (1d) + HTF (1w) + entry (1h) OHLCV
3. Build 40 features, match training column order
4. booster.predict(DMatrix(X_live))  →  [P(SELL), P(HOLD), P(BUY)]
5. Direction = argmax(probabilities), apply confidence threshold (0.45)
6. ATR_abs = raw ATR value for SL/TP computation
7. S/R levels = detect_sr_levels(last 50 bars)
8. SL/TP = tighter of (ATR × mult) and (nearest S/R level)
9. Confluence = score each TF (1h, 4h, 1d, 1w) on RSI/MACD/EMA → -100..+100%
10. Entry timing = check 1h RSI/Stoch/MACD alignment → excellent/good/neutral/poor
11. R:R gate = reject if risk:reward < 1.5
12. Return full JSON response with all analysis
```

### Key Design Decisions

- **Native xgb.Booster** for loading/inference instead of XGBClassifier wrapper (avoids sklearn `__sklearn_tags__` compatibility bug)
- **Zero-volume handling** for EUR/USD forex (volume features filled with neutral values)
- **HTF only uses higher TFs** than primary for training (4h skipped when primary is 1d)
- **HTF NaN fill with 0** during warm-up periods to avoid losing training rows
- **Per-pair SL/TP multipliers** tuned to each asset's volatility profile
- **Risk:reward gate** prevents low-quality trades from being suggested
