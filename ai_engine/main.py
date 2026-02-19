"""
Sorbot AI Engine — FastAPI Application  v2.0
===============================================
Multi-timeframe predictions with SL/TP, confluence analysis,
support/resistance, and best-entry timing for BTC, XAU, EUR.

Endpoints
---------
GET  /                      → health check
GET  /predict?symbol=X      → full prediction + trade plan
GET  /predict/all           → predictions for every active pair
GET  /analyze?symbol=X      → deep multi-TF analysis (no trade plan)
GET  /timeframes            → available timeframes
POST /train                 → retrain one or all models
GET  /models                → model metadata for all pairs
GET  /models/{symbol}       → model metadata for one pair
GET  /pairs                 → configured trading pairs
POST /data/refresh          → force re-download OHLCV data
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config import (
    TRADING_PAIRS, ACTIVE_SYMBOLS, API_HOST, API_PORT,
    TIMEFRAMES, PRIMARY_TIMEFRAME, CONFLUENCE_TIMEFRAMES, ENTRY_TIMEFRAME,
)
from ml_core.predictor import predict, predict_all, reload_models, get_model_info
from ml_core.trainer import train_symbol, train_all
from ml_core.data_loader import fetch_all, fetch_multi_timeframe

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
)
logger = logging.getLogger("sorbot.api")


# ── Lifespan ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Sorbot AI Engine v2.0 starting ...")
    try:
        reload_models()
        logger.info("Models loaded for: %s", list(TRADING_PAIRS.keys()))
    except Exception as exc:
        logger.warning("Some models not loaded: %s  (run /train first)", exc)
    yield
    logger.info("Sorbot AI Engine shutting down.")


# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Sorbot AI Engine",
    description="Multi-timeframe XGBoost prediction microservice — BTC, XAU, EUR",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────
#  ROUTES
# ──────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "sorbot-ai-engine",
        "version": "2.0.0",
        "status": "ok",
        "pairs": ACTIVE_SYMBOLS,
        "primary_timeframe": PRIMARY_TIMEFRAME,
        "features": [
            "multi-timeframe confluence",
            "stop-loss / take-profit proposals",
            "support & resistance detection",
            "best entry timing",
            "ADX / Ichimoku / EMA-200",
        ],
    }


@app.get("/pairs", tags=["Config"])
def list_pairs():
    """All configured trading pairs with SL/TP multipliers."""
    return TRADING_PAIRS


@app.get("/timeframes", tags=["Config"])
def list_timeframes():
    """Available timeframes and their roles."""
    return {
        "timeframes": TIMEFRAMES,
        "primary": PRIMARY_TIMEFRAME,
        "confluence": CONFLUENCE_TIMEFRAMES,
        "entry": ENTRY_TIMEFRAME,
    }


# ── Predict ───────────────────────────────────────────────

@app.get("/predict", tags=["Prediction"])
def get_prediction(symbol: str = Query(..., description="e.g. BTCUSD, XAUUSD, EURUSD")):
    """
    Full prediction with trade plan, SL/TP, confluence, timing.

    Response includes:
    - direction, confidence, probabilities
    - trade_plan (SL, TP, risk:reward, risk%, reward%)
    - confluence (multi-TF trend scores)
    - entry_timing (best entry assessment)
    - support_resistance (nearest levels)
    - indicators (RSI, MACD, ADX, Stochastic, BB, EMA-200 …)
    """
    sym = symbol.upper().replace("/", "").replace("-", "")
    if sym not in TRADING_PAIRS:
        raise HTTPException(400, f"Unknown symbol '{sym}'. Available: {ACTIVE_SYMBOLS}")
    try:
        return predict(sym)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Prediction error for %s: %s", sym, exc, exc_info=True)
        raise HTTPException(500, str(exc))


@app.get("/predict/all", tags=["Prediction"])
def get_all_predictions():
    """Full predictions for every active pair."""
    return predict_all()


# ── Analyse (lightweight, no model needed) ────────────────

@app.get("/analyze", tags=["Analysis"])
def analyze_symbol(symbol: str = Query(..., description="e.g. BTCUSD")):
    """
    Multi-timeframe technical analysis without the ML model.
    Useful for a market overview even before training.
    """
    sym = symbol.upper().replace("/", "").replace("-", "")
    if sym not in TRADING_PAIRS:
        raise HTTPException(400, f"Unknown symbol '{sym}'. Available: {ACTIVE_SYMBOLS}")

    from ml_core.data_loader import fetch_ohlcv
    from ml_core.feature_eng import detect_sr_levels, _rsi, _macd, _stochastic, _atr_raw, _adx
    from ml_core.predictor import _tf_trend, _confluence_analysis

    try:
        mtf_data = fetch_multi_timeframe(sym)
    except Exception as exc:
        raise HTTPException(500, f"Data fetch failed: {exc}")

    confluence = _confluence_analysis(mtf_data)

    # Primary candle details
    primary = mtf_data.get(PRIMARY_TIMEFRAME)
    if primary is None or primary.empty:
        raise HTTPException(500, "No primary-TF data")

    close = primary["Close"]
    current_price = round(float(close.iloc[-1]), TRADING_PAIRS[sym]["decimals"])
    sr_levels = detect_sr_levels(primary["High"], primary["Low"], close)

    atr = float(_atr_raw(primary["High"], primary["Low"], close).iloc[-1])
    rsi = float(_rsi(close).iloc[-1])
    adx = float(_adx(primary["High"], primary["Low"], close).iloc[-1])

    return {
        "symbol": sym,
        "current_price": current_price,
        "confluence": confluence,
        "support_resistance": sr_levels[:6],
        "primary_indicators": {
            "rsi": round(rsi, 2),
            "adx": round(adx, 2),
            "atr": round(atr, TRADING_PAIRS[sym]["decimals"]),
        },
        "timeframes_available": list(mtf_data.keys()),
    }


# ── Train ─────────────────────────────────────────────────

@app.post("/train", tags=["Training"])
def trigger_training(
    symbol: Optional[str] = Query(None, description="Train a specific pair, or omit for all"),
    refresh: bool = Query(False, description="Force-download fresh OHLCV data"),
):
    """Train (or retrain) XGBoost models with multi-TF features."""
    if symbol:
        sym = symbol.upper().replace("/", "").replace("-", "")
        if sym not in TRADING_PAIRS:
            raise HTTPException(400, f"Unknown: {sym}")
        result = {sym: train_symbol(sym, force_refresh=refresh)}
    else:
        result = train_all(force_refresh=refresh)

    reload_models()
    return {"status": "trained", "results": result}


# ── Model metadata ────────────────────────────────────────

@app.get("/models", tags=["Models"])
def list_models():
    return {sym: get_model_info(sym) for sym in ACTIVE_SYMBOLS}


@app.get("/models/{symbol}", tags=["Models"])
def model_detail(symbol: str):
    sym = symbol.upper().replace("/", "").replace("-", "")
    info = get_model_info(sym)
    if info is None:
        raise HTTPException(404, f"No model trained for {sym}")
    return info


# ── Data ──────────────────────────────────────────────────

@app.post("/data/refresh", tags=["Data"])
def refresh_data():
    """Force-download latest OHLCV data for all pairs (primary TF)."""
    results = {}
    data = fetch_all(force_refresh=True)
    for sym, df in data.items():
        results[sym] = {"rows": len(df), "latest": str(df.index[-1])}
    return results


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
