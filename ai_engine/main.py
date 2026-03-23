"""
Sorbot AI Engine v3.0 — FastAPI Server
======================================
Multi-symbol AI prediction engine for BTC/USD, EUR/USD, and XAU/USD.
Execution endpoints are virtual/paper-only (no live exchange keys required).
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import API_HOST, API_PORT, DEFAULT_SYMBOL, SYMBOLS
from ml_core.data_loader import fetch_all_timeframes, fetch_ohlcv
from ml_core.feature_eng import build_dataset
from ml_core.trainer import train_model
from ml_core.predictor import Predictor
from ml_core.retrainer import RetrainingScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("sorbot.api")


predictor = Predictor()
retrain_scheduler = RetrainingScheduler(predictor)


def _normalize_symbol(symbol: Optional[str]) -> str:
    key = (symbol or DEFAULT_SYMBOL).upper().replace("/", "")
    if key not in SYMBOLS:
        raise HTTPException(400, f"Unsupported symbol '{symbol}'. Allowed: {', '.join(SYMBOLS.keys())}")
    return key


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        predictor.load(DEFAULT_SYMBOL)
        logger.info("Model loaded on startup for %s", DEFAULT_SYMBOL)
    except FileNotFoundError:
        logger.warning("No trained model found for %s. Training initial model...", DEFAULT_SYMBOL)
        try:
            data = fetch_all_timeframes(symbol=DEFAULT_SYMBOL)
            htf_data = {"4h": data.get("4h"), "1d": data.get("1d")}
            dataset = build_dataset(data["1h"], include_target=True, htf_data=htf_data)
            train_model(dataset, symbol=DEFAULT_SYMBOL)
            predictor.load(DEFAULT_SYMBOL)
            logger.info("Initial model trained and loaded for %s", DEFAULT_SYMBOL)
        except Exception as e:
            logger.error("Initial training failed: %s", e)

    retrain_scheduler.start()
    yield
    retrain_scheduler.stop()


app = FastAPI(
    title="Sorbot AI Engine v3.0",
    description="Multi-symbol AI engine with virtual paper-trade execution",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health(symbol: Optional[str] = Query(default=None)):
    sym = _normalize_symbol(symbol) if symbol else DEFAULT_SYMBOL
    return {
        "status": "running",
        "engine": "Sorbot AI v3.0",
        "symbol": sym,
        "symbol_label": SYMBOLS[sym]["label"],
        "model_loaded": predictor._loaded,
        "retraining_enabled": retrain_scheduler._running,
        "total_retrains": retrain_scheduler._retrain_count,
    }


@app.get("/price")
async def get_price(symbol: str = Query(default=DEFAULT_SYMBOL)):
    sym = _normalize_symbol(symbol)
    try:
        df = fetch_ohlcv(symbol=sym, timeframe="1h", force_refresh=True)
        price = float(df["Close"].iloc[-1])
        return {"symbol": sym, "symbol_label": SYMBOLS[sym]["label"], "price": price}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/train")
async def train(symbol: str = Query(default=DEFAULT_SYMBOL)):
    sym = _normalize_symbol(symbol)
    try:
        logger.info("Starting training for %s", sym)
        data = fetch_all_timeframes(symbol=sym)
        htf_data = {"4h": data.get("4h"), "1d": data.get("1d")}
        dataset = build_dataset(data["1h"], include_target=True, htf_data=htf_data)
        meta = train_model(dataset, symbol=sym)
        predictor.load(sym)
        return {
            "status": "trained",
            "symbol": sym,
            "symbol_label": SYMBOLS[sym]["label"],
            "samples": meta["n_samples"],
            "cv_metrics": meta["cv_metrics"],
            "final_metrics": meta["final_metrics"],
            "top_features": meta["top_features"][:10],
        }
    except Exception as e:
        logger.error("Training error: %s", e)
        raise HTTPException(500, str(e))


@app.get("/predict")
async def predict(
    symbol: str = Query(default=DEFAULT_SYMBOL),
    virtual_balance: float = Query(default=10000.0),
):
    sym = _normalize_symbol(symbol)

    try:
        predictor.load(sym)
        data = fetch_all_timeframes(symbol=sym)
        htf_data = {"4h": data.get("4h"), "1d": data.get("1d")}
        dataset = build_dataset(data["1h"], include_target=False, htf_data=htf_data)
        result = predictor.predict_latest(
            dataset,
            data["1h"],
            symbol=sym,
            virtual_balance=virtual_balance,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(500, str(e))


class ExecuteTradeRequest(BaseModel):
    signal: str
    entry_price: float
    sl_price: float
    tp_price: float
    qty_btc: Optional[float] = None
    symbol: Optional[str] = None
    virtual_balance: float = 10000.0


@app.post("/execute")
async def execute_trade(req: ExecuteTradeRequest):
    sym = _normalize_symbol(req.symbol or DEFAULT_SYMBOL)
    if req.signal not in ("LONG", "SHORT"):
        return {"action": "NO_TRADE", "reason": "Signal is not tradeable"}

    sl_distance = abs(req.entry_price - req.sl_price)
    if sl_distance <= 0:
        return {"action": "ERROR", "error": "Invalid stop-loss distance"}

    if req.qty_btc and req.qty_btc > 0:
        qty = round(req.qty_btc, 5)
    else:
        risk_usd = max(req.virtual_balance, 0) * 0.015
        qty = round(risk_usd / sl_distance, 5)

    notional = qty * req.entry_price
    risk_usd = qty * sl_distance

    return {
        "action": "TRADE_EXECUTED",
        "symbol": sym,
        "symbol_label": SYMBOLS[sym]["label"],
        "signal": {
            "signal": req.signal,
            "current_price": req.entry_price,
            "sl_price": req.sl_price,
            "tp_price": req.tp_price,
        },
        "sizing": {
            "qty_btc": qty,
            "notional_usd": round(notional, 2),
            "risk_usd": round(risk_usd, 2),
            "capital_used_pct": round((notional / max(req.virtual_balance, 1e-9)) * 100, 2),
            "balance": round(req.virtual_balance, 2),
        },
        "orders": {
            "entry": {
                "status": "FILLED",
                "side": req.signal,
                "entry_price": req.entry_price,
                "virtual": True,
            }
        },
    }


@app.get("/status")
async def status(
    symbol: str = Query(default=DEFAULT_SYMBOL),
    virtual_balance: float = Query(default=10000.0),
):
    sym = _normalize_symbol(symbol)
    price = None
    try:
        price = float(fetch_ohlcv(symbol=sym, timeframe="1h")["Close"].iloc[-1])
    except Exception:
        pass

    return {
        "symbol": sym,
        "symbol_label": SYMBOLS[sym]["label"],
        "model_loaded": predictor._loaded,
        "virtual_account": True,
        "virtual_balance": round(virtual_balance, 2),
        "open_positions": 0,
        "current_price": price,
    }


@app.post("/close")
async def close_position(symbol: str = Query(default=DEFAULT_SYMBOL)):
    sym = _normalize_symbol(symbol)
    return {
        "action": "NO_POSITION",
        "symbol": sym,
        "symbol_label": SYMBOLS[sym]["label"],
        "message": "Position tracking is managed by backend virtual account logic.",
    }


@app.get("/model-info")
async def model_info(symbol: str = Query(default=DEFAULT_SYMBOL)):
    sym = _normalize_symbol(symbol)
    try:
        predictor.load(sym)
        info = predictor.get_model_info()
        info["symbol"] = sym
        info["symbol_label"] = SYMBOLS[sym]["label"]
        return info
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/retrain-status")
async def retrain_status():
    return retrain_scheduler.get_status()


@app.post("/retrain-now")
async def retrain_now():
    try:
        result = retrain_scheduler.force_retrain()
        return result
    except Exception as e:
        logger.error("Manual retrain error: %s", e)
        raise HTTPException(500, str(e))


@app.get("/retrain-history")
async def retrain_history():
    status = retrain_scheduler.get_status()
    return {
        "total_retrains": status["total_retrains"],
        "history": status["recent_history"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
