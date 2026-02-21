"""
Sorbot AI Engine v3.0 — FastAPI Server
=========================================
BTC/USD only · Binance Futures · High-conviction trades

Endpoints:
  GET  /              — health check
  POST /train         — retrain model
  GET  /predict       — get latest signal
  POST /trade         — execute trade on Binance
  GET  /status        — account & position status
  POST /close         — close open position
  GET  /model-info    — model metrics & top features
  GET  /retrain-status  — continuous retraining scheduler status
  POST /retrain-now     — manually trigger immediate retrain
  GET  /retrain-history — full retraining history log

Continuous Retraining:
  The model automatically retrains every RETRAIN_INTERVAL_HOURS (default 6h)
  on fresh market data. New models must pass a validation gate (metrics
  comparison) before replacing the current model.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from config import API_HOST, API_PORT, SYMBOL
from ml_core.data_loader import fetch_all_timeframes, fetch_ohlcv
from ml_core.feature_eng import build_dataset
from ml_core.trainer import train_model
from ml_core.predictor import Predictor
from ml_core.risk_manager import RiskManager, get_risk_manager
from ml_core.exchange import BinanceExchange, get_exchange
from ml_core.retrainer import RetrainingScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("sorbot.api")

# ── Globals ────────────────────────────────────
predictor = Predictor()
risk_mgr = get_risk_manager()
retrain_scheduler = RetrainingScheduler(predictor)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, start retraining scheduler."""
    try:
        predictor.load()
        logger.info("Model loaded on startup")
    except FileNotFoundError:
        logger.warning("No trained model found. Training initial model...")
        try:
            data = fetch_all_timeframes()
            htf_data = {"4h": data.get("4h"), "1d": data.get("1d")}
            dataset = build_dataset(data["1h"], include_target=True, htf_data=htf_data)
            train_model(dataset)
            predictor.load()
            logger.info("Initial model trained and loaded")
        except Exception as e:
            logger.error("Initial training failed: %s. Train manually via POST /train", e)

    # Start continuous retraining
    retrain_scheduler.start()

    yield

    # Shutdown
    retrain_scheduler.stop()


app = FastAPI(
    title="Sorbot AI Engine v3.0",
    description="BTC/USD AI Trading Engine — High-conviction signals with Binance Spot execution",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── HEALTH ─────────────────────────────────────

@app.get("/")
async def health():
    return {
        "status": "running",
        "engine": "Sorbot AI v3.0",
        "symbol": SYMBOL,
        "model_loaded": predictor._loaded,
        "retraining_enabled": retrain_scheduler._running,
        "total_retrains": retrain_scheduler._retrain_count,
    }


# ── TRAIN ──────────────────────────────────────

@app.post("/train")
async def train():
    """Retrain the model with latest data."""
    try:
        logger.info("Starting training...")
        data = fetch_all_timeframes()
        htf_data = {"4h": data.get("4h"), "1d": data.get("1d")}
        dataset = build_dataset(data["1h"], include_target=True, htf_data=htf_data)
        meta = train_model(dataset)

        # Reload predictor
        predictor.load()

        return {
            "status": "trained",
            "samples": meta["n_samples"],
            "cv_metrics": meta["cv_metrics"],
            "final_metrics": meta["final_metrics"],
            "top_features": meta["top_features"][:10],
        }
    except Exception as e:
        logger.error("Training error: %s", e)
        raise HTTPException(500, str(e))


# ── PREDICT ────────────────────────────────────

@app.get("/predict")
async def predict():
    """Get latest BTC/USD trading signal."""
    if not predictor._loaded:
        raise HTTPException(400, "Model not trained. POST /train first.")

    try:
        data = fetch_all_timeframes()
        htf_data = {"4h": data.get("4h"), "1d": data.get("1d")}
        dataset = build_dataset(data["1h"], include_target=False, htf_data=htf_data)
        result = predictor.predict_latest(dataset, data["1h"])
        return result
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(500, str(e))


# ── TRADE ──────────────────────────────────────

class ExecuteTradeRequest(BaseModel):
    """Parameters for executing a trade with specific prediction values."""
    signal: str            # LONG
    entry_price: float     # price from accepted prediction
    sl_price: float        # stop loss from accepted prediction
    tp_price: float        # take profit from accepted prediction
    qty_btc: Optional[float] = None   # pre-approved quantity from prediction estimate
    symbol: Optional[str] = None


@app.post("/execute")
async def execute_trade(req: ExecuteTradeRequest):
    """
    Execute a trade on Binance using the EXACT parameters from an accepted prediction.
    No new prediction is generated — uses the SL/TP/signal/qty the user accepted.
    """
    try:
        # Spot trading: only LONG (BUY) is allowed
        if req.signal == "SHORT":
            return {
                "action": "NO_TRADE",
                "reason": "SHORT signal — spot trading is BUY-only.",
            }

        # Fetch REAL Binance balance
        exchange = get_exchange()
        try:
            balance = exchange.get_available_balance()
            risk_mgr.update_balance(balance)
            logger.info("Binance available balance: $%.2f", balance)
        except Exception as e:
            logger.warning("Could not fetch Binance balance: %s", e)

        # Check risk
        can, reason = risk_mgr.can_trade()
        if not can:
            return {"action": "NO_TRADE", "reason": reason}

        # Use pre-approved quantity from prediction if provided,
        # otherwise recalculate (backward compatible)
        if req.qty_btc and req.qty_btc > 0:
            # User approved this exact quantity — use it directly
            qty_btc = round(req.qty_btc, 5)
            notional = qty_btc * req.entry_price

            # Safety cap: never exceed MAX_POSITION_PCT of CURRENT balance
            from config import MAX_POSITION_PCT
            max_notional = risk_mgr.balance * MAX_POSITION_PCT
            if notional > max_notional:
                qty_btc = round(max_notional / req.entry_price, 5)
                notional = qty_btc * req.entry_price
                logger.warning(
                    "Pre-approved qty exceeded current balance cap. "
                    "Reduced from %.5f to %.5f BTC ($%.2f)",
                    req.qty_btc, qty_btc, notional,
                )

            sl_distance = req.entry_price - req.sl_price
            potential_loss = qty_btc * sl_distance if sl_distance > 0 else 0
            capital_pct = (notional / risk_mgr.balance * 100) if risk_mgr.balance > 0 else 0

            sizing = {
                "qty_btc": qty_btc,
                "notional_usd": round(notional, 2),
                "risk_usd": round(potential_loss, 2),
                "capital_used_pct": round(capital_pct, 1),
                "balance": round(risk_mgr.balance, 2),
                "source": "pre_approved",
            }
            logger.info(
                "Using pre-approved qty: %.5f BTC ($%.2f, %.1f%% of balance)",
                qty_btc, notional, capital_pct,
            )
        else:
            # Fallback: calculate position size fresh
            sizing = risk_mgr.calculate_position_size(
                entry_price=req.entry_price,
                sl_price=req.sl_price,
                signal=req.signal,
            )

        if sizing.get("error"):
            return {"action": "ERROR", "error": sizing["error"]}

        # Place order with the EXACT SL/TP from the accepted prediction
        order_result = exchange.place_order(
            side=req.signal,
            qty=sizing["qty_btc"],
            sl_price=req.sl_price,
            tp_price=req.tp_price,
        )

        if "error" in order_result:
            return {"action": "ORDER_ERROR", "error": order_result["error"]}

        risk_mgr.register_open()

        return {
            "action": "TRADE_EXECUTED",
            "signal": {
                "signal": req.signal,
                "current_price": req.entry_price,
                "sl_price": req.sl_price,
                "tp_price": req.tp_price,
            },
            "sizing": sizing,
            "orders": order_result,
        }

    except Exception as e:
        logger.error("Execute trade error: %s", e)
        raise HTTPException(500, str(e))


@app.post("/trade")
async def trade():
    """
    Get prediction and if high-conviction, execute on Binance.
    Calculates position size, places market order + SL/TP.
    """
    if not predictor._loaded:
        raise HTTPException(400, "Model not trained. POST /train first.")

    try:
        # Get prediction
        data = fetch_all_timeframes()
        htf_data = {"4h": data.get("4h"), "1d": data.get("1d")}
        dataset = build_dataset(data["1h"], include_target=False, htf_data=htf_data)
        signal = predictor.predict_latest(dataset, data["1h"])

        if signal["signal"] == "NO_TRADE":
            return {
                "action": "NO_TRADE",
                "reason": signal.get("reject_reason", "Low confidence"),
                "probability": signal["probability_up"],
            }

        # Spot trading: only LONG (BUY) is allowed
        if signal["signal"] == "SHORT":
            return {
                "action": "NO_TRADE",
                "reason": "SHORT signal — spot trading is BUY-only. If holding BTC, consider closing.",
                "signal": signal,
            }

        # Fetch REAL Binance balance FIRST before any risk checks
        exchange = get_exchange()
        try:
            balance = exchange.get_available_balance()
            risk_mgr.update_balance(balance)
            logger.info("Binance available balance: $%.2f", balance)
        except Exception as e:
            logger.warning("Could not fetch Binance balance: %s — using last known balance", e)

        # Check risk (now using real balance)
        can, reason = risk_mgr.can_trade()
        if not can:
            return {"action": "NO_TRADE", "reason": reason, "signal": signal}

        # Calculate position size
        sizing = risk_mgr.calculate_position_size(
            entry_price=signal["current_price"],
            sl_price=signal["sl_price"],
            signal=signal["signal"],
        )

        if sizing.get("error"):
            return {"action": "ERROR", "error": sizing["error"], "signal": signal}

        # Place order
        order_result = exchange.place_order(
            side=signal["signal"],
            qty=sizing["qty_btc"],
            sl_price=signal["sl_price"],
            tp_price=signal["tp_price"],
        )

        if "error" in order_result:
            return {"action": "ORDER_ERROR", "error": order_result["error"], "signal": signal}

        risk_mgr.register_open()

        return {
            "action": "TRADE_EXECUTED",
            "signal": signal,
            "sizing": sizing,
            "orders": order_result,
        }

    except Exception as e:
        logger.error("Trade error: %s", e)
        raise HTTPException(500, str(e))


# ── STATUS ─────────────────────────────────────

@app.get("/status")
async def status():
    """Get account and position status."""
    result = {
        "symbol": SYMBOL,
        "model_loaded": predictor._loaded,
        "balance": risk_mgr.balance,
        "open_positions": risk_mgr.open_positions,
    }

    try:
        exchange = get_exchange()
        result["binance_balance"] = exchange.get_balance()
        result["binance_available"] = exchange.get_available_balance()
        pos = exchange.get_position()
        result["position"] = pos
        result["current_price"] = exchange.get_current_price()
    except Exception as e:
        result["binance_error"] = str(e)

    return result


# ── CLOSE POSITION ─────────────────────────────

@app.post("/close")
async def close_position():
    """Close any open position."""
    try:
        exchange = get_exchange()
        result = exchange.close_position()
        if result:
            risk_mgr.register_close(result.get("pnl", 0))
            return {"action": "CLOSED", "details": result}
        return {"action": "NO_POSITION", "message": "No open position"}
    except Exception as e:
        logger.error("Close error: %s", e)
        raise HTTPException(500, str(e))


# ── MODEL INFO ─────────────────────────────────

@app.get("/model-info")
async def model_info():
    """Get trained model metrics and top features."""
    if not predictor._loaded:
        raise HTTPException(400, "Model not trained. POST /train first.")
    return predictor.get_model_info()


# ── CONTINUOUS RETRAINING ──────────────────────

@app.get("/retrain-status")
async def retrain_status():
    """Get continuous retraining scheduler status and history."""
    return retrain_scheduler.get_status()


@app.post("/retrain-now")
async def retrain_now():
    """
    Manually trigger an immediate retrain cycle.
    This bypasses the schedule and validation gate — the new model
    is always accepted (force=True).
    """
    try:
        result = retrain_scheduler.force_retrain()
        return result
    except Exception as e:
        logger.error("Manual retrain error: %s", e)
        raise HTTPException(500, str(e))


@app.get("/retrain-history")
async def retrain_history():
    """Get full retraining history."""
    status = retrain_scheduler.get_status()
    return {
        "total_retrains": status["total_retrains"],
        "history": status["recent_history"],
    }


# ── RUN ────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
