"""
Sorbot AI Engine — Live Predictor  v2.0
=========================================
Multi-timeframe confluence, Stop-Loss / Take-Profit proposals,
support/resistance levels, best-timing analysis, and full trade plan.

Uses xgboost.Booster (native API) for inference to avoid scikit-learn
wrapper compatibility issues.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_DIR, TRADING_PAIRS, CONFIDENCE_THRESHOLD,
    PRIMARY_TIMEFRAME, CONFLUENCE_TIMEFRAMES, ENTRY_TIMEFRAME,
    SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER, MIN_RISK_REWARD,
    ATR_PERIOD,
)
from ml_core.data_loader import fetch_ohlcv, fetch_multi_timeframe
from ml_core.feature_eng import (
    build_features, build_base_features, detect_sr_levels,
    _atr_raw, _rsi, _macd, _stochastic,
)

logger = logging.getLogger("sorbot.predictor")

LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}

# ── Model cache ──────────────────────────────────────────
_model_cache: dict[str, xgb.Booster] = {}
_meta_cache: dict[str, dict] = {}


def _model_path(symbol: str) -> Path:
    return MODEL_DIR / f"{symbol}_xgb.json"


def _meta_path(symbol: str) -> Path:
    return MODEL_DIR / f"{symbol}_meta.json"


def _load_model(symbol: str, force: bool = False) -> xgb.Booster:
    if not force and symbol in _model_cache:
        return _model_cache[symbol]

    path = _model_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"No trained model for {symbol}. Run trainer.py first.")

    booster = xgb.Booster()
    booster.load_model(str(path))
    _model_cache[symbol] = booster
    logger.info("Loaded model for %s", symbol)

    mp = _meta_path(symbol)
    if mp.exists():
        _meta_cache[symbol] = json.loads(mp.read_text())

    return booster


def reload_models():
    """Force-reload all models from disk."""
    _model_cache.clear()
    _meta_cache.clear()
    for symbol in TRADING_PAIRS:
        if _model_path(symbol).exists():
            _load_model(symbol, force=True)
    logger.info("All models reloaded.")


# ──────────────────────────────────────────────────────────
#  STOP-LOSS / TAKE-PROFIT CALCULATOR
# ──────────────────────────────────────────────────────────

def _compute_sl_tp(
    symbol: str,
    direction: str,
    current_price: float,
    atr_value: float,
    sr_levels: list[dict],
) -> dict:
    """
    Calculate SL and TP based on ATR + nearest S/R, with risk:reward check.
    """
    pair_cfg = TRADING_PAIRS[symbol]
    sl_mult = pair_cfg.get("default_sl_atr_mult", SL_ATR_MULTIPLIER)
    tp_mult = pair_cfg.get("default_tp_atr_mult", TP_ATR_MULTIPLIER)
    decimals = pair_cfg["decimals"]

    atr_sl = atr_value * sl_mult
    atr_tp = atr_value * tp_mult

    supports = sorted([l for l in sr_levels if l["type"] == "support"],
                      key=lambda x: abs(x["price"] - current_price))
    resistances = sorted([l for l in sr_levels if l["type"] == "resistance"],
                         key=lambda x: abs(x["price"] - current_price))

    if direction == "BUY":
        # SL below recent support or ATR
        sl_atr = current_price - atr_sl
        sl_sr = supports[0]["price"] * 0.998 if supports else sl_atr  # tiny buffer
        sl = max(sl_atr, sl_sr)  # tighter SL = higher of the two

        # TP at resistance or ATR
        tp_atr = current_price + atr_tp
        tp_sr = resistances[0]["price"] * 0.998 if resistances else tp_atr
        tp = min(tp_atr, tp_sr)  # take profit before resistance

        risk = current_price - sl
        reward = tp - current_price

    elif direction == "SELL":
        sl_atr = current_price + atr_sl
        sl_sr = resistances[0]["price"] * 1.002 if resistances else sl_atr
        sl = min(sl_atr, sl_sr)

        tp_atr = current_price - atr_tp
        tp_sr = supports[0]["price"] * 1.002 if supports else tp_atr
        tp = max(tp_atr, tp_sr)

        risk = sl - current_price
        reward = current_price - tp
    else:
        # HOLD — no SL/TP
        return {
            "stop_loss": None,
            "take_profit": None,
            "risk_reward_ratio": None,
            "risk_pct": None,
            "reward_pct": None,
        }

    rr = round(reward / risk, 2) if risk > 0 else 0.0

    return {
        "stop_loss": round(sl, decimals),
        "take_profit": round(tp, decimals),
        "risk_reward_ratio": rr,
        "risk_pct": round((risk / current_price) * 100, 2),
        "reward_pct": round((reward / current_price) * 100, 2),
    }


# ──────────────────────────────────────────────────────────
#  MULTI-TIMEFRAME CONFLUENCE SCORER
# ──────────────────────────────────────────────────────────

def _tf_trend(df: pd.DataFrame) -> dict:
    """Quick trend assessment for a single timeframe."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    rsi = _rsi(close).iloc[-1]
    _, _, macd_h = _macd(close)
    macd_hist = macd_h.iloc[-1]
    k, d = _stochastic(high, low, close)
    stoch_k = k.iloc[-1]

    ema9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    ema_cross = "bullish" if ema9 > ema21 else "bearish"

    # Simple score: -3 to +3
    score = 0
    if rsi > 55: score += 1
    elif rsi < 45: score -= 1
    if macd_hist > 0: score += 1
    else: score -= 1
    if ema_cross == "bullish": score += 1
    else: score -= 1

    trend = "bullish" if score > 0 else ("bearish" if score < 0 else "neutral")

    return {
        "trend": trend,
        "score": score,
        "rsi": round(float(rsi), 2),
        "macd_hist": round(float(macd_hist), 6),
        "stoch_k": round(float(stoch_k), 2),
        "ema_cross": ema_cross,
    }


def _confluence_analysis(mtf_data: dict[str, pd.DataFrame]) -> dict:
    """
    Analyse multiple timeframes and compute a confluence score.
    Score range: -N*3 to +N*3 where N = number of timeframes.
    """
    tf_details = {}
    total_score = 0
    total_possible = 0

    for tf, df in mtf_data.items():
        if df is None or len(df) < 30:
            continue
        info = _tf_trend(df)
        tf_details[tf] = info
        total_score += info["score"]
        total_possible += 3

    if total_possible == 0:
        confluence_pct = 0
    else:
        confluence_pct = round((total_score / total_possible) * 100, 1)

    if confluence_pct > 30:
        overall = "bullish"
    elif confluence_pct < -30:
        overall = "bearish"
    else:
        overall = "mixed"

    return {
        "overall_trend": overall,
        "confluence_score": total_score,
        "confluence_pct": confluence_pct,
        "timeframes": tf_details,
    }


# ──────────────────────────────────────────────────────────
#  BEST TIMING / ENTRY QUALITY
# ──────────────────────────────────────────────────────────

def _entry_timing(df_entry: pd.DataFrame, direction: str) -> dict:
    """
    Assess entry timing quality on the entry timeframe (e.g. 1h).
    Returns timing score and recommendation.
    """
    if df_entry is None or len(df_entry) < 20:
        return {"timing": "unknown", "score": 0, "note": "Insufficient entry-TF data"}

    close = df_entry["Close"]
    high = df_entry["High"]
    low = df_entry["Low"]

    rsi = _rsi(close).iloc[-1]
    k, _ = _stochastic(high, low, close)
    stoch = k.iloc[-1]
    _, _, macd_h = _macd(close)
    macd_hist = macd_h.iloc[-1]

    score = 0
    notes = []

    if direction == "BUY":
        if rsi < 35:
            score += 2; notes.append("RSI oversold — ideal buy zone")
        elif rsi < 45:
            score += 1; notes.append("RSI approaching oversold")
        elif rsi > 70:
            score -= 1; notes.append("RSI overbought — risky to buy")

        if stoch < 20:
            score += 1; notes.append("Stochastic oversold")
        if macd_hist > 0 and macd_h.iloc[-2] < 0:
            score += 2; notes.append("MACD just crossed bullish")
        elif macd_hist > 0:
            score += 1

    elif direction == "SELL":
        if rsi > 65:
            score += 2; notes.append("RSI overbought — ideal sell zone")
        elif rsi > 55:
            score += 1; notes.append("RSI approaching overbought")
        elif rsi < 30:
            score -= 1; notes.append("RSI oversold — risky to sell")

        if stoch > 80:
            score += 1; notes.append("Stochastic overbought")
        if macd_hist < 0 and macd_h.iloc[-2] > 0:
            score += 2; notes.append("MACD just crossed bearish")
        elif macd_hist < 0:
            score += 1

    # Map score to timing label
    if score >= 3:
        timing = "excellent"
    elif score >= 1:
        timing = "good"
    elif score == 0:
        timing = "neutral"
    else:
        timing = "poor"

    return {
        "timing": timing,
        "score": score,
        "note": "; ".join(notes) if notes else "No strong signals on entry TF",
    }


# ──────────────────────────────────────────────────────────
#  MAIN PREDICT
# ──────────────────────────────────────────────────────────

def predict(symbol: str) -> dict:
    """
    Generate a live prediction with full trade plan.
    """
    if symbol not in TRADING_PAIRS:
        raise ValueError(f"Unknown symbol '{symbol}'")

    booster = _load_model(symbol)
    pair_cfg = TRADING_PAIRS[symbol]

    # 1. Fetch primary + HTF data
    primary_df = fetch_ohlcv(symbol, PRIMARY_TIMEFRAME)
    htf_data = {}
    for tf in CONFLUENCE_TIMEFRAMES:
        if tf == PRIMARY_TIMEFRAME:
            htf_data[tf] = primary_df
            continue
        try:
            htf_data[tf] = fetch_ohlcv(symbol, tf)
        except Exception:
            pass

    # Entry timeframe
    entry_df = None
    try:
        entry_df = fetch_ohlcv(symbol, ENTRY_TIMEFRAME)
    except Exception:
        pass

    # 2. Build features (match training)
    htf_for_features = {k: v for k, v in htf_data.items() if k != PRIMARY_TIMEFRAME}
    features = build_features(primary_df, include_target=False,
                              htf_dataframes=htf_for_features or None)
    if features.empty:
        raise RuntimeError("Feature matrix empty after NaN cleanup")

    X_live = features.iloc[[-1]]

    meta = _meta_cache.get(symbol, {})
    if "feature_names" in meta:
        # Ensure column order matches training; fill missing with 0
        for col in meta["feature_names"]:
            if col not in X_live.columns:
                X_live[col] = 0.0
        X_live = X_live[meta["feature_names"]]

    # 3. Predict
    dmat = xgb.DMatrix(X_live)
    proba = booster.predict(dmat)[0]
    class_idx = int(np.argmax(proba))
    confidence = float(proba[class_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        direction = "HOLD"
    else:
        direction = LABEL_MAP[class_idx]

    current_price = round(float(primary_df["Close"].iloc[-1]), pair_cfg["decimals"])

    # 4. ATR (absolute) for SL/TP
    atr_abs = _atr_raw(primary_df["High"], primary_df["Low"], primary_df["Close"]).iloc[-1]

    # 5. Support / Resistance levels
    sr_levels = detect_sr_levels(primary_df["High"], primary_df["Low"], primary_df["Close"])

    # 6. SL / TP
    sl_tp = _compute_sl_tp(symbol, direction, current_price, float(atr_abs), sr_levels)

    # 7. Multi-timeframe confluence
    confluence = _confluence_analysis(htf_data)

    # 8. Entry timing
    timing = _entry_timing(entry_df, direction)

    # 9. Risk:reward gate
    trade_viable = True
    trade_note = ""
    if direction in ("BUY", "SELL"):
        rr = sl_tp.get("risk_reward_ratio") or 0
        if rr < MIN_RISK_REWARD:
            trade_viable = False
            trade_note = f"R:R {rr} < minimum {MIN_RISK_REWARD} — consider waiting"

    # 10. Indicators snapshot
    last_feat = features.iloc[-1]
    rsi_val = round(float(last_feat.get("rsi", 0)), 2)
    macd_val = float(last_feat.get("macd_hist", 0))
    adx_val = round(float(last_feat.get("adx", 0)), 2)

    result = {
        "symbol": symbol,
        "direction": direction,
        "confidence": round(confidence, 4),
        "current_price": current_price,
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "trade_plan": {
            **sl_tp,
            "viable": trade_viable,
            "note": trade_note,
        },

        "confluence": confluence,

        "entry_timing": timing,

        "support_resistance": sr_levels[:6],

        "indicators": {
            "rsi": rsi_val,
            "macd": "bullish" if macd_val > 0 else "bearish",
            "macd_histogram": round(macd_val, 6),
            "adx": adx_val,
            "adx_trend": "strong" if adx_val > 25 else "weak",
            "atr_abs": round(float(atr_abs), pair_cfg["decimals"]),
            "bb_pctb": round(float(last_feat.get("bb_pctb", 0)), 4),
            "ema200_dist": round(float(last_feat.get("ema200_dist", 0)), 4),
            "stoch_k": round(float(last_feat.get("stoch_k", 0)), 2),
            "stoch_d": round(float(last_feat.get("stoch_d", 0)), 2),
        },

        "probabilities": {
            "SELL": round(float(proba[0]), 4),
            "HOLD": round(float(proba[1]), 4),
            "BUY":  round(float(proba[2]), 4),
        },

        "model_info": {
            "version": meta.get("version", "1.0"),
            "trained_at": meta.get("trained_at", "unknown"),
            "accuracy": meta.get("accuracy", 0),
            "n_features": meta.get("n_features", len(features.columns)),
        },
    }

    logger.info(
        "%s → %s (%.1f%%)  price=%s  SL=%s  TP=%s  R:R=%s  confluence=%s  timing=%s",
        symbol, direction, confidence * 100, current_price,
        sl_tp.get("stop_loss"), sl_tp.get("take_profit"),
        sl_tp.get("risk_reward_ratio"), confluence["overall_trend"], timing["timing"],
    )
    return result


def predict_all() -> list[dict]:
    """Run full predictions for every active pair."""
    results = []
    for symbol in TRADING_PAIRS:
        try:
            results.append(predict(symbol))
        except Exception as exc:
            logger.error("Prediction failed for %s: %s", symbol, exc)
            results.append({"symbol": symbol, "error": str(exc)})
    return results


def get_model_info(symbol: str) -> Optional[dict]:
    """Return training metadata for a symbol, or None."""
    mp = _meta_path(symbol)
    if mp.exists():
        return json.loads(mp.read_text())
    return None


# ── CLI test ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    for p in predict_all():
        print(json.dumps(p, indent=2))
