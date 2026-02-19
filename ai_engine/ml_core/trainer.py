"""
Sorbot AI Engine — Model Trainer  v2.0
========================================
Trains one XGBoost classifier per trading pair with multi-timeframe
feature enrichment.  Saves native xgb JSON + metadata.

Usage (CLI)
-----------
    python trainer.py                  # train all pairs
    python trainer.py --symbol BTCUSD  # train a single pair
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    TRADING_PAIRS, MODEL_DIR, XGB_PARAMS,
    ACTIVE_SYMBOLS, PRIMARY_TIMEFRAME, CONFLUENCE_TIMEFRAMES,
)
from ml_core.data_loader import fetch_ohlcv, fetch_multi_timeframe
from ml_core.feature_eng import build_features

logger = logging.getLogger("sorbot.trainer")

LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


def _model_path(symbol: str) -> Path:
    return MODEL_DIR / f"{symbol}_xgb.json"


def _meta_path(symbol: str) -> Path:
    return MODEL_DIR / f"{symbol}_meta.json"


def train_symbol(symbol: str, force_refresh: bool = False) -> dict:
    """
    Train an XGBoost model for a single trading pair using
    multi-timeframe confluence features.

    Returns
    -------
    dict   Metrics summary (accuracy, per-class precision/recall, rows used).
    """
    logger.info("══════ Training %s  (multi-TF) ══════", symbol)

    # 1. Fetch primary + higher-timeframe OHLCV
    primary_df = fetch_ohlcv(symbol, PRIMARY_TIMEFRAME, force_refresh=force_refresh)
    logger.info("Primary (%s) raw rows: %d", PRIMARY_TIMEFRAME, len(primary_df))

    # Fetch HTF data for confluence — only TFs higher than primary
    tf_order = ["1h", "4h", "1d", "1w"]
    primary_rank = tf_order.index(PRIMARY_TIMEFRAME) if PRIMARY_TIMEFRAME in tf_order else 0
    htf_data: dict[str, pd.DataFrame] = {}
    for tf in CONFLUENCE_TIMEFRAMES:
        if tf == PRIMARY_TIMEFRAME:
            continue
        # Skip timeframes lower than or equal to primary
        tf_rank = tf_order.index(tf) if tf in tf_order else -1
        if tf_rank <= primary_rank:
            logger.info("  Skipping %s (lower than primary %s)", tf, PRIMARY_TIMEFRAME)
            continue
        try:
            htf_data[tf] = fetch_ohlcv(symbol, tf, force_refresh=force_refresh)
            logger.info("  HTF %s: %d rows", tf, len(htf_data[tf]))
        except Exception as e:
            logger.warning("  HTF %s failed: %s", tf, e)

    # 2. Build features + target with multi-TF enrichment
    features = build_features(primary_df, include_target=True, htf_dataframes=htf_data or None)
    X = features.drop(columns=["target"])
    y = features["target"].astype(int)
    feature_names = list(X.columns)

    logger.info("Feature matrix: %s   Classes: %s",
                X.shape, dict(y.value_counts().sort_index()))

    if len(X) < 40:
        raise RuntimeError(f"Not enough data for {symbol}: only {len(X)} rows after features")

    # 3. Time-series train / test split (last fold = test)
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # 4. Fit XGBoost
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # 5. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        labels=[0, 1, 2],
        target_names=["SELL", "HOLD", "BUY"],
        output_dict=True,
        zero_division=0,
    )

    logger.info("Accuracy: %.4f", acc)
    logger.info("\n%s", classification_report(
        y_test, y_pred, labels=[0, 1, 2],
        target_names=["SELL", "HOLD", "BUY"], zero_division=0
    ))

    # 6. Feature importance (top 15)
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    top_features = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15])

    # 7. Persist model + metadata
    model_file = _model_path(symbol)
    model.save_model(str(model_file))
    logger.info("Model saved → %s", model_file)

    meta = {
        "symbol": symbol,
        "version": "2.0",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "primary_timeframe": PRIMARY_TIMEFRAME,
        "htf_timeframes": list(htf_data.keys()),
        "rows_total": len(features),
        "rows_train": len(X_train),
        "rows_test": len(X_test),
        "n_features": len(feature_names),
        "accuracy": round(acc, 4),
        "report": {k: v for k, v in report.items() if k in LABEL_MAP.values()},
        "top_features": top_features,
        "feature_names": feature_names,
        "xgb_params": XGB_PARAMS,
    }
    _meta_path(symbol).write_text(json.dumps(meta, indent=2))

    return meta


def train_all(force_refresh: bool = False) -> dict[str, dict]:
    """Train models for every active symbol. Returns {symbol: metrics}."""
    results = {}
    for sym in ACTIVE_SYMBOLS:
        try:
            results[sym] = train_symbol(sym, force_refresh=force_refresh)
        except Exception as exc:
            logger.error("Training failed for %s: %s", sym, exc, exc_info=True)
            results[sym] = {"error": str(exc)}
    return results


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser(description="Sorbot XGBoost Trainer v2.0")
    parser.add_argument("--symbol", type=str, default=None, help="Train a specific pair")
    parser.add_argument("--refresh", action="store_true", help="Force-download fresh data")
    args = parser.parse_args()

    if args.symbol:
        result = train_symbol(args.symbol.upper(), force_refresh=args.refresh)
        print(json.dumps(result, indent=2))
    else:
        results = train_all(force_refresh=args.refresh)
        for sym, meta in results.items():
            print(f"\n{'-'*40}")
            print(f"  {sym}: accuracy={meta.get('accuracy', 'N/A')}  features={meta.get('n_features', '?')}")
