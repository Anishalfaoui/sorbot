"""
Sorbot AI Engine v3.0 — Trainer
=================================
Walk-forward XGBoost training for BTC/USD:
  - 5-fold expanding-window time-series CV
  - Early stopping to prevent overfitting
  - Dynamic scale_pos_weight for class balance
  - Saves native xgb.Booster JSON (avoids sklearn compat bugs)
  - Persists feature list + training meta in meta.json
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_DIR, XGB_PARAMS,
    WF_N_SPLITS, WF_TEST_SIZE, EARLY_STOPPING_ROUNDS,
)

logger = logging.getLogger("sorbot.trainer")

MODEL_FILE = MODEL_DIR / "btc_model.json"
META_FILE = MODEL_DIR / "btc_meta.json"


def _walk_forward_splits(n_samples: int, n_splits: int, test_size: int):
    """
    Generate expanding-window walk-forward train/test indices.
    Training window grows; test window is fixed at `test_size`.
    """
    min_train = max(500, n_samples - n_splits * test_size)
    splits = []
    for i in range(n_splits):
        test_end = n_samples - (n_splits - 1 - i) * test_size
        test_start = test_end - test_size
        train_end = test_start
        if train_end < min_train:
            continue
        splits.append((list(range(0, train_end)), list(range(test_start, test_end))))
    return splits


def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_true, y_prob), 4) if len(np.unique(y_true)) > 1 else 0.0,
    }


def train_model(dataset: pd.DataFrame) -> dict:
    """
    Train XGBoost with walk-forward validation.

    Args:
        dataset: DataFrame with features + 'target' column
                 (NaN targets already removed by feature_eng.build_dataset)

    Returns:
        dict with training metrics and fold results
    """
    t0 = time.time()

    # Separate features and target
    feature_cols = [c for c in dataset.columns if c != "target"]
    X = dataset[feature_cols].values
    y = dataset["target"].values.astype(int)

    n_samples = len(X)
    n_up = int(y.sum())
    n_down = n_samples - n_up
    logger.info("Training data: %d samples (%d UP / %d DOWN)", n_samples, n_up, n_down)

    # Walk-forward splits
    splits = _walk_forward_splits(n_samples, WF_N_SPLITS, WF_TEST_SIZE)
    if not splits:
        raise ValueError(f"Not enough data for walk-forward. Have {n_samples} rows, need at least {WF_TEST_SIZE * WF_N_SPLITS + 500}")

    logger.info("Walk-forward: %d splits x %d test bars", len(splits), WF_TEST_SIZE)

    # Prepare XGB params for native API
    params = {
        "max_depth": XGB_PARAMS["max_depth"],
        "learning_rate": XGB_PARAMS["learning_rate"],
        "subsample": XGB_PARAMS["subsample"],
        "colsample_bytree": XGB_PARAMS["colsample_bytree"],
        "min_child_weight": XGB_PARAMS["min_child_weight"],
        "gamma": XGB_PARAMS["gamma"],
        "reg_alpha": XGB_PARAMS["reg_alpha"],
        "reg_lambda": XGB_PARAMS["reg_lambda"],
        "objective": XGB_PARAMS["objective"],
        "eval_metric": XGB_PARAMS["eval_metric"],
        "seed": XGB_PARAMS.get("random_state", 42),
        "nthread": -1,
        "verbosity": 0,
    }

    # Walk-forward cross validation
    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Dynamic class balance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        fold_params = params.copy()
        fold_params["scale_pos_weight"] = n_neg / max(n_pos, 1)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

        booster = xgb.train(
            fold_params,
            dtrain,
            num_boost_round=XGB_PARAMS["n_estimators"],
            evals=[(dtrain, "train"), (dtest, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

        y_prob = booster.predict(dtest)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = _compute_metrics(y_test, y_pred, y_prob)
        metrics["fold"] = fold_idx + 1
        metrics["train_size"] = len(train_idx)
        metrics["test_size"] = len(test_idx)
        metrics["best_iteration"] = booster.best_iteration
        fold_metrics.append(metrics)
        logger.info(
            "  Fold %d: acc=%.3f  auc=%.3f  f1=%.3f  (train=%d, test=%d, iters=%d)",
            fold_idx + 1, metrics["accuracy"], metrics["auc_roc"],
            metrics["f1"], len(train_idx), len(test_idx), booster.best_iteration,
        )

    # FINAL model — train on ALL data with early stopping on last 200 bars
    logger.info("Training final model on full dataset...")
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    params["scale_pos_weight"] = n_neg / max(n_pos, 1)

    # Use last WF_TEST_SIZE bars as eval set for early stopping
    final_train_idx = list(range(0, n_samples - WF_TEST_SIZE))
    final_eval_idx = list(range(n_samples - WF_TEST_SIZE, n_samples))

    dtrain_final = xgb.DMatrix(X[final_train_idx], label=y[final_train_idx], feature_names=feature_cols)
    deval_final = xgb.DMatrix(X[final_eval_idx], label=y[final_eval_idx], feature_names=feature_cols)

    final_booster = xgb.train(
        params,
        dtrain_final,
        num_boost_round=XGB_PARAMS["n_estimators"],
        evals=[(dtrain_final, "train"), (deval_final, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

    # Evaluate final model
    y_prob_final = final_booster.predict(deval_final)
    y_pred_final = (y_prob_final >= 0.5).astype(int)
    final_metrics = _compute_metrics(y[final_eval_idx], y_pred_final, y_prob_final)

    # Save model
    final_booster.save_model(str(MODEL_FILE))
    logger.info("Model saved to %s", MODEL_FILE)

    # Average CV metrics
    avg_metrics = {
        "accuracy": round(np.mean([m["accuracy"] for m in fold_metrics]), 4),
        "auc_roc": round(np.mean([m["auc_roc"] for m in fold_metrics]), 4),
        "f1": round(np.mean([m["f1"] for m in fold_metrics]), 4),
        "precision": round(np.mean([m["precision"] for m in fold_metrics]), 4),
        "recall": round(np.mean([m["recall"] for m in fold_metrics]), 4),
    }

    # Feature importance
    importance = final_booster.get_score(importance_type="gain")
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

    elapsed = round(time.time() - t0, 1)

    # Save meta
    meta = {
        "trained_at": datetime.utcnow().isoformat(),
        "training_time_sec": elapsed,
        "n_samples": n_samples,
        "n_up": n_up,
        "n_down": n_down,
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "wf_folds": len(splits),
        "wf_test_size": WF_TEST_SIZE,
        "cv_metrics": avg_metrics,
        "fold_details": fold_metrics,
        "final_metrics": final_metrics,
        "best_iteration": final_booster.best_iteration,
        "top_features": top_features,
        "params": {k: v for k, v in params.items() if k != "nthread"},
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Meta saved to %s", META_FILE)

    logger.info("Training complete in %.1fs", elapsed)
    logger.info("CV avg: acc=%.3f  auc=%.3f  f1=%.3f", avg_metrics["accuracy"], avg_metrics["auc_roc"], avg_metrics["f1"])
    logger.info("Final:  acc=%.3f  auc=%.3f  f1=%.3f", final_metrics["accuracy"], final_metrics["auc_roc"], final_metrics["f1"])

    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    from data_loader import fetch_all_timeframes
    from feature_eng import build_dataset

    data = fetch_all_timeframes()
    htf_data = {"4h": data.get("4h"), "1d": data.get("1d")}
    dataset = build_dataset(data["1h"], include_target=True, htf_data=htf_data)
    meta = train_model(dataset)
    print(f"\nTop features: {meta['top_features'][:10]}")
