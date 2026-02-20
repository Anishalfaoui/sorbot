"""
Sorbot AI Engine v3.0 — Backtester
=====================================
Walk-forward backtest with confidence filtering and ATR SL/TP.
Simulates real trading conditions day-by-day:
  - Train on past data only (no future leak)
  - Apply confidence gates (65% LONG / 35% SHORT)
  - ATR-based SL/TP with R:R validation
  - Position sizing with fixed risk %
  - Track PnL, win rate, drawdown, profit factor
"""

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    RISK_PER_TRADE,
    SL_ATR_MULT, TP_ATR_MULT, MIN_RR_RATIO,
    CONFIDENCE_LONG, CONFIDENCE_SHORT,
    LOOKAHEAD_CANDLES,
)
LEVERAGE = 1  # Spot trading — no leverage
from ml_core.data_loader import fetch_ohlcv
from ml_core.feature_eng import build_dataset, get_atr
from ml_core.trainer import train_model

import xgboost as xgb
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("sorbot.backtest")


def run_backtest(
    initial_balance: float = 500.0,
    risk_pct: float = RISK_PER_TRADE,
    lookback_days: int = 90,
    retrain_every: int = 168,  # retrain every 168 bars (1 week)
) -> dict:
    """
    Walk-forward backtest.

    Strategy:
      1. Train model on data up to current point
      2. Predict next bar
      3. If high confidence -> open position with SL/TP
      4. Track until SL/TP hit or LOOKAHEAD_CANDLES pass
      5. Move forward and repeat
    """
    t0 = time.time()

    # Fetch all data
    logger.info("Fetching historical data...")
    df_1h = fetch_ohlcv("1h")
    df_1d = fetch_ohlcv("1d")

    # Strip timezone
    if df_1h.index.tz:
        df_1h.index = df_1h.index.tz_localize(None)
    if df_1d.index.tz:
        df_1d.index = df_1d.index.tz_localize(None)

    n_total = len(df_1h)
    min_train = 500  # minimum bars to train on
    start_idx = min_train

    logger.info("Total 1h bars: %d, starting backtest at bar %d", n_total, start_idx)

    # State
    balance = initial_balance
    trades = []
    equity_curve = []
    in_trade = False
    current_trade = None
    booster = None
    feature_names = []
    last_train_bar = 0

    for i in range(start_idx, n_total):
        current_time = df_1h.index[i]

        # ── Retrain periodically ──────────────
        if booster is None or (i - last_train_bar) >= retrain_every:
            train_slice = df_1h.iloc[:i]
            # Build HTF data
            daily_mask = df_1d.index <= current_time
            htf_data = {"1d": df_1d[daily_mask]}

            dataset = build_dataset(train_slice, include_target=True, htf_data=htf_data)
            if len(dataset) < 200:
                continue

            try:
                # Quick train (fewer estimators for backtest speed)
                from config import XGB_PARAMS
                params = {
                    "max_depth": XGB_PARAMS["max_depth"],
                    "learning_rate": 0.02,  # slightly faster for backtest
                    "subsample": XGB_PARAMS["subsample"],
                    "colsample_bytree": XGB_PARAMS["colsample_bytree"],
                    "min_child_weight": XGB_PARAMS["min_child_weight"],
                    "gamma": XGB_PARAMS["gamma"],
                    "reg_alpha": XGB_PARAMS["reg_alpha"],
                    "reg_lambda": XGB_PARAMS["reg_lambda"],
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "seed": 42,
                    "verbosity": 0,
                }
                feature_cols = [c for c in dataset.columns if c != "target"]
                X = dataset[feature_cols].values
                y = dataset["target"].values.astype(int)

                # Dynamic class balance
                n_pos = y.sum()
                n_neg = len(y) - n_pos
                params["scale_pos_weight"] = n_neg / max(n_pos, 1)

                # Train/eval split
                split_idx = int(len(X) * 0.85)
                dtrain = xgb.DMatrix(X[:split_idx], label=y[:split_idx], feature_names=feature_cols)
                deval = xgb.DMatrix(X[split_idx:], label=y[split_idx:], feature_names=feature_cols)

                booster = xgb.train(
                    params, dtrain,
                    num_boost_round=400,
                    evals=[(deval, "val")],
                    early_stopping_rounds=30,
                    verbose_eval=False,
                )
                feature_names = feature_cols
                last_train_bar = i
            except Exception as e:
                logger.warning("Train error at bar %d: %s", i, e)
                continue

        # ── Check if in trade: resolve SL/TP ──
        if in_trade and current_trade:
            bar = df_1h.iloc[i]
            ct = current_trade

            # Check SL hit
            sl_hit = False
            tp_hit = False
            if ct["side"] == "LONG":
                if bar["Low"] <= ct["sl"]:
                    sl_hit = True
                if bar["High"] >= ct["tp"]:
                    tp_hit = True
            else:  # SHORT
                if bar["High"] >= ct["sl"]:
                    sl_hit = True
                if bar["Low"] <= ct["tp"]:
                    tp_hit = True

            # Time expiry
            bars_in_trade = i - ct["entry_bar"]
            expired = bars_in_trade >= max(LOOKAHEAD_CANDLES * 3, 12)

            if sl_hit or tp_hit or expired:
                if tp_hit:
                    exit_price = ct["tp"]
                    outcome = "TP"
                elif sl_hit:
                    exit_price = ct["sl"]
                    outcome = "SL"
                else:
                    exit_price = bar["Close"]
                    outcome = "EXPIRED"

                # Calculate PnL
                if ct["side"] == "LONG":
                    pnl_pct = (exit_price - ct["entry"]) / ct["entry"]
                else:
                    pnl_pct = (ct["entry"] - exit_price) / ct["entry"]

                pnl_usd = pnl_pct * ct["notional"]
                balance += pnl_usd

                trade_record = {
                    "entry_time": str(ct["entry_time"]),
                    "exit_time": str(current_time),
                    "side": ct["side"],
                    "entry": ct["entry"],
                    "exit": round(exit_price, 2),
                    "sl": ct["sl"],
                    "tp": ct["tp"],
                    "pnl_usd": round(pnl_usd, 2),
                    "pnl_pct": round(pnl_pct * 100, 3),
                    "outcome": outcome,
                    "balance_after": round(balance, 2),
                    "confidence": ct["confidence"],
                }
                trades.append(trade_record)
                in_trade = False
                current_trade = None

        # ── Generate signal if not in trade ───
        if not in_trade and booster is not None:
            # Build features for current bar
            slice_data = df_1h.iloc[max(0, i - 250):i + 1]
            daily_mask = df_1d.index <= current_time
            htf_data = {"1d": df_1d[daily_mask]}

            try:
                feats = build_dataset(slice_data, include_target=False, htf_data=htf_data)
                if feats.empty:
                    equity_curve.append({"time": str(current_time), "balance": round(balance, 2)})
                    continue

                # Align to trained features
                for col in feature_names:
                    if col not in feats.columns:
                        feats[col] = 0
                X_pred = feats[feature_names].iloc[[-1]].values
                dmat = xgb.DMatrix(X_pred, feature_names=feature_names)
                prob = float(booster.predict(dmat)[0])

                # Confidence gates
                signal = None
                confidence = 0.5
                if prob >= CONFIDENCE_LONG:
                    signal = "LONG"
                    confidence = prob
                elif prob <= CONFIDENCE_SHORT:
                    signal = "SHORT"
                    confidence = 1.0 - prob

                if signal:
                    bar = df_1h.iloc[i]
                    entry_price = bar["Close"]

                    # ATR
                    atr_val = float(get_atr(
                        df_1h["High"].iloc[max(0, i - 20):i + 1],
                        df_1h["Low"].iloc[max(0, i - 20):i + 1],
                        df_1h["Close"].iloc[max(0, i - 20):i + 1],
                    ).iloc[-1])

                    sl_dist = atr_val * SL_ATR_MULT
                    tp_dist = atr_val * TP_ATR_MULT
                    rr = tp_dist / sl_dist if sl_dist > 0 else 0

                    if rr >= MIN_RR_RATIO:
                        if signal == "LONG":
                            sl_price = entry_price - sl_dist
                            tp_price = entry_price + tp_dist
                        else:
                            sl_price = entry_price + sl_dist
                            tp_price = entry_price - tp_dist

                        # Position sizing
                        risk_usd = balance * risk_pct
                        qty = risk_usd / sl_dist if sl_dist > 0 else 0
                        notional = qty * entry_price

                        # Cap at leverage limit
                        max_notional = balance * LEVERAGE * 0.95
                        if notional > max_notional:
                            qty = max_notional / entry_price
                            notional = qty * entry_price

                        if qty > 0 and balance > 10:
                            current_trade = {
                                "entry_time": current_time,
                                "entry_bar": i,
                                "side": signal,
                                "entry": entry_price,
                                "sl": round(sl_price, 2),
                                "tp": round(tp_price, 2),
                                "qty": qty,
                                "notional": notional,
                                "confidence": round(confidence, 4),
                            }
                            in_trade = True

            except Exception as e:
                pass

        # Record equity
        if i % 24 == 0:  # daily equity snapshot
            equity_curve.append({"time": str(current_time), "balance": round(balance, 2)})

    # ── Results ────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    n_trades = len(trades)
    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0

    total_pnl = sum(t["pnl_usd"] for t in trades)
    gross_profit = sum(t["pnl_usd"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_usd"] for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Max drawdown
    peak = initial_balance
    max_dd = 0
    running = initial_balance
    for t in trades:
        running += t["pnl_usd"]
        peak = max(peak, running)
        dd = (peak - running) / peak
        max_dd = max(max_dd, dd)

    results = {
        "initial_balance": initial_balance,
        "final_balance": round(balance, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round((balance / initial_balance - 1) * 100, 2),
        "n_trades": n_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "avg_win": round(np.mean([t["pnl_usd"] for t in wins]), 2) if wins else 0,
        "avg_loss": round(np.mean([t["pnl_usd"] for t in losses]), 2) if losses else 0,
        "risk_pct": round(risk_pct * 100, 1),
        "leverage": LEVERAGE,
        "elapsed_sec": elapsed,
        "trades": trades,
        "equity_curve": equity_curve,
    }

    return results


def print_results(r: dict):
    """Pretty print backtest results."""
    print("\n" + "=" * 60)
    print("  SORBOT v3.0 BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Initial balance:  ${r['initial_balance']:.2f}")
    print(f"  Final balance:    ${r['final_balance']:.2f}")
    print(f"  Total PnL:        ${r['total_pnl']:.2f} ({r['total_return_pct']:+.2f}%)")
    print(f"  Trades:           {r['n_trades']} ({r['wins']}W / {r['losses']}L)")
    print(f"  Win rate:         {r['win_rate']:.1f}%")
    print(f"  Profit factor:    {r['profit_factor']:.2f}")
    print(f"  Max drawdown:     {r['max_drawdown_pct']:.2f}%")
    print(f"  Avg win:          ${r['avg_win']:.2f}")
    print(f"  Avg loss:         ${r['avg_loss']:.2f}")
    print(f"  Risk per trade:   {r['risk_pct']:.1f}%")
    print(f"  Leverage:         {r['leverage']}x")
    print(f"  Elapsed:          {r['elapsed_sec']:.1f}s")
    print("=" * 60)

    if r["trades"]:
        print("\n  Last 10 trades:")
        print(f"  {'Time':<20} {'Side':<6} {'Entry':>10} {'Exit':>10} {'PnL':>8} {'Out':<8}")
        for t in r["trades"][-10:]:
            print(f"  {t['entry_time'][:19]:<20} {t['side']:<6} {t['entry']:>10.2f} {t['exit']:>10.2f} {t['pnl_usd']:>+8.2f} {t['outcome']:<8}")


if __name__ == "__main__":
    results = run_backtest(
        initial_balance=500.0,
        risk_pct=0.03,
    )
    print_results(results)
