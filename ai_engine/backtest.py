"""
Sorbot AI Engine  -  Backtest Simulation
==========================================
Simulates trading from January 1, 2026 to February 19, 2026
with $10,000 starting capital using the trained XGBoost models.

Walk-forward day-by-day:
  1. Build features from data available up to that day (no future leak)
  2. Run model prediction
  3. Enter BUY/SELL trades with SL/TP
  4. Track open trades, close on SL/TP hit or after 5 days
  5. Position sizing: risk 2% of equity per trade

Usage:
    python backtest.py
"""

import json
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    TRADING_PAIRS, MODEL_DIR, PRIMARY_TIMEFRAME,
    CONFIDENCE_THRESHOLD, MIN_RISK_REWARD, ATR_PERIOD,
    SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER,
)
from ml_core.data_loader import fetch_ohlcv
from ml_core.feature_eng import (
    build_features, build_base_features, detect_sr_levels,
    _atr_raw, _rsi, _macd, _stochastic,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("sorbot.backtest")

# -------------------------------------------------------
#  CONSTANTS
# -------------------------------------------------------
START_DATE = pd.Timestamp("2025-01-01")
END_DATE   = pd.Timestamp("2026-02-19")
INITIAL_EQUITY = 10_000.0
RISK_PER_TRADE  = 0.05       # 5% of equity risked per trade
MAX_HOLD_DAYS   = 5           # close after 5 bars if SL/TP not hit
MAX_POSITION_PCT = 0.50       # max 50% of equity in one position
LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


@dataclass
class Trade:
    symbol: str
    direction: str          # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float    # units of the asset
    entry_date: pd.Timestamp
    confidence: float
    bars_held: int = 0
    exit_price: Optional[float] = None
    exit_date: Optional[pd.Timestamp] = None
    exit_reason: str = ""
    pnl: float = 0.0


# -------------------------------------------------------
#  LOAD MODELS
# -------------------------------------------------------
def load_model(symbol: str):
    """Load xgb.Booster + training meta."""
    model_path = MODEL_DIR / f"{symbol}_xgb.json"
    meta_path  = MODEL_DIR / f"{symbol}_meta.json"
    if not model_path.exists():
        return None, None
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    return booster, meta


# -------------------------------------------------------
#  COMPUTE SL / TP
# -------------------------------------------------------
def compute_sl_tp(symbol, direction, price, atr_val, sr_levels):
    """Simplified version of predictor._compute_sl_tp."""
    pair = TRADING_PAIRS[symbol]
    sl_mult = pair.get("default_sl_atr_mult", SL_ATR_MULTIPLIER)
    tp_mult = pair.get("default_tp_atr_mult", TP_ATR_MULTIPLIER)

    atr_sl = atr_val * sl_mult
    atr_tp = atr_val * tp_mult

    supports    = sorted([l for l in sr_levels if l["type"] == "support"],
                         key=lambda x: abs(x["price"] - price))
    resistances = sorted([l for l in sr_levels if l["type"] == "resistance"],
                         key=lambda x: abs(x["price"] - price))

    if direction == "BUY":
        sl_atr = price - atr_sl
        sl_sr  = supports[0]["price"] * 0.998 if supports else sl_atr
        sl = max(sl_atr, sl_sr)        # tighter SL
        tp_atr = price + atr_tp
        tp_sr  = resistances[0]["price"] * 0.998 if resistances else tp_atr
        tp = min(tp_atr, tp_sr)
    elif direction == "SELL":
        sl_atr = price + atr_sl
        sl_sr  = resistances[0]["price"] * 1.002 if resistances else sl_atr
        sl = min(sl_atr, sl_sr)
        tp_atr = price - atr_tp
        tp_sr  = supports[0]["price"] * 1.002 if supports else tp_atr
        tp = max(tp_atr, tp_sr)
    else:
        return None, None

    # Validate risk:reward
    if direction == "BUY":
        risk   = price - sl
        reward = tp - price
    else:
        risk   = sl - price
        reward = price - tp

    if risk <= 0:
        return None, None
    rr = reward / risk
    if rr < MIN_RISK_REWARD:
        return None, None

    return sl, tp


# -------------------------------------------------------
#  GENERATE SIGNAL FOR ONE DAY
# -------------------------------------------------------
def generate_signal(symbol, booster, meta, daily_data, weekly_data, sim_date):
    """
    Build features from data up to sim_date, run model prediction.
    Returns (direction, confidence, sl, tp, current_price) or None.
    """
    # Slice data up to sim_date (inclusive)
    df_daily = daily_data.loc[daily_data.index <= sim_date].copy()
    if len(df_daily) < 210:   # need 200+ bars for EMA-200 warmup
        return None

    # Weekly data up to sim_date
    htf_data = {}
    if weekly_data is not None and not weekly_data.empty:
        df_w = weekly_data.loc[weekly_data.index <= sim_date].copy()
        if len(df_w) > 20:
            htf_data["1w"] = df_w

    # Build features
    try:
        features = build_features(df_daily, include_target=False,
                                  htf_dataframes=htf_data if htf_data else None)
    except Exception:
        return None

    if features.empty:
        return None

    X = features.iloc[[-1]].copy()

    # Align columns to training order
    if "feature_names" in meta:
        for col in meta["feature_names"]:
            if col not in X.columns:
                X[col] = 0.0
        X = X[meta["feature_names"]]

    # Predict
    dmat = xgb.DMatrix(X)
    proba = booster.predict(dmat)[0]
    class_idx = int(np.argmax(proba))
    confidence = float(proba[class_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        direction = "HOLD"
    else:
        direction = LABEL_MAP[class_idx]

    if direction == "HOLD":
        return None   # no trade

    current_price = float(df_daily["Close"].iloc[-1])

    # ATR for SL/TP
    atr_val = float(_atr_raw(
        df_daily["High"], df_daily["Low"], df_daily["Close"]
    ).iloc[-1])

    # S/R levels
    sr = detect_sr_levels(df_daily["High"], df_daily["Low"], df_daily["Close"])

    sl, tp = compute_sl_tp(symbol, direction, current_price, atr_val, sr)
    if sl is None:
        return None  # R:R failed

    return {
        "direction": direction,
        "confidence": confidence,
        "price": current_price,
        "sl": sl,
        "tp": tp,
    }


# -------------------------------------------------------
#  MAIN SIMULATION
# -------------------------------------------------------
def run_backtest():
    print("=" * 70)
    print("  SORBOT AI ENGINE  -  BACKTEST SIMULATION")
    print("  Period: January 1, 2026  -->  February 19, 2026")
    print(f"  Starting Capital: ${INITIAL_EQUITY:,.2f}")
    print(f"  Risk per trade: {RISK_PER_TRADE*100:.0f}% of equity")
    print("=" * 70)

    # ----- 1. Load models -----
    models = {}
    metas  = {}
    for sym in TRADING_PAIRS:
        b, m = load_model(sym)
        if b is None:
            print(f"  [!] No model for {sym} - skipping")
            continue
        models[sym] = b
        metas[sym]  = m
        print(f"  Model loaded: {sym}  (accuracy={m.get('accuracy','?')}, "
              f"features={m.get('n_features','?')})")

    if not models:
        print("No models found. Train models first.")
        return

    # ----- 2. Fetch data -----
    print("\nFetching historical data...")
    daily_data = {}
    weekly_data = {}
    for sym in models:
        try:
            daily_data[sym]  = fetch_ohlcv(sym, "1d", force_refresh=True)
            weekly_data[sym] = fetch_ohlcv(sym, "1w", force_refresh=True)
        except Exception as e:
            print(f"  [!] Data fetch failed for {sym}: {e}")

    # Normalize indices to tz-naive for consistent comparison
    for sym in list(daily_data.keys()):
        if daily_data[sym].index.tz is not None:
            daily_data[sym].index = daily_data[sym].index.tz_localize(None)
        if weekly_data[sym].index.tz is not None:
            weekly_data[sym].index = weekly_data[sym].index.tz_localize(None)

    # ----- 3. Build trading calendar -----
    # Use BTCUSD daily index as the master calendar (crypto trades every day)
    all_dates = set()
    for sym in daily_data:
        mask = (daily_data[sym].index >= START_DATE) & (daily_data[sym].index <= END_DATE)
        all_dates |= set(daily_data[sym].index[mask])
    trading_days = sorted(all_dates)

    if not trading_days:
        print("No trading days found in the simulation period!")
        return

    print(f"  Trading days: {len(trading_days)}  "
          f"({trading_days[0].strftime('%Y-%m-%d')} to {trading_days[-1].strftime('%Y-%m-%d')})")

    # ----- 4. Walk-forward simulation -----
    equity = INITIAL_EQUITY
    equity_curve = [(START_DATE, equity)]
    open_trades: dict[str, Trade] = {}
    closed_trades: list[Trade] = []
    total_signals = 0
    daily_log = []

    for day in trading_days:
        day_events = []

        # -- 4a. Update open trades (check SL/TP/expiry) --
        symbols_to_close = []
        for sym, trade in open_trades.items():
            if sym not in daily_data:
                continue
            df = daily_data[sym]
            if day not in df.index:
                trade.bars_held += 1
                continue

            day_row = df.loc[day]
            day_high = float(day_row["High"])
            day_low  = float(day_row["Low"])
            day_close = float(day_row["Close"])
            trade.bars_held += 1

            closed = False

            if trade.direction == "BUY":
                # Check SL first (worst case)
                if day_low <= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.exit_reason = "STOP-LOSS"
                    closed = True
                elif day_high >= trade.take_profit:
                    trade.exit_price = trade.take_profit
                    trade.exit_reason = "TAKE-PROFIT"
                    closed = True
                elif trade.bars_held >= MAX_HOLD_DAYS:
                    trade.exit_price = day_close
                    trade.exit_reason = "MAX-HOLD (5d)"
                    closed = True

            elif trade.direction == "SELL":
                if day_high >= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.exit_reason = "STOP-LOSS"
                    closed = True
                elif day_low <= trade.take_profit:
                    trade.exit_price = trade.take_profit
                    trade.exit_reason = "TAKE-PROFIT"
                    closed = True
                elif trade.bars_held >= MAX_HOLD_DAYS:
                    trade.exit_price = day_close
                    trade.exit_reason = "MAX-HOLD (5d)"
                    closed = True

            if closed:
                trade.exit_date = day
                if trade.direction == "BUY":
                    trade.pnl = trade.position_size * (trade.exit_price - trade.entry_price)
                else:
                    trade.pnl = trade.position_size * (trade.entry_price - trade.exit_price)
                equity += trade.pnl
                symbols_to_close.append(sym)

                pnl_pct = (trade.pnl / equity) * 100
                emoji = "+" if trade.pnl >= 0 else ""
                day_events.append(
                    f"  CLOSE {sym} {trade.direction} @ {trade.exit_price:,.2f}  "
                    f"[{trade.exit_reason}]  PnL: {emoji}${trade.pnl:,.2f} ({emoji}{pnl_pct:.2f}%)  "
                    f"held {trade.bars_held}d"
                )

        for sym in symbols_to_close:
            closed_trades.append(open_trades.pop(sym))

        # -- 4b. Generate new signals --
        for sym in models:
            if sym in open_trades:
                continue  # already in a position
            if sym not in daily_data:
                continue
            if day not in daily_data[sym].index:
                continue

            sig = generate_signal(
                sym, models[sym], metas[sym],
                daily_data[sym], weekly_data.get(sym),
                day
            )
            total_signals += 1

            if sig is None:
                continue

            # Position sizing: risk 2% of equity
            direction = sig["direction"]
            entry_price = sig["price"]
            sl = sig["sl"]
            tp = sig["tp"]

            risk_per_unit = abs(entry_price - sl)
            if risk_per_unit <= 0:
                continue

            risk_amount = equity * RISK_PER_TRADE
            position_size = risk_amount / risk_per_unit

            # Cap position value at 50% of equity to avoid over-leverage
            position_value = position_size * entry_price
            max_position = equity * MAX_POSITION_PCT
            if position_value > max_position:
                position_size = max_position / entry_price

            trade = Trade(
                symbol=sym,
                direction=direction,
                entry_price=entry_price,
                stop_loss=round(sl, TRADING_PAIRS[sym]["decimals"]),
                take_profit=round(tp, TRADING_PAIRS[sym]["decimals"]),
                position_size=position_size,
                entry_date=day,
                confidence=sig["confidence"],
            )
            open_trades[sym] = trade

            day_events.append(
                f"  OPEN  {sym} {direction} @ {entry_price:,.2f}  "
                f"SL={trade.stop_loss:,.2f}  TP={trade.take_profit:,.2f}  "
                f"conf={sig['confidence']*100:.1f}%  size={position_size:.4f}"
            )

        equity_curve.append((day, equity))

        if day_events:
            daily_log.append((day, equity, day_events))

    # -- Close any remaining open trades at last available price --
    for sym, trade in list(open_trades.items()):
        df = daily_data[sym]
        last_price = float(df["Close"].iloc[-1])
        trade.exit_price = last_price
        trade.exit_date = trading_days[-1]
        trade.exit_reason = "END-OF-SIM"
        if trade.direction == "BUY":
            trade.pnl = trade.position_size * (last_price - trade.entry_price)
        else:
            trade.pnl = trade.position_size * (trade.entry_price - last_price)
        equity += trade.pnl
        closed_trades.append(trade)
        daily_log.append((trading_days[-1], equity,
            [f"  CLOSE {sym} {trade.direction} @ {last_price:,.2f} [END-OF-SIM]  "
             f"PnL: ${trade.pnl:,.2f}"]))

    equity_curve.append((END_DATE, equity))

    # -------------------------------------------------------
    #  PRINT RESULTS
    # -------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TRADE LOG")
    print("=" * 70)
    for day, eq, events in daily_log:
        print(f"\n  {day.strftime('%Y-%m-%d')}  |  Equity: ${eq:,.2f}")
        for e in events:
            print(e)

    # ---- Per-symbol breakdown ----
    print("\n" + "=" * 70)
    print("  PER-SYMBOL BREAKDOWN")
    print("=" * 70)
    for sym in TRADING_PAIRS:
        sym_trades = [t for t in closed_trades if t.symbol == sym]
        if not sym_trades:
            print(f"\n  {sym}: No trades taken")
            continue
        wins = [t for t in sym_trades if t.pnl > 0]
        losses = [t for t in sym_trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in sym_trades)
        win_rate = len(wins) / len(sym_trades) * 100 if sym_trades else 0

        print(f"\n  {sym}:")
        print(f"    Trades: {len(sym_trades)}  |  Wins: {len(wins)}  |  Losses: {len(losses)}  |  Win Rate: {win_rate:.1f}%")
        print(f"    Total PnL: ${total_pnl:,.2f}")
        if wins:
            print(f"    Avg Win:  ${sum(t.pnl for t in wins)/len(wins):,.2f}")
        if losses:
            print(f"    Avg Loss: ${sum(t.pnl for t in losses)/len(losses):,.2f}")
        print(f"    Trades:")
        for t in sym_trades:
            pnl_sign = "+" if t.pnl >= 0 else ""
            print(f"      {t.entry_date.strftime('%m/%d')} {t.direction:4s} "
                  f"@ {t.entry_price:>12,.2f} -> {t.exit_price:>12,.2f}  "
                  f"[{t.exit_reason:12s}]  {pnl_sign}${t.pnl:>10,.2f}  "
                  f"({t.bars_held}d, {t.confidence*100:.0f}%)")

    # ---- Summary ----
    total_pnl = equity - INITIAL_EQUITY
    total_return = (total_pnl / INITIAL_EQUITY) * 100
    total_trades = len(closed_trades)
    winning = [t for t in closed_trades if t.pnl > 0]
    losing  = [t for t in closed_trades if t.pnl <= 0]
    win_rate = len(winning) / total_trades * 100 if total_trades else 0

    # Max drawdown
    peak = INITIAL_EQUITY
    max_dd = 0
    for _, eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_profit = sum(t.pnl for t in winning) if winning else 0
    gross_loss   = abs(sum(t.pnl for t in losing)) if losing else 1

    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"""
  Starting Capital:   ${INITIAL_EQUITY:>12,.2f}
  Final Equity:       ${equity:>12,.2f}
  -----------------------------------------------
  Net P&L:            ${total_pnl:>12,.2f}  ({total_return:+.2f}%)
  -----------------------------------------------
  Total Trades:       {total_trades:>12d}
  Winning Trades:     {len(winning):>12d}
  Losing Trades:      {len(losing):>12d}
  Win Rate:           {win_rate:>11.1f}%
  -----------------------------------------------
  Gross Profit:       ${gross_profit:>12,.2f}
  Gross Loss:         ${gross_loss:>12,.2f}
  Profit Factor:      {gross_profit/gross_loss if gross_loss else float('inf'):>12.2f}
  -----------------------------------------------
  Max Drawdown:       {max_dd:>11.2f}%
  Avg Win:            ${gross_profit/len(winning) if winning else 0:>12,.2f}
  Avg Loss:           ${-gross_loss/len(losing) if losing else 0:>12,.2f}
  -----------------------------------------------
  Period:             {len(trading_days)} trading days
  Signals Evaluated:  {total_signals}
""")

    # ---- Equity curve (ASCII) ----
    print("  EQUITY CURVE")
    print("  " + "-" * 50)
    eq_vals = [eq for _, eq in equity_curve if eq > 0]
    if eq_vals:
        mn, mx = min(eq_vals), max(eq_vals)
        rng = mx - mn if mx != mn else 1
        step = max(1, len(equity_curve) // 25)
        for i in range(0, len(equity_curve), step):
            dt, eq = equity_curve[i]
            bar_len = int((eq - mn) / rng * 40)
            marker = "#" * max(bar_len, 1)
            print(f"  {dt.strftime('%m/%d')} ${eq:>10,.2f} |{marker}")
    print("  " + "-" * 50)

    print("\n  NOTE: This simulation uses the currently trained model.")
    print("  The model was trained on data that includes this period,")
    print("  so results may exhibit look-ahead bias. A true out-of-sample")
    print("  backtest would require retraining the model before Jan 1.\n")

    return {
        "risk": RISK_PER_TRADE,
        "equity": equity,
        "pnl": total_pnl,
        "return": total_return,
        "trades": total_trades,
        "win_rate": win_rate,
        "pf": gross_profit / gross_loss if gross_loss else float('inf'),
        "max_dd": max_dd,
    }


def run_comparison():
    """Run backtests at multiple risk levels for comparison."""
    global RISK_PER_TRADE, MAX_POSITION_PCT

    risk_levels = [0.02, 0.05, 0.10, 0.15]
    results_summary = []

    for risk in risk_levels:
        RISK_PER_TRADE = risk
        MAX_POSITION_PCT = min(0.30 + risk * 5, 0.80)  # scale position cap with risk
        print(f"\n{'#' * 70}")
        print(f"#  RUNNING SIMULATION  -  RISK = {risk*100:.0f}% per trade")
        print(f"{'#' * 70}")
        result = run_backtest()
        if result:
            results_summary.append(result)

    # ---- Side-by-side comparison ----
    if results_summary:
        print("\n" + "=" * 70)
        print("  RISK LEVEL COMPARISON")
        print("=" * 70)
        print(f"  {'Risk':>6s}  {'Final $':>12s}  {'P&L':>10s}  {'Return':>8s}  "
              f"{'Trades':>7s}  {'WinRate':>8s}  {'PF':>6s}  {'MaxDD':>7s}")
        print("  " + "-" * 68)
        for r in results_summary:
            print(f"  {r['risk']*100:>5.0f}%  ${r['equity']:>11,.2f}  "
                  f"${r['pnl']:>9,.2f}  {r['return']:>+7.2f}%  "
                  f"{r['trades']:>7d}  {r['win_rate']:>7.1f}%  "
                  f"{r['pf']:>5.2f}  {r['max_dd']:>6.2f}%")
        print()


if __name__ == "__main__":
    run_comparison()
