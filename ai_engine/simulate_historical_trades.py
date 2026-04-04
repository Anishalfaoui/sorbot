"""
Sorbot AI Engine — Historical Daily Trade Simulation
====================================================
Simulate exactly N trade attempts per day on historical data, using
point-in-time predictions (no future bars in feature generation).

Default scenario (requested):
- Initial balance: 10,000 USD
- Date range: 2026-03-01 -> today
- 2 trades per day
- Real OHLCV prices from yfinance cache/refresh
- Predictions recomputed for each slot in the past

Outputs:
- CSV trade log in ai_engine/data/
- JSON summary in ai_engine/data/
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import DATA_DIR, LOOKAHEAD_CANDLES, SYMBOLS
from ml_core.data_loader import fetch_all_timeframes
from ml_core.feature_eng import build_dataset
from ml_core.predictor import Predictor
from ml_core.risk_manager import RiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("sorbot.historical_sim")


@dataclass
class TradeResult:
    day: str
    slot_time: str
    symbol: str
    side: str
    confidence_pct: float
    reject_reason: str
    entry_time: str
    entry_price: float
    sl_price: float
    tp_price: float
    qty: float
    notional_usd: float
    exit_time: str
    exit_price: float
    outcome: str
    pnl_usd: float
    balance_after: float


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def _slot_datetimes(day: date, hours: List[int]) -> List[datetime]:
    return [datetime.combine(day, time(h, 0, 0)) for h in hours]


def _first_index_at_or_after(idx: pd.DatetimeIndex, ts: datetime) -> Optional[int]:
    pos = idx.searchsorted(ts)
    if pos >= len(idx):
        return None
    return int(pos)


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
        return default
    except Exception:
        return default


def _prepare_data(symbols: List[str], force_refresh: bool) -> Dict[str, Dict[str, pd.DataFrame]]:
    data_by_symbol: Dict[str, Dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        logger.info("Loading data for %s...", symbol)
        d = fetch_all_timeframes(symbol=symbol, force_refresh=force_refresh)
        if "1h" not in d or d["1h"] is None or d["1h"].empty:
            raise RuntimeError(f"Missing 1h data for {symbol}")

        d["1h"] = _normalize_index(d["1h"]) 
        if "4h" in d and d["4h"] is not None and not d["4h"].empty:
            d["4h"] = _normalize_index(d["4h"]) 
        if "1d" in d and d["1d"] is not None and not d["1d"].empty:
            d["1d"] = _normalize_index(d["1d"]) 

        data_by_symbol[symbol] = d
        logger.info("%s loaded: 1h=%d, 4h=%d, 1d=%d", symbol,
                    len(d.get("1h", [])), len(d.get("4h", [])), len(d.get("1d", [])))
    return data_by_symbol


def _predict_at_time(
    predictor: Predictor,
    symbol: str,
    ts: datetime,
    balance: float,
    data_by_symbol: Dict[str, Dict[str, pd.DataFrame]],
    min_history_bars: int,
) -> Optional[dict]:
    data = data_by_symbol[symbol]
    df_1h = data["1h"]
    hist_1h = df_1h[df_1h.index <= ts]
    if len(hist_1h) < min_history_bars:
        return None

    htf_data = {
        "4h": data.get("4h")[data.get("4h").index <= ts] if data.get("4h") is not None else None,
        "1d": data.get("1d")[data.get("1d").index <= ts] if data.get("1d") is not None else None,
    }

    predictor.load(symbol)
    feats = build_dataset(hist_1h, include_target=False, htf_data=htf_data)
    if feats.empty:
        return None

    pred = predictor.predict_latest(
        dataset=feats,
        ohlcv_1h=hist_1h,
        symbol=symbol,
        virtual_balance=balance,
    )
    return pred


def _resolve_trade(
    df_1h: pd.DataFrame,
    entry_idx: int,
    side: str,
    entry: float,
    sl: float,
    tp: float,
    qty: float,
    max_hold_hours: int,
) -> tuple[str, float, datetime, float]:
    # Start monitoring from the next bar to avoid intrabar ambiguity on entry candle.
    start = entry_idx + 1
    end = min(entry_idx + max_hold_hours, len(df_1h) - 1)

    if start > end:
        # No future bars available -> close immediately at entry.
        return "EXPIRED", entry, df_1h.index[entry_idx], 0.0

    for i in range(start, end + 1):
        bar = df_1h.iloc[i]
        high = _safe_float(bar.get("High"))
        low = _safe_float(bar.get("Low"))

        if side == "LONG":
            sl_hit = low <= sl
            tp_hit = high >= tp
            if sl_hit and tp_hit:
                # Conservative tie-break
                exit_price = sl
                pnl = (exit_price - entry) * qty
                return "SL", exit_price, df_1h.index[i], pnl
            if sl_hit:
                exit_price = sl
                pnl = (exit_price - entry) * qty
                return "SL", exit_price, df_1h.index[i], pnl
            if tp_hit:
                exit_price = tp
                pnl = (exit_price - entry) * qty
                return "TP", exit_price, df_1h.index[i], pnl
        else:
            sl_hit = high >= sl
            tp_hit = low <= tp
            if sl_hit and tp_hit:
                exit_price = sl
                pnl = (entry - exit_price) * qty
                return "SL", exit_price, df_1h.index[i], pnl
            if sl_hit:
                exit_price = sl
                pnl = (entry - exit_price) * qty
                return "SL", exit_price, df_1h.index[i], pnl
            if tp_hit:
                exit_price = tp
                pnl = (entry - exit_price) * qty
                return "TP", exit_price, df_1h.index[i], pnl

    close_price = _safe_float(df_1h.iloc[end].get("Close"), entry)
    if side == "LONG":
        pnl = (close_price - entry) * qty
    else:
        pnl = (entry - close_price) * qty
    return "EXPIRED", close_price, df_1h.index[end], pnl


def run_simulation(
    start_date: date,
    end_date: date,
    initial_balance: float,
    trades_per_day: int,
    slot_hours: List[int],
    symbols: List[str],
    force_refresh: bool,
    min_history_bars: int,
    max_hold_hours: int,
) -> dict:
    predictor = Predictor()
    data_by_symbol = _prepare_data(symbols, force_refresh=force_refresh)

    balance = float(initial_balance)
    trades: List[TradeResult] = []

    day = start_date
    while day <= end_date:
        slots = _slot_datetimes(day, slot_hours)[:trades_per_day]

        for slot_ts in slots:
            candidates = []
            for symbol in symbols:
                try:
                    pred = _predict_at_time(
                        predictor=predictor,
                        symbol=symbol,
                        ts=slot_ts,
                        balance=balance,
                        data_by_symbol=data_by_symbol,
                        min_history_bars=min_history_bars,
                    )
                    if pred is None:
                        continue
                    conf = _safe_float(pred.get("confidence_pct"), 0.0)
                    candidates.append((conf, symbol, pred))
                except Exception as e:
                    logger.debug("Prediction skipped %s @ %s: %s", symbol, slot_ts, e)

            if not candidates:
                continue

            # Pick highest-confidence signal across symbols for this slot.
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, symbol, pred = candidates[0]

            side = str(pred.get("signal", "")).upper()
            if side not in ("LONG", "SHORT"):
                continue

            entry = _safe_float(pred.get("current_price"))
            sl = _safe_float(pred.get("sl_price"))
            tp = _safe_float(pred.get("tp_price"))
            confidence = _safe_float(pred.get("confidence_pct"), 0.0)
            reject_reason = str(pred.get("reject_reason", "") or "")

            if entry <= 0 or sl <= 0 or tp <= 0:
                continue

            rm = RiskManager(balance=balance)
            sizing = rm.calculate_position_size(entry_price=entry, sl_price=sl, signal=side)
            if sizing.get("error"):
                logger.debug("Sizing rejected on %s @ %s: %s", symbol, slot_ts, sizing.get("error"))
                continue

            qty = _safe_float(sizing.get("qty_btc"))
            notional = _safe_float(sizing.get("notional_usd"))
            if qty <= 0 or notional <= 0:
                continue

            df_1h = data_by_symbol[symbol]["1h"]
            entry_idx = _first_index_at_or_after(df_1h.index, slot_ts)
            if entry_idx is None:
                continue

            entry_time = df_1h.index[entry_idx]
            entry_px = _safe_float(df_1h.iloc[entry_idx].get("Close"), entry)
            outcome, exit_px, exit_time, pnl = _resolve_trade(
                df_1h=df_1h,
                entry_idx=entry_idx,
                side=side,
                entry=entry_px,
                sl=sl,
                tp=tp,
                qty=qty,
                max_hold_hours=max_hold_hours,
            )

            balance += pnl

            trades.append(
                TradeResult(
                    day=str(day),
                    slot_time=str(slot_ts),
                    symbol=symbol,
                    side=side,
                    confidence_pct=round(confidence, 2),
                    reject_reason=reject_reason,
                    entry_time=str(entry_time),
                    entry_price=round(entry_px, 5),
                    sl_price=round(sl, 5),
                    tp_price=round(tp, 5),
                    qty=round(qty, 6),
                    notional_usd=round(notional, 2),
                    exit_time=str(exit_time),
                    exit_price=round(exit_px, 5),
                    outcome=outcome,
                    pnl_usd=round(float(pnl), 2),
                    balance_after=round(balance, 2),
                )
            )

        day += timedelta(days=1)

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gross_profit = sum(t.pnl_usd for t in wins)
    gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 0.0

    summary = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "initial_balance": round(initial_balance, 2),
        "final_balance": round(balance, 2),
        "total_pnl": round(balance - initial_balance, 2),
        "total_return_pct": round(((balance / initial_balance) - 1.0) * 100.0, 2) if initial_balance > 0 else 0.0,
        "trades_per_day_target": trades_per_day,
        "slots": slot_hours,
        "symbols": symbols,
        "trades_count": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round((len(wins) / len(trades) * 100.0), 2) if trades else 0.0,
        "avg_win": round(float(np.mean([t.pnl_usd for t in wins])), 2) if wins else 0.0,
        "avg_loss": round(float(np.mean([t.pnl_usd for t in losses])), 2) if losses else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else None,
    }

    return {
        "summary": summary,
        "trades": [t.__dict__ for t in trades],
    }


def _write_outputs(result: dict, output_prefix: str) -> tuple[Path, Path]:
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / f"{output_prefix}_trades.csv"
    json_path = DATA_DIR / f"{output_prefix}_summary.json"

    df = pd.DataFrame(result["trades"])
    if not df.empty:
        df.to_csv(csv_path, index=False)
    else:
        pd.DataFrame(columns=[
            "day", "slot_time", "symbol", "side", "confidence_pct", "reject_reason",
            "entry_time", "entry_price", "sl_price", "tp_price", "qty", "notional_usd",
            "exit_time", "exit_price", "outcome", "pnl_usd", "balance_after",
        ]).to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2)

    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    today = date.today()
    parser = argparse.ArgumentParser(description="Simulate 2 daily historical trades from past predictions")
    parser.add_argument("--start", default="2026-03-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=today.strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--trades-per-day", type=int, default=2)
    parser.add_argument("--slot-hours", default="10,18", help="Comma-separated UTC hours for daily entries")
    parser.add_argument("--symbols", default=",".join(SYMBOLS.keys()), help="Comma-separated symbols")
    parser.add_argument("--min-history-bars", type=int, default=500)
    parser.add_argument("--max-hold-hours", type=int, default=max(LOOKAHEAD_CANDLES * 3, 12))
    parser.add_argument("--force-refresh", action="store_true", help="Force yfinance refresh before sim")
    parser.add_argument("--output-prefix", default="historical_sim_2026_present")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)
    if end_date < start_date:
        raise ValueError("--end must be >= --start")

    slot_hours = [int(x.strip()) for x in args.slot_hours.split(",") if x.strip()]
    if not slot_hours:
        raise ValueError("--slot-hours cannot be empty")

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    unsupported = [s for s in symbols if s not in SYMBOLS]
    if unsupported:
        raise ValueError(f"Unsupported symbols: {unsupported}. Allowed: {list(SYMBOLS.keys())}")

    logger.info(
        "Running historical simulation: %s -> %s | initial=%.2f | trades/day=%d | slots=%s | symbols=%s",
        start_date,
        end_date,
        args.initial_balance,
        args.trades_per_day,
        slot_hours,
        symbols,
    )

    result = run_simulation(
        start_date=start_date,
        end_date=end_date,
        initial_balance=args.initial_balance,
        trades_per_day=args.trades_per_day,
        slot_hours=slot_hours,
        symbols=symbols,
        force_refresh=args.force_refresh,
        min_history_bars=args.min_history_bars,
        max_hold_hours=args.max_hold_hours,
    )

    csv_path, json_path = _write_outputs(result, args.output_prefix)
    summary = result["summary"]

    print("\n" + "=" * 64)
    print("HISTORICAL DAILY TRADE SIMULATION")
    print("=" * 64)
    print(f"Period:          {summary['start_date']} -> {summary['end_date']}")
    print(f"Initial balance: ${summary['initial_balance']:.2f}")
    print(f"Final balance:   ${summary['final_balance']:.2f}")
    print(f"Total PnL:       ${summary['total_pnl']:.2f} ({summary['total_return_pct']:+.2f}%)")
    print(f"Trades:          {summary['trades_count']} (wins={summary['wins']} losses={summary['losses']})")
    print(f"Win rate:        {summary['win_rate_pct']:.2f}%")
    print(f"Profit factor:   {summary['profit_factor']}")
    print(f"CSV output:      {csv_path}")
    print(f"JSON output:     {json_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
