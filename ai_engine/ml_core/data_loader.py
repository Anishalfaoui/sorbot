"""
Sorbot AI Engine — Data Loader  v2.0
======================================
Multi-timeframe OHLCV fetcher with smart caching.
Supports 1h, 4h, 1d, 1w candles for each trading pair.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TRADING_PAIRS, DATA_DIR, TIMEFRAMES, PRIMARY_TIMEFRAME

logger = logging.getLogger("sorbot.data_loader")


def _cache_path(symbol: str, timeframe: str) -> Path:
    """Return the CSV cache path for a given symbol + timeframe."""
    return DATA_DIR / f"{symbol}_{timeframe}_ohlcv.csv"


def _resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-hour candles into 4-hour candles."""
    df = df_1h.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()
    return df


def fetch_ohlcv(
    symbol: str,
    timeframe: str = PRIMARY_TIMEFRAME,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV data for *symbol* at the given *timeframe*.

    Parameters
    ----------
    symbol : str
        Internal symbol key, e.g. "BTCUSD", "XAUUSD", "EURUSD".
    timeframe : str
        One of: "1h", "4h", "1d", "1w".
    force_refresh : bool
        Force re-download even if cache exists.

    Returns
    -------
    pd.DataFrame   Columns: Open, High, Low, Close, Volume  (DatetimeIndex).
    """
    pair_cfg = TRADING_PAIRS.get(symbol)
    if pair_cfg is None:
        raise ValueError(f"Unknown symbol '{symbol}'. Available: {list(TRADING_PAIRS.keys())}")

    tf_cfg = TIMEFRAMES.get(timeframe)
    if tf_cfg is None:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Available: {list(TIMEFRAMES.keys())}")

    cache = _cache_path(symbol, timeframe)

    # Determine max cache age: intraday = 1h, daily+ = 12h
    max_age_hours = 1 if timeframe in ("1h", "4h") else 12

    if not force_refresh and cache.exists():
        mod_time = datetime.fromtimestamp(cache.stat().st_mtime)
        age_hours = (datetime.now() - mod_time).total_seconds() / 3600
        if age_hours < max_age_hours:
            logger.info("Cache hit  %s/%s  (%s)", symbol, timeframe, cache.name)
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            return df

    ticker = pair_cfg["yfinance_ticker"]

    # For 4h we download 1h and resample
    if timeframe == "4h":
        raw_interval = "1h"
        raw_period = tf_cfg["period"]
    else:
        raw_interval = tf_cfg["interval"]
        raw_period = tf_cfg["period"]

    logger.info("Downloading %s/%s  (%s  period=%s  interval=%s)",
                symbol, timeframe, ticker, raw_period, raw_interval)

    try:
        df: pd.DataFrame = yf.download(
            ticker, period=raw_period, interval=raw_interval,
            auto_adjust=True, progress=False,
        )
    except Exception as e:
        logger.warning("yf.download failed (%s), retrying with Ticker API...", e)
        t = yf.Ticker(ticker)
        df = t.history(period=raw_period, interval=raw_interval, auto_adjust=True)

    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} ({timeframe})")

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df.dropna(inplace=True)

    # Resample 1h → 4h if needed
    if timeframe == "4h":
        df = _resample_to_4h(df)

    df.to_csv(cache)
    logger.info("Saved %d rows → %s", len(df), cache.name)
    return df


def fetch_multi_timeframe(
    symbol: str,
    timeframes: list[str] | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for a symbol across multiple timeframes.

    Returns
    -------
    dict  {timeframe: DataFrame}  e.g. {"1h": df_1h, "1d": df_1d, "1w": df_1w}
    """
    if timeframes is None:
        timeframes = list(TIMEFRAMES.keys())
    results = {}
    for tf in timeframes:
        try:
            results[tf] = fetch_ohlcv(symbol, timeframe=tf, force_refresh=force_refresh)
        except Exception as exc:
            logger.error("Failed %s/%s: %s", symbol, tf, exc)
    return results


def fetch_all(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch primary-timeframe OHLCV for every active pair."""
    results = {}
    for symbol in TRADING_PAIRS:
        try:
            results[symbol] = fetch_ohlcv(symbol, PRIMARY_TIMEFRAME, force_refresh)
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", symbol, exc)
    return results


def get_latest_row(symbol: str, timeframe: str = PRIMARY_TIMEFRAME) -> Optional[pd.Series]:
    """Return the most recent OHLCV row for *symbol*, or None."""
    df = fetch_ohlcv(symbol, timeframe)
    if df.empty:
        return None
    return df.iloc[-1]


# ── CLI test ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for sym in TRADING_PAIRS:
        data = fetch_multi_timeframe(sym, force_refresh=True)
        for tf, frame in data.items():
            print(f"  {sym}/{tf}  —  {len(frame)} rows  |  latest: {frame.index[-1]}")
