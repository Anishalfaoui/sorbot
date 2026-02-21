"""
Sorbot AI Engine v3.0 — Data Loader (Extended)
=================================================
Fetches BTC/USD OHLCV data from yfinance.
Multi-timeframe: 1h (primary), 4h (resampled), 1d (context).

CHUNKED DOWNLOAD: yfinance limits 1h data to ~60 days per request.
We download in 59-day chunks going back up to 730 days (2 years),
then stitch together for a much larger training set (~17,000 bars).
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import YFINANCE_TICKER, DATA_DIR, TF_CONFIG, PRIMARY_TIMEFRAME

logger = logging.getLogger("sorbot.data_loader")


def _cache_path(timeframe: str) -> Path:
    return DATA_DIR / f"BTCUSD_{timeframe}_ohlcv.csv"


def _resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h candles to 4h."""
    return df_1h.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()


def _download_chunked(interval: str, max_days: int, chunk_days: int) -> pd.DataFrame:
    """
    Download intraday data in chunks to overcome yfinance's 60-day limit.
    Stitches multiple requests together, going back up to max_days.
    Returns ~17,000 rows for 730 days of 1h data.
    """
    all_frames = []
    now = datetime.utcnow()
    # yfinance 'end' is EXCLUSIVE — use tomorrow to include today's candles
    end_dt = now + timedelta(days=1)
    start_limit = now - timedelta(days=max_days)
    current_end = end_dt
    attempts = 0
    max_attempts = (max_days // chunk_days) + 3

    while current_end > start_limit and attempts < max_attempts:
        attempts += 1
        chunk_start = max(current_end - timedelta(days=chunk_days), start_limit)
        start_str = chunk_start.strftime("%Y-%m-%d")
        end_str = current_end.strftime("%Y-%m-%d")

        logger.info("  Chunk %d: %s -> %s", attempts, start_str, end_str)
        try:
            df = yf.download(
                YFINANCE_TICKER,
                start=start_str, end=end_str,
                interval=interval,
                auto_adjust=True, progress=False,
            )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                all_frames.append(df)
                logger.info("    -> %d rows", len(df))
        except Exception as e:
            logger.warning("  Chunk %d failed: %s", attempts, e)

        # Move window backward
        current_end = chunk_start - timedelta(days=1)

    if not all_frames:
        raise RuntimeError("No data received from chunked download")

    # Combine, sort, deduplicate
    combined = pd.concat(all_frames, axis=0)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)
    logger.info("Chunked download complete: %d total rows from %d chunks",
                len(combined), len(all_frames))
    return combined


def fetch_ohlcv(
    timeframe: str = PRIMARY_TIMEFRAME,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download BTC/USD OHLCV data for a given timeframe.

    For 1h data: downloads up to 730 days in chunks (~17k bars).
    For 1d data: downloads up to 5 years (~1800 bars).
    """
    tf_cfg = TF_CONFIG.get(timeframe)
    if tf_cfg is None:
        raise ValueError(f"Unknown timeframe '{timeframe}'")

    cache = _cache_path(timeframe)

    # Cache TTL: 1h for intraday, 12h for daily
    max_age_hours = 1 if timeframe in ("1h", "4h") else 12

    if not force_refresh and cache.exists():
        mod_time = datetime.fromtimestamp(cache.stat().st_mtime)
        age_hours = (datetime.now() - mod_time).total_seconds() / 3600
        if age_hours < max_age_hours:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            logger.info("Cache hit BTCUSD/%s (%d rows)", timeframe, len(df))
            return df

    # Chunked download for intraday data (1h, 4h)
    if "max_days" in tf_cfg:
        raw_interval = tf_cfg["interval"]
        max_days = tf_cfg["max_days"]
        chunk_days = tf_cfg["chunk_days"]
        logger.info("Downloading BTCUSD/%s chunked (interval=%s, %d days back)",
                    timeframe, raw_interval, max_days)
        df = _download_chunked(raw_interval, max_days, chunk_days)
    else:
        # Daily data - simple period download
        raw_interval = tf_cfg["interval"]
        raw_period = tf_cfg["period"]
        logger.info("Downloading BTCUSD/%s (interval=%s, period=%s)",
                    timeframe, raw_interval, raw_period)
        try:
            df = yf.download(
                YFINANCE_TICKER, period=raw_period, interval=raw_interval,
                auto_adjust=True, progress=False,
            )
        except Exception as e:
            logger.warning("yf.download failed (%s), trying Ticker API", e)
            t = yf.Ticker(YFINANCE_TICKER)
            df = t.history(period=raw_period, interval=raw_interval, auto_adjust=True)

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for BTCUSD/{timeframe}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df.dropna(inplace=True)

    if timeframe == "4h":
        df = _resample_to_4h(df)

    df.to_csv(cache)
    logger.info("Saved %d rows -> %s", len(df), cache.name)
    return df


def fetch_all_timeframes(force_refresh: bool = False) -> dict:
    """Fetch 1h, 4h, 1d data for BTC/USD."""
    result = {}
    for tf in ["1h", "4h", "1d"]:
        try:
            result[tf] = fetch_ohlcv(tf, force_refresh=force_refresh)
        except Exception as e:
            logger.error("Failed BTCUSD/%s: %s", tf, e)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    data = fetch_all_timeframes(force_refresh=True)
    for tf, df in data.items():
        print(f"{tf}: {len(df)} rows  ({df.index[0]} -> {df.index[-1]})")
