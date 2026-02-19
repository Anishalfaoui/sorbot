"""
Sorbot AI Engine — Feature Engineering  v2.0
===============================================
Multi-timeframe feature computation with support/resistance detection,
advanced indicators, and smart-money features.

Features  (50+ columns)
------------------------
Per-timeframe core features:
  RSI, MACD (line/signal/hist), Bollinger %B & BW, ATR, EMA cross (9/21),
  EMA-200 distance, Stochastic %K/%D, ROC-5/10/20, Volume ratio, OBV-norm,
  Candle body/upper/lower shadow ratios, ADX, Ichimoku signals

Higher-timeframe confluence:
  HTF trend direction (EMA cross), HTF RSI, HTF MACD hist for each
  confluence timeframe — aligned to the primary timeframe index.

Structural features:
  Distance to nearest support, distance to nearest resistance,
  number of nearby S/R levels, S/R zone strength.

Calendar:
  Day of week, month, hour (for intraday).

Target:
  Forward N-day return buckets: 0=SELL, 1=HOLD, 2=BUY
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BOLLINGER_PERIOD,
    BOLLINGER_STD,
    ATR_PERIOD,
    EMA_SHORT,
    EMA_LONG,
    EMA_200,
    VOLUME_SMA_PERIOD,
    LOOK_AHEAD_DAYS,
    SR_LOOKBACK,
    SR_TOUCH_THRESHOLD,
    SR_PROXIMITY_PCT,
)

logger = logging.getLogger("sorbot.feature_eng")

# ──────────────────────────────────────────────────────────
#  INDICATOR  HELPERS
# ──────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series):
    ema_fast = series.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = series.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(series: pd.Series):
    sma = series.rolling(BOLLINGER_PERIOD).mean()
    std = series.rolling(BOLLINGER_PERIOD).std()
    upper = sma + BOLLINGER_STD * std
    lower = sma - BOLLINGER_STD * std
    pct_b = (series - lower) / (upper - lower)
    bandwidth = (upper - lower) / sma
    return pct_b, bandwidth


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _atr_raw(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    """Return ATR in absolute price units (not normalised)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _stochastic(high, low, close, k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return k, d


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index — trend strength 0-100."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = _atr_raw(high, low, close, period)
    plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(period).mean()
    return adx


def _ichimoku_signals(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Compute Ichimoku Cloud signals: tenkan/kijun cross, price vs cloud.
    Returns a small DataFrame with 3 columns.
    """
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    ichi = pd.DataFrame(index=close.index)
    ichi["ichi_tk_cross"] = (tenkan - kijun) / close   # normalised cross
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    ichi["ichi_cloud_dist"] = (close - cloud_top) / close  # >0 = above cloud
    ichi["ichi_cloud_thick"] = (cloud_top - cloud_bot) / close
    return ichi


# ──────────────────────────────────────────────────────────
#  SUPPORT / RESISTANCE
# ──────────────────────────────────────────────────────────

def detect_sr_levels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = SR_LOOKBACK,
    touch_threshold: int = SR_TOUCH_THRESHOLD,
    proximity_pct: float = SR_PROXIMITY_PCT,
) -> list[dict]:
    """
    Find horizontal support / resistance levels from recent price action.

    Returns list of dicts:  {"price": float, "type": "support"|"resistance", "touches": int, "strength": float}
    """
    recent_high = high.tail(lookback)
    recent_low = low.tail(lookback)
    current = close.iloc[-1]

    # Candidate levels = pivot highs & lows
    candidate_prices = []
    for i in range(2, len(recent_high) - 2):
        h_vals = recent_high.iloc[i - 2 : i + 3]
        if recent_high.iloc[i] == h_vals.max():
            candidate_prices.append(recent_high.iloc[i])
        l_vals = recent_low.iloc[i - 2 : i + 3]
        if recent_low.iloc[i] == l_vals.min():
            candidate_prices.append(recent_low.iloc[i])

    if not candidate_prices:
        return []

    # Cluster nearby prices
    candidate_prices = sorted(set(candidate_prices))
    clusters: list[list[float]] = [[candidate_prices[0]]]
    for p in candidate_prices[1:]:
        if (p - clusters[-1][-1]) / clusters[-1][-1] < proximity_pct * 2:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    levels = []
    for cluster in clusters:
        level_price = np.mean(cluster)
        # Count how many candles "touched" this level
        touches = int(((recent_high - level_price).abs() / level_price < proximity_pct).sum() +
                      ((recent_low - level_price).abs() / level_price < proximity_pct).sum())
        if touches >= touch_threshold:
            sr_type = "support" if level_price < current else "resistance"
            levels.append({
                "price": round(float(level_price), 6),
                "type": sr_type,
                "touches": touches,
                "strength": round(touches / lookback, 4),
            })

    levels.sort(key=lambda x: abs(x["price"] - current))
    return levels[:10]  # top-10 nearest levels


def _sr_features(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Add rolling S/R distance features for each row (last SR_LOOKBACK window)."""
    out = pd.DataFrame(index=close.index)

    # Compute once for the last portion to avoid O(n²) over the whole series
    levels = detect_sr_levels(high, low, close)

    current = close.iloc[-1]
    supports = [l for l in levels if l["type"] == "support"]
    resistances = [l for l in levels if l["type"] == "resistance"]

    nearest_sup = supports[0]["price"] if supports else current * 0.99
    nearest_res = resistances[0]["price"] if resistances else current * 1.01

    # Fill last few rows; for training these will be static features (acceptable)
    out["sr_dist_support"] = (close - nearest_sup) / close
    out["sr_dist_resistance"] = (nearest_res - close) / close
    out["sr_num_levels"] = len(levels)
    return out


# ──────────────────────────────────────────────────────────
#  CORE SINGLE-TIMEFRAME FEATURES
# ──────────────────────────────────────────────────────────

def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 35+ features from a single OHLCV dataframe.
    Does NOT include target or HTF confluence.
    """
    out = pd.DataFrame(index=df.index)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    opn = df["Open"]
    vol = df.get("Volume", pd.Series(0, index=df.index))

    # ── Trend ──────────────────────────────────
    out["rsi"] = _rsi(close)

    macd_line, macd_sig, macd_hist = _macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist

    ema_s = close.ewm(span=EMA_SHORT, adjust=False).mean()
    ema_l = close.ewm(span=EMA_LONG, adjust=False).mean()
    ema_200 = close.ewm(span=EMA_200, adjust=False).mean()
    out["ema_cross"] = (ema_s - ema_l) / close
    out["ema200_dist"] = (close - ema_200) / close   # >0 = above 200 EMA

    # ── Volatility ─────────────────────────────
    pct_b, bw = _bollinger(close)
    out["bb_pctb"] = pct_b
    out["bb_bandwidth"] = bw
    out["atr"] = _atr(high, low, close) / close

    # ── Momentum ───────────────────────────────
    out["roc_5"] = close.pct_change(5)
    out["roc_10"] = close.pct_change(10)
    out["roc_20"] = close.pct_change(20)

    k, d = _stochastic(high, low, close)
    out["stoch_k"] = k
    out["stoch_d"] = d

    # ── ADX — trend strength ───────────────────
    out["adx"] = _adx(high, low, close)

    # ── Ichimoku ───────────────────────────────
    ichi = _ichimoku_signals(high, low, close)
    out = pd.concat([out, ichi], axis=1)

    # ── Volume ─────────────────────────────────
    has_volume = vol.sum() > 0
    if has_volume:
        vol_sma = vol.rolling(VOLUME_SMA_PERIOD).mean()
        out["vol_ratio"] = vol / vol_sma.replace(0, np.nan)
        out["obv_norm"] = _obv(close, vol) / vol.expanding().sum().replace(0, np.nan)
    else:
        out["vol_ratio"] = 1.0
        out["obv_norm"] = 0.0

    # ── Candle Anatomy ─────────────────────────
    body = (close - opn).abs()
    total_range = (high - low).replace(0, np.nan)
    out["body_ratio"] = body / total_range
    out["upper_shadow"] = (high - pd.concat([close, opn], axis=1).max(axis=1)) / total_range
    out["lower_shadow"] = (pd.concat([close, opn], axis=1).min(axis=1) - low) / total_range
    out["bullish_candle"] = (close > opn).astype(int)

    # ── Calendar ───────────────────────────────
    out["day_of_week"] = df.index.dayofweek
    out["month"] = df.index.month
    if hasattr(df.index, "hour"):
        out["hour"] = df.index.hour

    # ── Returns lags ───────────────────────────
    for lag in [1, 2, 3, 5, 10]:
        out[f"return_lag_{lag}"] = close.pct_change(lag)

    # ── Support / Resistance features ──────────
    sr_feats = _sr_features(high, low, close)
    out = pd.concat([out, sr_feats], axis=1)

    return out


# ──────────────────────────────────────────────────────────
#  HIGHER-TIMEFRAME CONFLUENCE FEATURES
# ──────────────────────────────────────────────────────────

def build_htf_features(
    primary_df: pd.DataFrame,
    htf_dataframes: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    For each higher timeframe, compute trend/momentum summaries and
    align them to the primary timeframe index using forward-fill.

    Parameters
    ----------
    primary_df : pd.DataFrame
        Primary-timeframe OHLCV (the one we train on).
    htf_dataframes : dict[str, pd.DataFrame]
        {timeframe_key: OHLCV DataFrame}

    Returns
    -------
    pd.DataFrame   index = primary_df.index, with HTF columns prefixed.
    """
    out = pd.DataFrame(index=primary_df.index)

    # Normalise primary index timezone for comparison
    primary_idx = primary_df.index
    if primary_idx.tz is not None:
        primary_idx = primary_idx.tz_localize(None)

    for tf_key, htf_df in htf_dataframes.items():
        if htf_df is None or htf_df.empty:
            continue

        prefix = f"htf_{tf_key}_"
        # Strip timezone from HTF index if present
        htf_index = htf_df.index
        if htf_index.tz is not None:
            htf_index = htf_index.tz_localize(None)

        htf_df_clean = htf_df.copy()
        htf_df_clean.index = htf_index

        close = htf_df_clean["Close"]
        high = htf_df_clean["High"]
        low = htf_df_clean["Low"]

        htf = pd.DataFrame(index=htf_index)
        htf[prefix + "rsi"] = _rsi(close)

        ema_s = close.ewm(span=EMA_SHORT, adjust=False).mean()
        ema_l = close.ewm(span=EMA_LONG, adjust=False).mean()
        htf[prefix + "ema_cross"] = (ema_s - ema_l) / close

        _, _, macd_h = _macd(close)
        htf[prefix + "macd_hist"] = macd_h

        htf[prefix + "adx"] = _adx(high, low, close)
        htf[prefix + "atr"] = _atr(high, low, close) / close

        # Align to primary index (forward fill)
        htf = htf.reindex(primary_idx, method="ffill")
        htf.index = primary_df.index  # restore original index
        out = pd.concat([out, htf], axis=1)

    return out


# ──────────────────────────────────────────────────────────
#  FULL PIPELINE  (for training)
# ──────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    include_target: bool = True,
    htf_dataframes: Optional[dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Produce the complete feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Primary-timeframe OHLCV with columns Open, High, Low, Close, Volume.
    include_target : bool
        If True, append target column (0=SELL, 1=HOLD, 2=BUY).
    htf_dataframes : dict | None
        Optional higher-TF OHLCV dict for confluence features.

    Returns
    -------
    pd.DataFrame   cleaned feature matrix.
    """
    # Base features on primary TF
    out = build_base_features(df)

    # Higher-timeframe confluence
    if htf_dataframes:
        htf = build_htf_features(df, htf_dataframes)
        out = pd.concat([out, htf], axis=1)

    # ── Target label ───────────────────────────
    if include_target:
        close = df["Close"]
        future_return = close.shift(-LOOK_AHEAD_DAYS) / close - 1
        out["target"] = np.where(
            future_return > 0.01, 2,
            np.where(future_return < -0.01, 0, 1)
        )

    # ── Clean-up ───────────────────────────────
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill HTF NaN (warm-up period) with 0 so we don't lose rows
    htf_cols = [c for c in out.columns if c.startswith("htf_")]
    if htf_cols:
        out[htf_cols] = out[htf_cols].fillna(0)

    out.dropna(inplace=True)

    logger.info("Built %d features × %d rows  (HTF=%s)",
                out.shape[1] - (1 if include_target else 0),
                out.shape[0],
                list(htf_dataframes.keys()) if htf_dataframes else "none")
    return out


def get_feature_names(include_target: bool = False) -> list[str]:
    """Return the ordered list of feature column names (base only)."""
    dummy = pd.DataFrame({
        "Open": [1.0]*250, "High": [2.0]*250, "Low": [0.5]*250,
        "Close": [1.5]*250, "Volume": [100.0]*250,
    }, index=pd.date_range("2020-01-01", periods=250))
    feats = build_features(dummy, include_target=include_target)
    return list(feats.columns)


# ── Quick CLI test ─────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_loader import fetch_ohlcv

    for sym in ["BTCUSD", "XAUUSD", "EURUSD"]:
        raw = fetch_ohlcv(sym)
        feats = build_features(raw)
        print(f"\n{sym}: {feats.shape}")
        print(feats.tail(3))
