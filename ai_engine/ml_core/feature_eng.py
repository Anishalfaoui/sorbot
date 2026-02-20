"""
Sorbot AI Engine v3.0 — Feature Engineering (Enhanced)
========================================================
100+ technical features for BTC/USD across multiple categories:

  TREND:       EMA(9,21,50,200), SMA(50,200), EMA cross, VWAP, Ichimoku
  MOMENTUM:    RSI, MACD, Stochastic, ROC, ADX, Williams %R, CCI, MFI
  VOLATILITY:  ATR, Bollinger Bands, Keltner Channel, Squeeze, historical vol
  VOLUME:      Volume ratio, OBV, VWAP deviation, accumulation/distribution
  STRUCTURE:   Support/Resistance, pivot points, higher highs/lower lows
  CANDLE:      Body ratio, shadows, patterns, engulfing, hammer/doji
  REGIME:      Market regime detection (trending vs ranging)
  DIVERGENCE:  RSI divergence, MACD divergence (price vs indicator)
  CALENDAR:    Hour, day-of-week, session (Asia/EU/US)
  HTF:         4h/1d trend signals (RSI, MACD, EMA, ADX, ATR, trend)

All features are strictly lagged (no future data leakage).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, ATR_PERIOD,
    EMA_FAST, EMA_SLOW, EMA_TREND, EMA_200,
    SMA_50, SMA_200, STOCH_K, STOCH_D, ADX_PERIOD, VOLUME_MA,
    LOOKAHEAD_CANDLES, UP_THRESHOLD, DOWN_THRESHOLD,
)

logger = logging.getLogger("sorbot.features")


# ──────────────────────────────────────────────
#  INDICATOR HELPERS (all return Series)
# ──────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def _rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series):
    ema_f = _ema(series, MACD_FAST)
    ema_s = _ema(series, MACD_SLOW)
    line = ema_f - ema_s
    signal = _ema(line, MACD_SIGNAL)
    hist = line - signal
    return line, signal, hist


def _bollinger(series: pd.Series):
    mid = _sma(series, BB_PERIOD)
    std = series.rolling(BB_PERIOD).std()
    upper = mid + BB_STD * std
    lower = mid - BB_STD * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    return pct_b, bandwidth, upper, lower


def _atr(high, low, close, period: int = ATR_PERIOD) -> pd.Series:
    """ATR in absolute price units."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _stochastic(high, low, close, k_period=STOCH_K, d_period=STOCH_D):
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def _adx(high, low, close, period=ADX_PERIOD):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_val = _atr(high, low, close, period)
    plus_di = 100 * plus_dm.rolling(period).mean() / atr_val.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(period).mean() / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean(), plus_di, minus_di


def _obv(close, volume) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _historical_volatility(close, window=20) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(24 * 365)


def _williams_r(high, low, close, period=14) -> pd.Series:
    """Williams %R oscillator (-100 to 0)."""
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def _cci(high, low, close, period=20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def _mfi(high, low, close, volume, period=14) -> pd.Series:
    """Money Flow Index (volume-weighted RSI)."""
    tp = (high + low + close) / 3
    mf = tp * volume
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0).rolling(period).sum()
    neg_mf = mf.where(delta <= 0, 0).rolling(period).sum()
    ratio = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def _keltner_channel(high, low, close, ema_period=20, atr_period=14, atr_mult=1.5):
    """Keltner Channel: EMA +/- ATR multiplier."""
    mid = _ema(close, ema_period)
    atr_val = _atr(high, low, close, atr_period)
    upper = mid + atr_mult * atr_val
    lower = mid - atr_mult * atr_val
    return upper, mid, lower


def _vwap(high, low, close, volume) -> pd.Series:
    """Rolling session VWAP approximation (24h rolling)."""
    tp = (high + low + close) / 3
    cum_tp_vol = (tp * volume).rolling(24).sum()
    cum_vol = volume.rolling(24).sum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def _accumulation_distribution(high, low, close, volume) -> pd.Series:
    """Accumulation/Distribution Line."""
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (clv * volume).cumsum()


def _detect_divergence(price, indicator, lookback=14) -> pd.Series:
    """
    Detect bearish/bullish divergence between price and indicator.
    Returns: +1 = bullish divergence, -1 = bearish divergence, 0 = none.
    """
    result = pd.Series(0, index=price.index)
    price_high = price.rolling(lookback).max()
    price_low = price.rolling(lookback).min()
    ind_high = indicator.rolling(lookback).max()
    ind_low = indicator.rolling(lookback).min()

    # Bearish: price makes new high but indicator doesn't
    bearish = (price >= price_high.shift(1)) & (indicator < ind_high.shift(1))
    # Bullish: price makes new low but indicator doesn't
    bullish = (price <= price_low.shift(1)) & (indicator > ind_low.shift(1))

    result[bearish] = -1
    result[bullish] = 1
    return result


def _ichimoku(high, low, close):
    """Ichimoku Cloud components."""
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(-26)
    return tenkan, kijun, senkou_a, senkou_b, chikou


def _pivot_points(high, low, close):
    """Classic pivot points from previous day (rolling 24h for hourly)."""
    h24 = high.rolling(24).max()
    l24 = low.rolling(24).min()
    c24 = close.shift(1)  # previous bar close as proxy
    pivot = (h24 + l24 + c24) / 3
    r1 = 2 * pivot - l24
    s1 = 2 * pivot - h24
    r2 = pivot + (h24 - l24)
    s2 = pivot - (h24 - l24)
    return pivot, r1, s1, r2, s2


# ──────────────────────────────────────────────
#  MAIN FEATURE BUILDER (100+ features)
# ──────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features from an OHLCV DataFrame (primary timeframe).
    Returns a DataFrame with feature columns only (no target).
    All features are strictly lagged - no future data.
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    opn = df["Open"]
    vol = df["Volume"]

    # ═══════════════════════════════════════════
    #  TREND FEATURES (20 features)
    # ═══════════════════════════════════════════
    ema9 = _ema(close, EMA_FAST)
    ema21 = _ema(close, EMA_SLOW)
    ema50 = _ema(close, EMA_TREND)
    ema200 = _ema(close, EMA_200)

    out["ema9_dist"] = (close - ema9) / close
    out["ema21_dist"] = (close - ema21) / close
    out["ema50_dist"] = (close - ema50) / close
    out["ema200_dist"] = (close - ema200) / close
    out["ema_cross_9_21"] = (ema9 - ema21) / close
    out["ema_cross_21_50"] = (ema21 - ema50) / close
    out["ema_cross_50_200"] = (ema50 - ema200) / close
    out["ema_slope_9"] = ema9.pct_change(5)
    out["ema_slope_21"] = ema21.pct_change(5)
    out["ema_slope_50"] = ema50.pct_change(10)

    sma50 = _sma(close, SMA_50)
    sma200 = _sma(close, SMA_200)
    out["sma50_dist"] = (close - sma50) / close
    out["sma200_dist"] = (close - sma200) / close
    out["golden_cross"] = (sma50 > sma200).astype(int)

    # VWAP
    vwap = _vwap(high, low, close, vol)
    out["vwap_dist"] = (close - vwap) / close
    out["above_vwap"] = (close > vwap).astype(int)

    # Ichimoku
    tenkan, kijun, senkou_a, senkou_b, _ = _ichimoku(high, low, close)
    out["ichi_tk_cross"] = ((tenkan - kijun) / close).fillna(0)
    out["ichi_cloud_dist"] = ((close - senkou_a) / close).fillna(0)
    out["ichi_above_cloud"] = ((close > senkou_a) & (close > senkou_b)).astype(int).fillna(0)

    # ═══════════════════════════════════════════
    #  MOMENTUM FEATURES (25 features)
    # ═══════════════════════════════════════════
    rsi = _rsi(close)
    out["rsi"] = rsi / 100.0
    out["rsi_slope"] = rsi.diff(3)
    out["rsi_oversold"] = (rsi < 30).astype(int)
    out["rsi_overbought"] = (rsi > 70).astype(int)

    macd_line, macd_sig, macd_hist = _macd(close)
    out["macd_line"] = macd_line / close
    out["macd_signal"] = macd_sig / close
    out["macd_hist"] = macd_hist / close
    out["macd_cross"] = (macd_line > macd_sig).astype(int)
    out["macd_hist_slope"] = macd_hist.diff(3) / close

    k, d = _stochastic(high, low, close)
    out["stoch_k"] = k / 100.0
    out["stoch_d"] = d / 100.0
    out["stoch_cross"] = (k > d).astype(int)
    out["stoch_oversold"] = (k < 20).astype(int)
    out["stoch_overbought"] = (k > 80).astype(int)

    adx, plus_di, minus_di = _adx(high, low, close)
    out["adx"] = adx / 100.0
    out["adx_strong"] = (adx > 25).astype(int)
    out["di_cross"] = ((plus_di - minus_di) / 100.0).fillna(0)

    # Williams %R
    wr = _williams_r(high, low, close)
    out["williams_r"] = wr / 100.0

    # CCI
    cci = _cci(high, low, close)
    out["cci"] = cci / 200.0  # normalize roughly to -1..+1

    # MFI (volume-weighted RSI)
    mfi = _mfi(high, low, close, vol)
    out["mfi"] = mfi / 100.0

    # Rate of Change at multiple periods
    for period in [6, 12, 24, 48]:
        out[f"roc_{period}"] = close.pct_change(period)

    # RSI divergence
    out["rsi_divergence"] = _detect_divergence(close, rsi, lookback=14)

    # MACD divergence
    out["macd_divergence"] = _detect_divergence(close, macd_hist, lookback=14)

    # ═══════════════════════════════════════════
    #  VOLATILITY FEATURES (15 features)
    # ═══════════════════════════════════════════
    atr_abs = _atr(high, low, close)
    out["atr_pct"] = atr_abs / close
    out["atr_change"] = atr_abs.pct_change(5)
    out["atr_ratio_20_50"] = _atr(high, low, close, 20) / _atr(high, low, close, 50).replace(0, np.nan)

    bb_pctb, bb_bw, bb_upper, bb_lower = _bollinger(close)
    out["bb_pctb"] = bb_pctb
    out["bb_bandwidth"] = bb_bw

    # Keltner Channel
    kc_upper, kc_mid, kc_lower = _keltner_channel(high, low, close)
    out["kc_position"] = (close - kc_lower) / (kc_upper - kc_lower).replace(0, np.nan)

    # Squeeze indicator: BB inside Keltner = low volatility squeeze
    squeeze = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).fillna(False)
    out["squeeze"] = squeeze.astype(int)
    out["squeeze_release"] = (squeeze.shift(1).fillna(False).astype(int) - squeeze.astype(int)).clip(lower=0)

    # Historical volatility multi-window
    out["hvol_10"] = _historical_volatility(close, 10)
    out["hvol_20"] = _historical_volatility(close, 20)
    out["hvol_50"] = _historical_volatility(close, 50)
    out["hvol_ratio"] = _historical_volatility(close, 10) / _historical_volatility(close, 50).replace(0, np.nan)

    # Intrabar volatility
    out["intrabar_vol"] = (high - low) / close

    # ═══════════════════════════════════════════
    #  VOLUME FEATURES (10 features)
    # ═══════════════════════════════════════════
    vol_ma = _sma(vol, VOLUME_MA)
    out["vol_ratio"] = vol / vol_ma.replace(0, np.nan)
    out["vol_spike"] = (vol > vol_ma * 2).astype(int)
    out["vol_trend"] = vol_ma.pct_change(10)

    obv = _obv(close, vol)
    obv_norm = obv / vol.expanding().sum().replace(0, np.nan)
    out["obv_norm"] = obv_norm
    out["obv_slope"] = obv.diff(5) / vol_ma.replace(0, np.nan)

    # VWAP deviation
    out["vwap_dev"] = ((close - vwap) / atr_abs).fillna(0)

    # A/D line
    ad = _accumulation_distribution(high, low, close, vol)
    ad_norm = ad / vol.expanding().sum().replace(0, np.nan)
    out["ad_norm"] = ad_norm
    out["ad_slope"] = ad.diff(5) / vol_ma.replace(0, np.nan)

    # Volume-price confirmation
    out["vol_price_confirm"] = (
        ((close > close.shift(1)) & (vol > vol.shift(1))).astype(int)
        - ((close < close.shift(1)) & (vol > vol.shift(1))).astype(int)
    )

    # ═══════════════════════════════════════════
    #  RETURNS (lagged) - 6 features
    # ═══════════════════════════════════════════
    for lag in [1, 2, 3, 5, 10, 20]:
        out[f"ret_{lag}"] = close.pct_change(lag)

    # ═══════════════════════════════════════════
    #  CANDLE ANATOMY & PATTERNS (12 features)
    # ═══════════════════════════════════════════
    body = (close - opn).abs()
    rng = (high - low).replace(0, np.nan)
    out["body_ratio"] = body / rng
    out["upper_shadow"] = (high - pd.concat([close, opn], axis=1).max(axis=1)) / rng
    out["lower_shadow"] = (pd.concat([close, opn], axis=1).min(axis=1) - low) / rng
    out["bullish_candle"] = (close > opn).astype(int)
    out["doji"] = (body / rng < 0.1).astype(int)

    # Engulfing pattern
    prev_body = (close.shift(1) - opn.shift(1)).abs()
    bullish_engulf = (close > opn) & (body > prev_body) & (close.shift(1) < opn.shift(1))
    bearish_engulf = (close < opn) & (body > prev_body) & (close.shift(1) > opn.shift(1))
    out["engulfing"] = bullish_engulf.astype(int) - bearish_engulf.astype(int)

    # Hammer / shooting star
    small_body = body / rng < 0.3
    long_lower = (pd.concat([close, opn], axis=1).min(axis=1) - low) / rng > 0.6
    long_upper = (high - pd.concat([close, opn], axis=1).max(axis=1)) / rng > 0.6
    out["hammer"] = (small_body & long_lower).astype(int)
    out["shooting_star"] = (small_body & long_upper).astype(int)

    # Consecutive direction
    direction = (close > close.shift(1)).astype(int)
    consec_up = direction.groupby((direction != direction.shift()).cumsum()).cumsum()
    consec_down = (1 - direction).groupby(((1 - direction) != (1 - direction).shift()).cumsum()).cumsum()
    out["consec_up"] = consec_up
    out["consec_down"] = consec_down

    # 3-bar pattern (higher-high-higher-close or lower-low-lower-close)
    out["hh_hc"] = ((high > high.shift(1)) & (close > close.shift(1)) &
                     (high.shift(1) > high.shift(2)) & (close.shift(1) > close.shift(2))).astype(int)
    out["ll_lc"] = ((low < low.shift(1)) & (close < close.shift(1)) &
                     (low.shift(1) < low.shift(2)) & (close.shift(1) < close.shift(2))).astype(int)

    # ═══════════════════════════════════════════
    #  STRUCTURE: S/R & PIVOT POINTS (7 features)
    # ═══════════════════════════════════════════
    pivot, r1, s1, r2, s2 = _pivot_points(high, low, close)
    out["pivot_dist"] = (close - pivot) / close
    out["r1_dist"] = (r1 - close) / close
    out["s1_dist"] = (close - s1) / close
    out["r2_dist"] = (r2 - close) / close
    out["s2_dist"] = (close - s2) / close

    # Price position within ranges
    roll_high_24 = high.rolling(24).max()
    roll_low_24 = low.rolling(24).min()
    out["price_pos_24h"] = (close - roll_low_24) / (roll_high_24 - roll_low_24).replace(0, np.nan)

    roll_high_7d = high.rolling(168).max()
    roll_low_7d = low.rolling(168).min()
    out["price_pos_7d"] = (close - roll_low_7d) / (roll_high_7d - roll_low_7d).replace(0, np.nan)

    # ═══════════════════════════════════════════
    #  MARKET REGIME DETECTION (5 features)
    # ═══════════════════════════════════════════
    # Regime based on ADX + volatility
    out["regime_trending"] = ((adx > 25) & (out["hvol_20"] > out["hvol_50"])).astype(int)
    out["regime_ranging"] = ((adx < 20) & (out["squeeze"] == 1)).astype(int)
    out["regime_volatile"] = ((out["hvol_ratio"] > 1.5) & (out["atr_change"] > 0.1)).astype(int)

    # Trend strength composite (normalized -1 to +1)
    trend_signals = (
        out["ema_cross_9_21"].clip(-0.01, 0.01) / 0.01 * 0.3 +
        out["macd_cross"].astype(float) * 0.2 +
        (out["rsi"] - 0.5) * 0.2 +
        out["golden_cross"].astype(float) * 0.15 +
        out["above_vwap"].astype(float) * 0.15
    )
    out["trend_strength"] = trend_signals.clip(-1, 1)

    # Mean reversion signal
    out["mean_reversion"] = (
        ((out["bb_pctb"] < 0.1) & (out["rsi"] < 0.3)).astype(int)
        - ((out["bb_pctb"] > 0.9) & (out["rsi"] > 0.7)).astype(int)
    )

    # ═══════════════════════════════════════════
    #  CALENDAR FEATURES (8 features)
    # ═══════════════════════════════════════════
    if hasattr(df.index, "hour"):
        out["hour"] = df.index.hour
        out["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        # Trading session: 0=Asia(0-8), 1=Europe(8-16), 2=US(16-24)
        hour = df.index.hour
        out["session_asia"] = ((hour >= 0) & (hour < 8)).astype(int)
        out["session_europe"] = ((hour >= 8) & (hour < 16)).astype(int)
        out["session_us"] = ((hour >= 16) & (hour < 24)).astype(int)

    out["dow"] = df.index.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    out["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    return out


# ──────────────────────────────────────────────
#  HTF CONFLUENCE FEATURES (enhanced)
# ──────────────────────────────────────────────

def build_htf_features(
    primary_df: pd.DataFrame,
    htf_data: dict,
) -> pd.DataFrame:
    """
    Build higher-timeframe trend signals aligned to the primary index.
    Enhanced: RSI, MACD, EMA cross, ADX, ATR, trend, BB, momentum.
    """
    out = pd.DataFrame(index=primary_df.index)

    primary_idx = primary_df.index
    if primary_idx.tz is not None:
        primary_idx = primary_idx.tz_localize(None)

    for tf_key, htf_df in htf_data.items():
        if htf_df is None or htf_df.empty or len(htf_df) < 50:
            continue

        prefix = f"htf_{tf_key}_"

        htf_idx = htf_df.index
        if htf_idx.tz is not None:
            htf_idx = htf_idx.tz_localize(None)
        htf_df = htf_df.copy()
        htf_df.index = htf_idx

        close = htf_df["Close"]
        high = htf_df["High"]
        low = htf_df["Low"]

        htf = pd.DataFrame(index=htf_idx)

        # RSI
        htf_rsi = _rsi(close)
        htf[prefix + "rsi"] = htf_rsi / 100.0

        # EMA cross
        ema_s = _ema(close, EMA_FAST)
        ema_l = _ema(close, EMA_SLOW)
        htf[prefix + "ema_cross"] = (ema_s - ema_l) / close

        # MACD
        _, _, macd_h = _macd(close)
        htf[prefix + "macd_hist"] = macd_h / close

        # ADX
        adx_val, _, _ = _adx(high, low, close)
        htf[prefix + "adx"] = adx_val / 100.0

        # ATR
        htf[prefix + "atr_pct"] = _atr(high, low, close) / close

        # Trend (above EMA50)
        htf[prefix + "trend"] = (close > _ema(close, EMA_TREND)).astype(int)

        # BB position
        bb_pctb, _, _, _ = _bollinger(close)
        htf[prefix + "bb_pctb"] = bb_pctb

        # Momentum (return over 5 bars)
        htf[prefix + "momentum"] = close.pct_change(5)

        # Align to primary index (forward fill)
        htf = htf.reindex(primary_idx, method="ffill")
        htf.index = primary_df.index
        out = pd.concat([out, htf], axis=1)

    return out


# ──────────────────────────────────────────────
#  TARGET BUILDER
# ──────────────────────────────────────────────

def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Binary target: 1 = price goes UP by >= UP_THRESHOLD in next N candles
                   0 = price goes DOWN by <= DOWN_THRESHOLD
                   NaN = flat (excluded from training)
    """
    close = df["Close"]
    future_ret = close.shift(-LOOKAHEAD_CANDLES) / close - 1

    target = pd.Series(np.nan, index=df.index, name="target")
    target[future_ret >= UP_THRESHOLD] = 1.0
    target[future_ret <= DOWN_THRESHOLD] = 0.0
    return target


# ──────────────────────────────────────────────
#  FULL PIPELINE
# ──────────────────────────────────────────────

def build_dataset(
    df: pd.DataFrame,
    include_target: bool = True,
    htf_data: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build complete feature matrix + optional target.
    Drops NaN rows from indicator warm-up.
    """
    features = build_features(df)

    if htf_data:
        htf_feats = build_htf_features(df, htf_data)
        features = pd.concat([features, htf_feats], axis=1)

    if include_target:
        features["target"] = build_target(df)

    # Clean up
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill HTF NaN (warm-up) with 0
    htf_cols = [c for c in features.columns if c.startswith("htf_")]
    if htf_cols:
        features[htf_cols] = features[htf_cols].fillna(0)

    features.dropna(inplace=True)

    n_feats = features.shape[1] - (1 if include_target else 0)
    logger.info("Dataset: %d rows x %d features", len(features), n_feats)

    return features


# ── ATR accessor for external use ──────────────
def get_atr(high, low, close, period=ATR_PERIOD):
    """Return ATR in absolute price units."""
    return _atr(high, low, close, period)


# ── Market analysis for prediction context ─────
def get_market_analysis(df: pd.DataFrame) -> dict:
    """
    Extract current market state for enriched prediction response.
    Returns human-readable analysis of all major indicators.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]
    price = float(close.iloc[-1])

    # RSI
    rsi_val = float(_rsi(close).iloc[-1])

    # MACD
    macd_line, macd_sig, macd_hist = _macd(close)
    macd_hist_val = float(macd_hist.iloc[-1])
    macd_cross_bullish = float(macd_line.iloc[-1]) > float(macd_sig.iloc[-1])

    # Stochastic
    k, d = _stochastic(high, low, close)
    stoch_k_val = float(k.iloc[-1])

    # ADX
    adx_val, plus_di, minus_di = _adx(high, low, close)
    adx_now = float(adx_val.iloc[-1])
    plus_di_now = float(plus_di.iloc[-1])
    minus_di_now = float(minus_di.iloc[-1])

    # Bollinger
    bb_pctb, bb_bw, _, _ = _bollinger(close)
    bb_pctb_val = float(bb_pctb.iloc[-1])
    bb_bw_val = float(bb_bw.iloc[-1])

    # EMAs
    ema9 = float(_ema(close, EMA_FAST).iloc[-1])
    ema21 = float(_ema(close, EMA_SLOW).iloc[-1])
    ema50 = float(_ema(close, EMA_TREND).iloc[-1])
    ema200 = float(_ema(close, EMA_200).iloc[-1])

    # ATR
    atr_val = float(_atr(high, low, close).iloc[-1])
    atr_pct = atr_val / price * 100

    # Volume
    vol_ma = float(_sma(vol, VOLUME_MA).iloc[-1])
    vol_ratio = float(vol.iloc[-1]) / vol_ma if vol_ma > 0 else 1.0

    # Williams %R
    wr_val = float(_williams_r(high, low, close).iloc[-1])

    # CCI
    cci_val = float(_cci(high, low, close).iloc[-1])

    # MFI
    mfi_val = float(_mfi(high, low, close, vol).iloc[-1])

    # Squeeze
    kc_upper, _, kc_lower = _keltner_channel(high, low, close)
    _, _, bb_upper, bb_lower = _bollinger(close)
    is_squeeze = bool((bb_lower.iloc[-1] > kc_lower.iloc[-1]) and (bb_upper.iloc[-1] < kc_upper.iloc[-1]))

    # Trend determination
    trend_direction = "BULLISH" if price > ema50 and ema9 > ema21 else (
        "BEARISH" if price < ema50 and ema9 < ema21 else "NEUTRAL"
    )

    # Market regime
    if adx_now > 25 and not is_squeeze:
        regime = "TRENDING"
    elif is_squeeze:
        regime = "SQUEEZE (low volatility, breakout imminent)"
    elif adx_now < 20:
        regime = "RANGING"
    else:
        regime = "TRANSITIONING"

    # Support / Resistance
    pivot, r1, s1, r2, s2 = _pivot_points(high, low, close)
    support_1 = float(s1.iloc[-1])
    resistance_1 = float(r1.iloc[-1])
    support_2 = float(s2.iloc[-1])
    resistance_2 = float(r2.iloc[-1])

    # RSI divergence
    rsi_div = int(_detect_divergence(close, _rsi(close)).iloc[-1])
    macd_div = int(_detect_divergence(close, macd_hist).iloc[-1])

    # Scoring: count bullish/bearish signals
    bullish_count = 0
    bearish_count = 0

    if rsi_val < 30: bullish_count += 1
    if rsi_val > 70: bearish_count += 1
    if macd_cross_bullish: bullish_count += 1
    else: bearish_count += 1
    if stoch_k_val < 20: bullish_count += 1
    if stoch_k_val > 80: bearish_count += 1
    if price > ema50: bullish_count += 1
    else: bearish_count += 1
    if price > ema200: bullish_count += 1
    else: bearish_count += 1
    if plus_di_now > minus_di_now: bullish_count += 1
    else: bearish_count += 1
    if mfi_val < 30: bullish_count += 1
    if mfi_val > 70: bearish_count += 1
    if rsi_div == 1: bullish_count += 1
    if rsi_div == -1: bearish_count += 1
    if wr_val > -20: bearish_count += 1
    if wr_val < -80: bullish_count += 1

    total_signals = bullish_count + bearish_count
    bull_pct = round(bullish_count / max(total_signals, 1) * 100, 1)
    bear_pct = round(bearish_count / max(total_signals, 1) * 100, 1)

    return {
        "price": round(price, 2),
        "trend_direction": trend_direction,
        "market_regime": regime,
        "indicators": {
            "rsi": round(rsi_val, 2),
            "rsi_zone": "OVERSOLD" if rsi_val < 30 else ("OVERBOUGHT" if rsi_val > 70 else "NEUTRAL"),
            "macd_histogram": round(macd_hist_val, 2),
            "macd_signal": "BULLISH" if macd_cross_bullish else "BEARISH",
            "stochastic_k": round(stoch_k_val, 2),
            "stoch_zone": "OVERSOLD" if stoch_k_val < 20 else ("OVERBOUGHT" if stoch_k_val > 80 else "NEUTRAL"),
            "adx": round(adx_now, 2),
            "adx_interpretation": "STRONG TREND" if adx_now > 25 else ("WEAK/NO TREND" if adx_now < 20 else "BUILDING"),
            "williams_r": round(wr_val, 2),
            "cci": round(cci_val, 2),
            "mfi": round(mfi_val, 2),
            "bollinger_pct_b": round(bb_pctb_val, 3),
            "bollinger_bandwidth": round(bb_bw_val, 4),
            "is_squeeze": is_squeeze,
            "volume_ratio": round(vol_ratio, 2),
        },
        "emas": {
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "ema50": round(ema50, 2),
            "ema200": round(ema200, 2),
        },
        "volatility": {
            "atr": round(atr_val, 2),
            "atr_pct": round(atr_pct, 3),
        },
        "structure": {
            "resistance_2": round(resistance_2, 2),
            "resistance_1": round(resistance_1, 2),
            "pivot": round(float(pivot.iloc[-1]), 2),
            "support_1": round(support_1, 2),
            "support_2": round(support_2, 2),
        },
        "divergences": {
            "rsi_divergence": {1: "BULLISH", -1: "BEARISH", 0: "NONE"}.get(rsi_div, "NONE"),
            "macd_divergence": {1: "BULLISH", -1: "BEARISH", 0: "NONE"}.get(macd_div, "NONE"),
        },
        "signal_score": {
            "bullish_signals": bullish_count,
            "bearish_signals": bearish_count,
            "bullish_pct": bull_pct,
            "bearish_pct": bear_pct,
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_loader import fetch_ohlcv
    df = fetch_ohlcv("1h")
    feats = build_dataset(df, include_target=True)
    print(f"\nFeatures: {feats.shape}")
    print(f"Target distribution:\n{feats['target'].value_counts()}")
    print(f"\nColumns ({len([c for c in feats.columns if c != 'target'])} features):")
    for c in sorted(feats.columns):
        if c != "target":
            print(f"  {c}")
