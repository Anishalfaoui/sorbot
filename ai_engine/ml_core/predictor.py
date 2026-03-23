"""
Sorbot AI Engine v3.0 — Predictor (Enhanced)
===============================================
Enriched predictions with:
  - Full market analysis (regime, trend, indicators)
  - Multi-timeframe alignment assessment
  - Support / Resistance levels
  - Indicator-by-indicator breakdown
  - Human-readable CONCLUSION message
  - Trade-level SL/TP with ATR-based risk
"""

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_DIR, DEFAULT_SYMBOL, SYMBOLS,
    CONFIDENCE_LONG, CONFIDENCE_SHORT,
    SL_ATR_MULT, TP_ATR_MULT, MIN_RR_RATIO,
)

from ml_core.feature_eng import get_atr, get_market_analysis

logger = logging.getLogger("sorbot.predictor")


def _normalize_symbol(symbol: str) -> str:
    key = (symbol or DEFAULT_SYMBOL).upper().replace("/", "")
    if key not in SYMBOLS:
        raise ValueError(f"Unsupported symbol '{symbol}'. Allowed: {', '.join(SYMBOLS.keys())}")
    return key


def _get_model_paths(symbol: str):
    s = _normalize_symbol(symbol)
    cfg = SYMBOLS[s]
    model_file = MODEL_DIR / cfg["model_file"]
    meta_file = MODEL_DIR / cfg["meta_file"]
    if model_file.exists() and meta_file.exists():
        return model_file, meta_file

    # Backward-compatibility for legacy BTC model names
    if cfg.get("legacy_model_file") and cfg.get("legacy_meta_file"):
        legacy_model = MODEL_DIR / cfg["legacy_model_file"]
        legacy_meta = MODEL_DIR / cfg["legacy_meta_file"]
        if legacy_model.exists() and legacy_meta.exists():
            return legacy_model, legacy_meta

    return model_file, meta_file


class Predictor:
    """
    Load a trained XGBoost native Booster and produce enriched predictions.
    Singleton usage: instantiate once, call predict_latest() per request.
    """

    def __init__(self):
        self._booster = None
        self._meta = None
        self._loaded = False
        self._loaded_symbol = None

    # ── Load model ───────────────────────────────

    def load(self, symbol: str = DEFAULT_SYMBOL):
        symbol = _normalize_symbol(symbol)
        if self._loaded and self._loaded_symbol == symbol:
            return

        model_file, meta_file = _get_model_paths(symbol)
        if not model_file.exists():
            raise FileNotFoundError(f"No model at {model_file}. Train first.")
        self._booster = xgb.Booster()
        self._booster.load_model(str(model_file))
        with open(meta_file) as f:
            self._meta = json.load(f)
        self._loaded = True
        self._loaded_symbol = symbol
        n_feat = self._meta.get("n_features", "?")
        logger.info("Model loaded for %s: %s features, trained %s", symbol, n_feat, self._meta.get("trained_at", "?"))

    # ── Raw predict ──────────────────────────────

    def predict(self, X: np.ndarray, feature_names: list) -> np.ndarray:
        """Return P(UP) for each row."""
        dmat = xgb.DMatrix(X, feature_names=feature_names)
        return self._booster.predict(dmat)

    # ── Enriched latest prediction ───────────────

    def predict_latest(
        self,
        dataset: pd.DataFrame,
        ohlcv_1h: pd.DataFrame,
        symbol: str = DEFAULT_SYMBOL,
        virtual_balance: float = 10000.0,
    ) -> dict:
        """
        Predict latest bar with ENRICHED context.

        Args:
            dataset: Feature matrix (no target col) from build_dataset
            ohlcv_1h: Raw 1h OHLCV for market analysis

        Returns:
            dict with signal, confidence, market analysis, and conclusion
        """
        symbol = _normalize_symbol(symbol)
        symbol_label = SYMBOLS[symbol]["label"]

        feature_cols = [c for c in dataset.columns if c != "target"]
        trained_features = self._meta.get("feature_names", feature_cols)

        # Align features: use only those the model was trained on
        available = [f for f in trained_features if f in feature_cols]
        missing = [f for f in trained_features if f not in feature_cols]
        if missing:
            logger.warning("Missing %d features: %s", len(missing), missing[:10])
            # Fill missing with 0 so DMatrix shape matches
            for m in missing:
                dataset[m] = 0.0
            available = trained_features

        row = dataset[available].iloc[[-1]]
        X = row.values

        prob_up = float(self.predict(X, available)[0])
        prob_down = 1 - prob_up

        # Price & ATR
        close = ohlcv_1h["Close"]
        high = ohlcv_1h["High"]
        low = ohlcv_1h["Low"]

        current_price = float(close.iloc[-1])

        atr_val = float(get_atr(high, low, close).iloc[-1])
        atr_pct = atr_val / current_price

        # Signal determination
        # Always output a direction (LONG/SHORT); lower confidence is surfaced as a warning.
        signal = "LONG" if prob_up >= 0.5 else "SHORT"
        reject_reason = None

        if signal == "LONG":
            sl_price = round(current_price - SL_ATR_MULT * atr_val, 2)
            tp_price = round(current_price + TP_ATR_MULT * atr_val, 2)
        else:
            sl_price = round(current_price + SL_ATR_MULT * atr_val, 2)
            tp_price = round(current_price - TP_ATR_MULT * atr_val, 2)

        if CONFIDENCE_SHORT < prob_up < CONFIDENCE_LONG:
            reject_reason = (
                f"Low-confidence directional call ({prob_up:.1%}); "
                f"outside strong-confidence zone ({CONFIDENCE_SHORT:.0%}-{CONFIDENCE_LONG:.0%})."
            )

        # R:R check
        if signal in ("LONG", "SHORT") and sl_price and tp_price:
            risk = abs(current_price - sl_price)
            reward = abs(tp_price - current_price)
            rr = reward / risk if risk > 0 else 0
            if rr < MIN_RR_RATIO:
                rr_warning = f"R:R ratio {rr:.2f} below preferred minimum {MIN_RR_RATIO}."
                reject_reason = f"{reject_reason} {rr_warning}".strip() if reject_reason else rr_warning

        # ── Market Analysis ──────────────────────
        try:
            market = get_market_analysis(ohlcv_1h)
        except Exception as e:
            logger.warning("Market analysis error: %s", e)
            market = {"error": str(e)}

        # ── Multi-TF Alignment ───────────────────
        # Check HTF features that were used in prediction
        htf_alignment = _assess_htf_alignment(dataset.iloc[-1], feature_cols)

        # ── Conclusion Message ───────────────────
        conclusion = _build_conclusion(
            signal=signal,
            symbol_label=symbol_label,
            prob_up=prob_up,
            current_price=current_price,
            market=market,
            htf_alignment=htf_alignment,
            atr_val=atr_val,
            sl_price=sl_price,
            tp_price=tp_price,
            reject_reason=reject_reason,
        )

        # ── Build Response ───────────────────────
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol_label,
            "signal": signal,
            "probability_up": round(prob_up, 4),
            "probability_down": round(prob_down, 4),
            "confidence_pct": round(max(prob_up, prob_down) * 100, 1),
            "current_price": current_price,
            "atr": round(atr_val, 2),
            "atr_pct": round(atr_pct * 100, 3),
        }

        result["sl_price"] = sl_price
        result["tp_price"] = tp_price
        risk = abs(current_price - sl_price)
        reward = abs(tp_price - current_price)
        result["risk_reward"] = round(reward / risk, 2) if risk > 0 else 0
        result["risk_usd"] = round(risk, 2)
        result["reward_usd"] = round(reward, 2)

        # Add estimated position sizing so user sees capital usage BEFORE accepting
        try:
            from ml_core.risk_manager import RiskManager
            _rm = RiskManager(balance=max(float(virtual_balance or 0), 0.0))
            sizing_est = _rm.calculate_position_size(
                entry_price=current_price,
                sl_price=sl_price,
                signal=signal,
            )
            if not sizing_est.get("error"):
                result["est_qty_btc"] = sizing_est["qty_btc"]
                result["est_notional_usd"] = sizing_est["notional_usd"]
                result["est_risk_usd"] = sizing_est["risk_usd"]
                result["est_capital_used_pct"] = sizing_est["capital_used_pct"]
                result["est_balance"] = sizing_est["balance"]
            else:
                result["sizing_warning"] = sizing_est["error"]
        except Exception as e:
            logger.warning("Could not estimate position size: %s", e)

        if reject_reason:
            result["reject_reason"] = reject_reason

        result["market_analysis"] = market
        result["htf_alignment"] = htf_alignment
        result["conclusion"] = conclusion

        return result

    # ── Model info ───────────────────────────────

    def get_model_info(self) -> dict:
        if not self._meta:
            return {}
        return {
            "trained_at": self._meta.get("trained_at"),
            "n_samples": self._meta.get("n_samples"),
            "n_features": self._meta.get("n_features"),
            "cv_metrics": self._meta.get("cv_metrics"),
            "final_metrics": self._meta.get("final_metrics"),
            "best_iteration": self._meta.get("best_iteration"),
            "top_features": self._meta.get("top_features", [])[:15],
        }


# ──────────────────────────────────────────────
#  HTF ALIGNMENT ANALYSIS
# ──────────────────────────────────────────────

def _assess_htf_alignment(features_row: pd.Series, feature_cols: list) -> dict:
    """
    Assess multi-timeframe alignment from HTF features embedded in the dataset.
    Returns bullish/bearish/neutral alignment per timeframe.
    """
    alignment = {}

    for tf in ["4h", "1d"]:
        prefix = f"htf_{tf}_"
        relevant = {k: v for k, v in features_row.items() if k.startswith(prefix)}
        if not relevant:
            continue

        bullish = 0
        bearish = 0

        # RSI
        rsi = relevant.get(f"{prefix}rsi")
        if rsi is not None:
            if rsi > 0.55:
                bullish += 1
            elif rsi < 0.45:
                bearish += 1

        # EMA cross
        ema_x = relevant.get(f"{prefix}ema_cross")
        if ema_x is not None:
            if ema_x > 0:
                bullish += 1
            else:
                bearish += 1

        # MACD histogram
        macd_h = relevant.get(f"{prefix}macd_hist")
        if macd_h is not None:
            if macd_h > 0:
                bullish += 1
            else:
                bearish += 1

        # Trend (above EMA50)
        trend = relevant.get(f"{prefix}trend")
        if trend is not None:
            if trend > 0.5:
                bullish += 1
            else:
                bearish += 1

        # ADX (trend strength)
        adx = relevant.get(f"{prefix}adx")
        adx_str = "WEAK"
        if adx is not None:
            if adx > 0.25:
                adx_str = "STRONG"
            elif adx > 0.20:
                adx_str = "MODERATE"

        # BB position
        bb = relevant.get(f"{prefix}bb_pctb")

        total = bullish + bearish
        if total == 0:
            tf_bias = "NEUTRAL"
        elif bullish > bearish:
            tf_bias = "BULLISH"
        elif bearish > bullish:
            tf_bias = "BEARISH"
        else:
            tf_bias = "NEUTRAL"

        alignment[tf] = {
            "bias": tf_bias,
            "bullish_signals": bullish,
            "bearish_signals": bearish,
            "trend_strength": adx_str,
            "rsi": round(float(rsi * 100), 1) if rsi is not None else None,
        }

    # Overall alignment
    biases = [v["bias"] for v in alignment.values()]
    if all(b == "BULLISH" for b in biases):
        alignment["overall"] = "ALL_BULLISH"
    elif all(b == "BEARISH" for b in biases):
        alignment["overall"] = "ALL_BEARISH"
    elif "BULLISH" in biases and "BEARISH" in biases:
        alignment["overall"] = "MIXED"
    else:
        alignment["overall"] = "NEUTRAL"

    return alignment


# ──────────────────────────────────────────────
#  CONCLUSION MESSAGE BUILDER
# ──────────────────────────────────────────────

def _build_conclusion(
    signal: str,
    symbol_label: str,
    prob_up: float,
    current_price: float,
    market: dict,
    htf_alignment: dict,
    atr_val: float,
    sl_price: float = None,
    tp_price: float = None,
    reject_reason: str = None,
) -> str:
    """
    Build a human-readable conclusion paragraph summarizing the prediction.
    """
    lines = []

    # Header
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"=== SORBOT {symbol_label} ANALYSIS — {ts} ===")
    lines.append("")

    # Price & confidence
    conf_pct = max(prob_up, 1 - prob_up) * 100
    lines.append(f"BTC is currently trading at ${current_price:,.2f}.")
    lines.append(f"Model confidence: {conf_pct:.1f}% (P(UP)={prob_up:.1%}, P(DOWN)={1-prob_up:.1%}).")
    lines.append("")

    # Market state
    if isinstance(market, dict) and "error" not in market:
        regime = market.get("market_regime", "UNKNOWN")
        trend = market.get("trend_direction", "UNKNOWN")
        indicators = market.get("indicators", {})
        rsi = indicators.get("rsi", 0)
        rsi_zone = indicators.get("rsi_zone", "")
        adx = indicators.get("adx", 0)
        adx_interp = indicators.get("adx_interpretation", "")
        squeeze = indicators.get("is_squeeze", False)
        vol_ratio = indicators.get("volume_ratio", 1.0)
        macd_sig = indicators.get("macd_signal", "")
        stoch_zone = indicators.get("stoch_zone", "")
        mfi = indicators.get("mfi", 50)
        cci = indicators.get("cci", 0)
        williams = indicators.get("williams_r", -50)

        lines.append(f"MARKET STATE: {regime} | TREND: {trend}")
        lines.append(f"  RSI: {rsi:.1f} ({rsi_zone}) | ADX: {adx:.1f} ({adx_interp})")
        lines.append(f"  MACD: {macd_sig} | Stochastic: {stoch_zone} | MFI: {mfi:.1f}")
        lines.append(f"  CCI: {cci:.1f} | Williams %R: {williams:.1f}")
        lines.append(f"  Volume: {vol_ratio:.1f}x average | Squeeze: {'YES' if squeeze else 'NO'}")

        # Divergences
        divs = market.get("divergences", {})
        rsi_div = divs.get("rsi_divergence", "NONE")
        macd_div = divs.get("macd_divergence", "NONE")
        if rsi_div != "NONE" or macd_div != "NONE":
            lines.append(f"  DIVERGENCES: RSI={rsi_div}, MACD={macd_div}")

        # Support/Resistance
        structure = market.get("structure", {})
        if structure:
            lines.append(f"  S/R LEVELS: R2=${structure.get('resistance_2', 0):,.2f} | R1=${structure.get('resistance_1', 0):,.2f} | "
                         f"Pivot=${structure.get('pivot', 0):,.2f} | S1=${structure.get('support_1', 0):,.2f} | S2=${structure.get('support_2', 0):,.2f}")

        # Signal score
        score = market.get("signal_score", {})
        bull_s = score.get("bullish_signals", 0)
        bear_s = score.get("bearish_signals", 0)
        bull_pct = score.get("bullish_pct", 50)
        bear_pct = score.get("bearish_pct", 50)
        lines.append(f"  SIGNAL SCORE: {bull_s} bullish ({bull_pct}%) vs {bear_s} bearish ({bear_pct}%)")
        lines.append("")

    # HTF alignment
    if htf_alignment:
        overall = htf_alignment.get("overall", "UNKNOWN")
        lines.append(f"MULTI-TIMEFRAME ALIGNMENT: {overall}")
        for tf in ["4h", "1d"]:
            tf_info = htf_alignment.get(tf, {})
            if tf_info:
                lines.append(
                    f"  {tf.upper()}: {tf_info.get('bias', '?')} "
                    f"(Bull:{tf_info.get('bullish_signals', 0)} Bear:{tf_info.get('bearish_signals', 0)} "
                    f"TrendStr:{tf_info.get('trend_strength', '?')})"
                )
        lines.append("")

    # Decision
    if signal == "NO_TRADE":
        lines.append(f"DECISION: NO TRADE")
        if reject_reason:
            lines.append(f"  Reason: {reject_reason}")
        lines.append("")
        lines.append("The AI model does not detect a high-conviction setup at this time. "
                      "The probability is too close to 50/50 or the risk/reward profile is unfavorable. "
                      "Patience is key — waiting for clearer signals preserves capital.")
    elif signal == "LONG":
        risk = abs(current_price - sl_price)
        reward = abs(tp_price - current_price)
        rr = reward / risk if risk > 0 else 0
        lines.append(f"DECISION: LONG (BUY)")
        lines.append(f"  Entry: ${current_price:,.2f}")
        lines.append(f"  Stop Loss: ${sl_price:,.2f} ({SL_ATR_MULT}x ATR = -${risk:,.2f})")
        lines.append(f"  Take Profit: ${tp_price:,.2f} ({TP_ATR_MULT}x ATR = +${reward:,.2f})")
        lines.append(f"  Risk:Reward = 1:{rr:.1f}")
        lines.append("")
        lines.append(f"The AI model sees a BULLISH opportunity with {prob_up:.1%} probability of upward movement. "
                      f"Price is expected to rise from ${current_price:,.2f} toward ${tp_price:,.2f} within "
                      f"the next few hours. ATR volatility is ${atr_val:.2f} ({atr_val/current_price*100:.2f}%). "
                      f"Trade with proper risk management - risk only what you can afford to lose.")
    elif signal == "SHORT":
        risk = abs(current_price - sl_price)
        reward = abs(tp_price - current_price)
        rr = reward / risk if risk > 0 else 0
        lines.append(f"DECISION: SHORT (SELL)")
        lines.append(f"  Entry: ${current_price:,.2f}")
        lines.append(f"  Stop Loss: ${sl_price:,.2f} ({SL_ATR_MULT}x ATR = -${risk:,.2f})")
        lines.append(f"  Take Profit: ${tp_price:,.2f} ({TP_ATR_MULT}x ATR = +${reward:,.2f})")
        lines.append(f"  Risk:Reward = 1:{rr:.1f}")
        lines.append("")
        lines.append(f"The AI model sees a BEARISH opportunity with {1-prob_up:.1%} probability of downward movement. "
                      f"A virtual short trade can be opened in the paper account environment. "
                      f"ATR volatility is ${atr_val:.2f} ({atr_val/current_price*100:.2f}%). "
                      f"Follow risk limits and position sizing discipline.")

    lines.append("")
    lines.append("--- This is an AI-generated analysis. Not financial advice. Trade at your own risk. ---")

    return "\n".join(lines)
