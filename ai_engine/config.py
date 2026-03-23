"""
Sorbot AI Engine v3.0 — Central Configuration
=============================================
Multi-symbol prediction engine (BTC/USD, EUR/USD, XAU/USD)
Paper-trading defaults for virtual accounts.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# SYMBOLS
# ──────────────────────────────────────────────
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "BTCUSD").upper()

SYMBOLS = {
    "BTCUSD": {
        "label": "BTC/USD",
        "yfinance": "BTC-USD",
        "decimals": 2,
        "model_file": "BTCUSD_xgb.json",
        "meta_file": "BTCUSD_meta.json",
        "legacy_model_file": "btc_model.json",
        "legacy_meta_file": "btc_meta.json",
    },
    "EURUSD": {
        "label": "EUR/USD",
        "yfinance": "EURUSD=X",
        "decimals": 5,
        "model_file": "EURUSD_xgb.json",
        "meta_file": "EURUSD_meta.json",
        "legacy_model_file": None,
        "legacy_meta_file": None,
    },
    "XAUUSD": {
        "label": "XAU/USD",
        "yfinance": "GC=F",
        "decimals": 2,
        "model_file": "XAUUSD_xgb.json",
        "meta_file": "XAUUSD_meta.json",
        "legacy_model_file": None,
        "legacy_meta_file": None,
    },
}

# ──────────────────────────────────────────────
# TIMEFRAMES
# ──────────────────────────────────────────────
PRIMARY_TIMEFRAME = "1h"         # main signal timeframe
HTF_TIMEFRAME = "4h"             # higher-TF confluence
DAILY_TIMEFRAME = "1d"           # daily context

# yfinance download settings per TF
# 1h data: download in 59-day chunks going back up to 730 days
TF_CONFIG = {
    "1h":  {"interval": "1h",  "max_days": 730,  "chunk_days": 59},
    "4h":  {"interval": "1h",  "max_days": 730,  "chunk_days": 59},
    "1d":  {"interval": "1d",  "period": "5y"},
}

# ──────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 50
EMA_200 = 200
SMA_50 = 50
SMA_200 = 200
STOCH_K = 14
STOCH_D = 3
ADX_PERIOD = 14
VOLUME_MA = 20

# ──────────────────────────────────────────────
# TARGET ENGINEERING
# ──────────────────────────────────────────────
# Predict: will price go UP by at least THRESHOLD in next N candles?
LOOKAHEAD_CANDLES = 3            # 3 x 1h = 3 hours ahead
UP_THRESHOLD = 0.003             # +0.3% = classified as UP
DOWN_THRESHOLD = -0.003          # -0.3% = classified as DOWN
# Binary: 1 = UP, 0 = DOWN  (skip FLAT periods in training)

# ──────────────────────────────────────────────
# XGBOOST HYPERPARAMETERS  (tuned for stability)
# ──────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators": 800,
    "max_depth": 5,
    "learning_rate": 0.01,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "gamma": 0.3,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "scale_pos_weight": 1.0,     # adjusted dynamically during training
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}

# Walk-forward validation
WF_N_SPLITS = 5                  # number of walk-forward folds
WF_TEST_SIZE = 200               # 200 hours test per fold
EARLY_STOPPING_ROUNDS = 50

# ──────────────────────────────────────────────
# PREDICTION & TRADE RULES
# ──────────────────────────────────────────────
# Only trade when model confidence exceeds this threshold
CONFIDENCE_LONG = 0.65           # probability > 65% = go LONG
CONFIDENCE_SHORT = 0.35          # probability < 35% = go SHORT
# Between 0.35 and 0.65 = NO TRADE (uncertain zone)

# ──────────────────────────────────────────────
# RISK MANAGEMENT (virtual account defaults)
# ──────────────────────────────────────────────
ACCOUNT_BALANCE = 10000.0        # default virtual balance per user in USD
RISK_PER_TRADE = 0.015           # risk 1.5% per trade ($7.50)
MAX_RISK_PER_TRADE = 0.02        # absolute max 2% ($10)
MAX_POSITION_PCT = 0.30          # max 30% of balance per trade (prevents all-in)
MAX_POSITIONS = 1                # only 1 open trade at a time

# SL/TP (ATR-based)
SL_ATR_MULT = 1.5                # SL = 1.5 x ATR below/above entry
TP_ATR_MULT = 3.0                # TP = 3.0 x ATR = R:R = 1:2
TRAILING_STOP_ATR = 1.0          # optional trailing stop at 1x ATR
MIN_RR_RATIO = 1.8               # minimum reward:risk to enter

# ──────────────────────────────────────────────
# CONTINUOUS RETRAINING
# ──────────────────────────────────────────────
RETRAIN_ENABLED = os.getenv("RETRAIN_ENABLED", "true").lower() == "true"
RETRAIN_INTERVAL_HOURS = int(os.getenv("RETRAIN_INTERVAL_HOURS", "6"))   # retrain every 6h
RETRAIN_FORCE_AFTER_HOURS = int(os.getenv("RETRAIN_FORCE_AFTER_HOURS", "24"))  # force retrain if model >24h old
RETRAIN_MIN_IMPROVEMENT = float(os.getenv("RETRAIN_MIN_IMPROVEMENT", "0.03"))  # allow up to 3% metric degradation

# ──────────────────────────────────────────────
# SERVER
# ──────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
