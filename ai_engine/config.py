"""
Sorbot AI Engine — Central Configuration  v2.0
=================================================
Multi-timeframe support, SL/TP calculation, smart money hunting.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# TRADING PAIRS CONFIGURATION
# ──────────────────────────────────────────────
TRADING_PAIRS = {
    "BTCUSD": {
        "yfinance_ticker": "BTC-USD",
        "display_name": "Bitcoin / US Dollar",
        "category": "crypto",
        "decimals": 2,
        "pip_value": 1.0,          # $1 per pip for BTC
        "default_sl_atr_mult": 1.5,
        "default_tp_atr_mult": 2.5,
    },
    "XAUUSD": {
        "yfinance_ticker": "GC=F",
        "display_name": "Gold / US Dollar",
        "category": "commodity",
        "decimals": 2,
        "pip_value": 0.01,
        "default_sl_atr_mult": 1.5,
        "default_tp_atr_mult": 2.0,
    },
    "EURUSD": {
        "yfinance_ticker": "EURUSD=X",
        "display_name": "Euro / US Dollar",
        "category": "forex",
        "decimals": 5,
        "pip_value": 0.0001,
        "default_sl_atr_mult": 1.2,
        "default_tp_atr_mult": 1.8,
    },
}

ACTIVE_SYMBOLS = list(TRADING_PAIRS.keys())

# ──────────────────────────────────────────────
# MULTI-TIMEFRAME CONFIGURATION
# ──────────────────────────────────────────────
# Each timeframe: (yfinance interval, yfinance max period, candle label)
TIMEFRAMES = {
    "1h":  {"interval": "1h",  "period": "60d",  "label": "1 Hour"},
    "4h":  {"interval": "1h",  "period": "60d",  "label": "4 Hours"},   # resample from 1h
    "1d":  {"interval": "1d",  "period": "2y",   "label": "1 Day"},
    "1w":  {"interval": "1wk", "period": "5y",   "label": "1 Week"},
}

# The PRIMARY timeframe used for training & main signal
PRIMARY_TIMEFRAME = "1d"

# Higher timeframes used to confirm the trend (confluence)
CONFLUENCE_TIMEFRAMES = ["4h", "1d", "1w"]

# The timeframe used to find precise entry/exit (sniper entry)
ENTRY_TIMEFRAME = "1h"

# ──────────────────────────────────────────────
# DATA SETTINGS
# ──────────────────────────────────────────────
HISTORY_PERIOD = os.getenv("HISTORY_PERIOD", "2y")
HISTORY_INTERVAL = os.getenv("HISTORY_INTERVAL", "1d")

# ──────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14
EMA_SHORT = 9
EMA_LONG = 21
EMA_200 = 200
VOLUME_SMA_PERIOD = 20

# ──────────────────────────────────────────────
# XGBOOST HYPERPARAMETERS  (tuned for trading)
# ──────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 7,
    "learning_rate": 0.03,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "min_child_weight": 5,
    "gamma": 0.2,
    "reg_alpha": 0.3,
    "reg_lambda": 1.5,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}

# ──────────────────────────────────────────────
# PREDICTION & TRADE MANAGEMENT
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45
LOOK_AHEAD_DAYS = 5

# Stop-Loss / Take-Profit (ATR multipliers)
SL_ATR_MULTIPLIER = 1.5      # default; overridden per-pair
TP_ATR_MULTIPLIER = 2.5      # default; overridden per-pair
MIN_RISK_REWARD = 1.5         # minimum R:R to accept a trade

# ──────────────────────────────────────────────
# SUPPORT / RESISTANCE DETECTION
# ──────────────────────────────────────────────
SR_LOOKBACK = 50              # bars to scan for S/R levels
SR_TOUCH_THRESHOLD = 3        # min touches to confirm a level
SR_PROXIMITY_PCT = 0.005      # 0.5% — how close price must be to "touch"

# ──────────────────────────────────────────────
# SERVER
# ──────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
