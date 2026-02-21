"""
Sorbot AI Engine v3.0 — Risk Manager
=======================================
Position sizing and risk controls for $500 spot account:
  - ATR-based position sizing
  - Risk per trade = 1.5% of equity ($7.50)
  - No leverage (spot only)
  - Max 1 open position
  - Pre-trade validation
"""

import logging

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    ACCOUNT_BALANCE, RISK_PER_TRADE, MAX_RISK_PER_TRADE,
    MAX_POSITIONS, MAX_POSITION_PCT, SL_ATR_MULT,
)

logger = logging.getLogger("sorbot.risk")


class RiskManager:
    """Manages position sizing and risk controls (spot, no leverage)."""

    def __init__(self, balance: float = ACCOUNT_BALANCE):
        self.balance = balance
        self.open_positions = 0

    def update_balance(self, new_balance: float):
        self.balance = new_balance
        logger.info("Balance updated: $%.2f", self.balance)

    # Binance BTCUSDT spot minimum notional is $10.
    # We require $12 to cover the order + trading fees + rounding.
    BINANCE_MIN_NOTIONAL = 12.0

    def can_trade(self) -> tuple:
        """Check if we can open a new trade. Returns (allowed, reason)."""
        if self.open_positions >= MAX_POSITIONS:
            return False, f"Max positions reached ({MAX_POSITIONS})"
        if self.balance <= 0:
            return False, "Zero balance"
        if self.balance < self.BINANCE_MIN_NOTIONAL:
            return False, (
                f"Insufficient balance: ${self.balance:.2f}. "
                f"Binance spot requires at least ${self.BINANCE_MIN_NOTIONAL:.0f} USDT "
                f"(minimum notional $10 + fees buffer). "
                f"Please deposit more USDT to continue trading."
            )
        return True, "OK"

    def calculate_position_size(
        self,
        entry_price: float,
        sl_price: float,
        signal: str,
    ) -> dict:
        """
        Calculate position size based on risk management rules.
        Spot trading: no leverage, BUY only.

        Args:
            entry_price: expected entry price
            sl_price: stop loss price
            signal: "LONG" only (SHORT not supported in spot)

        Returns:
            dict with qty, notional, risk_usd, etc.
        """
        if signal == "SHORT":
            return {"qty_btc": 0, "error": "SHORT not supported in spot trading"}

        # Risk amount in USD
        risk_pct = min(RISK_PER_TRADE, MAX_RISK_PER_TRADE)
        risk_usd = self.balance * risk_pct

        # Distance to SL (absolute)
        sl_distance = entry_price - sl_price

        if sl_distance <= 0:
            logger.warning("Invalid SL distance: %.2f", sl_distance)
            return {"qty_btc": 0, "error": "Invalid SL distance"}

        # Position size: how many BTC can we trade such that
        # if SL hits, we lose exactly risk_usd
        qty_btc = risk_usd / sl_distance

        # Notional value
        notional = qty_btc * entry_price

        # Cap at MAX_POSITION_PCT of balance (default 30%) — never go all-in
        max_notional = self.balance * MAX_POSITION_PCT
        if notional > max_notional:
            qty_btc = max_notional / entry_price
            notional = qty_btc * entry_price
            risk_usd = qty_btc * sl_distance
            logger.info(
                "Position capped at %.0f%% of balance ($%.2f). Qty: %.6f BTC ($%.2f)",
                MAX_POSITION_PCT * 100, max_notional, qty_btc, notional,
            )

        # Round to Binance precision (5 decimals for BTC spot)
        qty_btc = round(qty_btc, 5)
        if qty_btc <= 0:
            qty_btc = 0.00001  # minimum

        # Minimum notional check (Binance requires > $10 for BTCUSDT)
        if notional < self.BINANCE_MIN_NOTIONAL:
            return {
                "qty_btc": 0,
                "error": (
                    f"Order notional ${notional:.2f} below Binance minimum "
                    f"(${self.BINANCE_MIN_NOTIONAL:.0f}). Need more USDT."
                ),
            }

        capital_pct = (notional / self.balance * 100) if self.balance > 0 else 0
        potential_loss = qty_btc * sl_distance

        result = {
            "qty_btc": qty_btc,
            "notional_usd": round(notional, 2),
            "risk_usd": round(potential_loss, 2),
            "risk_pct": round(risk_pct * 100, 2),
            "capital_used_pct": round(capital_pct, 1),
            "sl_distance_usd": round(sl_distance, 2),
            "balance": round(self.balance, 2),
            "max_position_pct": round(MAX_POSITION_PCT * 100, 0),
        }
        logger.info(
            "Position size: %.5f BTC ($%.2f notional, $%.2f risk)",
            qty_btc, notional, risk_usd,
        )
        return result

    def register_open(self):
        """Mark a position as opened."""
        self.open_positions += 1

    def register_close(self, pnl: float):
        """Mark a position as closed and update balance."""
        self.open_positions = max(0, self.open_positions - 1)
        self.balance += pnl
        logger.info("Position closed. PnL: $%.2f, New balance: $%.2f", pnl, self.balance)


# Module-level singleton
_risk_manager = RiskManager()


def get_risk_manager() -> RiskManager:
    return _risk_manager
