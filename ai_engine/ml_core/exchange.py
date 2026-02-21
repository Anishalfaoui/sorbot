"""
Sorbot AI Engine v3.0 — Exchange (Binance Spot)
=====================================================
Handles all Binance SPOT API interactions:
  - Buy BTC with USDT (market orders)
  - Sell BTC back to USDT (close position)
  - OCO orders for SL/TP
  - Check USDT & BTC balances
  - Track entry price for PnL

No leverage, no short selling — spot only.
Requires BINANCE_API_KEY and BINANCE_API_SECRET in .env file.
"""

import logging
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET,
    SYMBOL, DECIMALS,
)

logger = logging.getLogger("sorbot.exchange")


class BinanceExchange:
    """Wrapper around python-binance for Spot trading (BUY only, no leverage)."""

    def __init__(self):
        self.client = None
        self._initialized = False
        self._entry_price = None  # track entry price for PnL

    def connect(self):
        """Initialize Binance Spot client."""
        try:
            from binance.client import Client
        except ImportError:
            raise ImportError("python-binance not installed. Run: pip install python-binance")

        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            raise ValueError(
                "Binance API keys not set. Add to .env file:\n"
                "  BINANCE_API_KEY=your_key\n"
                "  BINANCE_API_SECRET=your_secret"
            )

        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=BINANCE_TESTNET)
        self._initialized = True

        env_label = "TESTNET" if BINANCE_TESTNET else "LIVE"
        logger.info("Binance Spot %s connected", env_label)

    def _ensure_connected(self):
        if not self._initialized:
            self.connect()

    # ── Balances ───────────────────────────────

    def get_balance(self) -> float:
        """Get total USDT balance (free + locked)."""
        self._ensure_connected()
        info = self.client.get_asset_balance(asset="USDT")
        if info:
            return round(float(info["free"]) + float(info["locked"]), 2)
        return 0.0

    def get_available_balance(self) -> float:
        """Get free (available) USDT balance."""
        self._ensure_connected()
        info = self.client.get_asset_balance(asset="USDT")
        if info:
            return float(info["free"])
        return 0.0

    def get_btc_balance(self) -> float:
        """Get free BTC balance."""
        self._ensure_connected()
        info = self.client.get_asset_balance(asset="BTC")
        if info:
            return float(info["free"])
        return 0.0

    # ── Position ───────────────────────────────

    def get_position(self) -> Optional[dict]:
        """
        Check if we hold BTC (our 'position').
        In spot, holding BTC = LONG position.
        Returns None if BTC balance is negligible.
        """
        self._ensure_connected()
        btc_free = self.get_btc_balance()

        # Also check locked BTC (in open orders)
        info = self.client.get_asset_balance(asset="BTC")
        btc_locked = float(info["locked"]) if info else 0.0
        btc_total = btc_free + btc_locked

        # Ignore dust (< $1 worth)
        if btc_total < 0.00001:
            return None

        current_price = self.get_current_price()
        notional = btc_total * current_price

        # Calculate unrealized PnL
        pnl = 0.0
        if self._entry_price and self._entry_price > 0:
            pnl = (current_price - self._entry_price) * btc_total

        return {
            "symbol": SYMBOL,
            "side": "LONG",
            "qty": round(btc_total, 6),
            "qty_free": round(btc_free, 6),
            "entry_price": self._entry_price or 0.0,
            "current_price": current_price,
            "notional_usd": round(notional, 2),
            "unrealized_pnl": round(pnl, 2),
        }

    def get_current_price(self) -> float:
        """Get latest BTC/USDT spot price."""
        self._ensure_connected()
        ticker = self.client.get_symbol_ticker(symbol=SYMBOL)
        return float(ticker["price"])

    # ── Orders ─────────────────────────────────

    def place_order(
        self,
        side: str,
        qty: float,
        sl_price: float,
        tp_price: float,
    ) -> dict:
        """
        Place a spot market BUY order, then an OCO sell order for SL/TP.

        Spot trading is BUY-only (no short selling).
        If signal is SHORT, we skip (handled in main.py).

        Args:
            side: "LONG" only (SHORT is blocked upstream)
            qty: BTC amount to buy
            sl_price: stop loss price for OCO sell
            tp_price: take profit price for OCO sell

        Returns:
            dict with order details
        """
        self._ensure_connected()

        if side == "SHORT":
            return {"error": "SHORT not available in spot trading. Spot is BUY-only."}

        qty_str = f"{qty:.5f}"
        sl_str = f"{sl_price:.{DECIMALS}f}"
        tp_str = f"{tp_price:.{DECIMALS}f}"

        # Stop limit price slightly below stop price (for OCO)
        sl_limit = round(sl_price * 0.999, DECIMALS)
        sl_limit_str = f"{sl_limit:.{DECIMALS}f}"

        results = {}

        try:
            # 1) Market BUY
            entry_order = self.client.create_order(
                symbol=SYMBOL,
                side="BUY",
                type="MARKET",
                quantity=qty_str,
            )
            # Extract fill price
            fills = entry_order.get("fills", [])
            if fills:
                total_qty = sum(float(f["qty"]) for f in fills)
                total_cost = sum(float(f["qty"]) * float(f["price"]) for f in fills)
                self._entry_price = round(total_cost / total_qty, 2) if total_qty > 0 else 0
            else:
                self._entry_price = self.get_current_price()

            results["entry"] = {
                "orderId": entry_order["orderId"],
                "status": entry_order["status"],
                "side": "LONG",
                "qty": qty,
                "entry_price": self._entry_price,
            }
            logger.info("Spot BUY filled: %.5f BTC @ $%.2f", qty, self._entry_price)

            # 2) OCO sell order (TP + SL combined)
            # Binance OCO: price = TP limit, stopPrice = SL trigger, stopLimitPrice = SL limit
            try:
                oco = self.client.create_oco_order(
                    symbol=SYMBOL,
                    side="SELL",
                    quantity=qty_str,
                    price=tp_str,                     # Take profit limit price
                    stopPrice=sl_str,                 # Stop loss trigger price
                    stopLimitPrice=sl_limit_str,      # Stop loss limit price
                    stopLimitTimeInForce="GTC",
                )
                results["oco"] = {
                    "orderListId": oco["orderListId"],
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "status": oco["listOrderStatus"],
                }
                logger.info("OCO sell placed: TP=$%s, SL=$%s", tp_str, sl_str)
            except Exception as e:
                logger.error("OCO order failed: %s — placing separate limit orders", e)
                # Fallback: place just a stop-loss limit sell
                try:
                    sl_order = self.client.create_order(
                        symbol=SYMBOL,
                        side="SELL",
                        type="STOP_LOSS_LIMIT",
                        quantity=qty_str,
                        price=sl_limit_str,
                        stopPrice=sl_str,
                        timeInForce="GTC",
                    )
                    results["stop_loss"] = {
                        "orderId": sl_order["orderId"],
                        "stopPrice": sl_price,
                    }
                    logger.info("Fallback SL order placed at $%s", sl_str)
                except Exception as e2:
                    results["sl_error"] = str(e2)
                    logger.error("SL fallback also failed: %s", e2)

        except Exception as e:
            results["error"] = str(e)
            logger.error("Order error: %s", e)

        return results

    def close_position(self) -> Optional[dict]:
        """Close position by selling all BTC at market price."""
        self._ensure_connected()

        btc_balance = self.get_btc_balance()
        if btc_balance < 0.00001:
            logger.info("No BTC to sell — no open position")
            return None

        qty_str = f"{btc_balance:.5f}"

        # Cancel any open orders first (OCO SL/TP)
        self.cancel_all_orders()

        try:
            order = self.client.create_order(
                symbol=SYMBOL,
                side="SELL",
                type="MARKET",
                quantity=qty_str,
            )

            # Calculate PnL
            fills = order.get("fills", [])
            sell_price = 0.0
            if fills:
                total_qty = sum(float(f["qty"]) for f in fills)
                total_cost = sum(float(f["qty"]) * float(f["price"]) for f in fills)
                sell_price = total_cost / total_qty if total_qty > 0 else 0

            pnl = 0.0
            if self._entry_price and sell_price:
                pnl = (sell_price - self._entry_price) * btc_balance

            result = {
                "orderId": order["orderId"],
                "closed_side": "LONG",
                "qty": btc_balance,
                "sell_price": round(sell_price, 2),
                "entry_price": self._entry_price or 0,
                "pnl": round(pnl, 2),
            }
            logger.info("Position closed: SOLD %.5f BTC @ $%.2f, PnL: $%.2f",
                         btc_balance, sell_price, pnl)

            self._entry_price = None
            return result

        except Exception as e:
            logger.error("Close position error: %s", e)
            return {"error": str(e)}

    def cancel_all_orders(self):
        """Cancel all open orders for the symbol."""
        self._ensure_connected()
        try:
            open_orders = self.client.get_open_orders(symbol=SYMBOL)
            for order in open_orders:
                self.client.cancel_order(symbol=SYMBOL, orderId=order["orderId"])
            if open_orders:
                logger.info("Cancelled %d open orders for %s", len(open_orders), SYMBOL)
        except Exception as e:
            logger.error("Cancel orders error: %s", e)


# Module-level singleton
_exchange = BinanceExchange()


def get_exchange() -> BinanceExchange:
    return _exchange
