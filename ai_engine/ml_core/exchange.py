"""
Sorbot AI Engine v3.0 — Virtual Exchange Stub
==============================================
Backward-compatible paper-trading exchange API.
This module intentionally avoids any live broker/exchange credentials.
"""

from typing import Optional


class VirtualExchange:
    def __init__(self):
        self._entry_price = None

    def connect(self):
        return None

    def get_balance(self) -> float:
        return 0.0

    def get_available_balance(self) -> float:
        return 0.0

    def get_btc_balance(self) -> float:
        return 0.0

    def get_position(self) -> Optional[dict]:
        return None

    def get_current_price(self) -> float:
        return 0.0

    def place_order(self, side: str, qty: float, sl_price: float, tp_price: float) -> dict:
        return {
            "entry": {
                "status": "FILLED",
                "side": side,
                "qty": qty,
                "entry_price": self._entry_price,
                "virtual": True,
            },
            "sl_price": sl_price,
            "tp_price": tp_price,
        }

    def close_position(self) -> Optional[dict]:
        return None

    def cancel_all_orders(self):
        return None


_exchange = VirtualExchange()


def get_exchange() -> VirtualExchange:
    return _exchange
