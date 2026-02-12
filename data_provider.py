import functools
import logging
import numpy as np
import ccxt
from typing import Optional, Any, List
from config import BYBIT_API_KEY, BYBIT_API_SECRET

class CryptoDataProvider:
    def __init__(self, api_key: Optional[str], api_secret: Optional[str], symbols: List[str]):
        self.exchange = ccxt.bybit({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })
        self.symbols = symbols

    @functools.lru_cache(maxsize=128)
    def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> Optional[np.ndarray]:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                raise ValueError("No OHLCV data")
            logging.info("Fetched OHLCV for %s %s", symbol, timeframe)
            return np.array(ohlcv)
        except Exception as e:
            logging.error("get_ohlcv error for %s: %s", symbol, e)
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker.get("last"))
        except Exception as e:
            logging.error("get_current_price error for %s: %s", symbol, e)
            return None

    def get_order_book(self, symbol: str, limit: int = 50) -> Optional[dict]:
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=limit)
            logging.info("Fetched order book for %s", symbol)
            return ob
        except Exception as e:
            logging.error("get_order_book error for %s: %s", symbol, e)
            return None

    def execute_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None, order_type: str = "market") -> Optional[dict]:
        try:
            order = self.exchange.create_order(symbol, order_type, side, amount, price)
            logging.info("Order executed for %s: %s", symbol, order)
            return order
        except Exception as e:
            logging.error("execute_order error for %s: %s", symbol, e)
            return None

    def fetch_open_orders(self, symbol: Optional[str] = None):
        try:
            return self.exchange.fetch_open_orders(symbol) if symbol else self.exchange.fetch_open_orders()
        except Exception as e:
            logging.error("fetch_open_orders error: %s", e)
            return []

    def fetch_order(self, order_id: str, symbol: Optional[str] = None):
        try:
            return self.exchange.fetch_order(order_id, symbol) if order_id else None
        except Exception as e:
            logging.error("fetch_order error: %s", e)
            return None

    def clear_cache(self):
        try:
            self.get_ohlcv.cache_clear()
            logging.info("Cleared OHLCV cache")
        except Exception:
            pass
