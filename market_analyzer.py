import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional
import logging

class MarketAnalyzer:
    @staticmethod
    def convert_timestamps(timestamps):
        return [mdates.date2num(datetime.fromtimestamp(ts / 1000, tz=timezone.utc)) for ts in timestamps]

    @staticmethod
    def get_support_resistance(data):
        highs = data[:, 2]
        lows = data[:, 3]
        support = float(np.min(lows[-10:])) if len(lows) >= 10 else float(np.min(lows))
        resistance = float(np.max(highs[-10:])) if len(highs) >= 10 else float(np.max(highs))
        return support, resistance

    @staticmethod
    def calculate_cvd(data):
        return np.cumsum((data[:, 4] - data[:, 1]) * data[:, 5])

    @staticmethod
    def calculate_rsi(closes, period=14):
        if len(closes) < period + 1:
            return np.full(len(closes), 50.0)
        delta = np.diff(closes)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(up, np.ones(period) / period, mode="valid")
        avg_loss = np.convolve(down, np.ones(period) / period, mode="valid")
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate((np.full((period,), 50), rsi))

    @staticmethod
    def calculate_adx(data, period=14):
        high = pd.Series(data[:, 2])
        low = pd.Series(data[:, 3])
        close = pd.Series(data[:, 4])
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        up_move = high.diff()
        down_move = low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = -down_move.where((down_move > up_move) & (down_move < 0), 0.0)
        plus_di = 100 * plus_dm.rolling(window=period).sum() / atr
        minus_di = 100 * minus_dm.rolling(window=period).sum() / atr
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        return adx.fillna(0).values

    @staticmethod
    def calculate_pivots_from_ohlcv(prev_ohlcv):
        try:
            arr = np.array(prev_ohlcv)
            if arr.ndim == 1:
                H, L, C = float(arr[2]), float(arr[3]), float(arr[4])
            else:
                H, L, C = float(np.max(arr[:, 2])), float(np.min(arr[:, 3])), float(arr[-1, 4])
            P = (H + L + C) / 3.0
            R1 = 2 * P - L
            S1 = 2 * P - H
            R2 = P + (H - L)
            S2 = P - (H - L)
            R3 = H + 2 * (P - L)
            S3 = L - 2 * (H - P)
            return {"R3": R3, "R2": R2, "R1": R1, "P": P, "S1": S1, "S2": S2, "S3": S3}
        except Exception as e:
            logging.error("Pivot calc error: %s", e)
            return {}
