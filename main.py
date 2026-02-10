import os
import sys
import math
import json
import pickle
import logging
import functools
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # GUI backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import ccxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import requests  # Для отправки в Telegram

# Telegram constants (из исходного файла)
TELEGRAM_BOT_TOKEN = "**************************************"
TELEGRAM_CHAT_ID = "**************************************"

def send_telegram_message(message: str) -> None:
    """Функция для отправки сообщения в Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            logging.error(f"Failed to send Telegram message: {response.text}")
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

class TelegramHandler(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            msg = self.format(record)
            send_telegram_message(msg)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.addHandler(TelegramHandler())

class PersistenceManager:
    def __init__(self, base_dir: str, filename: str = 'state.json') -> None:
        self.state_path = os.path.join(base_dir, filename)
        self._lock = None

    @property
    def lock(self):
        if self._lock is None:
            import threading
            self._lock = threading.Lock()
        return self._lock

    def load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.state_path):
            logging.info("Файл состояния отсутствует: %s", self.state_path)
            return {}
        try:
            with open(self.state_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info("Состояние загружено: %s", self.state_path)
            return data
        except Exception as e:
            logging.error("Не удалось загрузить состояние: %s", e)
            return {}

    def save_state(self, state: Dict[str, Any]) -> None:
        with self.lock:
            tmp_path = self.state_path + '.tmp'
            try:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, self.state_path)
                logging.info("Состояние сохранено: %s", self.state_path)
            except Exception as e:
                logging.error("Не удалось сохранить состояние: %s", e)

class MLPredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_file = os.path.join(base_dir, 'ml_model.pkl')
        self.scaler_file = os.path.join(base_dir, 'scaler.pkl')
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.load_model()

    def prepare_features(self, data, order_book_data):
        closes = data[:, 4]
        volumes = data[:, 5]
        features = []
        for i in range(24, len(closes) - 1):
            price_window = closes[i - 10:i]
            avg_price = np.mean(price_window)
            volatility = np.std(price_window)
            volume_window = volumes[i - 10:i]
            volume_ratio = volumes[i - 1] / np.mean(volume_window) if np.mean(volume_window) != 0 else 0
            rsi = self.calculate_rsi(closes[:i])[-1]
            macd_line, _ = self.calculate_macd(closes[:i])
            macd = macd_line[-1] if len(macd_line) > 0 else 0
            features.append([avg_price, volatility, volume_ratio, rsi, macd])
        return np.array(features)

    def prepare_labels(self, data):
        closes = data[:, 4]
        labels = []
        for i in range(24, len(closes) - 1):
            future_close = closes[i + 1]
            current_close = closes[i]
            label = 1 if future_close > current_close else 0
            labels.append(label)
        return np.array(labels)

    def train(self, data, order_book_data):
        features = self.prepare_features(data, order_book_data)
        labels = self.prepare_labels(data)
        if len(features) > 0 and len(features) == len(labels):
            self.scaler = StandardScaler().fit(features)
            scaled_features = self.scaler.transform(features)
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=5,
                class_weight='balanced'
            ).fit(scaled_features, labels)
            self.is_trained = True
            logging.info("ML модель успешно обучена. Примеров: %d", len(features))
            self.save_model()
        else:
            logging.warning("Недостаточно данных для обучения.")

    def predict(self, data, order_book_data):
        if not self.is_trained:
            logging.warning("ML модель не обучена!")
            return 0
        features = self.prepare_features(data, order_book_data)
        if len(features) < 1:
            logging.warning("Недостаточно данных для прогноза!")
            return 0
        try:
            scaled_features = self.scaler.transform(features)
            return self.model.predict(scaled_features[-1].reshape(1, -1))[0]
        except Exception as e:
            logging.error("Ошибка предсказания: %s", str(e))
            return 0

    def predict_proba(self, data, order_book_data):
        if not self.is_trained:
            return 0.5
        features = self.prepare_features(data, order_book_data)
        if len(features) < 1:
            return 0.5
        try:
            scaled_features = self.scaler.transform(features)
            proba = self.model.predict_proba(scaled_features[-1].reshape(1, -1))[0]
            return proba[1]
        except Exception as e:
            logging.error("Ошибка вероятности: %s", str(e))
            return 0.5

    def calculate_rsi(self, closes, period=14):
        if len(closes) < period + 1:
            return np.full(len(closes), 50.0)
        delta = np.diff(closes)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(up, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(down, np.ones(period) / period, mode='valid')
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))
        rsi_full = np.concatenate((np.full((period,), 50), rsi))
        return rsi_full

    def calculate_macd(self, closes, n_fast=12, n_slow=26, n_signal=9):
        if len(closes) < n_slow + n_signal:
            return np.array([]), np.array([])
        ema_fast = pd.Series(closes).ewm(span=n_fast, adjust=False).mean()
        ema_slow = pd.Series(closes).ewm(span=n_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=n_signal, adjust=False).mean()
        return macd_line.values, signal_line.values

    def save_model(self):
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump({'model': self.model, 'scaler': self.scaler, 'version': '1.1'}, f)
            logging.info("ML модель сохранена: %s", self.model_file)
        except Exception as e:
            logging.error("Ошибка сохранения модели: %s", str(e))

    def load_model(self):
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = True
                logging.info("ML модель загружена (v%s)", data.get('version', '1.0'))
            except Exception as e:
                logging.error("Ошибка загрузки модели: %s", str(e))
                self.is_trained = False
        else:
            logging.info("Файл модели не найден. Будет создана новая модель.")
            self.is_trained = False

class CryptoDataProvider:
    def __init__(self, api_key, api_secret, symbols):
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        self.symbols = symbols

    @functools.lru_cache(maxsize=128)
    def get_ohlcv(self, symbol, timeframe='1h', limit=200):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                raise ValueError("Нет данных")
            logging.info(f"OHLCV получены для {symbol} с таймфреймом {timeframe}")
            return np.array(ohlcv)
        except Exception as e:
            logging.error(f"Ошибка получения данных для {symbol}: {e}")
            return None

    def get_current_price(self, symbol):
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logging.error(f"Ошибка получения текущей цены для {symbol}: {e}")
            return None

    def get_order_book(self, symbol, limit=50):
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit=limit)
            logging.info(f"Стакан ордеров получен для {symbol}")
            return order_book
        except Exception as e:
            logging.error(f"Ошибка получения стакана ордеров для {symbol}: {e}")
            return None

    def execute_order(self, symbol, side, amount, price=None, order_type='market'):
        try:
            balance = self.exchange.fetch_balance()
            base_currency, quote_currency = symbol.split('/')
            if side == 'buy':
                effective_price = price if price is not None else self.get_current_price(symbol)
                if effective_price is None:
                    raise ValueError("Не удалось получить текущую цену")
                required = amount * effective_price
                if balance['total'].get(quote_currency, 0) < required:
                    error_msg = f"Недостаточно средств для покупки: требуется {required} {quote_currency}, доступно {balance['total'].get(quote_currency, 0)}"
                    logging.error(error_msg)
                    send_telegram_message(error_msg)
                    return None
            elif side == 'sell':
                if balance['total'].get(base_currency, 0) < amount:
                    error_msg = f"Недостаточно средств для продажи: требуется {amount} {base_currency}, доступно {balance['total'].get(base_currency, 0)}"
                    logging.error(error_msg)
                    send_telegram_message(error_msg)
                    return None

            order = self.exchange.create_order(symbol, order_type, side, amount, price)
            logging.info("Ордер выполнен для %s: %s", symbol, order)
            return order
        except Exception as e:
            logging.error("Ошибка выполнения ордера для %s: %s", symbol, str(e))
            send_telegram_message(f"Ошибка выполнения ордера для {symbol}: {str(e)}")
            return None

    def fetch_open_orders(self, symbol=None):
        try:
            return self.exchange.fetch_open_orders(symbol) if symbol else self.exchange.fetch_open_orders()
        except Exception as e:
            logging.error("Ошибка получения открытых ордеров: %s", e)
            return []

    def fetch_order(self, order_id, symbol=None):
        try:
            return self.exchange.fetch_order(order_id, symbol) if order_id else None
        except Exception as e:
            logging.error("Ошибка fetch_order: %s", e)
            return None

    def clear_cache(self):
        try:
            self.get_ohlcv.cache_clear()
            logging.info("Кэш OHLCV очищен.")
        except Exception:
            pass

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
    def calculate_liquidity(data):
        volumes = data[:, 5]
        avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else np.mean(volumes)
        current_volume = volumes[-1]
        return float(current_volume / avg_volume) if avg_volume != 0 else 0.0

    @staticmethod
    def calculate_cvd(data):
        cvd = np.cumsum((data[:, 4] - data[:, 1]) * data[:, 5])
        return cvd

    @staticmethod
    def calculate_rsi(closes, period=14):
        if len(closes) < period + 1:
            return np.full(len(closes), 50.0)
        delta = np.diff(closes)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(up, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(down, np.ones(period) / period, mode='valid')
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))
        rsi_full = np.concatenate((np.full((period,), 50), rsi))
        return rsi_full

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
    def calculate_pivots_from_ohlcv(prev_ohlcv: np.ndarray) -> Dict[str, float]:
        """
        Рассчитывает уровни Pivot (R3..S3) по стандартным формулам на основе H,L,C.
        Принимает либо 2D-массив OHLCV, либо 1D-массив для одного бара.
        Формат строки: [ts, open, high, low, close, volume]
        """
        try:
            arr = np.array(prev_ohlcv)
            if arr.ndim == 1:
                # single row
                H = float(arr[2])
                L = float(arr[3])
                C = float(arr[4])
            else:
                # multiple rows
                H = float(np.max(arr[:, 2]))
                L = float(np.min(arr[:, 3]))
                C = float(arr[-1, 4])
            P = (H + L + C) / 3.0
            R1 = 2 * P - L
            S1 = 2 * P - H
            R2 = P + (H - L)
            S2 = P - (H - L)
            R3 = H + 2 * (P - L)
            S3 = L - 2 * (H - P)
            return {'R3': R3, 'R2': R2, 'R1': R1, 'P': P, 'S1': S1, 'S2': S2, 'S3': S3}
        except Exception as e:
            logging.error("Ошибка расчёта pivot: %s", e)
            return {}

    @staticmethod
    def generate_pivot_probs(pivots: Dict[str, float]) -> Dict[str, Tuple[int, int]]:
        """
        Возвращает простой шаблон вероятностей для подписи уровней.
        """
        probs = {}
        for level in ['R3', 'R2', 'R1', 'P', 'S1', 'S2', 'S3']:
            if level == 'P':
                probs[level] = (67, 34)
            elif level in ('R1', 'R2', 'R3'):
                probs[level] = (34, 67)
            else:
                probs[level] = (67, 34)
        return probs

    @staticmethod
    def plot_pivot_levels(ax, pivot_levels: Dict[str, float], probs: Optional[Dict[str, Tuple[int, int]]] = None, x_pos_ratio: float = 0.98, fontsize: int = 8):
        colors = {
            'R3':'#8B0000', 'R2':'#B22222', 'R1':'#FF4500',
            'P':'#000000',
            'S1':'#2E8B57', 'S2':'#228B22', 'S3':'#006400'
        }
        linestyles = {'R3':'--','R2':'--','R1':'-','P':':','S1':'-','S2':'--','S3':'--'}
        try:
            y_min, y_max = ax.get_ylim()
            x_min, x_max = ax.get_xlim()
            x_text = x_min + (x_max - x_min) * x_pos_ratio

            for level in ['R3','R2','R1','P','S1','S2','S3']:
                price = pivot_levels.get(level)
                if price is None:
                    continue
                color = colors.get(level, 'grey')
                ls = linestyles.get(level, '-')
                ax.axhline(price, color=color, linestyle=ls, linewidth=1.0, alpha=0.9, zorder=2)

                label_price = f"{price:,.2f}"
                if probs and level in probs:
                    bounce, breakout = probs[level]
                    label = f"{level} {label_price}  Отбой {bounce}% | Пробой {breakout}%"
                else:
                    label = f"{level} {label_price}"

                ax.text(x_text, price, label,
                        horizontalalignment='right',
                        verticalalignment='center',
                        fontsize=fontsize,
                        color=color,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                        zorder=3)

            # Расширим видимую область по Y, чтобы подписи не обрезались
            all_prices = [p for p in pivot_levels.values() if p is not None]
            if all_prices:
                margin = (y_max - y_min) * 0.05 if (y_max - y_min) != 0 else max(1.0, max(all_prices)*0.01)
                new_y_min = min(y_min, min(all_prices) - margin)
                new_y_max = max(y_max, max(all_prices) + margin)
                ax.set_ylim(new_y_min, new_y_max)
        except Exception as e:
            logging.error("Ошибка рисования уровней Pivot: %s", e)

    @staticmethod
    def plot_data(data, symbol, fig=None, current_price=None, pivot_levels: Optional[Dict[str, float]] = None, probs: Optional[Dict[str, Tuple[int, int]]] = None):
        dates = MarketAnalyzer.convert_timestamps(data[:, 0])
        if fig is None:
            fig = plt.figure(figsize=(12, 6))
            axes = fig.add_subplot(111)
        else:
            fig.clf()
            axes = fig.add_subplot(111)
        axes.plot(dates, data[:, 4], label='Цена закрытия', color='blue')
        ma10 = np.mean(data[:, 4][-10:]) if len(data[:,4])>=10 else np.mean(data[:,4])
        axes.plot(dates, np.full_like(dates, ma10), color='red', linestyle='--', label='МА 10')
        support, resistance = MarketAnalyzer.get_support_resistance(data)
        axes.axhline(support, color='green', linestyle='-', label='Поддержка')
        axes.axhline(resistance, color='red', linestyle='-', label='Сопротивление')
        axes.set_ylabel('Цена', fontsize=10)
        if current_price is not None:
            axes.set_title(f"{symbol} — текущий курс: {current_price:.2f} USDT", fontsize=12)
        else:
            axes.set_title(f"{symbol} — текущий курс: —", fontsize=12)
        axes.legend(loc='upper left', fontsize=8)
        axes.tick_params(axis='y', labelsize=8)
        axes.xaxis_date()
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        axes.tick_params(axis='x', rotation=45, labelsize=8)
        axes.tick_params(axis='y', labelsize=8)

        # Добавление уровней Pivot при наличии
        if pivot_levels:
            MarketAnalyzer.plot_pivot_levels(axes, pivot_levels, probs=probs, x_pos_ratio=0.98, fontsize=8)

        fig.tight_layout()
        return fig

class CryptoGUI:
    def __init__(self, data_provider, ml_predictor, persistence: PersistenceManager):
        self.data_provider = data_provider
        self.ml_predictor = ml_predictor
        self.persistence = persistence
        self.weights = {
            'trend': 0.25,
            'volume': 0.2,
            'cvd': 0.2,
            'rsi': 0.2,
            'support_resistance': 0.15,
            'ml_prediction': 0.2
        }
        self.auto_trade_enabled = True
        self.trade_cooldown = timedelta(minutes=1)
        self.initial_delay = timedelta(minutes=1)
        self.grid_orders = {}
        self.trade_history = []
        self.cumulative_return = 1.0
        self.last_trade_time = datetime.min
        self.start_time = datetime.now()
        self._load_persisted_state()
        self.create_gui()
        self._reconcile_open_orders_safely()

    def _serialize_dt(self, dt: datetime) -> str:
        try:
            return dt.isoformat()
        except Exception:
            return datetime.min.isoformat()

    def _deserialize_dt(self, value: Any) -> datetime:
        if not value:
            return datetime.min
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return datetime.min

    def _current_state(self) -> Dict[str, Any]:
        return {
            'grid_orders': self.grid_orders,
            'trade_history': self.trade_history,
            'cumulative_return': self.cumulative_return,
            'last_trade_time': self._serialize_dt(self.last_trade_time),
        }

    def _save_state(self) -> None:
        self.persistence.save_state(self._current_state())

    def _load_persisted_state(self) -> None:
        data = self.persistence.load_state()
        if not data:
            return
        self.grid_orders = data.get('grid_orders', {}) or {}
        self.trade_history = data.get('trade_history', []) or []
        self.cumulative_return = float(data.get('cumulative_return', 1.0) or 1.0)
        self.last_trade_time = self._deserialize_dt(data.get('last_trade_time'))
        logging.info("Восстановлено grid_orders: %s", list(self.grid_orders.keys()))

    def _reconcile_open_orders_safely(self) -> None:
        try:
            # Уточнённая логика: если сохранённый ордер имеет id, проверяем его статус на бирже.
            # Если ордер не найден и запись старая (>15m) — удаляем её. Если найден — оставляем.
            now = datetime.utcnow()
            for symbol, saved in list(self.grid_orders.items()):
                saved_order = saved.get('order') or {}
                saved_id = saved_order.get('id')
                created_at = self._deserialize_dt(saved.get('created_at'))
                open_orders = self.data_provider.fetch_open_orders(symbol)
                if saved_id:
                    found = any((o.get('id') == saved_id) for o in open_orders)
                    if found:
                        logging.info("Сохранённый TP %s для %s найден среди открытых.", saved_id, symbol)
                        continue
                    # если не найден - попытка получить статус ордера напрямую
                    order_obj = None
                    try:
                        order_obj = self.data_provider.fetch_order(saved_id, symbol)
                    except Exception:
                        order_obj = None
                    # если ордер подтверждён как closed/filled -> удаляем запись и добавляем в историю, если надо
                    if order_obj:
                        status = (order_obj.get('status') or '').lower()
                        if status in ('closed', 'filled', 'canceled', 'canc'):
                            logging.info("Ордер %s для %s имеет статус %s. Удаляем запись grid_orders.", saved_id, symbol, status)
                            # Если закрыт по TP, убедитесь, что trade_history отражает закрытие; если нет — добавьте.
                            if status in ('closed', 'filled'):
                                # если нет записи о закрытой сделке — добавим для согласованности
                                if not any(t.get('symbol') == symbol and abs(float(t.get('exit_price', 0)) - float(saved.get('tp_price', 0))) < 1e-6 for t in self.trade_history):
                                    try:
                                        entry = float(saved.get('raw_entry_price', saved.get('entry_price', 0) or 0))
                                        exit_p = float(saved.get('tp_price', 0) or 0)
                                        qty = float(saved.get('quantity', 0) or 0)
                                        self.record_closed_trade(symbol, entry, exit_p, qty)
                                    except Exception:
                                        pass
                            # удаляем
                            try:
                                del self.grid_orders[symbol]
                            except KeyError:
                                pass
                            self._save_state()
                            continue
                    # если ордер не найден и запись старше 15 минут — удаляем
                    if created_at != datetime.min and (now - created_at) > timedelta(minutes=15):
                        logging.info("Удаляем устаревшую запись grid_orders для %s (ордер %s не найден и старше 15м).", symbol, saved_id)
                        del self.grid_orders[symbol]
                        self._save_state()
                        continue
                    # иначе оставляем запись временно
                    logging.info("Оставляем запись grid_orders для %s временно; ордер %s не найден но не старый.", symbol, saved_id)
                else:
                    # запись без id — если старее 15 минут, удаляем; иначе оставляем
                    if created_at != datetime.min and (now - created_at) > timedelta(minutes=15):
                        logging.info("Удаляем запись без id для %s (старее 15м).", symbol)
                        del self.grid_orders[symbol]
                        self._save_state()
                    else:
                        logging.info("Есть запись без id для %s; оставляем в состоянии.", symbol)
        except Exception as e:
            logging.error("Ошибка reconcile: %s", e)

    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Анализатор крипто-пар")
        self.root.minsize(800, 600)
        self.trade_usdt_base = tk.DoubleVar(value=30.0)
        self.tp_percentage = tk.DoubleVar(value=0.007)
        self.tab_control = ttk.Notebook(self.root)
        
        self.main_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.main_tab, text='Главная')
        self.history_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.history_tab, text='История сделок')
        self.tab_control.pack(expand=1, fill='both')

        main_frame = ttk.Frame(self.main_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        ttk.Label(left_frame, text="Доступные пары:").pack(pady=5)
        listbox_frame = ttk.Frame(left_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set, width=30)
        scrollbar.config(command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

        ttk.Label(left_frame, text="Выберите таймфрейм:").pack(pady=5)
        self.timeframe_var = tk.StringVar(value='1h')
        timeframe_combobox = ttk.Combobox(left_frame, textvariable=self.timeframe_var, state='readonly')
        timeframe_combobox['values'] = ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M')
        timeframe_combobox.pack(pady=5)
        timeframe_combobox.bind("<<ComboboxSelected>>", self.on_timeframe_change)

        ttk.Label(left_frame, text="Сумма сделки (USDT):").pack(pady=5)
        self.trade_amount_label = ttk.Label(left_frame, text=f"{self.trade_usdt_base.get():.2f} USDT")
        self.trade_amount_label.pack()
        trade_amount_scale = ttk.Scale(left_frame, from_=30, to=100, orient=tk.HORIZONTAL, variable=self.trade_usdt_base,
                                      command=lambda x: self.trade_amount_label.config(text=f"{self.trade_usdt_base.get():.2f} USDT"))
        trade_amount_scale.pack(pady=5)

        ttk.Label(left_frame, text="Тейк-профит (%):").pack(pady=5)
        self.tp_percentage_label = ttk.Label(left_frame, text=f"{self.tp_percentage.get()*100:.2f}%")
        self.tp_percentage_label.pack()
        tp_percentage_scale = ttk.Scale(left_frame, from_=0.005, to=0.03, orient=tk.HORIZONTAL, variable=self.tp_percentage,
                                       command=lambda x: self.tp_percentage_label.config(text=f"{self.tp_percentage.get()*100:.2f}%"))
        tp_percentage_scale.pack(pady=5)

        indicators_frame = ttk.Frame(left_frame)
        indicators_frame.pack(pady=10)
        trend_frame = ttk.Frame(indicators_frame)
        trend_frame.pack(fill=tk.X, pady=5)
        self.up_arrow = tk.Button(trend_frame, text="⬆", state=tk.DISABLED, bg="grey", width=5)
        self.up_arrow.pack(side=tk.LEFT)
        self.trend_title_label = ttk.Label(trend_frame, text="Тренд", font=("Helvetica", 12))
        self.trend_title_label.pack(side=tk.LEFT, padx=5)
        self.down_arrow = tk.Button(trend_frame, text="⬇", state=tk.DISABLED, bg="grey", width=5)
        self.down_arrow.pack(side=tk.LEFT)

        volume_frame = ttk.Frame(indicators_frame)
        volume_frame.pack(fill=tk.X, pady=5)
        self.volume_up_button = tk.Button(volume_frame, text="Рост", state=tk.DISABLED, bg="grey", width=10)
        self.volume_up_button.pack(side=tk.LEFT)
        self.volume_label = ttk.Label(volume_frame, text="Объёмы", font=("Helvetica", 12))
        self.volume_label.pack(side=tk.LEFT, padx=5)
        self.volume_down_button = tk.Button(volume_frame, text="Падение", state=tk.DISABLED, bg="grey", width=10)
        self.volume_down_button.pack(side=tk.LEFT)

        cvd_frame = ttk.Frame(indicators_frame)
        cvd_frame.pack(fill=tk.X, pady=5)
        self.cvd_positive_button = tk.Button(cvd_frame, text="CVD ↑", state=tk.DISABLED, bg="grey", width=10)
        self.cvd_positive_button.pack(side=tk.LEFT)
        self.cvd_label = ttk.Label(cvd_frame, text="CVD", font=("Helvetica", 12))
        self.cvd_label.pack(side=tk.LEFT, padx=5)
        self.cvd_negative_button = tk.Button(cvd_frame, text="CVD ↓", state=tk.DISABLED, bg="grey", width=10)
        self.cvd_negative_button.pack(side=tk.LEFT)

        rsi_frame = ttk.Frame(indicators_frame)
        rsi_frame.pack(fill=tk.X, pady=5)
        self.rsi_overbought_button = tk.Button(rsi_frame, text="RSI ↑", state=tk.DISABLED, bg="grey", width=10)
        self.rsi_overbought_button.pack(side=tk.LEFT)
        self.rsi_label = ttk.Label(rsi_frame, text="RSI", font=("Helvetica", 12))
        self.rsi_label.pack(side=tk.LEFT, padx=5)
        self.rsi_oversold_button = tk.Button(rsi_frame, text="RSI ↓", state=tk.DISABLED, bg="grey", width=10)
        self.rsi_oversold_button.pack(side=tk.LEFT)

        ml_advisor_frame = ttk.Frame(indicators_frame)
        ml_advisor_frame.pack(fill=tk.X, pady=5)
        self.ml_advisor_label = ttk.Label(ml_advisor_frame, text="ML Прогноз: --", font=("Helvetica", 12))
        self.ml_advisor_label.pack(side=tk.LEFT, padx=5)

        trade_frame = ttk.Frame(left_frame)
        trade_frame.pack(pady=10)
        self.buy_recommend_button = tk.Button(trade_frame, text="Купить", state=tk.DISABLED, bg="grey",
                                             command=self.on_buy_recommend, width=10)
        self.buy_recommend_button.pack(side=tk.LEFT, padx=5)
        self.trade_recommend_label = ttk.Label(trade_frame, text="Шанс сделки", font=("Helvetica", 12))
        self.trade_recommend_label.pack(side=tk.LEFT, padx=5)

        order_book_button = tk.Button(left_frame, text="Показать стакан ордеров", command=self.open_order_book_window)
        order_book_button.pack(pady=5)

        self.pnl_label = ttk.Label(left_frame, text=f"Общая прибыль: {((self.cumulative_return - 1) * 100):.2f}%")
        self.pnl_label.pack(pady=5)

        balance_frame = ttk.Frame(left_frame)
        balance_frame.pack(pady=5)
        self.total_balance_label = ttk.Label(balance_frame, text="Общий баланс: -- USDT")
        self.total_balance_label.pack()
        self.usdt_balance_label = ttk.Label(balance_frame, text="USDT: --")
        self.usdt_balance_label.pack()
        self.btc_balance_label = ttk.Label(balance_frame, text="BTC: --")
        self.btc_balance_label.pack()

        self.canvas_frame = ttk.Frame(right_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.fig = plt.figure(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.create_history_tab()

        self.update_balance_display()

    def create_history_tab(self):
        history_frame = ttk.Frame(self.history_tab, padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True)

        columns = ('timestamp', 'symbol', 'entry_price', 'exit_price', 'quantity', 'profit_percent')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='headings')
        self.history_tree.heading('timestamp', text='Время (UTC)')
        self.history_tree.heading('symbol', text='Пара')
        self.history_tree.heading('entry_price', text='Цена входа (USDT)')
        self.history_tree.heading('exit_price', text='Цена выхода (USDT)')
        self.history_tree.heading('quantity', text='Количество')
        self.history_tree.heading('profit_percent', text='Прибыль (%)')
        self.history_tree.column('timestamp', width=150)
        self.history_tree.column('symbol', width=100)
        self.history_tree.column('entry_price', width=120)
        self.history_tree.column('exit_price', width=120)
        self.history_tree.column('quantity', width=100)
        self.history_tree.column('profit_percent', width=100)

        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.update_history_display()

    def update_history_display(self):
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        for trade in self.trade_history:
            try:
                timestamp = trade.get('timestamp', '')
                symbol = trade.get('symbol', '')
                entry_price = f"{trade.get('entry_price', 0):.2f}"
                exit_price = f"{trade.get('exit_price', 0):.2f}"
                quantity = f"{trade.get('quantity', 0):.6f}"
                profit_percent = f"{trade.get('profit_percent', 0):.2f}"
                self.history_tree.insert('', tk.END, values=(
                    timestamp, symbol, entry_price, exit_price, quantity, profit_percent
                ))
            except Exception as e:
                logging.error(f"Ошибка при обновлении истории сделок: {e}")

    def record_closed_trade(self, symbol, entry_price, exit_price, quantity):
        taker_fee = 0.0018
        maker_fee = 0.0010
        effective_entry_price = entry_price * (1 + taker_fee)
        effective_exit_price = exit_price * (1 - maker_fee)
        raw_profit_pct = (effective_exit_price - effective_entry_price) / effective_entry_price * 100

        # prevent duplicate closed-trade notification by checking if grid_orders existed and had a notified_closed flag
        notified_already = False
        if symbol in self.grid_orders:
            notified_already = self.grid_orders[symbol].get('notified_closed', False)

        self.trade_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'profit_percent': raw_profit_pct
        })
        self.cumulative_return *= (1 + raw_profit_pct / 100)
        overall_profit = (self.cumulative_return - 1) * 100
        self.pnl_label.config(text=f"Общая прибыль: {overall_profit:.2f}%")
        logging.info("Сделка %s закрыта по TP: вход %.2f, выход %.2f, прибыль %.2f%% (накопительный %.2f%%)",
                     symbol, entry_price, exit_price, raw_profit_pct, overall_profit)

        # send closed trade notification only once
        if not notified_already:
            send_telegram_message(f"Сделка {symbol} закрыта по TP: вход {entry_price:.2f}, выход {exit_price:.2f}, прибыль {raw_profit_pct:.2f}%")
            if symbol in self.grid_orders:
                self.grid_orders[symbol]['notified_closed'] = True

        if symbol in self.grid_orders:
            # remove grid order after TP triggered and ensure we don't re-notify about "existing TP"
            try:
                del self.grid_orders[symbol]
                logging.info("Удалена запись grid_orders для %s после закрытия сделки. Новая покупка возможна.", symbol)
                # notify once that new purchase possible
                send_telegram_message(f"TP для {symbol} сработал. Новая покупка возможна при сигнале ≥ 55%.")
            except KeyError:
                pass

        self._save_state()
        self.update_history_display()

    def update_balance_display(self):
        try:
            balance = self.data_provider.exchange.fetch_balance()
            usdt_balance = balance['total'].get('USDT', 0)
            btc_balance = balance['total'].get('BTC', 0)
            btc_price = self.data_provider.get_current_price('BTC/USDT')
            total_balance = usdt_balance
            if btc_price is not None:
                total_balance += btc_balance * btc_price
            self.total_balance_label.config(text=f"Общий баланс: {total_balance:.2f} USDT")
            self.usdt_balance_label.config(text=f"USDT: {usdt_balance:.2f}")
            self.btc_balance_label.config(text=f"BTC: {btc_balance:.8f}")
        except Exception as e:
            logging.error(f"Ошибка обновления баланса: {e}")
            self.total_balance_label.config(text="Общий баланс: Ошибка")
            self.usdt_balance_label.config(text="USDT: Ошибка")
            self.btc_balance_label.config(text="BTC: Ошибка")

    def update_interface(self):
        try:
            timeframe = self.timeframe_var.get()
            self.data_provider.clear_cache()
            self.find_pairs_thread(timeframe)
            self.update_balance_display()
            self.update_history_display()
            now = datetime.now()
            if now - getattr(self, 'last_training_time', datetime.min) >= timedelta(minutes=3) or not self.ml_predictor.is_trained:
                self.train_ml_model()
                self.last_training_time = now
        except Exception as e:
            logging.error(f"Ошибка обновления интерфейса: {e}", exc_info=True)

    def find_pairs_thread(self, timeframe):
        symbols = self.data_provider.symbols
        suitable_pairs = []
        with ThreadPoolExecutor(max_workers=max(1, len(symbols))) as executor:
            future_to_symbol = {executor.submit(self.data_provider.get_ohlcv, symbol, timeframe): symbol for symbol in symbols}
            for future in as_completed(future_to_symbol):
                sym = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        suitable_pairs.append(sym)
                except Exception as e:
                    logging.error(f"Ошибка при получении данных для пары {sym}: {e}", exc_info=True)
        self.update_listbox(suitable_pairs, timeframe)

    def update_listbox(self, pairs, timeframe):
        selection = self.listbox.curselection()
        current = self.listbox.get(selection[0]) if selection else None
        self.listbox.delete(0, tk.END)
        for symbol in pairs:
            self.listbox.insert(tk.END, symbol)
        if current in pairs:
            idx = pairs.index(current)
            self.listbox.selection_set(idx)
        elif pairs:
            self.listbox.selection_set(0)
        else:
            return
        self.listbox.event_generate('<<ListboxSelect>>')
        logging.info(f"Подходящие пары: {pairs} с таймфреймом {timeframe}")

    def on_select(self, event=None):
        selection = self.listbox.curselection()
        if selection:
            symbol = self.listbox.get(selection[0])
            self.update_data(symbol, self.timeframe_var.get())

    def update_data(self, symbol, timeframe):
        data = self.data_provider.get_ohlcv(symbol, timeframe)
        if data is not None:
            self.update_plots(data, symbol)
        else:
            messagebox.showerror("Ошибка", f"Нет данных для {symbol}")

    def update_plots(self, data, symbol):
        current_price = self.data_provider.get_current_price(symbol)

        # Автоматический расчёт pivot: пытаемся взять предыдущий дневной бар
        pivot_levels = {}
        probs = {}
        try:
            prev_day = self.data_provider.get_ohlcv(symbol, timeframe='1d', limit=2)
            if prev_day is not None and len(prev_day) >= 2:
                # используем предыдущий день (второй из массива)
                pivot_levels = MarketAnalyzer.calculate_pivots_from_ohlcv(prev_day[-2])
            else:
                # fallback: рассчитываем по видимым барам текущего timeframe
                pivot_levels = MarketAnalyzer.calculate_pivots_from_ohlcv(data)
        except Exception as e:
            logging.error("Ошибка при расчёте pivot_levels: %s", e)
            pivot_levels = MarketAnalyzer.calculate_pivots_from_ohlcv(data)

        probs = MarketAnalyzer.generate_pivot_probs(pivot_levels)

        MarketAnalyzer.plot_data(data, symbol, fig=self.fig, current_price=current_price, pivot_levels=pivot_levels, probs=probs)
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

        self.update_trend_indicator(data)
        self.update_volume_buttons(data)
        self.update_cvd_buttons(data)
        self.update_rsi_buttons(data)
        self.update_trade_recommendation(data, symbol)
        self.update_ml_advisor(data, symbol)
        if current_price is not None:
            self.trend_title_label.config(text=f"{symbol} — {current_price:.2f} USDT")

    def update_trend_indicator(self, data):
        closes = data[:, 4]
        ma = np.mean(closes[-10:]) if len(closes) >= 10 else np.mean(closes)
        if closes[-1] > ma:
            self.up_arrow.config(bg="green")
            self.down_arrow.config(bg="grey")
        elif closes[-1] < ma:
            self.up_arrow.config(bg="grey")
            self.down_arrow.config(bg="red")
        else:
            self.up_arrow.config(bg="grey")
            self.down_arrow.config(bg="grey")

    def update_volume_buttons(self, data):
        volumes = data[:, 5]
        current = volumes[-1]
        previous = volumes[-2] if len(volumes) >= 2 else current
        change = current - previous
        pct_change = (change / previous * 100) if previous != 0 else 0
        if change > 0:
            self.volume_up_button.config(bg="green", state=tk.NORMAL)
            self.volume_down_button.config(bg="grey", state=tk.DISABLED)
            self.volume_label.config(text=f"Объём {pct_change:.2f}% (+{change:.2f})")
        elif change < 0:
            self.volume_up_button.config(bg="grey", state=tk.DISABLED)
            self.volume_down_button.config(bg="red", state=tk.NORMAL)
            self.volume_label.config(text=f"Объём {pct_change:.2f}% ({change:.2f})")
        else:
            self.volume_up_button.config(bg="grey", state=tk.DISABLED)
            self.volume_down_button.config(bg="grey", state=tk.DISABLED)
            self.volume_label.config(text="Объём без изменений")

    def update_cvd_buttons(self, data):
        cvd = MarketAnalyzer.calculate_cvd(data)
        if len(cvd) >= 2 and cvd[-1] > cvd[-2]:
            self.cvd_positive_button.config(bg="green", state=tk.NORMAL)
            self.cvd_negative_button.config(bg="grey", state=tk.DISABLED)
            self.cvd_label.config(text="CVD Положительный")
        elif len(cvd) >= 2 and cvd[-1] < cvd[-2]:
            self.cvd_positive_button.config(bg="grey", state=tk.DISABLED)
            self.cvd_negative_button.config(bg="red", state=tk.NORMAL)
            self.cvd_label.config(text="CVD Отрицательный")
        else:
            self.cvd_positive_button.config(bg="grey", state=tk.DISABLED)
            self.cvd_negative_button.config(bg="grey", state=tk.DISABLED)
            self.cvd_label.config(text="CVD без изменений")

    def update_rsi_buttons(self, data):
        closes = data[:, 4]
        rsi = MarketAnalyzer.calculate_rsi(closes)
        current_rsi = rsi[-1]
        if current_rsi > 70:
            self.rsi_overbought_button.config(bg="red", state=tk.NORMAL)
            self.rsi_oversold_button.config(bg="grey", state=tk.DISABLED)
            self.rsi_label.config(text=f"RSI Перекуплен: {current_rsi:.2f}")
        elif current_rsi < 30:
            self.rsi_overbought_button.config(bg="grey", state=tk.DISABLED)
            self.rsi_oversold_button.config(bg="green", state=tk.NORMAL)
            self.rsi_label.config(text=f"RSI Перепродан: {current_rsi:.2f}")
        else:
            self.rsi_overbought_button.config(bg="grey", state=tk.DISABLED)
            self.rsi_oversold_button.config(bg="grey", state=tk.DISABLED)
            self.rsi_label.config(text=f"RSI Нейтральный: {current_rsi:.2f}")

    def update_trade_recommendation(self, data, symbol):
        closes = data[:, 4]
        ma = np.mean(closes[-10:]) if len(closes) >= 10 else np.mean(closes)
        trend_change = (closes[-1] - ma) / ma if ma != 0 else 0
        volumes = data[:, 5]
        volume_change = (volumes[-1] - (volumes[-2] if len(volumes) >= 2 else volumes[-1])) / (volumes[-2] if len(volumes) >= 2 and volumes[-2] != 0 else 1)
        cvd = MarketAnalyzer.calculate_cvd(data)
        cvd_change = (cvd[-1] - (cvd[-2] if len(cvd) >= 2 else cvd[-1])) / (cvd[-2] if len(cvd) >= 2 and cvd[-2] != 0 else 1)
        rsi = MarketAnalyzer.calculate_rsi(closes)
        rsi_signal = (rsi[-1] - 50) / 50
        support, resistance = MarketAnalyzer.get_support_resistance(data)
        support_resistance_signal = 0
        if closes[-1] > resistance:
            support_resistance_signal = 1
        elif closes[-1] < support:
            support_resistance_signal = -1

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))
        trend_signal = sigmoid(trend_change * 10) * 2 - 1
        volume_signal = sigmoid(volume_change * 10) * 2 - 1
        cvd_signal = sigmoid(cvd_change * 10) * 2 - 1
        rsi_signal = sigmoid(rsi_signal * 10) * 2 - 1

        order_book = self.data_provider.get_order_book(symbol, limit=50)
        ml_prediction = self.ml_predictor.predict(data, order_book) if order_book else 0
        ml_signal = ml_prediction * 2 - 1

        adx_arr = MarketAnalyzer.calculate_adx(data, period=14)
        adx_signal = 0
        if len(adx_arr) > 0:
            adx_value = adx_arr[-1]
            adx_signal = 1 if adx_value > 25 else 0

        additional_weight = 0.1
        total_signal = (
            trend_signal * self.weights['trend'] +
            volume_signal * self.weights['volume'] +
            cvd_signal * self.weights['cvd'] +
            rsi_signal * self.weights['rsi'] +
            support_resistance_signal * self.weights['support_resistance'] +
            ml_signal * self.weights['ml_prediction'] +
            adx_signal * additional_weight
        ) / (sum(self.weights.values()) + additional_weight)
        probability = (total_signal + 1) / 2 * 100
        probability = max(min(probability, 100), 0)

        now = datetime.now()
        if now - self.start_time < self.initial_delay:
            if probability > 55:
                self.buy_recommend_button.config(bg="green", state=tk.NORMAL)
                self.trade_recommend_label.config(text=f"Рекомендация: Покупка {probability:.2f}% (отложено)")
            else:
                self.buy_recommend_button.config(bg="grey", state=tk.DISABLED)
                self.trade_recommend_label.config(text=f"Рекомендация: Нейтрально ({probability:.2f}%)")
            return

        trade_threshold_buy = 55
        trade_usdt = self.trade_usdt_base.get()
        current_price = self.data_provider.get_current_price(symbol)
        if current_price is None:
            logging.error("Не удалось получить текущую цену для автоматической торговли.")
            self.trade_recommend_label.config(text="Ошибка: нет текущей цены")
            return
        quantity = trade_usdt / current_price

        taker_fee = 0.0018
        maker_fee = 0.0010
        effective_entry_price = current_price * (1 + taker_fee)

        ma20 = np.mean(data[:, 4][-20:]) if len(data[:,4])>=20 else np.mean(data[:,4])
        if current_price < ma20 * 0.95:
            logging.info("Цена %.2f ниже MA20 (%.2f) на ≥5%%, покупка заблокирована для %s.", current_price, ma20, symbol)
            existing = self.grid_orders.get(symbol, {})
            if not existing.get('notified_blocked_drop'):
                send_telegram_message(f"Покупка {symbol} заблокирована: цена {current_price:.2f} ниже MA20 ({ma20:.2f}) на ≥5%")
                existing['notified_blocked_drop'] = True
                self.grid_orders[symbol] = existing
                self._save_state()
            self.trade_recommend_label.config(text=f"Покупка заблокирована: падение ≥5% ({probability:.2f}%)")
            self.buy_recommend_button.config(bg="grey", state=tk.DISABLED)
            return
        else:
            if symbol in self.grid_orders and self.grid_orders[symbol].get('notified_blocked_drop'):
                self.grid_orders[symbol].pop('notified_blocked_drop', None)
                self._save_state()

        if current_price < ma20 * 0.98:
            logging.info("Цена %.2f не восстановилась до MA20*0.98 (%.2f) для %s. Покупка заблокирована.", current_price, ma20 * 0.98, symbol)
            existing = self.grid_orders.get(symbol, {})
            if not existing.get('notified_blocked_recovery'):
                send_telegram_message(f"Покупка {symbol} заблокирована: цена {current_price:.2f} не восстановилась до MA20*0.98 ({ma20 * 0.98:.2f})")
                existing['notified_blocked_recovery'] = True
                self.grid_orders[symbol] = existing
                self._save_state()
            self.trade_recommend_label.config(text=f"Ожидание восстановления: <MA20*0.98 ({probability:.2f}%)")
            self.buy_recommend_button.config(bg="grey", state=tk.DISABLED)
            return
        else:
            if symbol in self.grid_orders and self.grid_orders[symbol].get('notified_blocked_recovery'):
                self.grid_orders[symbol].pop('notified_blocked_recovery', None)
                self._save_state()

        # --- Обновлённая логика: разрешаем повторную покупку только если цена отличается > threshold_pct от предыдущей покупки ---
        if symbol in self.grid_orders:
            prev = self.grid_orders[symbol]
            # Для сравнения используем raw_entry_price (без комиссии) если он есть; иначе fallback на entry_price
            prev_entry = float(prev.get('raw_entry_price', prev.get('entry_price', 0) or 0))
            current_price_val = float(current_price or 0)
            threshold_pct = 1.0  # порог отклонения в процентах

            # Debug logging — показывает реальные числа
            logging.debug("DBG compare: symbol=%s prev_entry=%r current_price=%r tp_price=%r types=%s",
                          symbol,
                          prev_entry,
                          current_price_val,
                          self.grid_orders.get(symbol, {}).get('tp_price'),
                          (type(prev_entry), type(current_price_val)))

            # Если нет сохранённой цены предыдущей покупки — оставляем старое поведение (пропускаем)
            if prev_entry == 0 or current_price_val == 0:
                tp_price = self.grid_orders[symbol].get('tp_price', 0)
                if not self.grid_orders[symbol].get('notified_exists'):
                    logging.info("Для %s уже есть активный TP на уровне %.2f. Покупка пропущена.", symbol, tp_price)
                    send_telegram_message(f"Покупка {symbol} пропущена: уже есть активный TP на уровне {tp_price:.2f}")
                    self.grid_orders[symbol]['notified_exists'] = True
                    self._save_state()
                self.trade_recommend_label.config(text=f"Покупка пропущена: активный TP {tp_price:.2f} ({probability:.2f}%)")
                self.buy_recommend_button.config(bg="grey", state=tk.DISABLED)
                return

            price_diff_pct = (current_price_val - prev_entry) / prev_entry * 100  # положительное — выше, отрицательное — ниже

            if abs(price_diff_pct) <= threshold_pct:
                tp_price = self.grid_orders[symbol].get('tp_price', 0)
                if not self.grid_orders[symbol].get('notified_exists'):
                    logging.info("Покупка для %s пропущена: текущая цена %.4f слишком близка к предыдущей покупке %.4f (разница %.3f%%)", symbol, current_price_val, prev_entry, price_diff_pct)
                    send_telegram_message(f"Покупка {symbol} пропущена: цена {current_price_val:.4f} близка к предыдущей покупке {prev_entry:.4f} (Δ {price_diff_pct:.3f}%)")
                    self.grid_orders[symbol]['notified_exists'] = True
                    self._save_state()
                self.trade_recommend_label.config(text=f"Пропущено: цена близка к предыдущей ({price_diff_pct:.2f}%)")
                self.buy_recommend_button.config(bg="grey", state=tk.DISABLED)
                return
            else:
                logging.info("Для %s предыдущая покупка: %.4f, текущая цена: %.4f (Δ %.3f%%) — покупка разрешена", symbol, prev_entry, current_price_val, price_diff_pct)
                if self.grid_orders[symbol].get('notified_exists'):
                    self.grid_orders[symbol].pop('notified_exists', None)
                    self._save_state()
                # продолжаем выполнение — разрешаем новую покупку

        if probability > trade_threshold_buy:
            desired_tp_price = current_price * (1 + self.tp_percentage.get() + maker_fee)
            if desired_tp_price * (1 - maker_fee) <= effective_entry_price:
                logging.info("TP %.2f не обеспечит прибыль после комиссий для %s.", desired_tp_price, symbol)
                self.trade_recommend_label.config(text=f"Покупка заблокирована: убыточный TP ({probability:.2f}%)")
                return
            else:
                if now - getattr(self, 'last_trade_time', datetime.min) >= self.trade_cooldown:
                    order = self.data_provider.execute_order(symbol, 'buy', quantity, order_type='market')
                    if order:
                        logging.info("Авто-покупка выполнена для %s: цена %.2f, объём %.6f, TP %.2f", symbol, current_price, quantity, desired_tp_price)
                        self.trade_recommend_label.config(text=f"Авто-покупка выполнена по {current_price:.2f}")
                        self.last_trade_time = now
                        tp_order = self.data_provider.execute_order(
                            symbol, 'sell', quantity, price=desired_tp_price, order_type='limit'
                        )
                        if tp_order:
                            logging.info("TP установлен для %s: продажа по %.2f", symbol, desired_tp_price)
                            # Сохранение и нормализация: сохраняем raw_entry_price (без комиссии) и entry_price (с комиссией)
                            raw_entry = current_price
                            self.grid_orders[symbol] = {
                                'raw_entry_price': raw_entry,
                                'entry_price': effective_entry_price,
                                'tp_price': desired_tp_price,
                                'quantity': quantity,
                                'side': 'long',
                                'order': {
                                    'id': tp_order.get('id'),
                                    'type': tp_order.get('type', 'limit'),
                                    'price': tp_order.get('price', desired_tp_price),
                                },
                                'created_at': datetime.utcnow().isoformat(),
                                'notified_set': True,
                                'notified_exists': True,
                                'notified_closed': False,
                            }
                            # логируем полный ответ ордера для диагностики
                            logging.debug("TP order response for %s: %r", symbol, tp_order)
                            # send only one notification about TP set
                            send_telegram_message(f"Покупка {symbol}: цена {current_price:.2f}, TP {desired_tp_price:.2f}, объём {quantity:.6f}")
                            self._save_state()
                        else:
                            logging.error("Ошибка установки тейк-профита для покупки %s", symbol)
                    else:
                        logging.warning("Ошибка автоматической покупки для %s", symbol)
                        send_telegram_message(f"Ошибка авто-покупки для {symbol}")
        else:
            self.buy_recommend_button.config(bg="grey", state=tk.DISABLED)
            self.trade_recommend_label.config(text=f"Рекомендация: Нейтрально ({probability:.2f}%)")

        self._save_state()

    def update_ml_advisor(self, data, symbol):
        order_book = self.data_provider.get_order_book(symbol, limit=50)
        ml_probability = self.ml_predictor.predict_proba(data, order_book) if order_book else 0.5
        if ml_probability > 0.5:
            forecast = "Покупка"
            marker = "▲"
            confidence = ml_probability * 100
        else:
            forecast = "Продажа"
            marker = "▼"
            confidence = (1 - ml_probability) * 100
        ml_color = "green" if forecast == "Покупка" else "red"
        self.ml_advisor_label.config(text=f"ML Прогноз: {forecast} ({confidence:.2f}%) {marker}", foreground=ml_color)

    def train_ml_model(self):
        all_data = []
        all_order_books = []

        def fetch_symbol_data(symbol):
            ohlcv = self.data_provider.get_ohlcv(symbol, '1h', 200)
            order_book = self.data_provider.get_order_book(symbol, 50)
            return symbol, ohlcv, order_book

        symbols = self.data_provider.symbols
        with ThreadPoolExecutor(max_workers=max(1, len(symbols))) as executor:
            futures = [executor.submit(fetch_symbol_data, symbol) for symbol in symbols]
            for future in as_completed(futures):
                try:
                    symbol, ohlcv, order_book = future.result()
                    if ohlcv is not None and order_book is not None:
                        all_data.append(ohlcv)
                        all_order_books.append(order_book)
                except Exception as e:
                    logging.error(f"Ошибка при обучении ML для символа: {e}", exc_info=True)
        if all_data and all_order_books:
            combined = np.concatenate(all_data, axis=0)
            combined_order_books = {
                'bids': np.concatenate([ob.get('bids', []) for ob in all_order_books], axis=0) if all_order_books else [],
                'asks': np.concatenate([ob.get('asks', []) for ob in all_order_books], axis=0) if all_order_books else [],
            }
            self.ml_predictor.train(combined, combined_order_books)

    def on_buy_recommend(self):
        symbol = self.listbox.get(tk.ACTIVE)
        if not symbol:
            messagebox.showwarning("Внимание", "Выберите крипто-пару для покупки.")
            return
        self.open_trade_window(symbol, 'Купить')

    def open_trade_window(self, symbol, action):
        top = tk.Toplevel()
        top.title(f"{action} {symbol}")
        ttk.Label(top, text="Текущая цена (USDT):").pack()
        current_price = self.data_provider.get_current_price(symbol)
        if current_price is None:
            messagebox.showerror("Ошибка", "Не удалось получить цену.")
            top.destroy()
            return
        ttk.Label(top, text=f"{current_price:.2f} USDT").pack()
        ttk.Label(top, text="Сумма (USDT):").pack()
        amount_entry = tk.Entry(top)
        amount_entry.pack()
        quantity_var = tk.StringVar(value="0.0")

        def update_quantity(*args):
            try:
                amount = float(amount_entry.get())
                quantity = amount / current_price
                quantity_var.set(f"{quantity:.6f}")
            except ValueError:
                quantity_var.set("0.0")

        amount_entry.bind("<KeyRelease>", update_quantity)
        ttk.Label(top, text="Количество:").pack()
        ttk.Label(top, textvariable=quantity_var).pack()

        def confirm():
            try:
                amount = float(amount_entry.get())
                price = current_price
                quantity = amount / price
                messagebox.showinfo("Сделка", f"{action} {symbol} по цене {price:.2f} USDT на сумму {amount:.2f} USDT\nКоличество: {quantity:.6f}")
                top.destroy()
            except ValueError:
                messagebox.showerror("Ошибка", "Введите корректное число.")

        tk.Button(top, text="Подтвердить", command=confirm).pack()

    def open_order_book_window(self):
        symbol = self.listbox.get(tk.ACTIVE)
        if not symbol:
            messagebox.showwarning("Внимание", "Выберите крипто-пару для просмотра стакана ордеров.")
            return
        order_book = self.data_provider.get_order_book(symbol, limit=50)
        if order_book is None:
            messagebox.showerror("Ошибка", f"Не удалось получить стакан ордеров для {symbol}")
            return
        top = tk.Toplevel()
        top.title(f"Стакан ордеров для {symbol}")
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        bids_frame = ttk.Frame(top, padding="10")
        bids_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(bids_frame, text="Bids (покупка)").pack()
        bids_list = tk.Listbox(bids_frame, width=30)
        bids_list.pack(fill=tk.BOTH, expand=True)
        for bid in bids:
            try:
                bids_list.insert(tk.END, f"Цена: {bid[0]:.2f}, Кол-во: {bid[1]:.4f}")
            except Exception:
                pass
        asks_frame = ttk.Frame(top, padding="10")
        asks_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ttk.Label(asks_frame, text="Asks (продажа)").pack()
        asks_list = tk.Listbox(asks_frame, width=30)
        asks_list.pack(fill=tk.BOTH, expand=True)
        for ask in asks:
            try:
                asks_list.insert(tk.END, f"Цена: {ask[0]:.2f}, Кол-во: {ask[1]:.4f}")
            except Exception:
                pass

    def on_timeframe_change(self, event):
        self.data_provider.clear_cache()
        self.update_interface()

    def run(self):
        self.start_auto_update()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def start_auto_update(self):
        self.auto_update = True
        self.auto_update_loop()

    def auto_update_loop(self):
        if getattr(self, 'auto_update', False):
            try:
                self.update_interface()
            except Exception as e:
                logging.error(f"Ошибка в auto_update_loop: {e}", exc_info=True)
            self.root.after(30000, self.auto_update_loop)

    def on_closing(self):
        self.auto_update = False
        try:
            self._save_state()
        finally:
            self.root.destroy()

class CryptoApp:
    def __init__(self):
        api_key = "**************************************"
        api_secret = "**************************************"
        symbols = ['BTC/USDT']
        base_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_provider = CryptoDataProvider(api_key, api_secret, symbols)
        self.ml_predictor = MLPredictor()
        self.persistence = PersistenceManager(base_dir)
        self.gui = CryptoGUI(self.data_provider, self.ml_predictor, self.persistence)

    def run(self):
        self.gui.run()

if __name__ == "__main__":
    app = CryptoApp()
    app.run()
