import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from typing import Any

from market_analyzer import MarketAnalyzer

class CryptoGUI:
    """
    Упрощённый GUI-модуль. Содержит только отображение списка пар, график и историю.
    Логика торговли и reconcile вынесена из GUI.
    """
    def __init__(self, data_provider, ml_predictor, persistence):
        self.data_provider = data_provider
        self.ml_predictor = ml_predictor
        self.persistence = persistence
        self.grid_orders = {}
        self.trade_history = []
        self.cumulative_return = 1.0
        self.last_trade_time = datetime.min
        self.start_time = datetime.now()
        self.trade_usdt_base = tk.DoubleVar(value=30.0)
        self.tp_percentage = tk.DoubleVar(value=0.007)
        self._load_state()
        self._build_ui()

    def _load_state(self):
        data = self.persistence.load_state()
        if not data:
            return
        self.grid_orders = data.get("grid_orders", {}) or {}
        self.trade_history = data.get("trade_history", []) or []
        self.cumulative_return = float(data.get("cumulative_return", 1.0) or 1.0)
        last = data.get("last_trade_time")
        try:
            self.last_trade_time = datetime.fromisoformat(last) if last else datetime.min
        except Exception:
            self.last_trade_time = datetime.min

    def _current_state(self):
        return {
            "grid_orders": self.grid_orders,
            "trade_history": self.trade_history,
            "cumulative_return": self.cumulative_return,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else ""
        }

    def _save_state(self):
        self.persistence.save_state(self._current_state())

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Crypto Analyzer")
        self.root.minsize(800, 600)
        self.tab_control = ttk.Notebook(self.root)
        self.main_tab = ttk.Frame(self.tab_control)
        self.history_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.main_tab, text="Main")
        self.tab_control.add(self.history_tab, text="History")
        self.tab_control.pack(expand=1, fill="both")

        left = ttk.Frame(self.main_tab, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self.main_tab, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="Pairs").pack()
        self.listbox = tk.Listbox(left, width=30)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        ttk.Label(left, text="Timeframe").pack()
        self.timeframe_var = tk.StringVar(value="1h")
        cb = ttk.Combobox(left, textvariable=self.timeframe_var, state="readonly")
        cb["values"] = ("1m","5m","15m","30m","1h","4h","1d")
        cb.pack()

        self.fig = plt.Figure(figsize=(8,4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.create_history_tab()
        self.update_listbox(self.data_provider.symbols, self.timeframe_var.get())

    def create_history_tab(self):
        frame = ttk.Frame(self.history_tab, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        cols = ("timestamp","symbol","entry_price","exit_price","quantity","profit_percent")
        self.history_tree = ttk.Treeview(frame, columns=cols, show="headings")
        for c in cols:
            self.history_tree.heading(c, text=c)
            self.history_tree.column(c, width=120)
        self.history_tree.pack(fill=tk.BOTH, expand=True)

    def update_listbox(self, pairs, timeframe):
        self.listbox.delete(0, tk.END)
        for p in pairs:
            self.listbox.insert(tk.END, p)
        if pairs:
            self.listbox.selection_set(0)
            self.on_select()

    def on_select(self, event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        symbol = self.listbox.get(sel[0])
        self.update_data(symbol, self.timeframe_var.get())

    def update_data(self, symbol, timeframe):
        data = self.data_provider.get_ohlcv(symbol, timeframe)
        if data is None:
            messagebox.showerror("Error", f"No data for {symbol}")
            return
        self.update_plots(data, symbol)

    def update_plots(self, data, symbol):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        dates = MarketAnalyzer.convert_timestamps(data[:,0])
        ax.plot(dates, data[:,4], label="Close", color="blue")
        ax.set_title(symbol)
        ax.xaxis_date()
        self.canvas.draw_idle()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def on_close(self):
        try:
            self._save_state()
        finally:
            self.root.destroy()
