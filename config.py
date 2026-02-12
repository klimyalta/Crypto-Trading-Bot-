import os

"""
Конфигурация проекта "Crypto Trading Bot — Торговый бот с машинным обучением".
Здесь заданы реальные ключи/токены и базовые настройки.
"""

# === Ваши реальные ключи и токены ===
TELEGRAM_BOT_TOKEN = "7755283099:AAHQ5SZYzOgQZxC-fyDHxKvtTKzkmfaFdFQ"
TELEGRAM_CHAT_ID = "1467823964"
BYBIT_API_KEY = "UiAWRQeCZsM63ViQUY"
BYBIT_API_SECRET = "wywrp3tMO2VQDths20SmFqaIV2tJmgFaSPWj"

# === Прочие настройки ===
DEFAULT_SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT").split(",")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "ml_model.pkl")
STATE_FILENAME = os.getenv("STATE_FILENAME", "state.json")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
