import logging
from logging import Handler
from typing import Optional
import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

class TelegramHandler(Handler):
    def __init__(self, token: Optional[str], chat_id: Optional[str]):
        super().__init__()
        self.token = token
        self.chat_id = chat_id

    def emit(self, record):
        if not self.token or not self.chat_id:
            return
        if record.levelno >= logging.ERROR:
            try:
                msg = self.format(record)
                url = f"https://api.telegram.org/bot{self.token}/sendMessage"
                requests.get(url, params={"chat_id": self.chat_id, "text": msg}, timeout=5)
            except Exception:
                pass

def setup_logging():
    level = getattr(logging, __import__("config").LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    tg = TelegramHandler(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    tg.setLevel(logging.ERROR)
    logger.addHandler(tg)
    return logger
