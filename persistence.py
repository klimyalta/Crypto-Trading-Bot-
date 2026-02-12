import os
import json
import threading
import logging
from typing import Any, Dict

class PersistenceManager:
    def __init__(self, base_dir: str, filename: str = "state.json"):
        self.state_path = os.path.join(base_dir, filename)
        self._lock = threading.Lock()

    def load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.state_path):
            logging.info("State file not found: %s", self.state_path)
            return {}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error("Failed to load state: %s", e)
            return {}

    def save_state(self, state: Dict[str, Any]) -> None:
        tmp = self.state_path + ".tmp"
        with self._lock:
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self.state_path)
                logging.info("State saved: %s", self.state_path)
            except Exception as e:
                logging.error("Failed to save state: %s", e)
