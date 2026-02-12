import os
from logger_setup import setup_logging
from config import BYBIT_API_KEY, BYBIT_API_SECRET, DEFAULT_SYMBOLS
from persistence import PersistenceManager
from data_provider import CryptoDataProvider
from ml_predictor import MLPredictor
from gui import CryptoGUI

def main():
    logger = setup_logging()
    base_dir = os.path.dirname(os.path.realpath(__file__))
    persistence = PersistenceManager(base_dir)
    data_provider = CryptoDataProvider(BYBIT_API_KEY, BYBIT_API_SECRET, DEFAULT_SYMBOLS)
    ml = MLPredictor()
    gui = CryptoGUI(data_provider, ml, persistence)
    gui.run()

if __name__ == "__main__":
    main()
