"""
Модуль хранения буферов данных и вспомогательных функций доступа.
Содержит глобальные очереди, блокировки и утилиту получения DataFrame.
"""

from collections import deque
from threading import Lock
import pandas as pd
import hashlib  # Для хэша

from .config import config, logger

# Глобальные буферы и блокировки
buffer_lock = Lock()
fivesec_buffer = deque(maxlen=config["data"]["buffer_size"])
orderbook_buffer = deque(maxlen=config["data"]["buffer_orderbook_size"])  # Теперь хранит предвычисленные фичи
fivesec_predictions = deque(maxlen=config["data"]["buffer_size"])
fivesec_prediction_file_lock = Lock()

# Кэш для df
_cached_fivesec_df = None
_cached_fivesec_hash = None
_cached_orderbook_df = None
_cached_orderbook_hash = None

def _get_buffer_hash(buffer_deque):
    """Простой хэш по len и last timestamp"""
    if not buffer_deque:
        return None
    last_item = buffer_deque[-1]
    # Исправление: конкатенируем строки, затем encode
    hash_input = str(len(buffer_deque)) + str(last_item.get("timestamp", ""))
    return hashlib.md5(hash_input.encode()).hexdigest()

def get_current_buffer_df():
    global _cached_fivesec_df, _cached_fivesec_hash
    with buffer_lock:
        current_hash = _get_buffer_hash(fivesec_buffer)
        if current_hash == _cached_fivesec_hash and _cached_fivesec_df is not None:
            df = _cached_fivesec_df.copy()
        else:
            df = pd.DataFrame(list(fivesec_buffer))
            _cached_fivesec_df = df.copy()
            _cached_fivesec_hash = current_hash
            logger.info(f"Raw kline buffer df: shape={df.shape}, NaN={df.isna().sum().sum()}", extra={'source': 'buffers'})
    if df.empty:
        return None
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df.sort_index()

def get_current_orderbook_df():
    global _cached_orderbook_df, _cached_orderbook_hash
    with buffer_lock:
        current_hash = _get_buffer_hash(orderbook_buffer)
        if current_hash == _cached_orderbook_hash and _cached_orderbook_df is not None:
            df = _cached_orderbook_df.copy()
        else:
            df = pd.DataFrame(list(orderbook_buffer))
            _cached_orderbook_df = df.copy()
            _cached_orderbook_hash = current_hash
            logger.info(f"Raw orderbook buffer df: shape={df.shape}, NaN={df.isna().sum().sum()}", extra={'source': 'buffers'})
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df.sort_index()