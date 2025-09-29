"""
Модуль хранения буферов данных и вспомогательных функций доступа.
Содержит глобальные очереди, блокировки и утилиту получения DataFrame.
"""

from collections import deque
from threading import Lock
import pandas as pd

from .config import config

# Глобальные буферы и блокировки
buffer_lock = Lock()
fivesec_buffer = deque(maxlen=config["data"]["buffer_size"])
orderbook_buffer = deque(maxlen=config["data"]["buffer_orderbook_size"])  # Новый буфер для стакана
fivesec_predictions = deque(maxlen=config["data"]["buffer_size"])
fivesec_prediction_file_lock = Lock()

def get_current_buffer_df():
    """
    Преобразует текущий буфер 5-секундных данных в pandas.DataFrame
    с индексом timestamp (DatetimeIndex). Удаляет дубликаты.
    """
    with buffer_lock:
        df = pd.DataFrame(fivesec_buffer)
    if df.empty:
        return None
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df.sort_index()

def get_current_orderbook_df():
    """
    Преобразует текущий буфер стакана в pandas.DataFrame
    с индексом timestamp (DatetimeIndex).
    """
    with buffer_lock:
        df = pd.DataFrame(orderbook_buffer)
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df.sort_index()