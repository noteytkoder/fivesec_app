# data_handler/buffers.py
"""
Модуль хранения буферов данных и вспомогательных функций доступа.
Содержит глобальные очереди, блокировки и утилиту получения DataFrame.
"""

from collections import deque
from threading import Lock
from logger import pd
import hashlib
import numpy as np
from .config import config, logger

# Глобальные буферы и блокировки
buffer_lock = Lock()
fivesec_buffer = deque(maxlen=config["data"]["buffer_size"])
orderbook_buffer = deque(maxlen=config["data"]["buffer_size"])
fivesec_predictions = deque(maxlen=config["data"]["buffer_size"])
fivesec_prediction_file_lock = Lock()

# Кэш для df
_cached_fivesec_df = None
_cached_fivesec_hash = None
_cached_orderbook_df = None
_cached_orderbook_hash = None

def _get_buffer_hash(buffer_deque):
    if not buffer_deque:
        return None
    last_item = buffer_deque[-1]
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
            logger.debug(f"Raw kline buffer df: shape={df.shape}, NaN={df.isna().sum().sum()}", extra={'source': 'buffers'})
    if df.empty:
        return None
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    logger.debug(f"kline DF SAMPLE:\n{sample_tail_head(df)}")
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
            logger.debug(f"Raw orderbook buffer df: shape={df.shape}, NaN={df.isna().sum().sum()}", extra={'source': 'buffers'})
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    logger.debug(f"get_current_orderbook_df: shape={df.shape} NaN={df.isna().sum().sum()} Inf={np.isinf(df.values).any()}")
    logger.debug(f"OB DF SAMPLE:\n{sample_tail_head(df)}")
    repeated_mid = (df['mid_price'].diff() == 0).astype(int).groupby((df['mid_price'].diff() != 0).cumsum()).sum().max()
    logger.debug(f"OB max constant-mid run={repeated_mid}")
    if (df['imbalance_10'].abs() > 1).any():
        logger.warning("get_current_orderbook_df: imbalance >1 detected")
    return df.sort_index()

def sample_tail_head(df, n=3):
    if df.empty:
        return "Empty DataFrame"
    stats = f"shape={df.shape} NaN={df.isna().sum().sum()}"
    # Проверяем Inf только для числовых столбцов
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_check = np.isinf(df[numeric_cols].values).any() if not df[numeric_cols].empty else False
    stats += f" Inf={inf_check}"
    dups = f"duplicates_index={df.index.duplicated().sum()}"
    gaps = f"index_gaps_sec: {df.index.to_series().diff().dt.total_seconds().describe().round(2).to_dict()}" if isinstance(df.index, pd.DatetimeIndex) else "index_gaps_sec: Not a DatetimeIndex"
    return f"head:\n{df.head(n)}\n\ntail:\n{df.tail(n)}\n\n{stats}\n{dups}\n{gaps}"