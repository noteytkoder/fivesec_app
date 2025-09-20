"""
Утилиты для Dashboard: подготовка данных, кэш.
"""

import pandas as pd
from . import __name__ as pkgname
from data_handler import process_data_for_model
from config_manager import load_config

config = load_config()

_cached_df = None
_cached_timestamp = None

def prepare_data(data_copy):
    """
    Принимает список записей (из fivesec_buffer) и возвращает обработанный DataFrame
    ready для рисования (5s, с индикаторами).
    Кэширует последний результат по timestamp.
    """
    global _cached_df, _cached_timestamp
    latest_timestamp = data_copy[-1]["timestamp"] if data_copy else None

    if _cached_df is not None and _cached_timestamp == latest_timestamp:
        return _cached_df, latest_timestamp

    if not data_copy:
        return None, latest_timestamp

    df = pd.DataFrame(data_copy)
    if "timestamp" not in df.columns:
        return None, latest_timestamp

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    if df.empty:
        return None, latest_timestamp

    df = process_data_for_model(df, interval="5s")
    if df is None or df.empty:
        return None, latest_timestamp

    _cached_df = df
    _cached_timestamp = latest_timestamp
    return df, latest_timestamp

def prepare_pred_df(msk_tz, last_time, time_delta, fivesec_predictions, pred_file_lock):
    """
    Возвращает DataFrame предсказаний + метрики (mse, mae, count) для графика.
    """
    with pred_file_lock:
        if not fivesec_predictions:
            return pd.DataFrame(), None, None, 0, None
        pred_df = pd.DataFrame(list(fivesec_predictions))
    if pred_df.empty:
        return pred_df, None, None, 0, None

    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"]).dt.tz_convert(msk_tz)
    pred_df["fivesec_pred_time"] = pd.to_datetime(pred_df["fivesec_pred_time"]).dt.tz_convert(msk_tz)
    pred_df = pred_df[pred_df["timestamp"] >= (last_time - time_delta)]

    mse = mae = None
    valid = pred_df[pred_df["fivesec_error"].notna()]
    if not valid.empty:
        mse = (valid["fivesec_error"] ** 2).mean()
        mae = valid["fivesec_error"].abs().mean()
    return pred_df, mse, mae, len(pred_df), None
