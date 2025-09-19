"""
Модуль расчёта технических индикаторов и подготовки данных для модели.
Содержит функции преобразования timestamp, проверки индекса, расчёта RSI,
SMA, лагов и агрегации свечей.
"""

import pandas as pd
import numpy as np
from .config import config, MSK_TZ, logger

def process_timestamp(ms_timestamp):
    """Преобразует timestamp (мс) в pandas.Timestamp с TZ=MSK_TZ."""
    return pd.to_datetime(ms_timestamp, unit="ms", utc=True).tz_convert(MSK_TZ)

def ensure_datetime_index(df):
    """
    Убеждается, что индекс DataFrame — DatetimeIndex.
    Если нет — пытается установить по столбцу 'timestamp'.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        else:
            logger.error("No 'timestamp' column found in DataFrame")
            return None
    return df.sort_index()

def compute_rsi(data, periods=7):
    """Вычисляет RSI по ряду данных."""
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(df):
    """
    Добавляет RSI, SMA, логарифм объёма и лаги.
    Возвращает DataFrame без NaN.
    """
    try:
        df["rsi"] = compute_rsi(df["close"], config["model"].get("rsi_window", 7))
        df["sma"] = df["close"].rolling(window=config["model"].get("sma_window", 3)).mean()
        df["log_volume"] = np.log1p(df["volume"])
        for lag in range(1, 4):
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"rsi_lag_{lag}"] = df["rsi"].shift(lag)
            df[f"sma_lag_{lag}"] = df["sma"].shift(lag)
        return df.dropna()
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

def process_data_for_model(df, interval="5s"):
    """
    Ресемплирует DataFrame по заданному интервалу (1s/5s),
    агрегирует свечи и вызывает calculate_indicators.
    """
    try:
        df = ensure_datetime_index(df)
        if df is None:
            return None
        df = df.resample(interval).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).interpolate(method="linear").ffill().dropna()
        return calculate_indicators(df)
    except Exception as e:
        logger.error(f"Error processing data for interval {interval}: {e}", exc_info=True)
        return None
