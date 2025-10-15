# data_handler/indicators.py
"""
Модуль расчёта технических индикаторов и подготовки данных для модели.
Содержит функции преобразования timestamp, проверки индекса, расчёта RSI,
SMA, лагов и агрегации свечей.
"""

import pandas as pd
import numpy as np
from .config import config, MSK_TZ, logger
from .buffers import get_current_orderbook_df

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
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).tz_convert(MSK_TZ)
            df.set_index("timestamp", inplace=True)
        else:
            logger.error("No 'timestamp' column found in DataFrame")
            return None
    return df.sort_index()

def compute_rsi(data, periods=14):
    """Вычисляет RSI по ряду данных."""
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_indicators(df):
    """
    Добавляет RSI, SMA, логарифм объёма и лаги.
    Возвращает DataFrame без NaN, если данных достаточно.
    """
    try:
        if len(df) < config["model"].get("rsi_window", 14) + 3:  # Учитываем лаги и RSI
            logger.warning(f"Insufficient data for indicators: {len(df)} rows")
            return None
        original_len = len(df)
        df["rsi"] = compute_rsi(df["close"], config["model"].get("rsi_window", 14))
        df["sma"] = df["close"].rolling(window=config["model"].get("sma_window", 3)).mean()
        df["log_volume"] = np.log1p(df["volume"])
        lags = range(1, 4)
        for col in ["close", "rsi", "sma"]:
            shifts = pd.concat([df[col].shift(lag) for lag in lags], axis=1)
            shifts.columns = [f"{col}_lag_{lag}" for lag in lags]
            df = pd.concat([df, shifts], axis=1)
        df = df.dropna()
        logger.info(f"After indicators: shape={df.shape}, NaN={df.isna().sum().sum()}")
        return df if not df.empty else None
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
        if len(df) < config["data"]["min_records"]:
            logger.warning(f"Insufficient data for resample: {len(df)} rows")
            return None
        df = df.resample(interval).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).interpolate(method="linear").ffill(limit=2).dropna()
        logger.info(f"After resample ({interval}): shape={df.shape}, NaN={df.isna().sum().sum()}")
        return calculate_indicators(df)
    except Exception as e:
        logger.error(f"Error processing data for interval {interval}: {e}", exc_info=True)
        return None

def process_orderbook_for_model(orderbook_df, interval="5s"):
    """
    Обработка буфера стакана: ресэмплинг и добавление дельты mid_price.
    """
    try:
        if orderbook_df is None or orderbook_df.empty:
            logger.warning("No order book data available")
            return None
        orderbook_df = ensure_datetime_index(orderbook_df)
        orderbook_df["mid_price_delta"] = orderbook_df["mid_price"].diff().fillna(0.0)
        orderbook_df = orderbook_df.drop(columns=["mid_price", "bid_volume_10", "ask_volume_10"], errors="ignore")
        orderbook_df = orderbook_df.resample(interval).mean().interpolate(method="linear").ffill(limit=2).dropna()
        logger.info(f"Processed orderbook ({interval}): shape={orderbook_df.shape}, NaN={orderbook_df.isna().sum().sum()}")
        return orderbook_df
    except Exception as e:
        logger.error(f"Error processing order book: {e}", exc_info=True)
        return None

def merge_features(kline_df, orderbook_df):
    """
    Объединение признаков kline и стакана по времени.
    """
    try:
        kline_df = ensure_datetime_index(kline_df)
        orderbook_df = ensure_datetime_index(orderbook_df)
        if kline_df is None or orderbook_df is None:
            logger.error("Invalid input for merge_features")
            return None

        merged_df = pd.merge_asof(
            kline_df.reset_index(), orderbook_df.reset_index(),
            on="timestamp", direction="nearest", tolerance=pd.Timedelta(seconds=1)
        ).set_index("timestamp")
        if merged_df.empty:
            logger.error("Merged DataFrame is empty")
            return None

        # Новое: интерполяция и заполнение нулём вместо dropna()
        merged_df = merged_df.interpolate(method="linear").fillna(0)

        logger.info(f"After merge: shape={merged_df.shape}, NaN={merged_df.isna().sum().sum()}")
        return merged_df
    except Exception as e:
        logger.error(f"Error merging features: {e}", exc_info=True)
        return None