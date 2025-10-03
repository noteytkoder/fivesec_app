"""
Модуль расчёта технических индикаторов и подготовки данных для модели.
Содержит функции преобразования timestamp, проверки индекса, расчёта RSI,
SMA, лагов и агрегации свечей.
"""

import pandas as pd
import numpy as np
import logging
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
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        else:
            logger.error("No 'timestamp' column found in DataFrame", extra={'source': 'indicators'})
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
        lags = range(1, 4)
        for col in ["close", "rsi", "sma"]:
            shifts = pd.concat([df[col].shift(lag) for lag in lags], axis=1)
            shifts.columns = [f"{col}_lag_{lag}" for lag in lags]
            df = pd.concat([df, shifts], axis=1)
        df = df.dropna()
        logger.info(f"After indicators: shape={df.shape}, NaN={df.isna().sum().sum()}", extra={'source': 'indicators'})
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}", exc_info=True, extra={'source': 'indicators'})
        return None

def process_data_for_model(df, interval="5s"):
    """
    Ресемплирует DataFrame по заданному интервалу (1s/5s),
    агрегирует свечи и вызывает calculate_indicators.
    """
    try:
        df = ensure_datetime_index(df)
        if df is None:
            logger.error("Invalid kline DataFrame", extra={'source': 'indicators'})
            return None
        df = df.resample(interval).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).ffill().dropna()
        logger.info(f"After resample ({interval}): shape={df.shape}, NaN={df.isna().sum().sum()}", extra={'source': 'indicators'})
        return calculate_indicators(df)
    except Exception as e:
        logger.error(f"Error processing data for interval {interval}: {e}", exc_info=True, extra={'source': 'indicators'})
        return None

def process_orderbook_for_model(orderbook_df, interval="5s"):
    """
    Обработка буфера стакана: ресэмплинг, нормализация и добавление лагов.
    """
    try:
        if orderbook_df is None or orderbook_df.empty:
            logger.warning("Empty orderbook DataFrame", extra={'source': 'indicators'})
            return None
        orderbook_df["bid_ask_ratio"] = orderbook_df["bid_ask_ratio"].fillna(1.0)
        orderbook_df["imbalance"] = orderbook_df["imbalance"].fillna(0.0)
        
        # Логарифмирование bid_ask_ratio для снижения skew
        orderbook_df["bid_ask_ratio"] = np.log1p(orderbook_df["bid_ask_ratio"])
        
        # Z-score нормализация для imbalance и bid_ask_ratio
        for col in ["bid_ask_ratio", "imbalance"]:
            mean = orderbook_df[col].mean()
            std = orderbook_df[col].std()
            if std > 0:
                orderbook_df[col] = (orderbook_df[col] - mean) / std
            logger.debug(f"Normalized {col}: mean={mean:.4f}, std={std:.4f}", extra={'source': 'indicators'})
        
        # Клиппинг после нормализации
        orderbook_df["bid_ask_ratio"] = orderbook_df["bid_ask_ratio"].clip(-5, 5)
        orderbook_df["imbalance"] = orderbook_df["imbalance"].clip(-5, 5)
        
        # Добавление лагов для bid_ask_ratio и imbalance
        lags = range(1, 4)
        for col in ["bid_ask_ratio", "imbalance"]:
            shifts = pd.concat([orderbook_df[col].shift(lag) for lag in lags], axis=1)
            shifts.columns = [f"{col}_lag_{lag}" for lag in lags]
            orderbook_df = pd.concat([orderbook_df, shifts], axis=1)
        
        orderbook_df = orderbook_df.resample(interval).mean().ffill().dropna()
        ratio_outliers = (orderbook_df['bid_ask_ratio'].abs() > 5).sum()
        logger.info(f"Processed orderbook ({interval}): shape={orderbook_df.shape}, NaN={orderbook_df.isna().sum().sum()}, ratio_outliers={ratio_outliers}", extra={'source': 'indicators'})
        logger.debug(f"Orderbook stats: ratio_max={orderbook_df['bid_ask_ratio'].max():.2f}, imbalance_std={orderbook_df['imbalance'].std():.2f}", extra={'source': 'indicators'})
        return orderbook_df
    except Exception as e:
        logger.error(f"Error processing order book: {e}", exc_info=True, extra={'source': 'indicators'})
        return None

def merge_features(kline_df, orderbook_df):
    """
    Объединение признаков kline и стакана по времени с ffill и staleness check.
    """
    try:
        kline_df = ensure_datetime_index(kline_df)
        orderbook_df = ensure_datetime_index(orderbook_df)
        if kline_df is None or orderbook_df is None:
            logger.error("Invalid input for merge_features", extra={'source': 'indicators'})
            return None
        
        kline_df.index = kline_df.index.tz_convert(MSK_TZ)
        orderbook_df.index = orderbook_df.index.tz_convert(MSK_TZ)

        merged_df = pd.merge_asof(
            kline_df.reset_index(), orderbook_df.reset_index(),
            on="timestamp", direction="nearest", tolerance=pd.Timedelta(seconds=5)
        ).set_index("timestamp")
        if merged_df.empty:
            logger.error("Merged DataFrame is empty", extra={'source': 'indicators'})
            return None
        
        # Заполнение NaN с помощью ffill, но с проверкой staleness
        before_fill_na = merged_df.isna().sum().sum()
        merged_df = merged_df.ffill(limit=2)  # Ограничение: не более 2 строк (10 секунд)
        after_fill_na = merged_df.isna().sum().sum()
        logger.info(f"Merged: shape={merged_df.shape}, NaN before ffill={before_fill_na}, NaN after ffill={after_fill_na}", extra={'source': 'indicators'})
        
        if 'bid_ask_ratio' in merged_df.columns:
            corr_imbalance = merged_df['imbalance'].corr(merged_df['close'])
            corr_ratio = merged_df['bid_ask_ratio'].corr(merged_df['close'])
            logger.debug(f"Orderbook corr with close: imbalance={corr_imbalance:.2f}, bid_ask_ratio={corr_ratio:.2f}", extra={'source': 'indicators'})
        
        # Отбрасываем строки, где всё ещё есть NaN
        merged_df = merged_df.dropna()
        logger.info(f"After dropna: shape={merged_df.shape}, NaN={merged_df.isna().sum().sum()}", extra={'source': 'indicators'})
        return merged_df
    except Exception as e:
        logger.error(f"Error merging features: {e}", exc_info=True, extra={'source': 'indicators'})
        return None