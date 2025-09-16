import pandas as pd
from logger import setup_logger
from .indicators import calculate_indicators

logger = setup_logger()

def process_data_for_model(df, interval="5s"):
    """Ресэмплинг данных до 5-секундного интервала для модели"""
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            else:
                logger.error("No 'timestamp' column found in DataFrame")
                return None
        df = df.resample(interval).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).interpolate(method="linear").ffill().dropna()
        df = calculate_indicators(df)
        return df
    except Exception as e:
        logger.error(f"Error processing data for interval {interval}: {e}", exc_info=True)
        return None