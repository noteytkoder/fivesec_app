import pandas as pd
import numpy as np
from logger import setup_logger
from config_manager import load_config

config = load_config()
logger = setup_logger()

def process_timestamp(ms_timestamp):
    return pd.to_datetime(ms_timestamp, unit="ms", utc=True)\
             .tz_convert(config.get("timezone", "Europe/Moscow"))

def compute_rsi(data, periods=7):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(df):
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
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            else:
                logger.error("No 'timestamp' column found in DataFrame")
                return None
        df = df.resample(interval).agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum"
        }).interpolate(method="linear").ffill().dropna()
        df = calculate_indicators(df)
        return df
    except Exception as e:
        logger.error(f"Error processing data for interval {interval}: {e}", exc_info=True)
        return None
