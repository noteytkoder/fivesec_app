"""
Модуль для обучения и хранения моделей прогнозирования.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from logger import setup_logger
from config_manager import load_config
from data_handler.indicators import process_orderbook_for_model, merge_features
from data_handler.buffers import get_current_orderbook_df

config = load_config()
logger = setup_logger()

models = {
    "kline_only": None,
    "kline_with_orderbook": None
}
scalers = {
    "kline_only": None,
    "kline_with_orderbook": None
}

def train_fivesec_model(df, use_orderbook=False):
    """
    Обучение 5-секундной модели.
    """
    global models, scalers
    try:
        model_key = "kline_with_orderbook" if use_orderbook else "kline_only"
        logger.debug(f"train_fivesec_model ({model_key}): Input dataframe shape: {df.shape}, columns: {df.columns.tolist()}")
        
        window_seconds = config["model"]["fivesec_train_window_seconds"]
        df = df.tail(int(window_seconds / 5))
        logger.debug(f"train_fivesec_model ({model_key}): After window limit, shape: {df.shape}")
        
        if len(df) < config["model"]["min_fivesec_candles"]:
            logger.warning(f"Insufficient data for 5-sec training ({model_key}): {len(df)} candles, required: {config['model']['min_fivesec_candles']}")
            return
        
        if use_orderbook:
            orderbook_df = get_current_orderbook_df()
            if orderbook_df is None or orderbook_df.empty:
                logger.warning(f"No order book data available for training ({model_key})")
                return
            orderbook_df = process_orderbook_for_model(orderbook_df, interval="5s")
            if orderbook_df is None or orderbook_df.empty:
                logger.warning(f"Failed to process order book data for training ({model_key})")
                return
            df = merge_features(df, orderbook_df)
            if df is None or df.empty:
                logger.warning(f"Failed to merge kline and order book data ({model_key})")
                return
        
        features = [
            "close", "rsi", "sma", "volume", "log_volume",
            "close_lag_1", "close_lag_2", "close_lag_3",
            "rsi_lag_1", "rsi_lag_2", "rsi_lag_3",
            "sma_lag_1", "sma_lag_2", "sma_lag_3"
        ]
        if use_orderbook:
            features += [
                "imbalance_10", "rel_bid_volume_10", "rel_ask_volume_10",
                "delta_bid_vol_10", "delta_ask_vol_10"
            ]  # Включены все признаки стакана
        
        target = df["close"].shift(-1)
        valid_idx = target.notna()
        X = df[features][valid_idx]
        y = target[valid_idx]
        
        logger.debug(f"train_fivesec_model ({model_key}): Features shape: {X.shape}, Target shape: {y.shape}")
        
        if len(X) < config["model"]["min_fivesec_candles"]:
            logger.warning(f"Too few valid 5-sec samples ({model_key}): {len(X)}, required: {config['model']['min_fivesec_candles']}")
            return
        
        if X.isna().any().any() or np.any(np.isinf(X.values)):
            logger.error(f"NaN or Inf values found in features ({model_key}): {X.isna().sum()}")
            return
        
        if np.any(X.std() == 0):
            logger.warning(f"Zero standard deviation in features ({model_key}): {X.std()}")
            return
        
        logger.info(f"{model_key} before train: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"{model_key} cols: {X.columns.tolist()}")
        logger.info(f"{model_key} nan per col:\n{X.isna().sum()}")
        logger.info(f"{model_key} nunique per col:\n{X.nunique()}")
        logger.info(f"{model_key} std per col:\n{X.std()}")
        logger.info(f"{model_key} describe:\n{X.describe().T}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.debug(f"train_fivesec_model ({model_key}): Data normalized, X_scaled shape: {X_scaled.shape}")
        
        Xs_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        logger.info(f"{model_key} after scaling: shape={Xs_df.shape}")
        logger.info(f"{model_key} scaled mean (approx):\n{Xs_df.mean().round(6)}")
        logger.info(f"{model_key} scaled std (approx):\n{Xs_df.std().round(6)}")
        logger.info(f"{model_key} any NaN after scaling: {Xs_df.isna().any().any()}")
        
        max_depth = config["model"]["fivesec_max_depth"]
        if max_depth in (0, None):
            logger.info(f"fivesec_max_depth is 0 or null, setting to 8 ({model_key})")
            max_depth = 8
        elif not isinstance(max_depth, (int, type(None))) or (isinstance(max_depth, int) and max_depth < 1):
            logger.warning(f"Invalid fivesec_max_depth: {max_depth}, using default value 8 ({model_key})")
            max_depth = 8
        
        model = RandomForestRegressor(
            n_estimators=config["model"]["fivesec_n_estimators"],
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=config["model"].get("max_features", "sqrt"),
            random_state=42
        )
        logger.info(f"Target y describe: mean={y.mean()}, std={y.std()}, min={y.min()}, max={y.max()}")
        model.fit(X_scaled, y)
        
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        
        models[model_key] = model
        scalers[model_key] = scaler
        logger.info(f"{model_key} model trained, R^2={r2:.4f}, samples={len(X)}")
    except Exception as e:
        logger.error(f"Error training {model_key} model: {e}", exc_info=True)

def select_model(use_orderbook=False):
    model_key = "kline_with_orderbook" if use_orderbook else "kline_only"
    if models[model_key] is None:
        logger.warning(f"No {model_key} model available")
        return None
    return models[model_key]

def predict_fivesec(features, use_orderbook=False):
    model_key = "kline_with_orderbook" if use_orderbook else "kline_only"
    try:
        model = select_model(use_orderbook)
        scaler = scalers[model_key]
        if model is None or scaler is None:
            logger.warning(f"{model_key} model or scaler not initialized")
            return None
        
        expected_features = [
            "close", "rsi", "sma", "volume", "log_volume",
            "close_lag_1", "close_lag_2", "close_lag_3",
            "rsi_lag_1", "rsi_lag_2", "rsi_lag_3",
            "sma_lag_1", "sma_lag_2", "sma_lag_3"
        ]
        if use_orderbook:
            expected_features += [
                "imbalance_10", "rel_bid_volume_10", "rel_ask_volume_10",
                "delta_bid_vol_10", "delta_ask_vol_10"
            ]  # Включены все признаки стакана
        
        if not all(col in features.columns for col in expected_features):
            logger.error(f"Missing features in prediction input ({model_key}): {features.columns.tolist()}")
            return None
        if features.isna().any().any() or np.any(np.isinf(features.values)):
            logger.error(f"NaN or Inf values in prediction features ({model_key})")
            return None
        
        features = features[expected_features]
        features_scaled = scaler.transform(features)
        return model.predict(features_scaled)[0]
    except Exception as e:
        logger.error(f"{model_key} prediction error: {e}", exc_info=True)
        return None