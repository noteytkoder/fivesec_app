"""
Модуль для обучения и хранения моделей прогнозирования.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import joblib
import psutil
import time
import os
from logger import setup_logger
from config_manager import load_config
from data_handler.indicators import process_orderbook_for_model, merge_features
from data_handler.buffers import get_current_orderbook_df

logger = setup_logger()
config = load_config()

class PredictorModel:
    def __init__(self, model_type='random_forest', params=None, use_scaler=False, use_orderbook=False):
        self.model_type = model_type
        self.params = params or {}
        self.use_scaler = use_scaler
        self.use_orderbook = use_orderbook
        self.scaler = StandardScaler() if use_scaler else None
        self.model = self._build_model()
        self.feature_importances_ = None
        self.is_fitted = False  # Флаг для проверки, обучена ли модель

    def _build_model(self):
        if self.model_type == 'random_forest':
            return RandomForestRegressor(**self.params, random_state=42)
        elif self.model_type == 'xgboost':
            return xgb.XGBRegressor(**self.params, random_state=42, objective='reg:squarederror')  # Явно XGBRegressor
        elif self.model_type == 'lightgbm':
            return lgb.LGBMRegressor(**self.params, random_state=42, verbose=-1)
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")

    def fit(self, X, y, eval_set=None):
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / 1024**2  # МБ
        
        if self.use_scaler:
            X = self.scaler.fit_transform(X)
        
        if self.model_type in ['xgboost', 'lightgbm'] and eval_set:
            fit_params = {}
            if 'early_stopping_rounds' in self.model.fit.__code__.co_varnames:
                fit_params['early_stopping_rounds'] = self.params.get('early_stopping_rounds', 10)
            if self.model_type == 'xgboost':
                fit_params['verbose'] = False
            self.model.fit(X, y, eval_set=eval_set, **fit_params)
        else:
            self.model.fit(X, y)
        
        self.is_fitted = True  # Модель обучена
        end_time = time.time()
        end_mem = psutil.Process().memory_info().rss / 1024**2
        logger.info(f"Модель {self.model_type} обучена: время={end_time - start_time:.2f}с, дельта памяти={end_mem - start_mem:.2f}МБ")
        
        # Важность признаков
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        elif hasattr(self.model, 'booster_'):
            self.feature_importances_ = self.model.booster_.feature_importance()

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError(f"Модель {self.model_type} не обучена. Вызовите fit или load_model.")
        start_time = time.time()
        if self.use_scaler:
            X = self.scaler.transform(X)
        preds = self.model.predict(X)
        latency = (time.time() - start_time) / len(X) * 1000  # мс на сэмпл
        logger.debug(f"Задержка предсказания {self.model_type}: {latency:.4f}мс/сэмпл")
        return preds

    def score(self, X, y, metrics=['mae', 'rmse', 'r2', 'dir_acc']):
        preds = self.predict(X)
        results = {}
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(y, preds)
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y, preds))
        if 'r2' in metrics:
            results['r2'] = r2_score(y, preds)
        if 'dir_acc' in metrics:
            dir_pred = np.sign(preds - X['close'].values if 'close' in X else 0)
            dir_true = np.sign(y - X['close'].values)
            results['dir_acc'] = np.mean(dir_pred == dir_true)
        logger.info(f"Метрики {self.model_type}: {results}")
        return results

    def save(self, path):
        if self.model_type == 'random_forest':
            joblib.dump(self.model, path)
        else:
            self.model.save_model(path)
        if self.use_scaler:
            joblib.dump(self.scaler, path + '_scaler.pkl')

    def load(self, path):
        if self.model_type == 'random_forest':
            self.model = joblib.load(path)
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor()
            self.model.load_model(path)
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor()
            self.model.booster_ = lgb.Booster(model_file=path)
        if self.use_scaler and os.path.exists(path + '_scaler.pkl'):
            self.scaler = joblib.load(path + '_scaler.pkl')
        self.is_fitted = True

# Менеджер моделей
model_manager = {
    "kline_only": None,
    "kline_with_orderbook": None
}

def get_model(use_orderbook=False):
    key = "kline_with_orderbook" if use_orderbook else "kline_only"
    if model_manager[key] is None:
        model_type = config['model']['type']
        params = config['model']['params'].get(model_type, {})
        use_scaler = config['model'].get('use_scaler', False)
        model_manager[key] = PredictorModel(model_type, params, use_scaler, use_orderbook)
    return model_manager[key]

def train_fivesec_model(df, use_orderbook=False):
    """
    Обучение 5-секундной модели.
    """
    model = get_model(use_orderbook)
    model_key = "kline_with_orderbook" if use_orderbook else "kline_only"
    try:
        logger.debug(f"train_fivesec_model ({model_key}): Форма входного датафрейма: {df.shape}, столбцы: {df.columns.tolist()}")
        
        window_seconds = config["model"]["fivesec_train_window_seconds"]
        df = df.tail(int(window_seconds / 5))
        logger.debug(f"train_fivesec_model ({model_key}): После ограничения окна, форма: {df.shape}")
        
        if len(df) < config["model"]["min_fivesec_candles"]:
            logger.warning(f"Недостаточно данных для обучения 5-сек ({model_key}): {len(df)} свечей, требуется: {config['model']['min_fivesec_candles']}")
            return
        
        if use_orderbook:
            orderbook_df = get_current_orderbook_df()
            if orderbook_df is None or orderbook_df.empty:
                logger.warning(f"Нет данных стакана для обучения ({model_key})")
                return
            orderbook_df = process_orderbook_for_model(orderbook_df, interval="5s")
            if orderbook_df is None or orderbook_df.empty:
                logger.warning(f"Ошибка обработки данных стакана для обучения ({model_key})")
                return
            df = merge_features(df, orderbook_df)
            if df is None or df.empty:
                logger.warning(f"Ошибка слияния данных kline и стакана ({model_key})")
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
            ]
        
        target = df["close"].shift(-1)
        valid_idx = target.notna()
        X = df[features][valid_idx]
        y = target[valid_idx]
        
        logger.debug(f"train_fivesec_model ({model_key}): Форма признаков: {X.shape}, Форма цели: {y.shape}")
        
        if len(X) < config["model"]["min_fivesec_candles"]:
            logger.warning(f"Слишком мало валидных 5-сек сэмплов ({model_key}): {len(X)}, требуется: {config['model']['min_fivesec_candles']}")
            return
        
        if X.isna().any().any() or np.any(np.isinf(X.values)):
            logger.error(f"NaN или Inf в признаках ({model_key}): {X.isna().sum()}")
            return
        
        if np.any(X.std() == 0):
            logger.warning(f"Нулевое стандартное отклонение в признаках ({model_key}): {X.std()}")
            return
        
        logger.info(f"{model_key} перед обучением: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"{model_key} столбцы: {X.columns.tolist()}")
        logger.info(f"{model_key} NaN по столбцам:\n{X.isna().sum()}")
        logger.info(f"{model_key} уникальные по столбцам:\n{X.nunique()}")
        logger.info(f"{model_key} std по столбцам:\n{X.std()}")
        logger.info(f"{model_key} describe:\n{X.describe().T}")
        
        Xs_df = pd.DataFrame(X, index=X.index, columns=X.columns)
        logger.info(f"{model_key} форма: {Xs_df.shape}")
        logger.info(f"{model_key} средние (приблизительно):\n{Xs_df.mean().round(6)}")
        logger.info(f"{model_key} std (приблизительно):\n{Xs_df.std().round(6)}")
        logger.info(f"{model_key} NaN после: {Xs_df.isna().any().any()}")
        logger.info(f"Описание цели y: среднее={y.mean()}, std={y.std()}, мин={y.min()}, макс={y.max()}")
        
        if model.model_type in ['xgboost', 'lightgbm']:
            tscv = TimeSeriesSplit(n_splits=3)
            for train_idx, val_idx in tscv.split(X):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                eval_set = [(X_val, y_val)]
                model.fit(X_train, y_train, eval_set=eval_set)
                model.score(X_val, y_val)
        else:
            model.fit(X, y)
        
        model.save(f"models/{model.model_type}_{model_key}.model")
    except Exception as e:
        logger.error(f"Ошибка обучения модели {model_key}: {e}", exc_info=True)

def predict_fivesec(features, use_orderbook=False):
    model = get_model(use_orderbook)
    model_key = "kline_with_orderbook" if use_orderbook else "kline_only"
    try:
        if model is None or not model.is_fitted:
            logger.warning(f"Модель {model_key} не инициализирована или не обучена")
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
            ]
        
        if not all(col in features.columns for col in expected_features):
            logger.error(f"Отсутствуют признаки в входе предсказания ({model_key}): {features.columns.tolist()}")
            return None
        if features.isna().any().any() or np.any(np.isinf(features.values)):
            logger.error(f"NaN или Inf в признаках предсказания ({model_key})")
            return None
        
        features = features[expected_features]
        return model.predict(features)[0]
    except Exception as e:
        logger.error(f"Ошибка предсказания {model_key}: {e}", exc_info=True)
        return None