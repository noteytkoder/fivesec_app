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
import psutil
import time
from logger import setup_logger
from config_manager import load_config
from data_handler.indicators import process_orderbook_for_model, merge_features
from data_handler.buffers import get_current_orderbook_df, sample_tail_head
import hashlib

logger = setup_logger()
config = load_config()

# Глобальный словарь для хранения обученных моделей в памяти
FITTED_MODELS = {}  # Ключ: f"{model_key}_{m_type}" или просто model_key для одиночной

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
        # Общие параметры из config['model']
        common_params = {
            'max_depth': config['model'].get('fivesec_max_depth', 8),
            'n_estimators': config['model'].get('fivesec_n_estimators', 250),
            'min_samples_leaf': config['model'].get('min_samples_leaf', 1),
            'min_samples_split': config['model'].get('min_samples_split', 11),
            'learning_rate': config['model'].get('learning_rate', 0.05),
            'subsample': config['model'].get('subsample', 0.8),
            'colsample_bytree': config['model'].get('colsample_bytree', 0.8),
            'reg_alpha': config['model'].get('reg_alpha', 0.1),
            'reg_lambda': config['model'].get('reg_lambda', 1.0),
            'early_stopping_rounds': config['model'].get('early_stopping_rounds', 20)
        }
        # Обновляем общие параметры специфичными для модели
        model_specific_params = config['model']['params'].get(self.model_type, {})
        final_params = {**common_params, **model_specific_params}

        if self.model_type == 'random_forest':
            # Для RF используем только релевантные параметры
            rf_params = {
                'max_depth': final_params['max_depth'],
                'n_estimators': final_params['n_estimators'],
                'min_samples_leaf': final_params['min_samples_leaf'],
                'min_samples_split': final_params['min_samples_split'],
                'random_state': 42
            }
            return RandomForestRegressor(**rf_params)
        elif self.model_type == 'xgboost':
            xgb_params = {
                'max_depth': final_params['max_depth'],
                'n_estimators': final_params['n_estimators'],
                'learning_rate': final_params['learning_rate'],
                'subsample': final_params['subsample'],
                'colsample_bytree': final_params['colsample_bytree'],
                'reg_alpha': final_params['reg_alpha'],
                'reg_lambda': final_params['reg_lambda'],
                'random_state': 42,
                'objective': 'reg:squarederror'
            }
            return xgb.XGBRegressor(**xgb_params)
        elif self.model_type == 'lightgbm':
            lgb_params = {
                'max_depth': final_params['max_depth'],
                'n_estimators': final_params['n_estimators'],
                'learning_rate': final_params['learning_rate'],
                'subsample': final_params['subsample'],
                'colsample_bytree': final_params['colsample_bytree'],
                'reg_alpha': final_params['reg_alpha'],
                'reg_lambda': final_params['reg_lambda'],
                'num_leaves': final_params.get('num_leaves', 31),  # Специфично для LightGBM
                'random_state': 42,
                'verbose': -1
            }
            return lgb.LGBMRegressor(**lgb_params)
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
                fit_params['early_stopping_rounds'] = self.params.get('early_stopping_rounds', 20)
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
            raise ValueError(f"Модель {self.model_type} не обучена.")
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

def get_model(use_orderbook=False):
    config = load_config()
    test_all = config.get("test_all_models", False)
    model_key = "kline_with_orderbook" if use_orderbook else "kline_only"
    
    if test_all:
        models = {}
        for m_type in ['random_forest', 'xgboost', 'lightgbm']:
            key = f"{model_key}_{m_type}"
            if key in FITTED_MODELS:
                models[m_type] = FITTED_MODELS[key]
                logger.debug(f"Модель {m_type} для {model_key} взята из памяти (fitted={models[m_type].is_fitted})")
            else:
                models[m_type] = PredictorModel(model_type=m_type, params=config["model"]["params"].get(m_type, {}), use_scaler=False, use_orderbook=use_orderbook)
                logger.warning(f"Модель {m_type} для {model_key} не найдена в памяти, создана новая (не обучена)")
        return models  # dict {type: model}
    else:
        m_type = config["model"]["type"]
        key = model_key
        if key in FITTED_MODELS:
            model = FITTED_MODELS[key]
            logger.debug(f"Модель {m_type} для {model_key} взята из памяти (fitted={model.is_fitted})")
            return model
        else:
            model = PredictorModel(model_type=m_type, params=config["model"]["params"].get(m_type, {}), use_scaler=False, use_orderbook=use_orderbook)
            logger.warning(f"Модель {m_type} для {model_key} не найдена в памяти, создана новая (не обучена)")
            return model

def train_fivesec_model(df, use_orderbook=False):
    global FITTED_MODELS
    config = load_config()
    test_all = config.get("test_all_models", False)
    model_key = "kline_with_orderbook" if use_orderbook else "kline_only"
    if use_orderbook:
        orderbook_df = get_current_orderbook_df()
        if orderbook_df is None or orderbook_df.empty:
            logger.warning("No orderbook data for training")
            return
        orderbook_df = process_orderbook_for_model(orderbook_df, interval="5s")
        if orderbook_df is None or orderbook_df.empty:
            logger.warning("Failed to process orderbook data")
            return
        df = merge_features(df, orderbook_df)
        if df is None or df.empty:
            logger.warning("Failed to merge features")
            return
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
    try:
        y = df["close"].shift(-1)[:-1]
        X = df[expected_features][:-1]
        if X.empty or y.empty:
            logger.error(f"Empty X or y after preparing data: X.shape={X.shape}, y.shape={y.shape}")
            return
        logger.info(f"TRAIN INPUT: {sample_tail_head(X)} y head={y.head(3).values} tail={y.tail(3).values}")
        logger.info(f"TRAIN NAN per col:\n{X.isna().sum().to_dict()} Inf={np.isinf(X.values).any()}")
        new_hash = hashlib.md5(pd.util.hash_pandas_object(X, index=True).values).hexdigest()
        logger.debug(f"TRAIN X hash={new_hash}")
        repeated_mid = (X['mid_price_delta'].diff() == 0).astype(int).groupby((X['mid_price_delta'].diff() != 0).cumsum()).sum().max() if 'mid_price_delta' in X else 0
        logger.info(f"TRAIN max constant-mid_delta run={repeated_mid}")
        logger.info(f"{model_key} уникальные по столбцам:\n{X.nunique()}")
        logger.info(f"{model_key} std по столбцам:\n{X.std()}")
        logger.info(f"{model_key} describe:\n{X.describe().T}")
        Xs_df = pd.DataFrame(X, index=X.index, columns=X.columns)
        logger.info(f"{model_key} форма: {Xs_df.shape}")
        logger.info(f"{model_key} средние (приблизительно):\n{Xs_df.mean().round(6)}")
        logger.info(f"{model_key} std (приблизительно):\n{Xs_df.std().round(6)}")
        logger.info(f"{model_key} NaN после: {Xs_df.isna().any().any()}")
        logger.info(f"Описание цели y: среднее={y.mean()}, std={y.std()}, мин={y.min()}, макс={y.max()}")
        if test_all:
            for m_type in ['random_forest', 'xgboost', 'lightgbm']:
                model = PredictorModel(model_type=m_type, params=config["model"]["params"].get(m_type, {}), use_scaler=False, use_orderbook=use_orderbook)
                if m_type in ['xgboost', 'lightgbm']:
                    tscv = TimeSeriesSplit(n_splits=3)
                    for train_idx, val_idx in tscv.split(X):
                        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                        eval_set = [(X_val, y_val)]
                        model.fit(X_train, y_train, eval_set=eval_set)
                        model.score(X_val, y_val)
                else:
                    model.fit(X, y)
                key = f"{model_key}_{m_type}"
                FITTED_MODELS[key] = model  # Сохраняем в память
                logger.info(f"Обучена и сохранена в памяти модель {m_type} для {model_key}")
        else:
            m_type = config["model"]["type"]
            model = PredictorModel(model_type=m_type, params=config["model"]["params"].get(m_type, {}), use_scaler=False, use_orderbook=use_orderbook)
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
            key = model_key
            FITTED_MODELS[key] = model  # Сохраняем в память
            logger.info(f"Обучена и сохранена в памяти модель {m_type} для {model_key}")
    except Exception as e:
        logger.error(f"Error training model {model_key}: {e}", exc_info=True)

def predict_fivesec(features, use_orderbook=False):
    models = get_model(use_orderbook)
    config = load_config()
    test_all = config.get("test_all_models", False)
    model_key = "kline_with_orderbook" if use_orderbook else "kline_only"
    try:
        logger.debug(f"PREDICT INPUT: {sample_tail_head(features, n=1)}")
        if test_all:
            if not isinstance(models, dict):
                logger.warning(f"Модели {model_key} не инициализированы или не обучены")
                return None
            predictions = {}
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
            for m_type, model in models.items():
                if not model.is_fitted:
                    logger.warning(f"Модель {m_type} не обучена")
                    continue
                predictions[m_type] = model.predict(features)[0]
            return predictions
        else:
            model = models
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