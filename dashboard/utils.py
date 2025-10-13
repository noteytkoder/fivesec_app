"""
Утилиты для Dashboard: подготовка данных, кэш.
"""

import pandas as pd
from . import __name__ as pkgname
from data_handler import process_data_for_model
from config_manager import load_config
from logger import setup_logger

config = load_config()
logger = setup_logger()

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
        logger.debug("Используется кэшированный DataFrame для kline", extra={'source': 'utils'})
        return _cached_df, latest_timestamp

    if not data_copy:
        logger.debug("Пустой буфер kline", extra={'source': 'utils'})
        return None, latest_timestamp

    df = pd.DataFrame(data_copy)
    if "timestamp" not in df.columns:
        logger.error("Отсутствует столбец 'timestamp' в данных kline", extra={'source': 'utils'})
        return None, latest_timestamp

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    if df.empty:
        logger.debug("DataFrame kline пуст после установки индекса", extra={'source': 'utils'})
        return None, latest_timestamp

    df = process_data_for_model(df, interval="5s")
    if df is None or df.empty:
        logger.warning("Ошибка обработки данных kline или пустой результат", extra={'source': 'utils'})
        return None, latest_timestamp

    _cached_df = df
    _cached_timestamp = latest_timestamp
    logger.debug(f"Обновлён кэш kline: shape={df.shape}, columns={df.columns.tolist()}", extra={'source': 'utils'})
    return df, latest_timestamp

def prepare_pred_df(msk_tz, last_time, time_delta, fivesec_predictions, pred_file_lock):
    """
    Возвращает DataFrame предсказаний + метрики (mse, mae, count) для графика.
    """
    with pred_file_lock:
        if not fivesec_predictions:
            logger.debug("Нет предсказаний в fivesec_predictions", extra={'source': 'utils'})
            return pd.DataFrame(), None, None, 0, None
        pred_df = pd.DataFrame(list(fivesec_predictions))
    if pred_df.empty:
        logger.debug("DataFrame предсказаний пуст", extra={'source': 'utils'})
        return pred_df, None, None, 0, None

    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"]).dt.tz_convert(msk_tz)
    pred_df["fivesec_pred_time"] = pd.to_datetime(pred_df["fivesec_pred_time"]).dt.tz_convert(msk_tz)
    filtered = pred_df[pred_df["timestamp"] >= (last_time - time_delta)]
    logger.debug(
        f"Фильтрованные предсказания: shape={filtered.shape}, "
        f"timestamp_range=[{filtered['timestamp'].min() if not filtered.empty else 'N/A'}, "
        f"{filtered['timestamp'].max() if not filtered.empty else 'N/A'}]",
        extra={'source': 'utils'}
    )

    config_local = load_config()
    test_all = config_local.get("test_all_models", False)
    if test_all:
        mse = {}
        mae = {}
        model_map = {'rf': 'random_forest', 'xgb': 'xgboost', 'lgb': 'lightgbm'}
        for m_type in ['rf', 'xgb', 'lgb']:
            error_col = f"fivesec_error_{m_type}"
            if error_col in filtered.columns:
                valid = filtered[filtered[error_col].notna()]
                logger.debug(
                    f"Для {m_type}: valid_rows={len(valid)}, "
                    f"error_col={error_col}, non_na_count={len(valid)}",
                    extra={'source': 'utils'}
                )
                if not valid.empty:
                    errors = valid[error_col]
                    mse[model_map[m_type]] = (errors ** 2).mean()
                    mae[model_map[m_type]] = errors.abs().mean()
                    logger.debug(
                        f"Метрики для {m_type}: MSE={mse[model_map[m_type]]:.4f}, MAE={mae[model_map[m_type]]:.4f}",
                        extra={'source': 'utils'}
                    )
                else:
                    mse[model_map[m_type]] = 0.0
                    mae[model_map[m_type]] = 0.0
                    logger.debug(f"Нет валидных ошибок для {m_type}, установлены MSE=0, MAE=0", extra={'source': 'utils'})
        return pred_df, mse, mae, len(filtered), None
    else:
        mse = mae = None
        if "fivesec_error" in filtered.columns:
            valid = filtered[filtered["fivesec_error"].notna()]
            logger.debug(
                f"Для одиночной модели: valid_rows={len(valid)}, non_na_count={len(valid)}",
                extra={'source': 'utils'}
            )
            if not valid.empty:
                errors = valid["fivesec_error"]
                mse = (errors ** 2).mean()
                mae = errors.abs().mean()
                logger.debug(f"Метрики: MSE={mse:.4f}, MAE={mae:.4f}", extra={'source': 'utils'})
        return pred_df, mse, mae, len(filtered), None