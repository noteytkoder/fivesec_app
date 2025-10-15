# data_handler/prediction_loop.py
"""
Модуль цикла предсказаний.
Формирует прогнозы на 5 секунд, пишет их в CSV и в лог.
"""

import os
import time
import pandas as pd
from collections import deque
import asyncio
from config_manager import load_config
from logger import setup_logger, setup_predictions_logger
from .buffers import fivesec_predictions, fivesec_prediction_file_lock, get_current_buffer_df, orderbook_buffer, get_current_orderbook_df
from .indicators import process_data_for_model, process_orderbook_for_model, merge_features
from model import predict_fivesec, get_model
from .config import LOGS_DIR, MSK_TZ, INTERVAL_SECONDS

async def fivesec_prediction_loop(root_dir):
    """
    Асинхронный цикл предсказаний 5-секундной модели.
    Сохраняет данные в буфер и CSV.
    """
    global fivesec_predictions
    logger = setup_logger()
    predictions_logger = setup_predictions_logger(log_dir=LOGS_DIR)
    wait_seconds = INTERVAL_SECONDS["5s"]
    max_predictions = 10000
    csv_file_path = os.path.join(LOGS_DIR, "fivesec_predictions.csv")
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
    with fivesec_prediction_file_lock:
        columns = [
            "timestamp", "actual_price", "current_close", "fivesec_pred_time",
            "fivesec_pred_rf", "fivesec_pred_xgb", "fivesec_pred_lgb",
            "fivesec_change_pct_rf", "fivesec_change_pct_xgb", "fivesec_change_pct_lgb",
            "fivesec_error_rf", "fivesec_error_xgb", "fivesec_error_lgb",
            "fivesec_trend_pred_rf", "fivesec_trend_actual_rf", "fivesec_trend_accuracy_rf",
            "fivesec_trend_pred_xgb", "fivesec_trend_actual_xgb", "fivesec_trend_accuracy_xgb",
            "fivesec_trend_pred_lgb", "fivesec_trend_actual_lgb", "fivesec_trend_accuracy_lgb",
            "fivesec_pred", "fivesec_change_pct", "fivesec_error",
            "fivesec_trend_pred", "fivesec_trend_actual", "fivesec_trend_accuracy"
        ]
        pd.DataFrame(columns=columns).to_csv(csv_file_path, index=False, encoding='utf-8')

    last_csv_write_time = 0
    while True:
        start = time.time()
        try:
            config = load_config()
            use_orderbook = config["model"].get("use_orderbook", False)
            test_all = config.get("test_all_models", False)
            
            df = get_current_buffer_df()
            if df is None or len(df) < config["data"]["min_records"]:
                logger.warning(f"Недостаточно данных kline: {len(df) if df is not None else 'None'}", extra={'source': 'prediction_loop'})
                await asyncio.sleep(wait_seconds)
                continue
            df = process_data_for_model(df, interval="5s")
            if df is None or df.empty:
                logger.warning("Ошибка обработки данных kline", extra={'source': 'prediction_loop'})
                await asyncio.sleep(wait_seconds)
                continue

            logger.debug(f"Обработанные kline: форма={df.shape}, столбцы={df.columns.tolist()}", extra={'source': 'prediction_loop'})

            if use_orderbook:
                if not orderbook_buffer:
                    logger.warning("Буфер стакана пуст", extra={'source': 'prediction_loop'})
                    await asyncio.sleep(wait_seconds)
                    continue
                
                orderbook_df = get_current_orderbook_df()
                if orderbook_df is None or orderbook_df.empty:
                    logger.warning("Нет данных стакана", extra={'source': 'prediction_loop'})
                    await asyncio.sleep(wait_seconds)
                    continue
                orderbook_df = process_orderbook_for_model(orderbook_df, interval="5s")
                if orderbook_df is None or orderbook_df.empty:
                    logger.warning("Ошибка обработки данных стакана", extra={'source': 'prediction_loop'})
                    await asyncio.sleep(wait_seconds)
                    continue
                
                features_df = merge_features(df, orderbook_df)
                if features_df is None or features_df.empty:
                    logger.warning("Ошибка слияния данных kline и стакана", extra={'source': 'prediction_loop'})
                    await asyncio.sleep(wait_seconds)
                    continue
                logger.info(f"Признаки для предсказания: форма={features_df.shape}, NaN={features_df.isna().sum().sum()}", extra={'source': 'prediction_loop'})
                feature_columns = [
                    "close", "rsi", "sma", "volume", "log_volume",
                    "close_lag_1", "close_lag_2", "close_lag_3",
                    "rsi_lag_1", "rsi_lag_2", "rsi_lag_3",
                    "sma_lag_1", "sma_lag_2", "sma_lag_3",
                     "mid_price_delta", "imbalance_10",
                    "rel_bid_volume_10", "rel_ask_volume_10",
                    "delta_bid_vol_10", "delta_ask_vol_10"
                ]
            else:
                latest_row = df.iloc[-1]
                feature_columns = [
                    "close", "rsi", "sma", "volume", "log_volume",
                    "close_lag_1", "close_lag_2", "close_lag_3",
                    "rsi_lag_1", "rsi_lag_2", "rsi_lag_3",
                    "sma_lag_1", "sma_lag_2", "sma_lag_3"
                ]
                features_df = pd.DataFrame([latest_row[feature_columns]])

            missing_features = [col for col in feature_columns if col not in features_df.columns]
            if missing_features:
                logger.error(f"Отсутствуют признаки: {missing_features}", extra={'source': 'prediction_loop'})
                await asyncio.sleep(wait_seconds)
                continue

            model = get_model(use_orderbook)
            if test_all:
                if not isinstance(model, dict):
                    logger.warning(f"Модели (use_orderbook={use_orderbook}) не обучены, пропуск предсказания", extra={'source': 'prediction_loop'})
                    await asyncio.sleep(wait_seconds)
                    continue
            else:
                if not model.is_fitted:
                    logger.warning(f"Модель (use_orderbook={use_orderbook}) не обучена, пропуск предсказания", extra={'source': 'prediction_loop'})
                    await asyncio.sleep(wait_seconds)
                    continue

            current_close = features_df.iloc[-1]["close"]
            fivesec_prediction = predict_fivesec(features_df, use_orderbook=use_orderbook)
            if fivesec_prediction is None:
                await asyncio.sleep(wait_seconds)
                continue

            pred_timestamp = pd.Timestamp.now(tz=MSK_TZ)
            fivesec_pred_time = pred_timestamp + pd.Timedelta(seconds=5)

            if test_all:
                rf_pred = fivesec_prediction.get('random_forest', None)
                xgb_pred = fivesec_prediction.get('xgboost', None)
                lgb_pred = fivesec_prediction.get('lightgbm', None)
                
                rf_change = ((rf_pred - current_close) / current_close * 100) if rf_pred and current_close > 0 else None
                xgb_change = ((xgb_pred - current_close) / current_close * 100) if xgb_pred and current_close > 0 else None
                lgb_change = ((lgb_pred - current_close) / current_close * 100) if lgb_pred and current_close > 0 else None
                
                rf_str = f"{rf_pred:.4f}" if rf_pred is not None else "N/A"
                rf_chg = f"{rf_change:+.2f}%" if rf_change is not None else "N/A"
                xgb_str = f"{xgb_pred:.4f}" if xgb_pred is not None else "N/A"
                xgb_chg = f"{xgb_change:+.2f}%" if xgb_change is not None else "N/A"
                lgb_str = f"{lgb_pred:.4f}" if lgb_pred is not None else "N/A"
                lgb_chg = f"{lgb_change:+.2f}%" if lgb_change is not None else "N/A"

                predictions_logger.info(
                    f"время={pred_timestamp}, цена={current_close:.4f}, "
                    f"прогноз_rf={rf_str} ({rf_chg}), "
                    f"прогноз_xgb={xgb_str} ({xgb_chg}), "
                    f"прогноз_lgb={lgb_str} ({lgb_chg})",
                    extra={'source': 'prediction_loop'}
                )

                prediction_record = {
                    "timestamp": pred_timestamp,
                    "actual_price": current_close,
                    "current_close": current_close,
                    "fivesec_pred_rf": rf_pred,
                    "fivesec_pred_xgb": xgb_pred,
                    "fivesec_pred_lgb": lgb_pred,
                    "fivesec_change_pct_rf": rf_change,
                    "fivesec_change_pct_xgb": xgb_change,
                    "fivesec_change_pct_lgb": lgb_change,
                    "fivesec_pred_time": fivesec_pred_time,
                    "fivesec_error_rf": None,
                    "fivesec_error_xgb": None,
                    "fivesec_error_lgb": None,
                    "fivesec_trend_pred_rf": None,
                    "fivesec_trend_actual_rf": None,
                    "fivesec_trend_accuracy_rf": None,
                    "fivesec_trend_pred_xgb": None,
                    "fivesec_trend_actual_xgb": None,
                    "fivesec_trend_accuracy_xgb": None,
                    "fivesec_trend_pred_lgb": None,
                    "fivesec_trend_actual_lgb": None,
                    "fivesec_trend_accuracy_lgb": None
                }
            else:
                fivesec_change_pct = ((fivesec_prediction - current_close) / current_close * 100) if current_close > 0 else 0
                predictions_logger.info(
                    f"время={pred_timestamp}, цена={current_close:.4f}, прогноз_на_5сек={fivesec_prediction:.4f}, целевое_время_5сек={fivesec_pred_time}, отклонение_5сек={fivesec_change_pct:+.2f}%",
                    extra={'source': 'prediction_loop'}
                )
                prediction_record = {
                    "timestamp": pred_timestamp,
                    "actual_price": current_close,
                    "current_close": current_close,
                    "fivesec_pred": fivesec_prediction,
                    "fivesec_change_pct": fivesec_change_pct,
                    "fivesec_pred_time": fivesec_pred_time,
                    "fivesec_error": None,
                    "fivesec_trend_pred": None,
                    "fivesec_trend_actual": None,
                    "fivesec_trend_accuracy": None
                }

            with fivesec_prediction_file_lock:
                fivesec_predictions.append(prediction_record)
                if len(fivesec_predictions) > max_predictions:
                    fivesec_predictions = deque(list(fivesec_predictions)[-max_predictions:], maxlen=max_predictions)

            current_time = time.time()
            if current_time - last_csv_write_time >= config.get("data", {}).get("csv_write_interval", 30):
                with fivesec_prediction_file_lock:
                    pd.DataFrame(list(fivesec_predictions)).to_csv(csv_file_path, mode='w', index=False, encoding='utf-8')
                    last_csv_write_time = current_time
        except Exception as e:
            logger.error(f"Ошибка в fivesec_prediction_loop: {e}", exc_info=True, extra={'source': 'prediction_loop'})
        elapsed = time.time() - start
        await asyncio.sleep(max(0, wait_seconds - elapsed))