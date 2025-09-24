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
from .buffers import fivesec_predictions, fivesec_prediction_file_lock, get_current_buffer_df, get_current_orderbook_df
from .indicators import process_data_for_model, merge_features
from model import predict_fivesec
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
        pd.DataFrame(columns=[
            "timestamp", "actual_price", "current_close", "fivesec_pred", "fivesec_change_pct",
            "fivesec_pred_time", "fivesec_actual_price", "fivesec_error",
            "fivesec_trend_pred", "fivesec_trend_actual", "fivesec_trend_accuracy"
        ]).to_csv(csv_file_path, index=False, encoding='utf-8')

    last_csv_write_time = 0
    while True:
        start = time.time()
        try:
            # Перезагружаем конфигурацию
            config = load_config()
            use_orderbook = config["model"].get("use_orderbook", False)
            
            # Получаем данные kline
            df = get_current_buffer_df()
            if df is None or len(df) < config["data"]["min_records"]:
                await asyncio.sleep(wait_seconds)
                continue
            df = process_data_for_model(df, interval="5s")
            if df is None or df.empty:
                await asyncio.sleep(wait_seconds)
                continue

            # Формируем признаки для предсказания
            if use_orderbook:
                # Получаем данные стакана и объединяем с kline
                orderbook_df = get_current_orderbook_df()
                if orderbook_df is None or orderbook_df.empty:
                    logger.warning("No order book data available for prediction")
                    await asyncio.sleep(wait_seconds)
                    continue
                features_df = merge_features(df, orderbook_df)
                if features_df is None or features_df.empty:
                    logger.warning("Failed to merge kline and order book data for prediction")
                    await asyncio.sleep(wait_seconds)
                    continue
                latest_row = features_df.iloc[-1]
                feature_columns = [
                    "close", "rsi", "sma", "volume", "log_volume",
                    "close_lag_1", "close_lag_2", "close_lag_3",
                    "rsi_lag_1", "rsi_lag_2", "rsi_lag_3",
                    "sma_lag_1", "sma_lag_2", "sma_lag_3",
                    "spread", "mid_price", "bid_ask_ratio", "imbalance",
                    "bid_volume_10", "ask_volume_10"
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

            # Проверяем наличие всех необходимых признаков
            missing_features = [col for col in feature_columns if col not in features_df.columns]
            if missing_features:
                logger.error(f"Missing features in prediction input (kline_with_orderbook): {missing_features}")
                await asyncio.sleep(wait_seconds)
                continue

            current_close = latest_row["close"]
            fivesec_prediction = predict_fivesec(features_df, use_orderbook=use_orderbook)
            if fivesec_prediction is None:
                await asyncio.sleep(wait_seconds)
                continue

            pred_timestamp = pd.Timestamp.now(tz=MSK_TZ)
            fivesec_pred_time = pred_timestamp + pd.Timedelta(seconds=5)
            fivesec_change_pct = ((fivesec_prediction - current_close) / current_close * 100) if current_close > 0 else 0

            predictions_logger.info(
                f"время={pred_timestamp}, цена={current_close:.4f}, прогноз_на_5сек={fivesec_prediction:.4f}, целевое_время_5сек={fivesec_pred_time}, отклонение_5сек={fivesec_change_pct:+.2f}%"
            )

            prediction_record = {
                "timestamp": pred_timestamp,
                "actual_price": current_close,
                "current_close": current_close,
                "fivesec_pred": fivesec_prediction,
                "fivesec_error": None,
                "fivesec_pred_time": fivesec_pred_time,
                "fivesec_change_pct": fivesec_change_pct,
                "fivesec_actual_price": None,
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
            logger.error(f"Error in fivesec_prediction_loop: {e}", exc_info=True)
        elapsed = time.time() - start
        await asyncio.sleep(max(0, wait_seconds - elapsed))