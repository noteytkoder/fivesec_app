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
            config = load_config()
            use_orderbook = config["model"].get("use_orderbook", False)
            
            df = get_current_buffer_df()
            if df is None or len(df) < config["data"]["min_records"]:
                logger.warning(f"Insufficient kline data: {len(df) if df is not None else 'None'}", extra={'source': 'prediction_loop'})
                await asyncio.sleep(wait_seconds)
                continue
            df = process_data_for_model(df, interval="5s")
            if df is None or df.empty:
                logger.warning("Failed to process kline data", extra={'source': 'prediction_loop'})
                await asyncio.sleep(wait_seconds)
                continue

            logger.debug(f"Processed kline: shape={df.shape}, columns={df.columns.tolist()}", extra={'source': 'prediction_loop'})

            if use_orderbook:
                if orderbook_buffer:
                    logger.debug(f"Last orderbook item: {orderbook_buffer[-1]}", extra={'source': 'prediction_loop'})
                else:
                    logger.warning("Orderbook buffer empty", extra={'source': 'prediction_loop'})
                    await asyncio.sleep(wait_seconds)
                    continue
                
                orderbook_df = get_current_orderbook_df()
                if orderbook_df is not None and not orderbook_df.empty:
                    orderbook_df = process_orderbook_for_model(orderbook_df, interval="5s")
                    if orderbook_df is not None and not orderbook_df.empty:
                        logger.debug(f"Processed orderbook: shape={orderbook_df.shape}, columns={orderbook_df.columns.tolist()}", extra={'source': 'prediction_loop'})
                        ratio_outliers = (orderbook_df['rel_bid_volume_10'].abs() > 0.99).sum()
                        logger.info(f"Orderbook stats: imbalance_10_mean={orderbook_df['imbalance_10'].mean():.2f}, ratio_outliers={ratio_outliers}", extra={'source': 'prediction_loop'})
                    else:
                        logger.warning("Failed to process orderbook data", extra={'source': 'prediction_loop'})
                        await asyncio.sleep(wait_seconds)
                        continue
                else:
                    logger.warning("No orderbook data available", extra={'source': 'prediction_loop'})
                    await asyncio.sleep(wait_seconds)
                    continue
                
                features_df = merge_features(df, orderbook_df)
                if features_df is None or features_df.empty:
                    logger.warning("Failed to merge kline and orderbook data", extra={'source': 'prediction_loop'})
                    await asyncio.sleep(wait_seconds)
                    continue
                logger.info(f"Features for pred: shape={features_df.shape}, NaN={features_df.isna().sum().sum()}", extra={'source': 'prediction_loop'})
                if use_orderbook:
                    corr_imbalance = features_df['imbalance_10'].corr(features_df['close'])
                    logger.debug(f"Orderbook corr with close: imbalance_10={corr_imbalance:.2f}", extra={'source': 'prediction_loop'})
                latest_row = features_df.iloc[-1]
                feature_columns = [
                    "close", "rsi", "sma", "volume", "log_volume",
                    "close_lag_1", "close_lag_2", "close_lag_3",
                    "rsi_lag_1", "rsi_lag_2", "rsi_lag_3",
                    "sma_lag_1", "sma_lag_2", "sma_lag_3",
                    "spread_5", "mid_price_delta", "imbalance_10",
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
                logger.error(f"Missing features: {missing_features}", extra={'source': 'prediction_loop'})
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
                f"время={pred_timestamp}, цена={current_close:.4f}, прогноз_на_5сек={fivesec_prediction:.4f}, целевое_время_5сек={fivesec_pred_time}, отклонение_5сек={fivesec_change_pct:+.2f}%",
                extra={'source': 'prediction_loop'}
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
            logger.error(f"Error in fivesec_prediction_loop: {e}", exc_info=True, extra={'source': 'prediction_loop'})
        elapsed = time.time() - start
        await asyncio.sleep(max(0, wait_seconds - elapsed))