# data_handler/errors_loop.py
"""
Модуль обновления ошибок и метрик.
Сопоставляет прогнозы с фактическими ценами, вычисляет MAE и точность тренда,
обновляет CSV.
"""

import os
import pandas as pd

from .config import LOGS_DIR, MSK_TZ, logger
from .buffers import fivesec_predictions, fivesec_prediction_file_lock, fivesec_buffer
import asyncio

async def update_fivesec_errors_loop(root_dir):
    """
    Асинхронный цикл обновления ошибок предсказаний и метрик (MAE, точность тренда).
    """
    csv_file_path = os.path.join(LOGS_DIR, "fivesec_predictions.csv")
    tolerance_seconds = {"fivesec": 10}
    cached_mae_10min = None
    cached_trend_accuracy_10min = None

    while True:
        try:
            now = pd.Timestamp.now(tz=MSK_TZ)
            data_df = pd.DataFrame(list(fivesec_buffer))
            if data_df.empty:
                await asyncio.sleep(5)
                continue
            data_df["timestamp"] = pd.to_datetime(data_df["timestamp"]).dt.tz_convert(MSK_TZ)
            data_df = data_df.sort_values("timestamp")

            pending_predictions = [p for p in list(fivesec_predictions) if p.get("fivesec_actual_price") is None]
            for prediction in pending_predictions:
                pred_time = prediction.get("fivesec_pred_time")
                if not pred_time:
                    continue
                pred_time = pd.to_datetime(pred_time).tz_convert(MSK_TZ)
                if pred_time > now:
                    continue
                idx = data_df["timestamp"].searchsorted(pred_time)
                if idx == 0 or idx >= len(data_df):
                    continue
                candidates = data_df.iloc[max(0, idx - 1):idx + 1]
                time_diff = (candidates["timestamp"] - pred_time).abs()
                if time_diff.empty:
                    continue
                closest_pos = time_diff.idxmin()
                min_diff = time_diff.loc[closest_pos]
                if pd.isna(min_diff) or min_diff.total_seconds() > tolerance_seconds["fivesec"]:
                    continue
                actual_price = data_df.loc[closest_pos, "close"]
                prediction["fivesec_actual_price"] = actual_price
                if prediction.get("fivesec_pred") is not None:
                    prediction["fivesec_error"] = abs(actual_price - prediction["fivesec_pred"])
                    current_close = prediction.get("current_close")
                    if current_close is not None:
                        pred_dir = 1 if prediction["fivesec_pred"] > current_close else (-1 if prediction["fivesec_pred"] < current_close else 0)
                        act_dir = 1 if actual_price > current_close else (-1 if actual_price < current_close else 0)
                        prediction["fivesec_trend_pred"] = pred_dir
                        prediction["fivesec_trend_actual"] = act_dir
                        prediction["fivesec_trend_accuracy"] = 1 if pred_dir == act_dir else 0

            pred_df = pd.DataFrame(list(fivesec_predictions))
            if not pred_df.empty:
                pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
                ten_min_ago = pd.Timestamp.now(tz=MSK_TZ) - pd.Timedelta(minutes=10)
                recent_preds = pred_df[pred_df["timestamp"] >= ten_min_ago]
                if not recent_preds.empty:
                    valid_err = pd.to_numeric(recent_preds["fivesec_error"], errors="coerce").dropna()
                    cached_mae_10min = valid_err.mean() if not valid_err.empty else None
                    valid_trend = pd.to_numeric(recent_preds["fivesec_trend_accuracy"], errors="coerce").dropna()
                    cached_trend_accuracy_10min = valid_trend.mean() * 100.0 if not valid_trend.empty else None
                else:
                    cached_mae_10min = None
                    cached_trend_accuracy_10min = None

                with fivesec_prediction_file_lock:
                    pred_df.to_csv(csv_file_path, mode='w', index=False, encoding='utf-8')

            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error in update_fivesec_errors_loop: {e}", exc_info=True)
            await asyncio.sleep(5)