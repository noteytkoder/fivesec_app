# data_handler/errors_loop.py
"""
Модуль для обновления ошибок предсказаний.
Периодически проверяет предсказания, время которых прошло, и обновляет ошибки/тренд на основе фактической цены.
"""

import asyncio
import pandas as pd
import numpy as np
from config_manager import load_config
from logger import setup_logger
from .buffers import fivesec_predictions, fivesec_prediction_file_lock, get_current_buffer_df
from .config import MSK_TZ

logger = setup_logger()

async def update_fivesec_errors_loop():
    """
    Асинхронный цикл обновления ошибок предсказаний каждые 5 секунд.
    """
    config = load_config()
    test_all = config.get("test_all_models", False)
    wait_seconds = 5  # Период проверки (5 секунд)

    while True:
        try:
            now = pd.Timestamp.now(tz=MSK_TZ)
            
            with fivesec_prediction_file_lock:
                if not fivesec_predictions:
                    await asyncio.sleep(wait_seconds)
                    continue
                preds = pd.DataFrame(list(fivesec_predictions))
            
            if preds.empty:
                await asyncio.sleep(wait_seconds)
                continue

            # Получаем актуальные данные kline для сравнения
            kline_df = get_current_buffer_df()
            if kline_df is None or kline_df.empty:
                logger.warning("Нет данных kline для обновления ошибок", extra={'source': 'errors_loop'})
                await asyncio.sleep(wait_seconds)
                continue

            # Индексируем kline по timestamp для поиска ближайшей цены
            kline_df.index = pd.to_datetime(kline_df.index).tz_convert(MSK_TZ)

            # Находим предсказания, время которых прошло (fivesec_pred_time <= now)
            to_update = preds[pd.to_datetime(preds['fivesec_pred_time']) <= now]
            if to_update.empty:
                await asyncio.sleep(wait_seconds)
                continue

            for idx, row in to_update.iterrows():
                pred_time = pd.to_datetime(row['fivesec_pred_time'])
                # Находим ближайшую актуальную цену на момент pred_time (или ближайшую после)
                closest_kline = kline_df.loc[kline_df.index >= pred_time].iloc[0] if not kline_df.loc[kline_df.index >= pred_time].empty else None
                if closest_kline is None:
                    continue  # Ещё нет данных
                
                actual_price = closest_kline['close']
                current_close = row['current_close']  # Цена на момент предсказания

                if test_all:
                    for m_type in ['rf', 'xgb', 'lgb']:
                        pred_col = f'fivesec_pred_{m_type}'
                        error_col = f'fivesec_error_{m_type}'
                        trend_pred_col = f'fivesec_trend_pred_{m_type}'
                        trend_actual_col = f'fivesec_trend_actual_{m_type}'
                        trend_acc_col = f'fivesec_trend_accuracy_{m_type}'
                        
                        if pred_col in row and not pd.isna(row[pred_col]):
                            pred = row[pred_col]
                            error = actual_price - pred
                            trend_pred = np.sign(pred - current_close) if current_close > 0 else 0
                            trend_actual = np.sign(actual_price - current_close) if current_close > 0 else 0
                            trend_acc = 1 if trend_pred == trend_actual else 0
                            
                            preds.at[idx, error_col] = error
                            preds.at[idx, trend_pred_col] = trend_pred
                            preds.at[idx, trend_actual_col] = trend_actual
                            preds.at[idx, trend_acc_col] = trend_acc
                else:
                    if 'fivesec_pred' in row and not pd.isna(row['fivesec_pred']):
                        pred = row['fivesec_pred']
                        error = actual_price - pred
                        trend_pred = np.sign(pred - current_close) if current_close > 0 else 0
                        trend_actual = np.sign(actual_price - current_close) if current_close > 0 else 0
                        trend_acc = 1 if trend_pred == trend_actual else 0
                        
                        preds.at[idx, 'fivesec_error'] = error
                        preds.at[idx, 'fivesec_trend_pred'] = trend_pred
                        preds.at[idx, 'fivesec_trend_actual'] = trend_actual
                        preds.at[idx, 'fivesec_trend_accuracy'] = trend_acc

            # Сохраняем обновлённые предсказания обратно
            with fivesec_prediction_file_lock:
                fivesec_predictions.clear()
                fivesec_predictions.extend(preds.to_dict(orient='records'))
                # Опционально: сохранить в CSV, если нужно

            logger.info(f"Обновлены ошибки для {len(to_update)} предсказаний", extra={'source': 'errors_loop'})

        except Exception as e:
            logger.error(f"Ошибка в update_fivesec_errors_loop: {e}", exc_info=True, extra={'source': 'errors_loop'})
        
        await asyncio.sleep(wait_seconds)