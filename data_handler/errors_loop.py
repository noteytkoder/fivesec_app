# data_handler/errors_loop.py
"""
Модуль для обновления ошибок предсказаний.
Периодически проверяет предсказания, время которых прошло, и обновляет ошибки/тренд на основе фактической цены.
"""

import asyncio
import os
from logger import pd
import numpy as np
from collections import deque
from config_manager import load_config
from logger import setup_logger
from .buffers import fivesec_predictions, fivesec_prediction_file_lock, get_current_buffer_df
from .config import MSK_TZ, LOGS_DIR

logger = setup_logger()

async def update_fivesec_errors_loop(root_dir=None):
    """
    Асинхронный цикл обновления ошибок предсказаний каждые 5 секунд.
    """
    logger.info("Цикл update_fivesec_errors_loop успешно запущен!", extra={'source': 'errors_loop'})
    
    config = load_config()
    test_all = config.get("test_all_models", False)
    wait_seconds = 5
    tolerance_seconds = 10
    max_prediction_age_seconds = config.get("data", {}).get("max_prediction_age_seconds", 3600)

    while True:
        try:
            now = pd.Timestamp.now(tz=MSK_TZ)
            logger.debug(f"Начало цикла обновления ошибок, текущее время: {now}", extra={'source': 'errors_loop'})

            # --- ЧИТАЕМ И ОЧИЩАЕМ ПРЕДСКАЗАНИЯ ПОД БЛОКИРОВКОЙ ---
            with fivesec_prediction_file_lock:
                # Проверяем, есть ли данные
                if not fivesec_predictions:
                    logger.debug("Нет предсказаний в fivesec_predictions", extra={'source': 'errors_loop'})
                    await asyncio.sleep(wait_seconds)
                    continue

                # Копируем для безопасной обработки
                current_predictions = list(fivesec_predictions)
                old_count = len(current_predictions)

                # Очистка по возрасту
                max_age = pd.Timedelta(seconds=max_prediction_age_seconds)
                cleaned_predictions = [
                    p for p in current_predictions
                    if pd.to_datetime(p["fivesec_pred_time"]) > now - max_age
                ]

                # Обновляем буфер (только если что-то изменилось)
                if len(cleaned_predictions) < old_count:
                    fivesec_predictions.clear()
                    fivesec_predictions.extend(cleaned_predictions)
                    cleaned_count = old_count - len(cleaned_predictions)
                    logger.info(f"Очищено {cleaned_count} старых предсказаний (старше {max_prediction_age_seconds}с)", 
                                extra={'source': 'errors_loop'})

                preds = pd.DataFrame(cleaned_predictions)

            # --- Если после очистки ничего не осталось ---
            if preds.empty:
                logger.debug("DataFrame предсказаний пуст после очистки", extra={'source': 'errors_loop'})
                await asyncio.sleep(wait_seconds)
                continue

            logger.debug(f"preds: shape={preds.shape}, columns={preds.columns.tolist()}", extra={'source': 'errors_loop'})

            # --- Получаем kline-данные ---
            kline_df = get_current_buffer_df()
            if kline_df is None or kline_df.empty:
                logger.warning("Нет данных kline для обновления ошибок", extra={'source': 'errors_loop'})
                await asyncio.sleep(wait_seconds)
                continue

            kline_df.index = pd.to_datetime(kline_df.index).tz_convert(MSK_TZ)
            logger.debug(f"kline_df: shape={kline_df.shape}, min_time={kline_df.index.min()}, max_time={kline_df.index.max()}")

            # --- Находим предсказания, время которых прошло ---
            preds['fivesec_pred_time'] = pd.to_datetime(preds['fivesec_pred_time'])
            to_update = preds[preds['fivesec_pred_time'] <= now]
            if to_update.empty:
                logger.debug("Нет предсказаний, время которых прошло", extra={'source': 'errors_loop'})
                await asyncio.sleep(wait_seconds)
                continue

            logger.debug(f"Найдено {len(to_update)} предсказаний для обновления")

            updated_count = 0
            for idx, row in to_update.iterrows():
                pred_time = row['fivesec_pred_time']
                try:
                    loc = kline_df.index.get_indexer([pred_time], method='nearest')[0]
                    if loc == -1:
                        continue
                    closest_time = kline_df.index[loc]
                    time_diff = abs((closest_time - pred_time).total_seconds())
                    if time_diff > tolerance_seconds:
                        continue
                    closest_kline = kline_df.iloc[loc]
                except Exception as e:
                    logger.error(f"Ошибка поиска closest_kline для idx={idx}: {e}", exc_info=True, extra={'source': 'errors_loop'})
                    continue

                actual_price = closest_kline['close']
                current_close = row['current_close']

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
                            updated_count += 1
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
                        updated_count += 1
                        logger.info(
                            f"Обновлено: idx={idx}, pred={pred:.4f}, actual={actual_price:.4f}, "
                            f"error={error:.4f}, trend_pred={trend_pred}, trend_actual={trend_actual}, trend_acc={trend_acc}",
                            extra={'source': 'errors_loop'}
                        )

            # --- Сохраняем обратно и в CSV ---
            with fivesec_prediction_file_lock:
                fivesec_predictions.clear()
                fivesec_predictions.extend(preds.to_dict(orient='records'))

                csv_path = os.path.join(LOGS_DIR, "fivesec_predictions.csv")
                try:
                    preds.to_csv(csv_path, index=False, encoding='utf-8')
                    logger.debug(f"CSV обновлён: {csv_path}", extra={'source': 'errors_loop'})
                except Exception as e:
                    logger.error(f"Ошибка записи CSV: {e}", exc_info=True, extra={'source': 'errors_loop'})

            logger.info(f"Обновлены ошибки для {updated_count} предсказаний (из {len(to_update)} проверенных)", extra={'source': 'errors_loop'})

        except Exception as e:
            logger.error(f"Ошибка в update_fivesec_errors_loop: {e}", exc_info=True, extra={'source': 'errors_loop'})

        await asyncio.sleep(wait_seconds)