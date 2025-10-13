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

async def update_fivesec_errors_loop(root_dir=None):
    """
    Асинхронный цикл обновления ошибок предсказаний каждые 5 секунд.
    :param root_dir: Корневая директория (для совместимости с system_control.py, не используется).
    """
    logger.info("Цикл update_fivesec_errors_loop успешно запущен!", extra={'source': 'errors_loop'})
    
    config = load_config()
    test_all = config.get("test_all_models", False)
    wait_seconds = 5  # Период проверки (5 секунд)
    tolerance_seconds = 10  # Максимальная разница для "nearest" (если >, пропуск)

    while True:
        try:
            now = pd.Timestamp.now(tz=MSK_TZ)
            logger.debug(f"Начало цикла обновления ошибок, текущее время: {now}", extra={'source': 'errors_loop'})
            
            with fivesec_prediction_file_lock:
                if not fivesec_predictions:
                    logger.debug("Нет предсказаний в fivesec_predictions", extra={'source': 'errors_loop'})
                    await asyncio.sleep(wait_seconds)
                    continue
                preds = pd.DataFrame(list(fivesec_predictions))
            
            if preds.empty:
                logger.debug("DataFrame предсказаний пуст", extra={'source': 'errors_loop'})
                await asyncio.sleep(wait_seconds)
                continue

            logger.debug(f"preds: shape={preds.shape}, columns={preds.columns.tolist()}", extra={'source': 'errors_loop'})

            # Получаем актуальные данные kline для сравнения
            kline_df = get_current_buffer_df()
            if kline_df is None or kline_df.empty:
                logger.warning("Нет данных kline для обновления ошибок", extra={'source': 'errors_loop'})
                await asyncio.sleep(wait_seconds)
                continue

            # Индексируем kline по timestamp для поиска ближайшей цены
            kline_df.index = pd.to_datetime(kline_df.index).tz_convert(MSK_TZ)
            logger.debug(f"kline_df: shape={kline_df.shape}, min_time={kline_df.index.min()}, max_time={kline_df.index.max()}, columns={kline_df.columns.tolist()}", extra={'source': 'errors_loop'})

            # Находим предсказания, время которых прошло (fivesec_pred_time <= now)
            preds['fivesec_pred_time'] = pd.to_datetime(preds['fivesec_pred_time'])
            to_update = preds[preds['fivesec_pred_time'] <= now]
            if to_update.empty:
                logger.debug("Нет предсказаний, время которых прошло (все fivesec_pred_time > now)", extra={'source': 'errors_loop'})
                await asyncio.sleep(wait_seconds)
                continue

            logger.info(f"Найдено {len(to_update)} предсказаний для обновления ошибок, пример pred_time: {to_update['fivesec_pred_time'].head().tolist()}", extra={'source': 'errors_loop'})

            for idx, row in to_update.iterrows():
                pred_time = row['fivesec_pred_time']
                logger.debug(f"Обработка idx={idx}, pred_time={pred_time}, current_close={row.get('current_close', 'N/A')}", extra={'source': 'errors_loop'})
                try:
                    # Ищем ближайшую цену (nearest) с помощью get_indexer
                    loc = kline_df.index.get_indexer([pred_time], method='nearest')[0]
                    if loc == -1:
                        logger.warning(f"Не найдена ближайшая метка времени для pred_time={pred_time}, idx={idx}", extra={'source': 'errors_loop'})
                        continue
                    closest_time = kline_df.index[loc]
                    time_diff = abs((closest_time - pred_time).total_seconds())
                    logger.debug(f"Ближайшая kline_time={closest_time}, time_diff={time_diff}s для idx={idx}", extra={'source': 'errors_loop'})
                    if time_diff > tolerance_seconds:
                        logger.warning(f"Ближайшая цена слишком далеко ({time_diff}s) для idx={idx}, pred_time={pred_time}, skip", extra={'source': 'errors_loop'})
                        continue
                    closest_kline = kline_df.iloc[loc]
                except Exception as e:
                    logger.error(f"Ошибка при поиске closest_kline для idx={idx}: {e}", exc_info=True, extra={'source': 'errors_loop'})
                    continue
                
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
                            logger.info(
                                f"Обновлено для {m_type}: idx={idx}, pred={pred:.4f}, actual={actual_price:.4f}, "
                                f"error={error:.4f}, trend_pred={trend_pred}, trend_actual={trend_actual}, trend_acc={trend_acc}",
                                extra={'source': 'errors_loop'}
                            )
                        else:
                            logger.debug(f"Нет валидного предсказания для {m_type} в idx={idx}", extra={'source': 'errors_loop'})
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
                        logger.info(
                            f"Обновлено: idx={idx}, pred={pred:.4f}, actual={actual_price:.4f}, "
                            f"error={error:.4f}, trend_pred={trend_pred}, trend_actual={trend_actual}, trend_acc={trend_acc}",
                            extra={'source': 'errors_loop'}
                        )
                    else:
                        logger.debug(f"Нет валидного предсказания для одиночной модели в idx={idx}", extra={'source': 'errors_loop'})

            # Сохраняем обновлённые предсказания обратно
            with fivesec_prediction_file_lock:
                fivesec_predictions.clear()
                fivesec_predictions.extend(preds.to_dict(orient='records'))
                # Сохраняем в CSV для отладки
                try:
                    preds.to_csv(config.get("logs_dir", "logs") + "/fivesec_predictions.csv", index=False, encoding='utf-8')
                    logger.debug("Обновлённый DataFrame сохранён в CSV", extra={'source': 'errors_loop'})
                except Exception as e:
                    logger.error(f"Ошибка при сохранении CSV: {e}", exc_info=True, extra={'source': 'errors_loop'})

            logger.info(f"Обновлены ошибки для {len(to_update)} предсказаний", extra={'source': 'errors_loop'})

        except Exception as e:
            logger.error(f"Ошибка в update_fivesec_errors_loop: {e}", exc_info=True, extra={'source': 'errors_loop'})
        
        await asyncio.sleep(wait_seconds)