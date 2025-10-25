# data_handler/retrain_loop.py
"""
Модуль цикла переобучения модели.
С периодичностью retrain_interval переобучает 5-секундную модель на новых данных.
"""

import time
import numpy as np
import asyncio
from config_manager import load_config
from .buffers import get_current_buffer_df
from .indicators import process_data_for_model
from model import train_fivesec_model, get_model
from logger import setup_logger

logger = setup_logger()

async def fivesec_retrain_loop():
    """
    Асинхронный цикл переобучения модели.
    """
    last_train_time = time.time()
    cached_processed_df = None
    last_buffer_hash = None

    while True:
        try:
            config = load_config()
            train_interval = config["data"]["fivesec_train_interval"]
            test_all = config.get("test_all_models", False)
            use_orderbook = config["model"].get("use_orderbook", False)

            current_time = time.time()
            df = get_current_buffer_df()
            if df is None or len(df) < config["data"]["min_records"]:
                logger.warning(f"Недостаточно данных: {len(df) if df is not None else 'None'}", extra={'source': 'retrain_loop'})
                await asyncio.sleep(train_interval)
                continue

            current_hash = (len(df), df.index[-1] if not df.empty else None)
            if current_hash == last_buffer_hash:
                df = cached_processed_df
            else:
                df = process_data_for_model(df, interval="5s")
                if df is None or df.empty:
                    logger.warning("Ошибка обработки данных для переобучения", extra={'source': 'retrain_loop'})
                    await asyncio.sleep(train_interval)
                    continue
                cached_processed_df = df
                last_buffer_hash = current_hash

            logger.debug(f"Обработанный df для переобучения: форма={df.shape}, NaN={df.isna().sum().sum()}", extra={'source': 'retrain_loop'})
            if df.isna().any().any() or np.any(np.isinf(df.values)):
                logger.warning("NaN/Inf в обработанном df, пропуск переобучения", extra={'source': 'retrain_loop'})
                await asyncio.sleep(train_interval)
                continue

            if current_time - last_train_time >= train_interval and len(df) >= config["model"].get("min_fivesec_candles", 1):
                train_fivesec_model(df, use_orderbook=use_orderbook)  # Обучает все три или одну
                models = get_model(use_orderbook)
                if test_all:
                    for m_type, model in models.items():
                        if model.is_fitted:
                            logger.info(f"5-секундная модель {m_type} переобучена (use_orderbook={use_orderbook}), сэмплов={len(df)}", extra={'source': 'retrain_loop'})
                        else:
                            logger.warning(f"Модель {m_type} (use_orderbook={use_orderbook}) не обучена, проверьте ошибки", extra={'source': 'retrain_loop'})
                else:
                    if models.is_fitted:
                        logger.info(f"5-секундная модель переобучена (use_orderbook={use_orderbook}), сэмплов={len(df)}", extra={'source': 'retrain_loop'})
                    else:
                        logger.warning(f"Модель (use_orderbook={use_orderbook}) не обучена, проверьте ошибки", extra={'source': 'retrain_loop'})
                last_train_time = current_time
            await asyncio.sleep(train_interval)
        except Exception as e:
            logger.error(f"Ошибка в fivesec_retrain_loop: {e}", exc_info=True, extra={'source': 'retrain_loop'})
            await asyncio.sleep(train_interval)