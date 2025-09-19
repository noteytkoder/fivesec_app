"""
Модуль цикла переобучения модели.
С периодичностью retrain_interval переобучает 5-секундную модель на новых данных.
"""

import time
import numpy as np

from .config import config, logger
from .buffers import get_current_buffer_df
from .indicators import process_data_for_model
from model import train_fivesec_model
import asyncio

async def fivesec_retrain_loop():
    """
    Асинхронный цикл переобучения модели.
    """
    last_train_time = time.time()
    cached_processed_df = None
    last_buffer_hash = None
    train_interval = config["data"]["fivesec_train_interval"]

    while True:
        try:
            current_time = time.time()
            df = get_current_buffer_df()
            if df is None or len(df) < config["data"]["min_records"]:
                await asyncio.sleep(train_interval)
                continue

            current_hash = (len(df), df.index[-1] if not df.empty else None)
            if current_hash == last_buffer_hash:
                df = cached_processed_df
            else:
                df = process_data_for_model(df, interval="5s")
                if df is None or df.empty:
                    await asyncio.sleep(train_interval)
                    continue
                cached_processed_df = df
                last_buffer_hash = current_hash

            if current_time - last_train_time >= train_interval and len(df) >= config["model"].get("min_fivesec_candles", 1):
                if df.isna().any().any() or np.any(np.isinf(df.values)):
                    logger.warning("NaN/Inf in processed df, skipping retrain")
                else:
                    train_fivesec_model(df)
                    last_train_time = current_time
                    logger.info(f"5-second model retrained, samples={len(df)}")
            await asyncio.sleep(train_interval)
        except Exception as e:
            logger.error(f"Error in fivesec_retrain_loop: {e}", exc_info=True)
            await asyncio.sleep(train_interval)
