"""
Модуль управления системой.
Запускает и останавливает фоновые задачи (REST, WebSocket, предсказания, переобучение, обновление ошибок).
"""

import asyncio
import os
from pathlib import Path

from .config import logger

from .binance_api import producer_ws, consumer_loop, producer_orderbook_ws
from .prediction_loop import fivesec_prediction_loop
from .retrain_loop import fivesec_retrain_loop
from .errors_loop import update_fivesec_errors_loop

MAIN_LOOP = None
SYSTEM_STATE = "RUNNING"
INTENTIONAL_STOP = False
RUNNING_TASKS = []
RUNNING_TASKS_LOCK = asyncio.Lock()
ACTIVE_QUEUE = None
ORDERBOOK_QUEUE = None  # Новая очередь для стакана

def set_main_loop(loop):
    global MAIN_LOOP
    MAIN_LOOP = loop
    logger.info(f"MAIN_LOOP set: {MAIN_LOOP}")

def stop_system():
    """
    Синхронный вызов остановки системы из другого потока.
    """
    global MAIN_LOOP
    logger.info(f"stop_system called, MAIN_LOOP={MAIN_LOOP}")
    if MAIN_LOOP is None:
        raise RuntimeError("Main loop not set")

    # ставим отмену всех фоновых задач в очередь event loop
    asyncio.run_coroutine_threadsafe(_stop_system_async(), MAIN_LOOP)

async def _spawn_all_tasks(root_dir):
    """
    Создает все фоновые задачи и возвращает список asyncio.Task.
    """
    global RUNNING_TASKS, ACTIVE_QUEUE, ORDERBOOK_QUEUE
    ACTIVE_QUEUE = asyncio.Queue(maxsize=10000)
    ORDERBOOK_QUEUE = asyncio.Queue(maxsize=10000)
    fivesec_kline_uri = "wss://stream.binance.com:443/ws/btcusdt@kline_1s"
    orderbook_uri = "wss://stream.binance.com:443/ws/btcusdt@depth@1000ms"

    tasks = [
        asyncio.create_task(producer_ws(fivesec_kline_uri, "fivesec_kline", ACTIVE_QUEUE)),
        asyncio.create_task(producer_orderbook_ws(orderbook_uri, "orderbook_diff", ORDERBOOK_QUEUE)),
        asyncio.create_task(consumer_loop(ACTIVE_QUEUE)),
        asyncio.create_task(consumer_loop(ORDERBOOK_QUEUE)),  # Повторно используем consumer_loop для стакана
        asyncio.create_task(fivesec_prediction_loop(root_dir)),
        asyncio.create_task(fivesec_retrain_loop()),
        asyncio.create_task(update_fivesec_errors_loop(root_dir)),
    ]
    async with RUNNING_TASKS_LOCK:
        RUNNING_TASKS = tasks
    logger.info("All websocket tasks started")
    return tasks

async def start_binance_websocket(root_dir):
    """
    Стартует систему. Создает все таски.
    """
    global SYSTEM_STATE, INTENTIONAL_STOP
    INTENTIONAL_STOP = False
    SYSTEM_STATE = "RUNNING"

    # просто создаём таски
    await _spawn_all_tasks(root_dir)

    logger.info("All websocket tasks started (not awaiting gather)")
    # ничего не await'им, корутина сразу вернёт управление

async def _stop_system_async():
    """
    Асинхронно отменяет все фоновые задачи и переводит систему в STOPPED.
    """
    global INTENTIONAL_STOP, SYSTEM_STATE
    if SYSTEM_STATE == "STOPPED":
        logger.info("System already STOPPED")
        return
    INTENTIONAL_STOP = True
    SYSTEM_STATE = "STOPPED"

    async with RUNNING_TASKS_LOCK:
        tasks = list(RUNNING_TASKS)
    logger.warning(f"Cancelling {len(tasks)} tasks...")

    for t in tasks:
        try:
            t.cancel()
        except Exception as e:
            logger.error(f"Failed to cancel task {t}: {e}")

    if tasks:
        _ = await asyncio.gather(*tasks, return_exceptions=True)
    async with RUNNING_TASKS_LOCK:
        RUNNING_TASKS.clear()

    logger.warning("All tasks cancelled. System is STOPPED.")
    # loop остаётся крутиться, чтобы можно было возобновить

async def _resume_system_async(root_dir):
    """
    Асинхронно возобновляет систему, если она была остановлена.
    """
    global INTENTIONAL_STOP, SYSTEM_STATE
    if SYSTEM_STATE == "RUNNING":
        logger.info("System already RUNNING")
        return
    INTENTIONAL_STOP = False
    SYSTEM_STATE = "RUNNING"
    await start_binance_websocket(root_dir)

def resume_system(root_dir):
    if MAIN_LOOP is None:
        raise RuntimeError("Main loop not set")
    asyncio.run_coroutine_threadsafe(_resume_system_async(root_dir), MAIN_LOOP)

async def _resume_system_async(root_dir):
    global INTENTIONAL_STOP, SYSTEM_STATE
    if SYSTEM_STATE == "RUNNING":
        logger.info("System already RUNNING")
        return
    INTENTIONAL_STOP = False
    SYSTEM_STATE = "RUNNING"
    await start_binance_websocket(root_dir)