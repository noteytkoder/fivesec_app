"""
Модуль управления системой.
Запускает и останавливает фоновые задачи (REST, WebSocket, предсказания, переобучение, обновление ошибок).
"""

import asyncio
import os
from pathlib import Path

from .config import logger, MAIN_LOOP
from . import config

from .binance_api import producer_ws, consumer_loop
from .prediction_loop import fivesec_prediction_loop
from .retrain_loop import fivesec_retrain_loop
from .errors_loop import update_fivesec_errors_loop

# Глобальные переменные для управления
# MAIN_LOOP = None
SYSTEM_STATE = "RUNNING"
INTENTIONAL_STOP = False
RUNNING_TASKS = []
RUNNING_TASKS_LOCK = asyncio.Lock()
ACTIVE_QUEUE = None


def stop_system():
    global MAIN_LOOP
    if MAIN_LOOP is None:
        raise RuntimeError("Main loop not set")
    
def set_main_loop(loop):
    """
    Сохраняет ссылку на основной asyncio loop для запуска/остановки задач.
    """
    global MAIN_LOOP
    MAIN_LOOP = loop

async def _spawn_all_tasks(root_dir):
    """
    Создает все фоновые задачи и возвращает список asyncio.Task.
    """
    global RUNNING_TASKS, ACTIVE_QUEUE
    ACTIVE_QUEUE = asyncio.Queue(maxsize=10000)
    fivesec_kline_uri = "wss://stream.binance.com:443/ws/btcusdt@kline_1s"

    tasks = [
        asyncio.create_task(producer_ws(fivesec_kline_uri, "fivesec_kline", ACTIVE_QUEUE)),
        asyncio.create_task(consumer_loop(ACTIVE_QUEUE)),
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
    Стартует систему. Создает все таски и следит за их завершением.
    Если завершились не по стопу – создается флаг рестарта.
    """
    global SYSTEM_STATE, INTENTIONAL_STOP
    INTENTIONAL_STOP = False
    SYSTEM_STATE = "RUNNING"

    tasks = await _spawn_all_tasks(root_dir)

    try:
        _ = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        async with RUNNING_TASKS_LOCK:
            RUNNING_TASKS = []

    if not INTENTIONAL_STOP:
        config.logger.error("start_binance_websocket exited unexpectedly, creating restart flag")
        Path(os.path.join(root_dir, "fivesec_restart.flag")).touch()
        os._exit(0)
    else:
        config.logger.info("System stopped intentionally — staying down")

async def _stop_system_async():
    """
    Асинхронно отменяет все задачи и переводит систему в STOPPED.
    """
    global INTENTIONAL_STOP, SYSTEM_STATE
    if SYSTEM_STATE == "STOPPED":
        config.logger.info("System already STOPPED")
        return
    INTENTIONAL_STOP = True
    SYSTEM_STATE = "STOPPED"

    async with RUNNING_TASKS_LOCK:
        tasks = list(RUNNING_TASKS)
    config.logger.warning(f"Cancelling {len(tasks)} tasks...")

    for t in tasks:
        try:
            t.cancel()
        except Exception as e:
            config.logger.error(f"Failed to cancel task {t}: {e}")

    if tasks:
        _ = await asyncio.gather(*tasks, return_exceptions=True)
    async with RUNNING_TASKS_LOCK:
        RUNNING_TASKS.clear()

    config.logger.warning("All tasks cancelled. System is STOPPED.")

async def _resume_system_async(root_dir):
    """
    Асинхронно возобновляет систему, если она была остановлена.
    """
    global INTENTIONAL_STOP, SYSTEM_STATE
    if SYSTEM_STATE == "RUNNING":
        config.logger.info("System already RUNNING")
        return
    INTENTIONAL_STOP = False
    SYSTEM_STATE = "RUNNING"
    await start_binance_websocket(root_dir)


def resume_system(root_dir):
    """
    Возобновляет систему из внешнего кода (через run_coroutine_threadsafe).
    """
    if MAIN_LOOP is None:
        raise RuntimeError("Main loop not set")
    asyncio.run_coroutine_threadsafe(_resume_system_async(root_dir), MAIN_LOOP)
