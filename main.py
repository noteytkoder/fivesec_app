import asyncio
import threading
import signal
import sys
import os
from pathlib import Path
import logging

from data_handler import (
    start_binance_websocket,
    fetch_fivesec_historical_data,
    fetch_orderbook_snapshot,
    fivesec_buffer,
)
from dashboard import create_dash_app, register_online_callbacks
from logger import setup_logger
from config_manager import load_config, load_environment_config

# Установить корневую директорию проекта
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

config = load_config()
env_config = load_environment_config()
logger = setup_logger(log_dir=os.path.join(ROOT_DIR, "logs"))
RESTART_FLAG = Path(os.path.join(ROOT_DIR, "fivesec_restart.flag"))

def signal_handler(sig, frame):
    """Обработчик сигналов завершения"""
    logger.info(f"Received signal {sig}, shutting down")
    RESTART_FLAG.touch()
    sys.exit(0)

def run_websocket():
    """Запуск WebSocket соединения"""
    logger.debug("Starting WebSocket thread")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        from data_handler import system_control
        system_control.set_main_loop(loop)

        # Создаем старт системы как таск
        loop.create_task(system_control.start_binance_websocket(ROOT_DIR))
        logger.debug("WebSocket task created")

        loop.run_forever()  # Loop живет, обрабатывает все корутины
    except Exception as e:
        logger.error(f"WebSocket thread error: {e}", exc_info=True)
        RESTART_FLAG.touch()
        sys.exit(1)

def run_fivesec_dash():
    """Запуск Dash сервера для основного приложения"""
    try:
        logger.info("Starting 5-second Dash server")
        dash_app = create_dash_app(mode='main')
        register_online_callbacks(dash_app, config, fivesec_buffer)

        env_name = config.get("app_env", "prod")
        port = env_config.get(env_name, {}).get("port_dash", 8052)
        logger.info(f"Starting main Dash server on port {port}")
        dash_app.run(host="0.0.0.0", port=port)
    except Exception as e:
        logger.error(f"5-second Dash thread error: {e}", exc_info=True)
        RESTART_FLAG.touch()
        sys.exit(1)

def run_analysis_dash():
    """Запуск Dash сервера для приложения анализа"""
    try:
        logger.info("Starting Analysis Dash server")
        dash_app = create_dash_app(mode='analysis')

        env_name = config.get("app_env", "prod")
        port = env_config.get(env_name, {}).get("port_analysis", 8070)
        logger.info(f"Starting Analysis Dash server on port {port}")
        dash_app.run(host="0.0.0.0", port=port)
    except Exception as e:
        logger.error(f"Analysis Dash thread error: {e}", exc_info=True)
        RESTART_FLAG.touch()
        sys.exit(1)

def main():
    logger.debug("Starting main function")
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.debug("Signal handlers registered")

        if RESTART_FLAG.exists():
            RESTART_FLAG.unlink()
            logger.info("Restart flag deleted on startup")

        logger.info(f"Starting 5-second application with app_env={config.get('app_env')}")

        # Загрузка исторических данных
        try:
            logger.info("Fetching historical data")
            asyncio.run(fetch_fivesec_historical_data())
            logger.debug("Historical data fetched successfully")
            asyncio.run(fetch_orderbook_snapshot())
            logger.debug("Orderbook snapshot fetched successfully")
        except Exception as e:
            logger.error(f"Error fetching historical data or orderbook snapshot: {e}", exc_info=True)

        # Запуск потоков
        websocket_thread = threading.Thread(target=run_websocket, daemon=True, name="WebSocketThread")
        dash_thread = threading.Thread(target=run_fivesec_dash, daemon=True, name="DashThread")
        analysis_thread = threading.Thread(target=run_analysis_dash, daemon=True, name="AnalysisDashThread")

        logger.debug("Starting threads: WebSocket, Dash, Analysis")
        websocket_thread.start()
        dash_thread.start()
        analysis_thread.start()

        # Главный поток ожидает завершения
        try:
            logger.debug("Main thread waiting for threads to join")
            websocket_thread.join()
            dash_thread.join()
            analysis_thread.join()
        except KeyboardInterrupt:
            logger.info("Main thread received KeyboardInterrupt")
            RESTART_FLAG.touch()
            sys.exit(0)
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        RESTART_FLAG.touch()
        sys.exit(1)

if __name__ == "__main__":
    logger.debug("Initializing main script")
    main()