"""
Модуль конфигурации для системы обработки данных Binance.
Загружает конфиг, определяет корневые пути, логгер и глобальные константы.
"""

import os
import pytz
from logger import setup_logger
from config_manager import load_config

# Загружаем YAML-конфиг
config = load_config()

# Пути и логгер
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
logger = setup_logger(log_dir=LOGS_DIR)


# Часовой пояс и интервалы
MSK_TZ = pytz.timezone(config.get("timezone", "Europe/Moscow"))
INTERVAL_SECONDS = {"1s": 1, "5s": 5}
