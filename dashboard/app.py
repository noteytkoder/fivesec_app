# dashboard/app.py
"""
Фабрика Dash-приложения.
Создаёт dash.Dash с BasicAuth, секретным ключом и layout.
"""
from dash import Dash
from dash_auth import BasicAuth
import secrets
import os
from config_manager import load_config, load_environment_config
from logger import setup_logger
from .layout import build_layout
import time
import dash_bootstrap_components as dbc  # Для тем и tooltips
import logging

def create_dash_app(mode='main'):
    """
    Создает Dash приложение в зависимости от режима ('main' или 'analysis').
    """
    logger = setup_logger()
    logger.debug(f"Запуск create_dash_app с mode={mode}")
    
    try:
        config = load_config()
        env_name = config.get("app_env")
        env_conf = load_environment_config()
        logger.info(f"Конфигурация загружена: app_env={env_name}, env_conf={env_conf}")
        
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        logger.debug(f"ROOT_DIR установлен: {ROOT_DIR}")
        
        if mode == 'analysis':
            logger.debug("Переключение на создание приложения анализа")
            from .analysis import create_analysis_app
            return create_analysis_app()
        
        logger.debug("Создание основного Dash приложения")
        dash_app = Dash(__name__, assets_folder="static", external_stylesheets=[dbc.themes.DARKLY])
        logger.debug("Dash приложение инициализировано с темой DARKLY")
        
        dash_app.server.secret_key = secrets.token_hex(16)
        logger.debug("Секретный ключ сервера установлен")
        
        # Время для статуса сервера
        dash_app.server.start_time = time.time()
        logger.debug(f"Установлено время старта сервера: {dash_app.server.start_time}")
        
        # Basic auth
        creds = {config["auth"]["username"]: config["auth"]["password"]}
        BasicAuth(dash_app, creds)
        logger.debug("BasicAuth настроен")
        
        # Layout uses config
        dash_app.layout = build_layout(config)
        logger.debug("Layout приложения создан")
        
        logger.info("Основное Dash приложение успешно создано")
        return dash_app
    except Exception as e:
        logger.error(f"Ошибка при создании Dash приложения: {str(e)}", exc_info=True)
        raise