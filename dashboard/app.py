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

def create_dash_app():
    config = load_config()
    env_name = config.get("app_env")
    env_conf = load_environment_config()
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # тут мб не нужно на шаг выше, но хз
    logger = setup_logger(log_dir=os.path.join(ROOT_DIR, "logs"))

    dash_app = Dash(__name__, assets_folder="static")
    dash_app.server.secret_key = secrets.token_hex(16)

    #время для статуса сервера
    dash_app.server.start_time = time.time()
    
    # Basic auth
    creds = {config["auth"]["username"]: config["auth"]["password"]}
    BasicAuth(dash_app, creds)

    # layout uses config
    dash_app.layout = build_layout(config)
    return dash_app
