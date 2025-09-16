from dash import Dash
from dash_auth import BasicAuth
import secrets
from config_manager import load_config, load_environment_config
from logger import setup_logger
import os
from pathlib import Path

config = load_config()
env_name = config["app_env"]
env_config = load_environment_config()
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
logger = setup_logger(log_dir=os.path.join(ROOT_DIR, "logs"))

dash_app = Dash(__name__, assets_folder="../static")
SECRET_KEY = secrets.token_hex(16)
dash_app.server.secret_key = SECRET_KEY

BasicAuth(dash_app, {
    config["auth"]["username"]: config["auth"]["password"]
})

# layout задаём здесь:
from .layout import create_layout
dash_app.layout = create_layout()

def start_fivesec_dash():
    logger.info("Starting 5-second Dash server")
    dash_app.run(port=env_config[env_name]["port_dash"], host="0.0.0.0")
