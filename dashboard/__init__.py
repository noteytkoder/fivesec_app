"""
Пакет dashboard: фабрика Dash-приложения и регистрация callbacks/routes.
"""
from .app import create_dash_app
from .callbacks_online import register_online_callbacks
from .layout import build_layout

__all__ = ["create_dash_app", "register_online_callbacks", "build_layout"]
