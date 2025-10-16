# layout.py
"""
Layout Dash-приложения.
Формирует основной layout, использует панели из panels.py.
"""
from dash import html, dcc
from .panels import create_fivesec_layout, create_server_status_panel, create_orderbook_status_panel

def build_layout(config):
    return html.Div([
        html.Div(id="main-content", children=create_fivesec_layout(config)),
        html.Div(id="server-status", children=create_server_status_panel()),
        html.Div(id="orderbook-status", children=create_orderbook_status_panel()),  # Новая панель статуса стакана
        dcc.Checklist(
            id="model-selector",
            options=[{"label": "Использовать модель со стаканом", "value": "orderbook"}],
            value=["orderbook"] if config["model"].get("use_orderbook", False) else []
        ),
        html.P(id="model-status", children="Модель: Только Kline"),  # Статус модели
        dcc.Interval(id="interval-component", interval=config["visual"]["update_interval"], n_intervals=0),
        dcc.Interval(id="server-status-interval", interval=5000, n_intervals=0),
        dcc.Store(id="graph-layout", data={}),
        dcc.Store(id="interval-update", data=config["visual"]["update_interval"]),
        dcc.Store(id="pred-fivesec-layout", data={}),
    ])