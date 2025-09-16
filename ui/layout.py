from dash import html, dcc
from .settings_panel import create_settings_panel  # см. ниже
from config_manager import load_config

config = load_config()

def create_fivesec_layout():
    # dcc.Tabs + графики (как у тебя)
    pass

def create_server_status_panel():
    # твоя create_server_status_panel
    pass

def create_layout():
    return html.Div([
        html.Div(id="main-content", children=create_fivesec_layout()),
        html.Div(id="server-status", children=create_server_status_panel()),
        dcc.Interval(id="interval-component", interval=config["visual"]["update_interval"], n_intervals=0),
        dcc.Interval(id="server-status-interval", interval=5000, n_intervals=0),
        dcc.Store(id="graph-layout", data={}),
        dcc.Store(id="interval-update", data=config["visual"]["update_interval"]),
        dcc.Store(id="pred-fivesec-layout", data={}),
    ])
