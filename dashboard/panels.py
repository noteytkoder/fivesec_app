"""
UI-панели: элементы управления, настройки и панель статуса.
"""
from dash import html, dcc

def create_fivesec_layout(config):
    return html.Div([
        html.Div([
            dcc.Tabs(id="control-tabs", value="controls", children=[
                dcc.Tab(label="Управление", value="controls", children=[
                    dcc.Checklist(id="show-candles", options=[{"label": "Показать свечи", "value": "candles"}], value=[]),
                    dcc.Checklist(id="show-error-band", options=[{"label": "Зона погрешности", "value": "show"}],
                                  value=["show"] if config["visual"]["show_error_band"] else []),
                    html.Label("Диапазон автоскейлинга:"),
                    dcc.Dropdown(id="autoscale-range", options=[
                        {"label": "1 минута", "value": "1min"},
                        {"label": "10 минут", "value": "10min"},
                        {"label": "1 час", "value": "1hour"},
                        {"label": "Не ограниченно", "value": "unlimited"},
                    ], value="10min"),
                    html.Button("Скачать данные", id="download-btn"),
                    html.Button("Перезапустить приложение", id="restart-btn", n_clicks=0),
                    html.Button("Стоп (полный)", id="stop-btn", n_clicks=0,
                                style={"backgroundColor": "#8a0606", "color": "white"}),
                ]),
                dcc.Tab(label="Настройки", value="settings", children=create_settings_panel(config)),
            ]),
        ], style={"width": "20%", "display": "inline-block", "vertical-align": "top"}),
        html.Div([
            dcc.Graph(id="main-graph", config={"displayModeBar": True, "scrollZoom": True, "modeBarButtonsToAdd": ["zoom2d", "pan2d", "select2d", "lasso2d"]}),
            dcc.Graph(id="predictions-graph-fivesec", style={"display": "none"}, config={"displayModeBar": True, "scrollZoom": True}),
            dcc.Graph(id="orderbook-graph", style={"display": "block"}, config={"displayModeBar": True, "scrollZoom": True}),  # Новый график стакана
        ], style={"width": "80%", "display": "inline-block"}),
    ])

def create_settings_panel(config):
    return html.Div([
        html.H3("Настройки", style={"margin-top": "20px"}),
        html.Div([html.Label("Размер буфера:"), dcc.Input(id="buffer-size", type="number", value=config["data"]["buffer_size"], style={"width": "100px", "margin": "10px"})]),
        html.Div([html.Label("Интервал переобучения (сек):"), dcc.Input(id="fivesec-train-interval", type="number", value=config["data"]["fivesec_train_interval"], style={"width": "100px", "margin": "10px"})]),
        html.Div([html.Label("Максимальная глубина модели:"), dcc.Input(id="fivesec-max-depth", type="number", value=config["model"]["fivesec_max_depth"], style={"width": "100px", "margin": "10px"})]),
        html.Div([html.Label("Количество деревьев:"), dcc.Input(id="fivesec-n_estimators", type="number", value=config["model"]["fivesec_n_estimators"], style={"width": "100px", "margin": "10px"})]),
        html.Div([html.Label("Период обучения (сек):"), dcc.Input(id="fivesec-train-window-seconds", type="number", value=config["model"]["fivesec_train_window_seconds"], style={"width": "100px", "margin": "10px"})]),
        html.Div([html.Label("Интервал записи в CSV (сек):"), dcc.Input(id="csv-write-interval", type="number", value=config["data"]["csv_write_interval"], style={"width": "100px", "margin": "10px"})]),
        html.Button("Применить", id="apply-settings"),
    ])

def create_server_status_panel():
    return html.Div([
        html.H3("Статус сервера", style={"margin-top": "20px"}),
        html.Div(id="server-status-content", children=[
            html.P("Загрузка CPU: ...%", id="cpu-usage"),
            html.P("Использование памяти: ...%", id="memory-usage"),
            html.P("Время работы сервера: ...", id="uptime"),
            html.P("Статус: ...", id="server-health")
        ], style={"border": "1px solid #444", "padding": "10px", "margin": "10px"}),
    ])

def create_orderbook_status_panel():
    """
    Панель статуса стакана.
    """
    return html.Div([
        html.H3("Order Book Status"),
        html.P(id="orderbook-buffer-size", children="Buffer Size: ..."),
        html.P(id="orderbook-last-update", children="Last Update: ..."),
        html.P(id="orderbook-imbalance", children="Bid/Ask Imbalance: ...")
    ])