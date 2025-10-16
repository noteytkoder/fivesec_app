"""
UI-панели: элементы управления, настройки и панель статуса.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc  # Для tooltips и стилизации

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
                        {"label": "Без ограничений", "value": "unlimited"},
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
            #dcc.Graph(id="orderbook-graph", style={"display": "block"}, config={"displayModeBar": True, "scrollZoom": True}),  # Новый график стакана
        ], style={"width": "80%", "display": "inline-block"}),
    ])

def create_settings_panel(config):
    tooltip_style = {
        'backgroundColor': '#333',  # Тёмно-серый фон
        'border': '1px solid #fff',  # Белая граница
        'padding': '5px',
        'fontSize': '14px',
        'color': '#fff',  # Белый текст
        'zIndex': 1000  # Поднимаем над другими элементами
    }

    return html.Div([
        html.H3("Настройки", style={"margin-top": "20px"}),
        
        # Секция: Данные
        html.H4("Данные", style={"margin-top": "15px"}),
        html.Div([
            html.Label("Размер буфера:"),
            dcc.Input(id="buffer-size", type="number", value=config["data"]["buffer_size"], style={"width": "100px", "margin": "10px"}),
            dbc.Tooltip("Максимальный размер буфера для хранения данных Kline (свечей).", target="buffer-size", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("Интервал записи в CSV (сек):"),
            dcc.Input(id="csv-write-interval", type="number", value=config["data"]["csv_write_interval"], style={"width": "100px", "margin": "10px"}),
            dbc.Tooltip("Интервал сохранения предсказаний в CSV файл (в секундах).", target="csv-write-interval", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("Интервал переобучения (сек):"),
            dcc.Input(id="fivesec-train-interval", type="number", value=config["data"]["fivesec_train_interval"], style={"width": "100px", "margin": "10px"}),
            dbc.Tooltip("Интервал между переобучениями модели (в секундах).", target="fivesec-train-interval", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        
        # Секция: Модель
        html.H4("Модель", style={"margin-top": "15px"}),
        html.Div([
            html.Label("Тип модели:"),
            dcc.Dropdown(id="model-type", options=[
                {"label": "Random Forest", "value": "random_forest"},
                {"label": "XGBoost", "value": "xgboost"},
                {"label": "LightGBM", "value": "lightgbm"}
            ], value=config["model"]["type"], style={"width": "150px", "margin": "10px"}),
            dbc.Tooltip("Тип модели для предсказаний (если не тестировать все модели).", target="model-type", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("Тестировать все модели:"),
            dcc.Checklist(id="test-all-models", options=[{"label": "Да", "value": "test_all"}], 
                          value=["test_all"] if config.get("test_all_models", False) else []),
            dbc.Tooltip("Если включено, обучает и использует все три модели (RF, XGB, LGB).", target="test-all-models", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("Максимальная глубина:"),
            dcc.Input(id="fivesec-max-depth", type="number", value=config["model"]["fivesec_max_depth"], style={"width": "100px", "margin": "10px"}),
            dbc.Tooltip("Максимальная глубина деревьев в модели.", target="fivesec-max-depth", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("Количество оценщиков:"),
            dcc.Input(id="fivesec-n_estimators", type="number", value=config["model"]["fivesec_n_estimators"], style={"width": "100px", "margin": "10px"}),
            dbc.Tooltip("Количество деревьев/оценщиков в ансамбле.", target="fivesec-n_estimators", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("Период обучения (сек):"),
            dcc.Input(id="fivesec-train-window-seconds", type="number", value=config["model"]["fivesec_train_window_seconds"], style={"width": "100px", "margin": "10px"}),
            dbc.Tooltip("Окно данных для обучения модели (в секундах).", target="fivesec-train-window-seconds", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("Скорость обучения:"),
            dcc.Input(id="learning-rate", type="number", value=config["model"]["learning_rate"], step=0.01, style={"width": "100px", "margin": "10px"}),
            dbc.Tooltip("Скорость обучения для градиентных моделей (XGBoost/LightGBM).", target="learning-rate", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("Подвыборка строк:"),
            dcc.Input(id="subsample", type="number", value=config["model"]["subsample"], step=0.1, style={"width": "100px", "margin": "10px"}),
            dbc.Tooltip("Доля строк, используемых для обучения каждого дерева.", target="subsample", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("Подвыборка столбцов:"),
            dcc.Input(id="colsample-bytree", type="number", value=config["model"]["colsample_bytree"], step=0.1, style={"width": "100px", "margin": "10px"}),
            dbc.Tooltip("Доля признаков, используемых для обучения каждого дерева.", target="colsample-bytree", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("Ранний стоп (раунды):"),
            dcc.Input(id="early-stopping-rounds", type="number", value=config["model"]["early_stopping_rounds"], style={"width": "100px", "margin": "10px"}),
            dbc.Tooltip("Количество раундов без улучшения для ранней остановки (XGBoost/LightGBM).", target="early-stopping-rounds", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("L1 регуляризация (alpha):"),
            dcc.Input(id="reg-alpha", type="number", value=config["model"]["reg_alpha"], step=0.1, style={"width": "100px", "margin": "10px"}),
            dbc.Tooltip("L1 регуляризация для XGBoost/LightGBM.", target="reg-alpha", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("L2 регуляризация (lambda):"),
            dcc.Input(id="reg-lambda", type="number", value=config["model"]["reg_lambda"], step=0.1, style={"width": "100px", "margin": "10px"}),
            dbc.Tooltip("L2 регуляризация для XGBoost/LightGBM.", target="reg-lambda", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("Мин. сэмплов в листе:"),
            dcc.Input(id="min-samples-leaf", type="number", value=config["model"]["min_samples_leaf"], style={"width": "100px", "margin": "10px"}),
            dbc.Tooltip("Минимальное количество сэмплов в листе (для Random Forest).", target="min-samples-leaf", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("Мин. сэмплов для сплита:"),
            dcc.Input(id="min-samples-split", type="number", value=config["model"]["min_samples_split"], style={"width": "100px", "margin": "10px"}),
            dbc.Tooltip("Минимальное количество сэмплов для разбиения узла (для Random Forest).", target="min-samples-split", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("Использовать стакан:"),
            dcc.Checklist(id="use-orderbook", options=[{"label": "Да", "value": "use_orderbook"}], 
                          value=["use_orderbook"] if config["model"].get("use_orderbook", False) else []),
            dbc.Tooltip("Включить признаки из стакана ордеров в модель.", target="use-orderbook", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("Использовать scaler:"),
            dcc.Checklist(id="use-scaler", options=[{"label": "Да", "value": "use_scaler"}], 
                          value=["use_scaler"] if config["model"].get("use_scaler", False) else []),
            dbc.Tooltip("Нормализовать признаки с помощью StandardScaler.", target="use-scaler", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
        html.Div([
            html.Label("Число листьев (LightGBM):"),
            dcc.Input(id="num-leaves", type="number", value=config["model"]["params"]["lightgbm"].get("num_leaves", 31), style={"width": "100px", "margin": "10px"}),
            dbc.Tooltip("Максимальное количество листьев в дереве для LightGBM.", target="num-leaves", placement="bottom", style=tooltip_style)
        ], style={"margin-bottom": "10px"}),
 
        html.Button("Применить", id="apply-settings", style={"margin-top": "20px", "width": "150px"}),
        dbc.Tooltip("Сохранить все настройки и применить их.", target="apply-settings", placement="bottom", style=tooltip_style)
    ], style={"padding": "15px", "border": "1px solid #ccc", "border-radius": "5px"})

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
    Панель статуса стакана ордеров.
    """
    return html.Div([
        html.H3("Статус стакана ордеров"),
        html.P(id="orderbook-buffer-size", children="Размер буфера: ..."),
        html.P(id="orderbook-last-update", children="Последнее обновление: ..."),
        html.P(id="orderbook-imbalance", children="Дисбаланс Bid/Ask: ...")
    ])