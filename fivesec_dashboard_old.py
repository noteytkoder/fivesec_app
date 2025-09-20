from dash import Dash, dcc, html, Input, Output, callback, State
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
import os
import psutil
from pathlib import Path
import pytz
from dash_auth import BasicAuth
import secrets
from flask import Response

from data_handler import (
    fivesec_buffer, buffer_lock, fivesec_prediction_file_lock,
    fivesec_predictions, process_data_for_model,
    stop_system, resume_system, cached_mae_10min, cached_trend_accuracy_10min
)
from model import predict_fivesec
from config_manager import load_config, load_environment_config, save_config
from logger import setup_logger

config = load_config()
env_name = config["app_env"]
env_config = load_environment_config()
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
logger = setup_logger(log_dir=os.path.join(ROOT_DIR, "logs"))
RESTART_FLAG = Path(os.path.join(ROOT_DIR, "fivesec_restart.flag"))
APP_START_TIME = time.time()

# --- dash init ---
DASH_AUTH_CREDENTIALS = {config["auth"]["username"]: config["auth"]["password"]}
SECRET_KEY = secrets.token_hex(16)
dash_app = Dash(__name__, assets_folder="static")
dash_app.server.secret_key = SECRET_KEY
BasicAuth(dash_app, DASH_AUTH_CREDENTIALS)

# --- helpers ---
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

def create_fivesec_layout():
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
                dcc.Tab(label="Настройки", value="settings", children=create_settings_panel()),
            ]),
        ], style={"width": "20%", "display": "inline-block", "vertical-align": "top"}),
        html.Div([
            dcc.Graph(id="main-graph",
                      config={"displayModeBar": True, "scrollZoom": True, "modeBarButtonsToAdd": ["zoom2d", "pan2d"]}),
            dcc.Graph(id="predictions-graph-fivesec", style={"display": "none"},
                      config={"displayModeBar": True, "scrollZoom": True}),
        ], style={"width": "80%", "display": "inline-block"}),
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

def create_settings_panel():
    return html.Div([
        html.H3("Настройки", style={"margin-top": "20px"}),
        html.Div([html.Label("Размер буфера:"),
                  dcc.Input(id="buffer-size", type="number", value=config["data"]["buffer_size"], style={"width": "100px", "margin": "10px"})]),
        html.Div([html.Label("Интервал переобучения (сек):"),
                  dcc.Input(id="fivesec-train-interval", type="number", value=config["data"]["fivesec_train_interval"], style={"width": "100px", "margin": "10px"})]),
        html.Div([html.Label("Максимальная глубина модели:"),
                  dcc.Input(id="fivesec-max-depth", type="number", value=config["model"]["fivesec_max_depth"], style={"width": "100px", "margin": "10px"})]),
        html.Div([html.Label("Количество деревьев:"),
                  dcc.Input(id="fivesec-n_estimators", type="number", value=config["model"]["fivesec_n_estimators"], style={"width": "100px", "margin": "10px"})]),
        html.Div([html.Label("Период обучения (сек):"),
                  dcc.Input(id="fivesec-train-window-seconds", type="number", value=config["model"]["fivesec_train_window_seconds"], style={"width": "100px", "margin": "10px"})]),
        html.Div([html.Label("Интервал записи в CSV (сек):"),
                  dcc.Input(id="csv-write-interval", type="number", value=config["data"]["csv_write_interval"], style={"width": "100px", "margin": "10px"})]),
        html.Button("Применить", id="apply-settings"),
    ])

def prepare_df_for_graph(data_copy, msk_tz):
    df = pd.DataFrame(data_copy)
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    if df.empty:
        return None
    return process_data_for_model(df, interval="5s")

def prepare_pred_df(msk_tz, last_time, time_delta):
    with fivesec_prediction_file_lock:
        if not fivesec_predictions:
            return pd.DataFrame(), None, None, 0, None
        pred_df = pd.DataFrame(list(fivesec_predictions))
    if pred_df.empty:
        return pred_df, None, None, 0, None
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"]).dt.tz_convert(msk_tz)
    pred_df["fivesec_pred_time"] = pd.to_datetime(pred_df["fivesec_pred_time"]).dt.tz_convert(msk_tz)
    pred_df = pred_df[pred_df["timestamp"] >= (last_time - time_delta)]
    mse, mae, pred_count = None, None, len(pred_df)
    fivesec_valid = pred_df[pred_df["fivesec_error"].notna()]
    if not fivesec_valid.empty:
        mse = np.mean(fivesec_valid["fivesec_error"] ** 2)
        mae = np.mean(fivesec_valid["fivesec_error"])
    return pred_df, mse, mae, pred_count, cached_mae_10min

def create_main_figure(df, show_candles, show_error_band, last_time, error_band_width):
    fig = go.Figure()
    if show_candles:
        fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                     name="Свечи", increasing_line_color="green", decreasing_line_color="red"))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Цена закрытия", line=dict(color="blue")))
    features = df.iloc[-1][[
        "close","rsi","sma","volume","log_volume",
        "close_lag_1","close_lag_2","close_lag_3",
        "rsi_lag_1","rsi_lag_2","rsi_lag_3",
        "sma_lag_1","sma_lag_2","sma_lag_3"
    ]]
    prediction = predict_fivesec(pd.DataFrame([features]))
    pred_time = last_time + pd.Timedelta(seconds=5)
    if prediction is not None:
        fig.add_trace(go.Scatter(x=[last_time, pred_time], y=[df["close"].iloc[-1], prediction],
                                 mode="lines", name="Прогноз (5 сек)", line=dict(color=config["visual"]["predicted_price_color"])))
        if show_error_band:
            fig.add_trace(go.Scatter(
                x=[last_time, pred_time, pred_time, last_time],
                y=[df["close"].iloc[-1], prediction + error_band_width, prediction - error_band_width, df["close"].iloc[-1]],
                fill="toself", fillcolor=config["visual"]["error_band_color"],
                line=dict(color="rgba(255,255,255,0)"),
                name=f"Зона погрешности (±{error_band_width:.2f} USDT)"
            ))
    return fig

def create_prediction_figure(pred_df, mse, mae, pred_count, last_time, time_delta, x_range_pred, y_range):
    pred_fig = go.Figure()
    pred_style = {"display": "block" if not pred_df.empty else "none"}
    if not pred_df.empty:
        filtered = pred_df[pred_df["timestamp"] >= (last_time - time_delta)]
        pred_fig.add_trace(go.Scatter(x=filtered["timestamp"], y=filtered["actual_price"], mode="lines",
                                      name="Фактическая цена", line=dict(color=config["visual"]["real_price_color"])))
        pred_fig.add_trace(go.Scatter(x=filtered["fivesec_pred_time"], y=filtered["fivesec_pred"], mode="lines",
                                      name="Предсказанная цена (5 сек)", line=dict(color=config["visual"]["predicted_price_color"])))
        annotation_text = (f"MSE: {mse:.2f}, MAE: {mae:.2f}, Количество: {pred_count}"
                           if mse is not None and mae is not None else "Ожидание данных")
        pred_fig.add_annotation(xref="paper", yref="paper", x=0.05, y=0.95, text=annotation_text,
                                showarrow=False, font=dict(size=12, color="white"))
        pred_fig.update_layout(title="BTC/USDT: Фактические и предсказанные цены (5 секунд)",
                               xaxis_title="Время (MSK)", yaxis_title="Цена (USDT)",
                               xaxis_range=x_range_pred, yaxis_range=y_range, showlegend=True,
                               height=400, template="plotly_dark", dragmode="zoom",
                               uirevision="predictions-graph-fivesec",
                               xaxis=dict(tickformat="%Y-%m-%d %H:%M:%S", tickangle=45))
    return pred_fig, pred_style

dash_app.layout = create_layout()

# --- callbacks ---
@callback(
    Output("main-graph", "figure"),
    Output("predictions-graph-fivesec", "figure"),
    Output("predictions-graph-fivesec", "style"),
    Output("graph-layout", "data"),
    Output("pred-fivesec-layout", "data"),
    Input("interval-component", "n_intervals"),
    Input("show-candles", "value"),
    Input("show-error-band", "value"),
    Input("autoscale-range", "value"),
    Input("main-graph", "relayoutData"),
    Input("predictions-graph-fivesec", "relayoutData"),
    State("graph-layout", "data"),
    State("pred-fivesec-layout", "data"),
    prevent_initial_call=True
)
def update_graph(n, show_candles, show_error_band, autoscale_range,
                 main_relayout_data, pred_relayout_data, main_stored_layout, pred_stored_layout):
    msk_tz = pytz.timezone(config.get("timezone", "Europe/Moscow"))
    try:
        with buffer_lock:
            if not fivesec_buffer:
                return go.Figure(), go.Figure(), {"display": "none"}, main_stored_layout, pred_stored_layout
            data_copy = list(fivesec_buffer)
        df = prepare_df_for_graph(data_copy, msk_tz)
        if df is None:
            return go.Figure(), go.Figure(), {"display": "none"}, main_stored_layout, pred_stored_layout
        last_time, first_time = df.index.max(), df.index.min()
        if autoscale_range == "1min":
            time_delta = pd.Timedelta(minutes=1)
        elif autoscale_range == "10min":
            time_delta = pd.Timedelta(minutes=10)
        elif autoscale_range == "unlimited":
            time_delta = last_time - first_time
        else:
            time_delta = pd.Timedelta(hours=1)
        default_x_range = [last_time - time_delta, last_time + pd.Timedelta(seconds=5)]
        default_x_range_pred = default_x_range
        x_range = ([pd.to_datetime(main_stored_layout["xaxis.range[0]"]),
                    pd.to_datetime(main_stored_layout["xaxis.range[1]"])]
                   if main_stored_layout and "xaxis.range[0]" in main_stored_layout
                   else default_x_range)
        x_range_pred = ([pd.to_datetime(pred_stored_layout["xaxis.range[0]"]),
                         pd.to_datetime(pred_stored_layout["xaxis.range[1]"])]
                        if pred_stored_layout and "xaxis.range[0]" in pred_stored_layout
                        else default_x_range_pred)
        show_candles_flag = "candles" in (show_candles or [])
        show_band_flag = "show" in (show_error_band or [])
        pred_df, mse, mae, pred_count, mae10 = prepare_pred_df(msk_tz, last_time, time_delta)
        error_band_width = max((mae or 0) * config["visual"]["error_band_multiplier"], config["visual"]["error_band_min"])
        y_range = [df["close"].min() * 0.995, df["close"].max() * 1.005]
        fig = create_main_figure(df, show_candles_flag, show_band_flag, last_time, error_band_width)
        fig.update_layout(title="BTC/USDT: Цены и прогноз (5 секунд)", xaxis_title="Время (MSK)",
                          yaxis_title="Цена (USDT)", xaxis_range=x_range, yaxis_range=y_range, showlegend=True,
                          height=700, template="plotly_dark", dragmode="zoom", uirevision="main-graph",
                          xaxis=dict(tickformat="%Y-%m-%d %H:%M:%S", tickangle=45), margin=dict(b=100))
        pred_fig, pred_style = create_prediction_figure(pred_df, mse, mae, pred_count, last_time, time_delta, x_range_pred, y_range)
        return fig, pred_fig, pred_style, main_stored_layout, pred_stored_layout
    except Exception as e:
        logger.error(f"Error in update_graph: {e}", exc_info=True)
        return go.Figure(), go.Figure(), {"display": "none"}, main_stored_layout, pred_stored_layout

@callback(Output("download-btn", "n_clicks"), Input("download-btn", "n_clicks"))
def download_data(n_clicks):
    if n_clicks:
        with buffer_lock:
            df = pd.DataFrame(fivesec_buffer)
        os.makedirs(os.path.join(ROOT_DIR, "data", "downloads"), exist_ok=True)
        df.to_csv(os.path.join(ROOT_DIR, "data", "downloads", f"fivesec_{int(time.time())}.csv"), index=False)
    return n_clicks

@callback(Output("apply-settings", "n_clicks"),
          Input("apply-settings", "n_clicks"),
          Input("buffer-size", "value"),
          Input("fivesec-train-interval", "value"),
          Input("fivesec-max-depth", "value"),
          Input("fivesec-n_estimators", "value"),
          Input("fivesec-train-window-seconds", "value"),
          Input("csv-write-interval", "value"))
def update_settings(n_clicks, buffer_size, train_interval, max_depth, n_estimators, window_seconds, csv_interval):
    if n_clicks:
        try:
            new_config = load_config()
            if buffer_size: new_config["data"]["buffer_size"] = int(buffer_size)
            if train_interval: new_config["data"]["fivesec_train_interval"] = int(train_interval)
            if max_depth: new_config["model"]["fivesec_max_depth"] = int(max_depth)
            if n_estimators: new_config["model"]["fivesec_n_estimators"] = int(n_estimators)
            if window_seconds: new_config["model"]["fivesec_train_window_seconds"] = int(window_seconds)
            if csv_interval: new_config["data"]["csv_write_interval"] = int(csv_interval)
            save_config(new_config)
            logger.info("Settings saved to config.yaml")
        except Exception as e:
            logger.error(f"Error in update_settings: {e}", exc_info=True)
            raise
    return n_clicks

@callback(Output("restart-btn", "n_clicks"), Input("restart-btn", "n_clicks"))
def restart_app(n_clicks):
    if n_clicks:
        RESTART_FLAG.touch()
        os._exit(0)
    return n_clicks

@callback(Output("stop-btn", "n_clicks"), Input("stop-btn", "n_clicks"))
def stop_app(n_clicks):
    if n_clicks:
        stop_system()
    return n_clicks

@dash_app.server.route("/logs")
def serve_logtotal():
    log_dir = os.path.join(ROOT_DIR, "logs")
    return Response(
        "\n".join(open(os.path.join(log_dir, f), encoding="utf-8").read() for f in os.listdir(log_dir) if f.endswith(".log")),
        mimetype="text/plain")

@dash_app.server.route("/fivesec_predictions.csv")
def serve_predictions_csv():
    csv_file_path = os.path.join(ROOT_DIR, "logs", "fivesec_predictions.csv")
    if not os.path.exists(csv_file_path):
        return Response("CSV file not found", status=404)
    return Response(open(csv_file_path, "rb"), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=fivesec_predictions.csv"})

@dash_app.server.route("/predictions_log")
def serve_predictions_log():
    log_dir = os.path.join(ROOT_DIR, "logs")
    file_path = os.path.join(log_dir, "predictions.log")
    if not os.path.exists(file_path):
        return Response("predictions.log not found", status=404)
    return Response(open(file_path, "rb"), mimetype="text/plain")

@dash_app.server.route("/fivesec_predictions_table")
def serve_fivesec_predictions_table():
    with fivesec_prediction_file_lock:
        df = pd.DataFrame(list(fivesec_predictions))
    if df.empty:
        return "<p>Нет данных</p>"
    return df.tail(100).to_html(index=False, escape=False)

@callback(Output("server-status-content", "children"), Input("server-status-interval", "n_intervals"))
def update_server_status(n):
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    uptime = time.strftime("%H:%M:%S", time.gmtime(time.time() - APP_START_TIME))
    status = "Работает" if not RESTART_FLAG.exists() else "Перезапуск..."
    return [
        html.P(f"Загрузка CPU: {cpu:.1f}%"),
        html.P(f"Использование памяти: {mem:.1f}%"),
        html.P(f"Время работы сервера: {uptime}"),
        html.P(f"Статус: {status}")
    ]

if __name__ == "__main__":
    dash_app.run(debug=False, host=config["app"]["host"], port=config["app"]["port"])

def start_fivesec_dash():
    env_name = config.get("app_env", "dev")
    env_config = load_environment_config()
    env_settings = env_config[env_name]
    dash_app.run(debug=False,
                        host="0.0.0.0",
                        port=env_settings["port_dash"])
