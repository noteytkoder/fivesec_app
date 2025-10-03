"""
Регистрация online callback'ов и серверных маршрутов.
Сохраняет совместимость с оригинальным fivesec_dashboard.py.
"""
import os
import time
import psutil
import pandas as pd
from dash import no_update, html
from dash.dependencies import Input, Output, State
from flask import Response
from config_manager import load_config, load_environment_config, save_config
from logger import setup_logger
from .utils import prepare_data, prepare_pred_df
from .figures import create_main_figure, create_prediction_figure
from data_handler import fivesec_buffer, buffer_lock, fivesec_prediction_file_lock, fivesec_predictions, stop_system, resume_system, get_current_orderbook_df, MSK_TZ, orderbook_buffer
import pytz
from pathlib import Path

def _get_file_reversed(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(reversed(lines))
    except Exception:
        return ""
_cached_status = None
_cached_buffer_hash = None

def register_online_callbacks(app, config, buffer_deque):
    """
    Регистрирует callback'ы и flask routes на переданном app.
    :param app: dash.Dash instance
    :param config: dict from load_config()
    :param buffer_deque: ссылка на fivesec_buffer
    """
    ROOT_DIR = os.path.abspath(os.path.dirname(app.server.root_path))
    logger = setup_logger(log_dir=os.path.join(ROOT_DIR, "logs"))
    env_name = config.get("app_env")
    env_config = load_environment_config()

    # update_graph (replicates original complexity)
    @app.callback(
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
    def update_graph(n, show_candles, show_error_band, autoscale_range, main_relayout_data, pred_relayout_data, main_stored_layout, pred_stored_layout):
        msk_tz = pytz.timezone(config.get("timezone", "Europe/Moscow"))
        try:
            # store relayouts if provided
            if main_relayout_data and "xaxis.range[0]" in main_relayout_data:
                try:
                    start = pd.to_datetime(main_relayout_data["xaxis.range[0]"])
                    end = pd.to_datetime(main_relayout_data["xaxis.range[1]"])
                    main_stored_layout = {"xaxis.range[0]": start.isoformat(), "xaxis.range[1]": end.isoformat()}
                except Exception:
                    main_stored_layout = {}

            if pred_relayout_data and "xaxis.range[0]" in pred_relayout_data:
                try:
                    start = pd.to_datetime(pred_relayout_data["xaxis.range[0]"])
                    end = pd.to_datetime(pred_relayout_data["xaxis.range[1]"])
                    pred_stored_layout = {"xaxis.range[0]": start.isoformat(), "xaxis.range[1]": end.isoformat()}
                except Exception:
                    pred_stored_layout = {}

            with buffer_lock:
                if not buffer_deque:
                    return create_main_figure(None, False, False, None, 0), create_prediction_figure(pd.DataFrame(), None, None, 0, pd.Timestamp.now(tz=msk_tz), pd.Timedelta(minutes=1), None, None)[0], {"display": "none"}, main_stored_layout, pred_stored_layout
                data_copy = list(buffer_deque)

            df, latest_timestamp = prepare_data(data_copy)
            if df is None or df.empty:
                return no_update, no_update, {"display": "none"}, main_stored_layout, pred_stored_layout

            last_time = df.index.max()
            first_time = df.index.min()

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

            x_range = (
                [
                    pd.to_datetime(main_stored_layout["xaxis.range[0]"]),
                    pd.to_datetime(main_stored_layout["xaxis.range[1]"])
                ]
                if main_stored_layout and "xaxis.range[0]" in main_stored_layout and "xaxis.range[1]" in main_stored_layout
                else default_x_range
            )

            # (above line kept minimal; later we use default if parse fails)
            # Prepare pred df
            pred_df, mse, mae, pred_count, _ = prepare_pred_df(msk_tz, last_time, time_delta, fivesec_predictions, fivesec_prediction_file_lock)

            error_band_width = config["visual"]["error_band_min"]
            if mae is not None:
                error_band_width = max(mae * config["visual"]["error_band_multiplier"], config["visual"]["error_band_min"])

            y_range = [df["close"].min() * 0.995, df["close"].max() * 1.005]

            show_candles_flag = "candles" in (show_candles or [])
            show_band_flag = "show" in (show_error_band or [])

            fig = create_main_figure(df, show_candles_flag, show_band_flag, last_time, error_band_width)
            fig.update_layout(title="BTC/USDT: Цены и прогноз (5 секунд)",
                              xaxis_title="Время (MSK)",
                              yaxis_title="Цена (USDT)",
                              xaxis_range=default_x_range,
                              yaxis_range=y_range,
                              showlegend=True,
                              height=700,
                              template="plotly_dark",
                              dragmode="zoom",
                              uirevision="main-graph",
                              xaxis=dict(tickformat="%Y-%m-%d %H:%M:%S", tickangle=45),
                              margin=dict(b=100))

            pred_fig, pred_style = create_prediction_figure(pred_df, mse, mae, pred_count, last_time, time_delta, default_x_range_pred, y_range)

            return fig, pred_fig, pred_style, main_stored_layout or {}, pred_stored_layout or {}
        except Exception as e:
            logger.error(f"Error in update_graph: {e}", exc_info=True)
            return no_update, no_update, {"display": "none"}, {}, {}

    # download button
    @app.callback(Output("download-btn", "n_clicks"), Input("download-btn", "n_clicks"))
    def download_data(n_clicks):
        if n_clicks:
            with buffer_lock:
                df = pd.DataFrame(buffer_deque)
            os.makedirs(os.path.join(ROOT_DIR, "data", "downloads"), exist_ok=True)
            df.to_csv(os.path.join(ROOT_DIR, "data", "downloads", f"fivesec_{int(time.time())}.csv"), index=False)
            logger.info("5-second data downloaded")
        return n_clicks

    # apply settings
    @app.callback(
        Output("apply-settings", "n_clicks"),
        Input("apply-settings", "n_clicks"),
        State("buffer-size", "value"),
        State("fivesec-train-interval", "value"),
        State("fivesec-max-depth", "value"),
        State("fivesec-n_estimators", "value"),
        State("fivesec-train-window-seconds", "value"),
        State("csv-write-interval", "value")
    )
    def update_settings(n_clicks, buffer_size, train_interval, max_depth, n_estimators, window_seconds, csv_write_interval):
        if n_clicks:
            try:
                new_conf = load_config()
                if buffer_size: new_conf["data"]["buffer_size"] = int(buffer_size)
                if train_interval: new_conf["data"]["fivesec_train_interval"] = int(train_interval)
                if max_depth: new_conf["model"]["fivesec_max_depth"] = int(max_depth)
                if n_estimators: new_conf["model"]["fivesec_n_estimators"] = int(n_estimators)
                if window_seconds: new_conf["model"]["fivesec_train_window_seconds"] = int(window_seconds)
                if csv_write_interval: new_conf["data"]["csv_write_interval"] = int(csv_write_interval)
                save_config(new_conf)
                logger.info("Settings saved to config.yaml")
            except Exception as e:
                logger.error(f"Error in update_settings: {e}", exc_info=True)
                raise
        return n_clicks

    @app.callback(Output("restart-btn", "n_clicks"), Input("restart-btn", "n_clicks"))
    def restart_app(n_clicks):
        if n_clicks:
            Path(os.path.join(ROOT_DIR, "fivesec_restart.flag")).touch()
            os._exit(0)
        return n_clicks

    @app.callback(Output("stop-btn", "n_clicks"), Input("stop-btn", "n_clicks"))
    def stop_app(n_clicks):
        if n_clicks:
            stop_system()
        return n_clicks

    # server routes (mirrors original endpoints)
    @app.server.route(env_config[env_name]["logtotal_endpoint"], methods=["GET"])
    def serve_logtotal():
        log_file = os.path.join(ROOT_DIR, "logs", "fivesec_app.log")
        if not os.path.exists(log_file):
            return Response("Лог отсутствует", status=404, mimetype="text/plain")
        return Response(_get_file_reversed(log_file), mimetype="text/plain")

    @app.server.route(env_config[env_name]["csv_endpoint"], methods=["GET"])
    def serve_predictions_csv():
        csv_file = os.path.join(ROOT_DIR, "logs", "fivesec_predictions.csv")
        if not os.path.exists(csv_file):
            return Response("Файл предсказаний отсутствует", status=404, mimetype="text/plain")
        return Response(_get_file_reversed(csv_file), mimetype="text/plain")

    @app.server.route(env_config[env_name]["predictions_log_endpoint"], methods=["GET"])
    def serve_predictions_log():
        log_file = os.path.join(ROOT_DIR, "logs", "fivesec_predictions.log")
        if not os.path.exists(log_file):
            return Response("Лог предсказаний отсутствует", status=404, mimetype="text/plain")
        return Response(_get_file_reversed(log_file), mimetype="text/plain")

    @app.server.route(env_config[env_name]["table_endpoint"], methods=["GET"])
    def serve_fivesec_predictions_table():
        with fivesec_prediction_file_lock:
            df = pd.DataFrame(list(fivesec_predictions))
        if df.empty:
            return Response("Логи отсутствуют", status=404, mimetype="text/plain")
        # same rendering as original minimal table
        last_pred = df.iloc[-1]
        timestamp = pd.to_datetime(last_pred["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        actual_price = float(last_pred.get("actual_price", 0.0))
        fivesec_pred = float(last_pred.get("fivesec_pred", 0.0))
        fivesec_change_str = f"{last_pred.get('fivesec_change_pct', 0.0):+.4f}"
        fivesec_pred_time = pd.to_datetime(last_pred.get("fivesec_pred_time")).strftime("%Y-%m-%d %H:%M:%S")
        # compute MAE/Trend over 10 min
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        tz = df["timestamp"].dt.tz
        now_ts = pd.Timestamp.now(tz=tz) if tz is not None else pd.Timestamp.now()
        ten_min_ago = now_ts - pd.Timedelta(minutes=10)
        recent = df[df["timestamp"] >= ten_min_ago]
        mae_10min = None
        trend_acc_10min = None
        if not recent.empty:
            if "fivesec_error" in recent:
                ve = pd.to_numeric(recent["fivesec_error"], errors="coerce").dropna()
                if not ve.empty:
                    mae_10min = ve.mean()
            if "fivesec_trend_accuracy" in recent:
                vt = pd.to_numeric(recent["fivesec_trend_accuracy"], errors="coerce").dropna()
                if not vt.empty:
                    trend_acc_10min = vt.mean() * 100.0
        mae_str = f"{mae_10min:.4f}" if mae_10min is not None else "..."
        trend_str = f"{trend_acc_10min:.2f}" if trend_acc_10min is not None else "..."
        html_row = f"""
            <tr>
                <td>{timestamp}</td>
                <td>{actual_price:.4f}</td>
                <td>{fivesec_pred:.4f} ({fivesec_change_str})<br><small>{fivesec_pred_time}</small></td>
                <td>{mae_str}</td>
                <td>{trend_str}%</td>
            </tr>
        """
        template_path = os.path.join(ROOT_DIR, "logs_fivesec_template.html")
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
            html_content = template.replace("{{TABLE_ROWS}}", html_row)
            return Response(html_content, mimetype="text/html")
        else:
            return Response(f"<table>{html_row}</table>", mimetype="text/html")

    import hashlib

    @app.callback(
        [
            Output("orderbook-buffer-size", "children"),
            Output("orderbook-last-update", "children"),
            Output("orderbook-imbalance", "children"),
        ],
        [Input("interval-component", "n_intervals")]
    )
    def update_orderbook_status(n):
        global _cached_status, _cached_buffer_hash
        with buffer_lock:
            # Используем хэш для проверки изменений в буфере
            if not orderbook_buffer:
                current_hash = None
            else:
                last_item = orderbook_buffer[-1]
                hash_input = str(len(orderbook_buffer)) + str(last_item.get("timestamp", ""))
                current_hash = hashlib.md5(hash_input.encode()).hexdigest()

            # Если хэш не изменился, возвращаем кэшированный статус
            if current_hash == _cached_buffer_hash and _cached_status is not None:
                logger.debug("Using cached orderbook status")
                return _cached_status
            
            # Формируем статус
            if not orderbook_buffer:
                status = ("Buffer Size: 0", "Last Update: N/A", "Bid/Ask Imbalance (Top-10): N/A")
            else:
                latest = orderbook_buffer[-1]
                buffer_size = len(orderbook_buffer)
                last_update = pd.to_datetime(latest["timestamp"]).tz_convert(MSK_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
                imbalance_10 = latest["imbalance_10"]
                status = (
                    f"Buffer Size: {buffer_size}",
                    f"Last Update: {last_update}",
                    f"Bid/Ask Imbalance (Top-10): {imbalance_10:.2f}"
                )
            
            # Обновляем кэш
            _cached_status = status
            _cached_buffer_hash = current_hash
        
        logger.debug(f"Orderbook status updated: {status}")
        return status
    
    # Колбэк для переключения модели
    @app.callback(
        Output("model-status", "children"),
        Input("model-selector", "value")
    )
    def update_model_selection(values):
        use_orderbook = "orderbook" in values
        config["model"]["use_orderbook"] = use_orderbook
        save_config(config)  # Сохраняем изменения в конфиг
        return f"Model: {'Order Book' if use_orderbook else 'Kline Only'}"
    
    # server status callback
    @app.callback(Output("server-status-content", "children"), Input("server-status-interval", "n_intervals"))
    def update_server_status(n):
        if n == 0:
            return [
                html.P("Загрузка CPU: Ожидание..."),
                html.P("Использование памяти: Ожидание..."),
                html.P("Время работы сервера: Ожидание..."),
                html.P("Статус: Ожидание...")
            ]
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        uptime = time.time() - app.server.start_time if hasattr(app.server, "start_time") else 0
        days, rem = divmod(uptime, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, seconds = divmod(rem, 60)
        uptime_str = f"{int(days)}д {int(hours)}ч {int(minutes)}м {int(seconds)}с"
        status = "ОК" if cpu < 90 and mem < 90 else "Высокая нагрузка"
        return [
            html.P(f"Загрузка CPU: {cpu:.1f}%"),
            html.P(f"Использование памяти: {mem:.1f}%"),
            html.P(f"Время работы сервера: {uptime_str}"),
            html.P(f"Статус: {status}")
        ]
