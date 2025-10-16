"""
Регистрация online callback'ов и серверных маршрутов.
Сохраняет совместимость с оригинальным fivesec_dashboard.py.
"""
import os
import time
import psutil
from logger import pd
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
import hashlib

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

    # update_graph
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

            # Prepare pred df
            pred_df, mse, mae, pred_count, _ = prepare_pred_df(msk_tz, last_time, time_delta, fivesec_predictions, fivesec_prediction_file_lock)
            pred_fig, pred_style = create_prediction_figure(pred_df, mse, mae, pred_count, last_time, time_delta, default_x_range_pred, None)

            # Main figure
            error_band_width = config["visual"]["error_band_multiplier"] * (mae.get('random_forest', mae) if isinstance(mae, dict) else mae or config["visual"]["error_band_min"]) if mae else config["visual"]["error_band_min"]
            main_fig = create_main_figure(df, "candles" in show_candles, "show" in show_error_band, last_time, error_band_width)

            return main_fig, pred_fig, pred_style, main_stored_layout, pred_stored_layout
        except Exception as e:
            logger.error(f"Ошибка в update_graph: {e}", exc_info=True)
            return no_update, no_update, no_update, no_update, no_update

    # Flask route for predictions table
    @app.server.route("/predictions")
    def predictions():
        try:
            with fivesec_prediction_file_lock:
                if not fivesec_predictions:
                    return Response("<table></table>", mimetype="text/html")
                df = pd.DataFrame(list(fivesec_predictions))
            if df.empty:
                return Response("<table></table>", mimetype="text/html")

            latest = df.iloc[-1]
            timestamp = pd.to_datetime(latest["timestamp"]).tz_convert(MSK_TZ).strftime("%Y-%m-%d %H:%M:%S")
            actual_price = latest["actual_price"]
            fivesec_pred_time = pd.to_datetime(latest["fivesec_pred_time"]).tz_convert(MSK_TZ).strftime("%Y-%m-%d %H:%M:%S")
            if config.get("test_all_models", False):
                rf_pred = latest["fivesec_pred_rf"]
                xgb_pred = latest["fivesec_pred_xgb"]
                lgb_pred = latest["fivesec_pred_lgb"]
                rf_change = latest["fivesec_change_pct_rf"]
                xgb_change = latest["fivesec_change_pct_xgb"]
                lgb_change = latest["fivesec_change_pct_lgb"]
                fivesec_pred_str = (
                    f"RF: {rf_pred:.4f} ({rf_change:+.2f}% if rf_change else 'N/A')<br>"
                    f"XGB: {xgb_pred:.4f} ({xgb_change:+.2f}% if xgb_change else 'N/A')<br>"
                    f"LGB: {lgb_pred:.4f} ({lgb_change:+.2f}% if lgb_change else 'N/A')<br>"
                    f"<small>{fivesec_pred_time}</small>"
                )
            else:
                fivesec_pred = latest["fivesec_pred"]
                fivesec_change = latest["fivesec_change_pct"]
                fivesec_pred_str = f"{fivesec_pred:.4f} ({fivesec_change:+.2f}%)<br><small>{fivesec_pred_time}</small>"

            ten_min_ago = pd.Timestamp.now(tz=MSK_TZ) - pd.Timedelta(minutes=10)
            recent = df[df["timestamp"] >= ten_min_ago]
            mae_10min = {}
            trend_acc_10min = {}
            if config.get("test_all_models", False):
                for m_type in ['rf', 'xgb', 'lgb']:
                    error_col = f"fivesec_error_{m_type}"
                    trend_col = f"fivesec_trend_accuracy_{m_type}"
                    if error_col in recent:
                        ve = pd.to_numeric(recent[error_col], errors="coerce").dropna()
                        if not ve.empty:
                            mae_10min[m_type] = ve.mean()
                    if trend_col in recent:
                        vt = pd.to_numeric(recent[trend_col], errors="coerce").dropna()
                        if not vt.empty:
                            trend_acc_10min[m_type] = vt.mean() * 100.0
                mae_str = f"RF: {mae_10min.get('rf', 'N/A'):.4f}, XGB: {mae_10min.get('xgb', 'N/A'):.4f}, LGB: {mae_10min.get('lgb', 'N/A'):.4f}"
                trend_str = f"RF: {trend_acc_10min.get('rf', 'N/A'):.2f}%, XGB: {trend_acc_10min.get('xgb', 'N/A'):.2f}%, LGB: {trend_acc_10min.get('lgb', 'N/A'):.2f}%"
            else:
                mae_10min = None
                trend_acc_10min = None
                if "fivesec_error" in recent:
                    ve = pd.to_numeric(recent["fivesec_error"], errors="coerce").dropna()
                    if not ve.empty:
                        mae_10min = ve.mean()
                if "fivesec_trend_accuracy" in recent:
                    vt = pd.to_numeric(recent["fivesec_trend_accuracy"], errors="coerce").dropna()
                    if not vt.empty:
                        trend_acc_10min = vt.mean() * 100.0
                mae_str = f"{mae_10min:.4f}" if mae_10min is not None else "..."
                trend_str = f"{trend_acc_10min:.2f}%" if trend_acc_10min is not None else "..."

            html_row = f"""
                <tr>
                    <td>{timestamp}</td>
                    <td>{actual_price:.4f}</td>
                    <td>{fivesec_pred_str}</td>
                    <td>{mae_str}</td>
                    <td>{trend_str}</td>
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
        except Exception as e:
            logger.error(f"Ошибка в /predictions: {e}", exc_info=True)
            return Response("<table></table>", mimetype="text/html")

    # Orderbook status
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
            if not orderbook_buffer:
                current_hash = None
            else:
                last_item = orderbook_buffer[-1]
                hash_input = str(len(orderbook_buffer)) + str(last_item.get("timestamp", ""))
                current_hash = hashlib.md5(hash_input.encode()).hexdigest()

            if current_hash == _cached_buffer_hash and _cached_status is not None:
                logger.debug("Using cached orderbook status")
                return _cached_status
            
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
            
            _cached_status = status
            _cached_buffer_hash = current_hash
        
        logger.debug(f"Orderbook status updated: {status}")
        return status
    
    # Model selection
    @app.callback(
        Output("model-status", "children"),
        Input("model-selector", "value")
    )
    def update_model_selection(values):
        use_orderbook = "orderbook" in values
        config["model"]["use_orderbook"] = use_orderbook
        save_config(config)
        return f"Model: {'Order Book' if use_orderbook else 'Kline Only'}"
    
    # Server status
    @app.callback(
        Output("server-status-content", "children"),
        Input("server-status-interval", "n_intervals")
    )
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

    # Stop system
    @app.callback(
        Output("stop-btn", "n_clicks"),
        Input("stop-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def stop_system_callback(n_clicks):
        if n_clicks and n_clicks > 0:
            stop_system()
            logger.info("Система остановлена через UI")
        return 0

    # Restart system
    @app.callback(
        Output("restart-btn", "n_clicks"),
        Input("restart-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def restart_system_callback(n_clicks):
        if n_clicks and n_clicks > 0:
            resume_system()
            logger.info("Система перезапущена через UI")
        return 0

    # Apply settings
    @app.callback(
        Output("apply-settings", "n_clicks"),
        [
            Input("apply-settings", "n_clicks"),
            Input("buffer-size", "value"),
            Input("fivesec-train-interval", "value"),
            Input("fivesec-max-depth", "value"),
            Input("fivesec-n_estimators", "value"),
            Input("fivesec-train-window-seconds", "value"),
            Input("csv-write-interval", "value"),
        ],
        prevent_initial_call=True
    )
    def apply_settings(n_clicks, buffer_size, train_interval, max_depth, n_estimators, train_window, csv_interval):
        if n_clicks and n_clicks > 0:
            config["data"]["buffer_size"] = buffer_size
            config["data"]["fivesec_train_interval"] = train_interval
            config["model"]["fivesec_max_depth"] = max_depth
            config["model"]["fivesec_n_estimators"] = n_estimators
            config["model"]["fivesec_train_window_seconds"] = train_window
            config["data"]["csv_write_interval"] = csv_interval
            save_config(config)
            logger.info("Настройки обновлены через UI")
        return 0

    # Download data
    @app.callback(
        Output("download-btn", "n_clicks"),
        Input("download-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def download_data(n_clicks):
        if n_clicks and n_clicks > 0:
            with buffer_lock:
                if not buffer_deque:
                    logger.warning("Буфер пуст, нет данных для скачивания")
                    return 0
                df = pd.DataFrame(list(buffer_deque))
            if not df.empty:
                csv_path = os.path.join(ROOT_DIR, "logs", f"data_download_{pd.Timestamp.now(tz=MSK_TZ).strftime('%Y%m%d_%H%M%S')}.csv")
                df.to_csv(csv_path, index=False, encoding='utf-8')
                logger.info(f"Данные скачаны в {csv_path}")
        return 0