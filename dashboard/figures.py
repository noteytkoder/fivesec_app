"""
Функции построения графиков.
"""
import plotly.graph_objects as go
import pandas as pd
from model import predict_fivesec
from config_manager import load_config

config = load_config()

def create_main_figure(df, show_candles, show_error_band, last_time, error_band_width):
    if df is None or df.empty:
        return go.Figure()
    if not isinstance(df.index, pd.DatetimeIndex):
        return go.Figure()

    fig = go.Figure()
    if show_candles:
        fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                     name="Свечи", increasing_line_color="green", decreasing_line_color="red"))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Цена закрытия"))

    required_cols = [
        "close","rsi","sma","volume","log_volume",
        "close_lag_1","close_lag_2","close_lag_3",
        "rsi_lag_1","rsi_lag_2","rsi_lag_3",
        "sma_lag_1","sma_lag_2","sma_lag_3"
    ]
    if not all(col in df.columns for col in required_cols) or len(df) == 0:
        return fig

    features = df.iloc[-1][required_cols]
    try:
        prediction = predict_fivesec(pd.DataFrame([features]))
        if isinstance(prediction, dict):
            # Выбираем прогноз от основной модели (из config) или среднее
            main_type = config["model"]["type"]
            prediction = prediction.get(main_type, sum(prediction.values()) / len(prediction))
    except Exception:
        prediction = None

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
    fig = go.Figure()
    style = {"display": "block" if not pred_df.empty else "none"} if isinstance(pred_df, pd.DataFrame) else {"display": "none"}
    if pred_df is not None and not pred_df.empty:
        filtered = pred_df[pred_df["timestamp"] >= (last_time - time_delta)]
        fig.add_trace(go.Scatter(x=filtered["timestamp"], y=filtered["actual_price"], mode="lines", name="Фактическая цена", line=dict(color="blue")))
        
        config_local = load_config()
        test_all = config_local.get("test_all_models", False)
        if test_all:
            fig.add_trace(go.Scatter(x=filtered["fivesec_pred_time"], y=filtered["fivesec_pred_rf"], mode="lines", name="Предсказанная (RF)", line=dict(color="orange")))
            fig.add_trace(go.Scatter(x=filtered["fivesec_pred_time"], y=filtered["fivesec_pred_xgb"], mode="lines", name="Предсказанная (XGB)", line=dict(color="purple")))
            fig.add_trace(go.Scatter(x=filtered["fivesec_pred_time"], y=filtered["fivesec_pred_lgb"], mode="lines", name="Предсказанная (LGB)", line=dict(color="green")))
            
            # Для метрик предполагаем, что mse и mae - dict { 'random_forest': value, ... }
            if isinstance(mse, dict) and isinstance(mae, dict):
                annotation_text = (
                    f"RF: MSE={mse.get('random_forest', 0):.2f}, MAE={mae.get('random_forest', 0):.2f}; "
                    f"XGB: MSE={mse.get('xgboost', 0):.2f}, MAE={mae.get('xgboost', 0):.2f}; "
                    f"LGB: MSE={mse.get('lightgbm', 0):.2f}, MAE={mae.get('lightgbm', 0):.2f}; Количество: {pred_count}"
                )
            else:
                annotation_text = "Ожидание данных"
        else:
            fig.add_trace(go.Scatter(x=filtered["fivesec_pred_time"], y=filtered["fivesec_pred"], mode="lines", name="Предсказанная цена (5 сек)"))
            annotation_text = (f"MSE: {mse:.2f}, MAE: {mae:.2f}, Количество: {pred_count}" if mse is not None and mae is not None else "Ожидание данных")
        
        fig.add_annotation(xref="paper", yref="paper", x=0.05, y=0.95, text=annotation_text, showarrow=False, font=dict(size=12, color="white"))
        fig.update_layout(title="BTC/USDT: Фактические и предсказанные цены (5 секунд)",
                          xaxis_title="Время (MSK)", yaxis_title="Цена (USDT)",
                          xaxis_range=x_range_pred, yaxis_range=y_range, showlegend=True, height=400, template="plotly_dark")
    return fig, style

# def create_orderbook_figure(orderbook_df):
#     """
#     Создаёт график для отображения данных стакана (например, временной ряд imbalance).
#     """
#     if orderbook_df is None or orderbook_df.empty:
#         return go.Figure()
#     
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=orderbook_df.index,
#         y=orderbook_df["imbalance"],
#         mode="lines",
#         name="Bid/Ask Imbalance"
#     ))
#     fig.update_layout(
#         title="Order Book Imbalance",
#         xaxis_title="Time",
#         yaxis_title="Imbalance",
#         template="plotly_dark"
#     )
#     return fig