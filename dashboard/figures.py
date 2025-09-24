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
        fig.add_trace(go.Scatter(x=filtered["timestamp"], y=filtered["actual_price"], mode="lines", name="Фактическая цена"))
        fig.add_trace(go.Scatter(x=filtered["fivesec_pred_time"], y=filtered["fivesec_pred"], mode="lines", name="Предсказанная цена (5 сек)"))
        annotation_text = (f"MSE: {mse:.2f}, MAE: {mae:.2f}, Количество: {pred_count}" if mse is not None and mae is not None else "Ожидание данных")
        fig.add_annotation(xref="paper", yref="paper", x=0.05, y=0.95, text=annotation_text, showarrow=False, font=dict(size=12, color="white"))
        fig.update_layout(title="BTC/USDT: Фактические и предсказанные цены (5 секунд)",
                          xaxis_title="Время (MSK)", yaxis_title="Цена (USDT)",
                          xaxis_range=x_range_pred, yaxis_range=y_range, showlegend=True, height=400, template="plotly_dark")
    return fig, style

def create_orderbook_figure(orderbook_df):
    """
    Построение графика глубины стакана.
    """
    if orderbook_df is None or orderbook_df.empty:
        return go.Figure()
    fig = go.Figure()
    latest = orderbook_df.iloc[-1]
    bids = pd.DataFrame(latest["b"], columns=["price", "quantity"]).astype(float)
    asks = pd.DataFrame(latest["a"], columns=["price", "quantity"]).astype(float)
    fig.add_trace(go.Bar(x=bids["price"], y=bids["quantity"], name="Bids", marker_color="blue"))
    fig.add_trace(go.Bar(x=asks["price"], y=asks["quantity"], name="Asks", marker_color="red"))
    fig.update_layout(title="Order Book Depth", xaxis_title="Price", yaxis_title="Quantity")
    return fig