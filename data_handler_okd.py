import pandas as pd
import numpy as np
import websockets
import json
import asyncio
import time
import requests
import pytz
from collections import deque
from threading import Lock
from logger import setup_logger, setup_predictions_logger
from config_manager import load_config
from pathlib import Path
import os
from model import train_fivesec_model, predict_fivesec

# --- Конфигурация и константы ---
config = load_config()
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
logger = setup_logger(log_dir=LOGS_DIR)

MSK_TZ = pytz.timezone(config.get("timezone", "Europe/Moscow"))
INTERVAL_SECONDS = {"1s": 1, "5s": 5}

# --- Глобальные структуры ---
buffer_lock = Lock()
fivesec_buffer = deque(maxlen=config["data"]["buffer_size"])
fivesec_predictions = deque(maxlen=config["data"]["buffer_size"])
fivesec_prediction_file_lock = Lock()

last_fivesec_train_time = time.time()
cached_processed_df = None
last_buffer_hash = None
last_csv_write_time = 0
cached_mae_10min = None
cached_trend_accuracy_10min = None

MAIN_LOOP = None
SYSTEM_STATE = "RUNNING"
INTENTIONAL_STOP = False
RUNNING_TASKS = []
RUNNING_TASKS_LOCK = Lock()
ACTIVE_QUEUE = None

# --- Утилиты ---
def set_main_loop(loop):
    global MAIN_LOOP
    MAIN_LOOP = loop

def process_timestamp(ms_timestamp):
    return pd.to_datetime(ms_timestamp, unit="ms", utc=True).tz_convert(MSK_TZ)

def ensure_datetime_index(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        else:
            logger.error("No 'timestamp' column found in DataFrame")
            return None
    return df.sort_index()

def compute_rsi(data, periods=7):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(df):
    try:
        df["rsi"] = compute_rsi(df["close"], config["model"].get("rsi_window", 7))
        df["sma"] = df["close"].rolling(window=config["model"].get("sma_window", 3)).mean()
        df["log_volume"] = np.log1p(df["volume"])
        for lag in range(1, 4):
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"rsi_lag_{lag}"] = df["rsi"].shift(lag)
            df[f"sma_lag_{lag}"] = df["sma"].shift(lag)
        return df.dropna()
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

def process_data_for_model(df, interval="5s"):
    try:
        df = ensure_datetime_index(df)
        if df is None:
            return None
        df = df.resample(interval).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).interpolate(method="linear").ffill().dropna()
        return calculate_indicators(df)
    except Exception as e:
        logger.error(f"Error processing data for interval {interval}: {e}", exc_info=True)
        return None

def get_current_buffer_df():
    with buffer_lock:
        df = pd.DataFrame(fivesec_buffer)
    if df.empty:
        return None
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df.sort_index()

# --- Основные асинхронные функции ---
async def fetch_fivesec_historical_data():
    global fivesec_buffer
    try:
        range_ms = 60*60*1000  # 1 час
        interval = "1s"
        expected_records = range_ms // (INTERVAL_SECONDS[interval] * 1000)
        end_time = int(time.time() * 1000)
        start_time = end_time - range_ms
        limit = 1000
        klines, last_timestamp = [], None

        while start_time < end_time:
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&startTime={start_time}&endTime={end_time}&limit={limit}"
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
            except requests.exceptions.HTTPError:
                if response.status_code == 429:
                    logger.warning("Rate limit exceeded, sleeping for 60 seconds")
                    await asyncio.sleep(60)
                    continue
                raise
            new_klines = response.json()
            if not new_klines:
                break
            for kline in new_klines:
                if last_timestamp is None or kline[0] > last_timestamp:
                    klines.append(kline)
                    last_timestamp = kline[0]
            start_time = last_timestamp + (INTERVAL_SECONDS[interval] * 1000)
            await asyncio.sleep(0.5)

        if not klines:
            logger.error("No historical data fetched")
            return

        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ])
        df["timestamp"] = df["timestamp"].apply(process_timestamp)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.set_index("timestamp", inplace=True)
        df = df.sort_index().interpolate(method="linear")

        if df.isna().any().any() or np.any(np.isinf(df.values)) or (df < 0).any().any():
            logger.error("Invalid data detected in historical data")
            return

        with buffer_lock:
            fivesec_buffer.clear()
            fivesec_buffer.extend(df.reset_index().to_dict("records"))
        logger.info(f"5-second buffer updated with {len(fivesec_buffer)} records")

        if len(fivesec_buffer) >= config["data"]["min_records"]:
            df = get_current_buffer_df()
            df = process_data_for_model(df, interval="5s")
            if df is not None:
                train_fivesec_model(df)
                logger.info("Initial 5-second model trained")
    except Exception as e:
        logger.error(f"Error fetching 5-second historical data: {e}", exc_info=True)

async def producer_ws(uri, name, queue):
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                logger.info(f"WebSocket {name} connected")
                while True:
                    message = await websocket.recv()
                    await queue.put((name, message))
        except Exception as e:
            logger.error(f"WebSocket {name} error: {e}")
            await asyncio.sleep(5)

async def consumer_loop(raw_queue):
    global fivesec_buffer
    message_count, last_fivesec_timestamp = 0, None

    while True:
        try:
            name, raw = await raw_queue.get()
            data = json.loads(raw)
            if name == "fivesec_kline" and "k" in data:
                k = data["k"]
                timestamp = process_timestamp(k["t"])
                item = {
                    "timestamp": timestamp,
                    "open": float(k["o"]),
                    "high": float(k["h"]),
                    "low": float(k["l"]),
                    "close": float(k["c"]),
                    "volume": float(k["v"])
                }
                if last_fivesec_timestamp and (timestamp - last_fivesec_timestamp).total_seconds() > INTERVAL_SECONDS["1s"] * 2:
                    logger.warning(f"[consumer] GAP DETECTED: {timestamp} vs {last_fivesec_timestamp}")
                last_fivesec_timestamp = timestamp
                with buffer_lock:
                    fivesec_buffer.append(item)
                message_count += 1
                if message_count % 100 == 0:
                    logger.info(f"[consumer] Klines processed: {message_count}, buffer size: {len(fivesec_buffer)}")
        except Exception as e:
            logger.error(f"[consumer] Error: {e}")

# --- Фоновые циклы предсказаний, обучения, метрик ---
async def fivesec_prediction_loop(root_dir):
    global fivesec_predictions, last_csv_write_time, cached_trend_accuracy_10min
    logger.info("fivesec_prediction_loop started")
    predictions_logger = setup_predictions_logger(log_dir=LOGS_DIR)
    interval = "5s"
    wait_seconds = INTERVAL_SECONDS[interval]
    max_predictions = 10000
    csv_file_path = os.path.join(LOGS_DIR, "fivesec_predictions.csv")
    csv_write_interval = config.get("data", {}).get("csv_write_interval", 30)

    os.makedirs(LOGS_DIR, exist_ok=True)
    try:
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)
        with fivesec_prediction_file_lock:
            pd.DataFrame(columns=[
                "timestamp", "actual_price", "current_close", "fivesec_pred", "fivesec_change_pct",
                "fivesec_pred_time", "fivesec_actual_price", "fivesec_error",
                "fivesec_trend_pred", "fivesec_trend_actual", "fivesec_trend_accuracy"
            ]).to_csv(csv_file_path, index=False, encoding='utf-8')
    except Exception as e:
        logger.error(f"Failed to initialize fivesec_predictions.csv: {e}", exc_info=True)
        return

    while True:
        start = time.time()
        try:
            df = get_current_buffer_df()
            if df is None or len(df) < config["data"]["min_records"]:
                await asyncio.sleep(wait_seconds)
                continue
            df = process_data_for_model(df, interval="5s")
            if df is None or df.empty:
                await asyncio.sleep(wait_seconds)
                continue
            latest_row = df.iloc[-1]
            current_close = latest_row["close"]
            features = latest_row[[
                "close", "rsi", "sma", "volume", "log_volume",
                "close_lag_1", "close_lag_2", "close_lag_3",
                "rsi_lag_1", "rsi_lag_2", "rsi_lag_3",
                "sma_lag_1", "sma_lag_2", "sma_lag_3"
            ]]
            features_df = pd.DataFrame([features])

            fivesec_prediction = predict_fivesec(features_df)
            if fivesec_prediction is None:
                await asyncio.sleep(wait_seconds)
                continue

            pred_timestamp = pd.Timestamp.now(tz=MSK_TZ)
            fivesec_pred_time = pred_timestamp + pd.Timedelta(seconds=5)
            fivesec_change_pct = ((fivesec_prediction - current_close) / current_close * 100) if current_close > 0 else 0

            predictions_logger.info(
                f"время={pred_timestamp}, цена={current_close:.4f}, прогноз_на_5сек={fivesec_prediction:.4f}, целевое_время_5сек={fivesec_pred_time}, отклонение_5сек={fivesec_change_pct:+.2f}%"
            )

            prediction_record = {
                "timestamp": pred_timestamp,
                "actual_price": current_close,
                "current_close": current_close,
                "fivesec_pred": fivesec_prediction,
                "fivesec_error": None,
                "fivesec_pred_time": fivesec_pred_time,
                "fivesec_change_pct": fivesec_change_pct,
                "fivesec_actual_price": None,
                "fivesec_trend_pred": None,
                "fivesec_trend_actual": None,
                "fivesec_trend_accuracy": None
            }
            with fivesec_prediction_file_lock:
                fivesec_predictions.append(prediction_record)
                if len(fivesec_predictions) > max_predictions:
                    fivesec_predictions = deque(list(fivesec_predictions)[-max_predictions:], maxlen=max_predictions)

            current_time = time.time()
            if current_time - last_csv_write_time >= csv_write_interval:
                with fivesec_prediction_file_lock:
                    pd.DataFrame(list(fivesec_predictions)).to_csv(csv_file_path, mode='w', index=False, encoding='utf-8')
                    last_csv_write_time = current_time
        except Exception as e:
            logger.error(f"Error in fivesec_prediction_loop: {e}", exc_info=True)
        elapsed = time.time() - start
        sleep_time = max(0, wait_seconds - elapsed)
        await asyncio.sleep(sleep_time)

async def fivesec_retrain_loop():
    global last_fivesec_train_time, cached_processed_df, last_buffer_hash
    train_interval = config["data"]["fivesec_train_interval"]
    while True:
        try:
            current_time = time.time()
            df = get_current_buffer_df()
            if df is None or len(df) < config["data"]["min_records"]:
                await asyncio.sleep(train_interval)
                continue
            current_hash = (len(df), df.index[-1] if not df.empty else None)
            if current_hash == last_buffer_hash:
                df = cached_processed_df
            else:
                df = process_data_for_model(df, interval="5s")
                if df is None or df.empty:
                    await asyncio.sleep(train_interval)
                    continue
                cached_processed_df = df
                last_buffer_hash = current_hash

            if current_time - last_fivesec_train_time >= train_interval and len(df) >= config["model"].get("min_fivesec_candles", 1):
                if df.isna().any().any() or np.any(np.isinf(df.values)):
                    logger.warning("NaN/Inf in processed df, skipping retrain")
                else:
                    train_fivesec_model(df)
                    last_fivesec_train_time = current_time
                    logger.info(f"5-second model retrained, samples={len(df)}")
            await asyncio.sleep(train_interval)
        except Exception as e:
            logger.error(f"Error in fivesec_retrain_loop: {e}", exc_info=True)
            await asyncio.sleep(train_interval)

async def update_fivesec_errors_loop(root_dir):
    global fivesec_predictions, cached_mae_10min, cached_trend_accuracy_10min
    logger.info("update_fivesec_errors_loop started")
    csv_file_path = os.path.join(LOGS_DIR, "fivesec_predictions.csv")
    tolerance_seconds = {"fivesec": 10}

    while True:
        try:
            now = pd.Timestamp.now(tz=MSK_TZ)
            data_df = pd.DataFrame(list(fivesec_buffer))
            if data_df.empty:
                await asyncio.sleep(5)
                continue
            data_df["timestamp"] = pd.to_datetime(data_df["timestamp"]).dt.tz_convert(MSK_TZ)
            data_df = data_df.sort_values("timestamp")

            # Обновляем pending predictions
            pending_predictions = [p for p in list(fivesec_predictions) if p.get("fivesec_actual_price") is None]
            for prediction in pending_predictions:
                pred_time = prediction.get("fivesec_pred_time")
                if not pred_time:
                    continue
                pred_time = pd.to_datetime(pred_time).tz_convert(MSK_TZ)
                if pred_time > now:
                    continue
                idx = data_df["timestamp"].searchsorted(pred_time)
                if idx == 0 or idx >= len(data_df):
                    continue
                candidates = data_df.iloc[max(0, idx - 1):idx + 1]
                time_diff = (candidates["timestamp"] - pred_time).abs()
                if time_diff.empty:
                    continue
                closest_pos = time_diff.idxmin()
                min_diff = time_diff.loc[closest_pos]
                if pd.isna(min_diff) or min_diff.total_seconds() > tolerance_seconds["fivesec"]:
                    continue
                actual_price = data_df.loc[closest_pos, "close"]
                # Найдена актуальная цена — обновляем запись в deque (по ссылке)
                prediction["fivesec_actual_price"] = actual_price
                prediction["fivesec_error"] = abs(actual_price - prediction["fivesec_pred"]) if prediction.get("fivesec_pred") is not None else None
                current_close = prediction.get("current_close")
                if current_close is not None and prediction.get("fivesec_pred") is not None:
                    pred_value = prediction["fivesec_pred"]
                    pred_dir = 1 if pred_value > current_close else (-1 if pred_value < current_close else 0)
                    act_dir = 1 if actual_price > current_close else (-1 if actual_price < current_close else 0)
                    prediction["fivesec_trend_pred"] = pred_dir
                    prediction["fivesec_trend_actual"] = act_dir
                    prediction["fivesec_trend_accuracy"] = 1 if pred_dir == act_dir else 0

            # Пересчёт агрегатов (MAE, trend acc) по последним 10 минутам
            pred_df = pd.DataFrame(list(fivesec_predictions))
            if not pred_df.empty:
                pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
                ten_min_ago = pd.Timestamp.now(tz=MSK_TZ) - pd.Timedelta(minutes=10)
                recent_preds = pred_df[pred_df["timestamp"] >= ten_min_ago]
                if not recent_preds.empty:
                    if "fivesec_error" in recent_preds:
                        valid_err = pd.to_numeric(recent_preds["fivesec_error"], errors="coerce").dropna()
                        cached_mae_10min = valid_err.mean() if not valid_err.empty else None
                    if "fivesec_trend_accuracy" in recent_preds:
                        valid_trend = pd.to_numeric(recent_preds["fivesec_trend_accuracy"], errors="coerce").dropna()
                        cached_trend_accuracy_10min = valid_trend.mean() * 100.0 if not valid_trend.empty else None
                else:
                    cached_mae_10min = None
                    cached_trend_accuracy_10min = None

                # Перезапись CSV
                with fivesec_prediction_file_lock:
                    try:
                        pred_df.to_csv(csv_file_path, mode='w', index=False, encoding='utf-8')
                    except Exception as e:
                        logger.error(f"Failed to write predictions CSV in update loop: {e}", exc_info=True)

            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error in update_fivesec_errors_loop: {e}", exc_info=True)
            await asyncio.sleep(5)

# --- Управление задачами ---
async def _spawn_all_tasks(root_dir):
    global RUNNING_TASKS, ACTIVE_QUEUE
    ACTIVE_QUEUE = asyncio.Queue(maxsize=10000)
    fivesec_kline_uri = f"wss://stream.binance.com:443/ws/btcusdt@kline_1s"

    tasks = [
        asyncio.create_task(producer_ws(fivesec_kline_uri, "fivesec_kline", ACTIVE_QUEUE)),
        asyncio.create_task(consumer_loop(ACTIVE_QUEUE)),
        asyncio.create_task(fivesec_prediction_loop(root_dir)),
        asyncio.create_task(fivesec_retrain_loop()),
        asyncio.create_task(update_fivesec_errors_loop(root_dir)),
    ]
    with RUNNING_TASKS_LOCK:
        RUNNING_TASKS = tasks
    logger.info("All websocket tasks started")
    return tasks

async def start_binance_websocket(root_dir):
    global SYSTEM_STATE, INTENTIONAL_STOP
    INTENTIONAL_STOP = False
    SYSTEM_STATE = "RUNNING"

    tasks = await _spawn_all_tasks(root_dir)

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        with RUNNING_TASKS_LOCK:
            RUNNING_TASKS = []

    # Если не стопнули руками — значит ошибка, рестартуем
    if not INTENTIONAL_STOP:
        logger.error("start_binance_websocket exited unexpectedly, creating restart flag")
        Path(os.path.join(root_dir, "fivesec_restart.flag")).touch()
        os._exit(0)
    else:
        logger.info("System stopped intentionally — staying down")

async def _stop_system_async():
    global INTENTIONAL_STOP, SYSTEM_STATE
    if SYSTEM_STATE == "STOPPED":
        logger.info("System already STOPPED")
        return
    INTENTIONAL_STOP = True
    SYSTEM_STATE = "STOPPED"

    with RUNNING_TASKS_LOCK:
        tasks = list(RUNNING_TASKS)
    logger.warning(f"Cancelling {len(tasks)} tasks...")

    for t in tasks:
        try:
            t.cancel()
        except Exception as e:
            logger.error(f"Failed to cancel task {t}: {e}")

    if tasks:
        _ = await asyncio.gather(*tasks, return_exceptions=True)
    with RUNNING_TASKS_LOCK:
        RUNNING_TASKS.clear()

    logger.warning("All tasks cancelled. System is STOPPED.")

async def _resume_system_async(root_dir):
    global INTENTIONAL_STOP, SYSTEM_STATE
    if SYSTEM_STATE == "RUNNING":
        logger.info("System already RUNNING")
        return
    INTENTIONAL_STOP = False
    SYSTEM_STATE = "RUNNING"
    await start_binance_websocket(root_dir)

def stop_system():
    if MAIN_LOOP is None:
        raise RuntimeError("Main loop not set")
    asyncio.run_coroutine_threadsafe(_stop_system_async(), MAIN_LOOP)

def resume_system(root_dir):
    if MAIN_LOOP is None:
        raise RuntimeError("Main loop not set")
    asyncio.run_coroutine_threadsafe(_resume_system_async(root_dir), MAIN_LOOP)
