"""
Модуль API Binance. Содержит функции загрузки исторических данных
и работы с WebSocket (producer/consumer).
"""

import pandas as pd
import requests
import asyncio
import json
import websockets
import numpy as np

from .config import config, INTERVAL_SECONDS, logger, MSK_TZ
from .buffers import buffer_lock, fivesec_buffer, orderbook_buffer
from .indicators import process_timestamp, process_data_for_model
from model import train_fivesec_model
import time  

async def fetch_fivesec_historical_data():
    """
    Загружает час исторических данных BTCUSDT@1s через REST API Binance.
    Сохраняет в буфер и обучает начальную модель.
    """
    try:
        range_ms = 60 * 60 * 1000
        interval = "1s"
        end_time = int(time.time() * 1000)  # Unix-timestamp в мс
        start_time = end_time - range_ms
        limit = 1000
        klines, last_timestamp = [], None

        while start_time < end_time:
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&startTime={start_time}&endTime={end_time}&limit={limit}"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
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
        df = df.astype({c: float for c in ["open", "high", "low", "close", "volume"]})
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.set_index("timestamp", inplace=True)
        df = df.sort_index().interpolate(method="linear")

        if df.isna().any().any() or np.any(np.isinf(df.values)) or (df < 0).any().any():
            logger.error("Invalid data detected in historical data")
            return

        with buffer_lock:
            fivesec_buffer.clear()
            fivesec_buffer.extend(df.reset_index().to_dict("records"))
        logger.info(f"Buffer updated with {len(fivesec_buffer)} records")

        if len(fivesec_buffer) >= config["data"]["min_records"]:
            df = process_data_for_model(df, interval="5s")
            if df is not None:
                train_fivesec_model(df)
                logger.info("Initial 5-second model trained")
    except Exception as e:
        logger.error(f"Error fetching 5-second historical data: {e}", exc_info=True)

async def fetch_orderbook_snapshot():
    """
    Загружает моментальный снимок стакана (order book) через REST API Binance.
    """
    try:
        url = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=100@1000ms"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        data["timestamp"] = pd.Timestamp.now(tz=MSK_TZ)
        logger.info("Order book snapshot fetched")
        return data
    except Exception as e:
        logger.error(f"Error fetching order book snapshot: {e}", exc_info=True)
        return None

async def producer_ws(uri, name, queue):
    """
    Подключается к WebSocket Binance, получает сообщения и помещает в очередь.
    """
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

async def producer_orderbook_ws(uri, name, queue):
    """
    Подключается к WebSocket Binance для обновлений стакана, получает сообщения и помещает в очередь.
    """
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                logger.info(f"WebSocket {name} connected")
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)  # Парсим JSON здесь для добавления timestamp
                    data["timestamp"] = pd.Timestamp.now(tz=MSK_TZ).isoformat()
                    await queue.put((name, json.dumps(data)))  # Помещаем сериализованный JSON в очередь
        except Exception as e:
            logger.error(f"WebSocket {name} error: {e}")
            await asyncio.sleep(5)

async def consumer_loop(raw_queue):
    """
    Обрабатывает поток kline-сообщений из очереди и пополняет буфер.
    """
    message_count, last_fivesec_timestamp = 0, None

    while True:
        try:
            name, raw = await raw_queue.get()
            data = json.loads(raw)  # Ожидаем строку JSON
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
                with buffer_lock:
                    fivesec_buffer.append(item)
                message_count += 1
                if message_count % 100 == 0:
                    logger.info(f"[consumer] Klines processed: {message_count}, buffer size: {len(fivesec_buffer)}")
            if name == "orderbook_diff" and "b" in data and "a" in data:
                with buffer_lock:
                    orderbook_buffer.append(data)
                # logger.info(f"Order book buffer updated, size: {len(orderbook_buffer)}")
        except Exception as e:
            logger.error(f"[consumer] Error: {e}")