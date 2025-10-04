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
from sortedcontainers import SortedDict

last_orderbook_update_id = None
last_orderbook_update_id = None  # Глобальная переменная для lastUpdateId снапшота

local_bids = SortedDict(reverse=True)  # Цены bids убывание, значение — volume
local_asks = SortedDict()  # Цены asks возрастание, значение — volume
last_update_id = None

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
    global last_update_id, local_bids, local_asks
    try:
        url = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=100"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        data["timestamp"] = pd.Timestamp.now(tz=MSK_TZ).isoformat()
        logger.info("Order book snapshot fetched")

        # Обновляем last_update_id
        last_update_id = data.get("lastUpdateId")
        logger.debug(f"Updated last_update_id: {last_update_id}")

        # Очищаем и загружаем локальный стакан
        local_bids.clear()
        local_asks.clear()
        for price, volume in data["bids"]:
            price = float(price)
            volume = float(volume)
            if volume > 0:
                local_bids[price] = volume
        for price, volume in data["asks"]:
            price = float(price)
            volume = float(volume)
            if volume > 0:
                local_asks[price] = volume

        # Вычисляем фичи на полном стакане
        item = calculate_orderbook_features(data["timestamp"])
        if item is not None:
            with buffer_lock:
                orderbook_buffer.append(item)
    except Exception as e:
        logger.error(f"Error fetching orderbook snapshot: {e}", exc_info=True)

def calculate_orderbook_features(timestamp):
    if not local_bids or not local_asks:
        logger.warning("Local orderbook empty, skipping features")
        return None

    # Best levels
    best_bid = local_bids.peekitem(0)[0]  # Highest bid
    best_ask = local_asks.peekitem(0)[0]  # Lowest ask

    # Top 5 spread avg
    bid_prices = list(local_bids.keys())[:5]
    ask_prices = list(local_asks.keys())[:5]
    spread_5 = np.mean([ask_prices[i] - bid_prices[i] for i in range(min(5, len(bid_prices), len(ask_prices)))]) if bid_prices and ask_prices else best_ask - best_bid

    if spread_5 <= 0:
        logger.warning(f"Invalid spread_5 in local orderbook: best_bid={best_bid}, best_ask={best_ask}, spread_5={spread_5}")
        return None

    # Top 10 volumes
    bid_volume_10 = sum(local_bids[price] for price in list(local_bids.keys())[:10])
    ask_volume_10 = sum(local_asks[price] for price in list(local_asks.keys())[:10])
    total_volume_10 = bid_volume_10 + ask_volume_10
    imbalance_10 = (bid_volume_10 - ask_volume_10) / total_volume_10 if total_volume_10 > 0 else 0.0
    rel_bid_volume_10 = bid_volume_10 / total_volume_10 if total_volume_10 > 0 else 0.5
    rel_ask_volume_10 = ask_volume_10 / total_volume_10 if total_volume_10 > 0 else 0.5

    # Deltas (от предыдущего, если есть)
    delta_bid_vol_10 = 0.0
    delta_ask_vol_10 = 0.0
    with buffer_lock:
        if orderbook_buffer:
            last_item = orderbook_buffer[-1]
            delta_bid_vol_10 = bid_volume_10 - last_item.get("bid_volume_10", bid_volume_10)
            delta_ask_vol_10 = ask_volume_10 - last_item.get("ask_volume_10", ask_volume_10)

    item = {
        "timestamp": timestamp,
        "spread_5": spread_5,
        "mid_price": (best_ask + best_bid) / 2,
        "imbalance_10": imbalance_10,
        "rel_bid_volume_10": rel_bid_volume_10,
        "rel_ask_volume_10": rel_ask_volume_10,
        "bid_volume_10": bid_volume_10,
        "ask_volume_10": ask_volume_10,
        "delta_bid_vol_10": delta_bid_vol_10,
        "delta_ask_vol_10": delta_ask_vol_10
    }

    logger.debug(f"Orderbook features: spread_5={spread_5:.5f}, imbalance_10={imbalance_10:.3f}, rel_bid_volume_10={rel_bid_volume_10:.3f}")
    if abs(rel_bid_volume_10 - rel_ask_volume_10) > 0.99:
        logger.warning(f"Orderbook outlier: rel_bid_volume_10={rel_bid_volume_10}, rel_ask_volume_10={rel_ask_volume_10}")

    return item

async def orderbook_snapshot_loop(interval: int = 5):
    """
    Периодически подтягивает полный снапшот стакана через REST
    и кладет в общий буфер, чтобы устранить дрейф дельт.
    """
    while True:
        try:
            await fetch_orderbook_snapshot()
        except Exception as e:
            logger.error(f"Error in orderbook_snapshot_loop: {e}", exc_info=True)
        await asyncio.sleep(interval)

async def producer_ws(uri, name, queue):
    """
    Подключается к WebSocket Binance, получает сообщения и помещает в очередь.
    """
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                logger.info(f"WebSocket {name} connected to {uri}")
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
                logger.info(f"WebSocket {name} connected to {uri}")
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    logger.debug(f"Orderbook WS message: keys={list(data.keys())}, data={data}")
                    data["timestamp"] = pd.Timestamp.now(tz=MSK_TZ).isoformat()
                    await queue.put((name, json.dumps(data)))  # Помещаем сериализованный JSON в очередь
        except Exception as e:
            logger.error(f"WebSocket {name} error: {e}")
            await asyncio.sleep(5)

async def consumer_loop(raw_queue):
    """
    Обрабатывает сообщения из очереди WebSocket, добавляет данные в буферы.
    """
    global last_update_id
    message_count, last_fivesec_timestamp = 0, None
    while True:
        try:
            name, raw = await raw_queue.get()
            data = json.loads(raw)
            logger.debug(f"Consumer received: name={name}, data_keys={list(data.keys())}", extra={'source': 'binance_api'})
            
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
                    logger.info(f"Klines processed: {message_count}, buffer size: {len(fivesec_buffer)}", extra={'source': 'binance_api'})
            
            elif name == "orderbook_diff" and data.get("e") == "depthUpdate" and "b" in data and "a" in data:
                # Проверка update IDs
                U = data.get("U")
                u = data.get("u")
                if last_update_id is None:
                    logger.warning(f"Diff received before snapshot: U={U}, u={u}, last_update_id={last_update_id}", extra={'source': 'binance_api'})
                    asyncio.create_task(fetch_orderbook_snapshot())  # Запускаем снапшот
                    continue
                if u < last_update_id:
                    logger.debug(f"Ignoring outdated orderbook_diff: U={U}, u={u}, last_update_id={last_update_id}", extra={'source': 'binance_api'})
                    continue
                if U <= last_update_id + 1 <= u:
                    # Применяем дельту к локальному стакану
                    for price, volume in data["b"]:
                        price = float(price)
                        volume = float(volume)
                        if volume == 0:
                            local_bids.pop(price, None)
                        else:
                            local_bids[price] = volume
                    for price, volume in data["a"]:
                        price = float(price)
                        volume = float(volume)
                        if volume == 0:
                            local_asks.pop(price, None)
                        else:
                            local_asks[price] = volume

                    # Обновляем last_update_id
                    last_update_id = u
                    logger.debug(f"Applied orderbook_diff: U={U}, u={u}, last_update_id={last_update_id}", extra={'source': 'binance_api'})

                    # Вычисляем фичи на обновлённом стакане
                    item = calculate_orderbook_features(data["timestamp"])
                    if item is not None:
                        with buffer_lock:
                            orderbook_buffer.append(item)
                else:
                    logger.warning(f"Out-of-sync orderbook_diff: U={U}, u={u}, last_update_id={last_update_id}", extra={'source': 'binance_api'})
                    asyncio.create_task(fetch_orderbook_snapshot())  # Запускаем ресинхронизацию
                    continue
            else:
                logger.warning(f"Invalid message: name={name}, keys={list(data.keys())}", extra={'source': 'binance_api'})
        except Exception as e:
            logger.error(f"Consumer error: {e}", exc_info=True, extra={'source': 'binance_api'})