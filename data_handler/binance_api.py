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
    Предвычисляет фичи и добавляет в буфер.
    """
    try:
        url = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=100"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        data["timestamp"] = pd.Timestamp.now(tz=MSK_TZ).isoformat()
        logger.info("Order book snapshot fetched")

        if "bids" in data and "asks" in data:
            bids = np.array(data["bids"], dtype=float)
            asks = np.array(data["asks"], dtype=float)

            if len(bids) > 0 and len(asks) > 0:
                # Binance гарантирует сортировку snapshot-а:
                # bids в порядке убывания, asks в порядке возрастания.
                best_bid = bids[0, 0]
                best_ask = asks[0, 0]
                
                # Средний спред по топ-5 уровням
                spread_5 = np.mean(asks[:5, 0] - bids[:5, 0]) if len(bids) >= 5 and len(asks) >= 5 else best_ask - best_bid
                
                # Дисбаланс по топ-10 уровням
                bid_volume_10 = bids[:10, 1].sum() if len(bids) >= 10 else bids[:, 1].sum()
                ask_volume_10 = asks[:10, 1].sum() if len(asks) >= 10 else asks[:, 1].sum()
                total_volume_10 = bid_volume_10 + ask_volume_10
                imbalance_10 = (bid_volume_10 - ask_volume_10) / total_volume_10 if total_volume_10 > 0 else 0.0
                
                # Нормализованные объёмы
                rel_bid_volume_10 = bid_volume_10 / total_volume_10 if total_volume_10 > 0 else 0.5
                rel_ask_volume_10 = ask_volume_10 / total_volume_10 if total_volume_10 > 0 else 0.5
                
                # Для дельт объёмов сравниваем с последним снимком (если есть)
                delta_bid_vol_10 = 0.0
                delta_ask_vol_10 = 0.0
                with buffer_lock:
                    if orderbook_buffer:
                        last_item = orderbook_buffer[-1]
                        delta_bid_vol_10 = bid_volume_10 - last_item.get("bid_volume_10", bid_volume_10)
                        delta_ask_vol_10 = ask_volume_10 - last_item.get("ask_volume_10", ask_volume_10)

                # Проверка на валидность спреда
                if spread_5 <= 0:
                    logger.warning(f"Invalid snapshot spread_5: best_bid={best_bid}, best_ask={best_ask}, spread_5={spread_5}")
                    return None

                item = {
                    "timestamp": data["timestamp"],
                    "spread_5": spread_5,
                    "mid_price": (best_ask + best_bid) / 2,  # Сохраняем для расчёта дельты в дальнейшем
                    "imbalance_10": imbalance_10,
                    "rel_bid_volume_10": rel_bid_volume_10,
                    "rel_ask_volume_10": rel_ask_volume_10,
                    "bid_volume_10": bid_volume_10,
                    "ask_volume_10": ask_volume_10,
                    "delta_bid_vol_10": delta_bid_vol_10,
                    "delta_ask_vol_10": delta_ask_vol_10
                }

                with buffer_lock:
                    orderbook_buffer.append(item)
                logger.info(f"Order book snapshot added to buffer, size: {len(orderbook_buffer)}")
            else:
                logger.warning("Empty bids or asks in orderbook snapshot")
        else:
            logger.warning(f"Invalid orderbook snapshot: keys={list(data.keys())}")

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
                bids = np.array(data["b"], dtype=float)
                asks = np.array(data["a"], dtype=float)

                if len(bids) > 0 and len(asks) > 0:
                    # Сортировка по цене
                    bids = bids[bids[:, 0].argsort()[::-1]]  # убывание (лучшая цена первой)
                    asks = asks[asks[:, 0].argsort()]        # возрастание (лучшая цена первой)

                    best_bid = bids[0, 0]
                    best_ask = asks[0, 0]
                    
                    # Средний спред по топ-5 уровням
                    spread_5 = np.mean(asks[:5, 0] - bids[:5, 0]) if len(bids) >= 5 and len(asks) >= 5 else best_ask - best_bid
                    
                    # Дисбаланс по топ-10 уровням
                    bid_volume_10 = bids[:10, 1].sum() if len(bids) >= 10 else bids[:, 1].sum()
                    ask_volume_10 = asks[:10, 1].sum() if len(asks) >= 10 else asks[:, 1].sum()
                    total_volume_10 = bid_volume_10 + ask_volume_10
                    imbalance_10 = (bid_volume_10 - ask_volume_10) / total_volume_10 if total_volume_10 > 0 else 0.0
                    
                    # Нормализованные объёмы
                    rel_bid_volume_10 = bid_volume_10 / total_volume_10 if total_volume_10 > 0 else 0.5
                    rel_ask_volume_10 = ask_volume_10 / total_volume_10 if total_volume_10 > 0 else 0.5
                    
                    # Дельты объёмов
                    delta_bid_vol_10 = 0.0
                    delta_ask_vol_10 = 0.0
                    with buffer_lock:
                        if orderbook_buffer:
                            last_item = orderbook_buffer[-1]
                            delta_bid_vol_10 = bid_volume_10 - last_item.get("bid_volume_10", bid_volume_10)
                            delta_ask_vol_10 = ask_volume_10 - last_item.get("ask_volume_10", ask_volume_10)

                    # Проверка на валидность спреда
                    if spread_5 <= 0:
                        logger.warning(f"Invalid spread_5: best_bid={best_bid}, best_ask={best_ask}, spread_5={spread_5}")
                        continue

                    item = {
                        "timestamp": data["timestamp"],
                        "spread_5": spread_5,
                        "mid_price": (best_ask + best_bid) / 2,  # Для расчёта дельты
                        "imbalance_10": imbalance_10,
                        "rel_bid_volume_10": rel_bid_volume_10,
                        "rel_ask_volume_10": rel_ask_volume_10,
                        "bid_volume_10": bid_volume_10,
                        "ask_volume_10": ask_volume_10,
                        "delta_bid_vol_10": delta_bid_vol_10,
                        "delta_ask_vol_10": delta_ask_vol_10
                    }
                    logger.debug(f"Orderbook features: spread_5={spread_5:.5f}, imbalance_10={item['imbalance_10']:.3f}, rel_bid_volume_10={item['rel_bid_volume_10']:.3f}", extra={'source': 'binance_api'})
                    if abs(item['rel_bid_volume_10'] - item['rel_ask_volume_10']) > 0.99:  # Проверка на выбросы
                        logger.warning(f"Orderbook outlier: rel_bid_volume_10={item['rel_bid_volume_10']}, rel_ask_volume_10={item['rel_ask_volume_10']}", extra={'source': 'binance_api'})

                    with buffer_lock:
                        orderbook_buffer.append(item)
                else:
                    logger.warning(f"Empty bids or asks: bids={len(bids)}, asks={len(asks)}", extra={'source': 'binance_api'})
            else:
                logger.warning(f"Invalid message: name={name}, keys={list(data.keys())}", extra={'source': 'binance_api'})
        except Exception as e:
            logger.error(f"Consumer error: {e}", exc_info=True, extra={'source': 'binance_api'})