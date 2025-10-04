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

last_update_id = None
local_bids = SortedDict(reverse=True)  # Цены bids убывание, значение — volume
local_asks = SortedDict()  # Цены asks возрастание, значение — volume
sync_issues_count = 0  # Счётчик рассинхронизаций

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

        logger.info(f"Raw kline data: unique close values={df['close'].nunique()}")
        with buffer_lock:
            fivesec_buffer.clear()
            fivesec_buffer.extend(df.reset_index().to_dict("records"))
        logger.info(f"Buffer updated with {len(fivesec_buffer)} records")

        if len(fivesec_buffer) >= config["data"]["min_records"]:
            df = process_data_for_model(df, interval="5s")
            if df is not None:
                train_fivesec_model(df, use_orderbook=False)  # Обучение kline_only
                train_fivesec_model(df, use_orderbook=True)  # Обучение kline_with_orderbook
                logger.info("Initial 5-second models trained")
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
        logger.debug(f"Snapshot data: bids={data['bids'][:5]}, asks={data['asks'][:5]}")

        last_update_id = data.get("lastUpdateId")
        logger.debug(f"Updated last_update_id: {last_update_id}")

        local_bids.clear()
        local_asks.clear()
        if not data["bids"] or not data["asks"]:
            logger.warning("Empty bids or asks in snapshot")
            return

        best_bid = float(data["bids"][0][0])  # Преобразуем в float
        best_ask = float(data["asks"][0][0])  # Преобразуем в float
        for price, volume in data["bids"]:
            price = float(price)
            volume = float(volume)
            if price > 1e6 or price < 0 or abs(price - best_bid) > 300:  # Уменьшен порог
                logger.warning(f"Anomalous bid price: {price}")
                continue
            if volume > 0:
                local_bids[price] = volume
        for price, volume in data["asks"]:
            price = float(price)
            volume = float(volume)
            if price > 1e6 or price < 0 or abs(price - best_ask) > 300:
                logger.warning(f"Anomalous ask price: {price}")
                continue
            if volume > 0:
                local_asks[price] = volume

        if not local_bids or not local_asks:
            logger.warning("No valid bids or asks after filtering")
            return

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

    best_bid = local_bids.peekitem(0)[0]  # Highest bid
    best_ask = local_asks.peekitem(0)[0]  # Lowest ask
    spread_5 = best_ask - best_bid  # Используем только лучший спред
    if spread_5 > 300 or spread_5 < 0:  # Уменьшен порог
        logger.warning(f"Anomalous spread_5: {spread_5}, best_bid={best_bid}, best_ask={best_ask}")
        return None
    logger.debug(f"best_bid={best_bid}, best_ask={best_ask}, spread_5={spread_5}")

    mid_price = (best_bid + best_ask) / 2
    bid_volume_10 = sum(v for _, v in list(local_bids.items())[:10])
    ask_volume_10 = sum(v for _, v in list(local_asks.items())[:10])
    imbalance_10 = (bid_volume_10 - ask_volume_10) / (bid_volume_10 + ask_volume_10 + 1e-10)
    rel_bid_volume_10 = bid_volume_10 / (bid_volume_10 + ask_volume_10 + 1e-10)
    rel_ask_volume_10 = ask_volume_10 / (bid_volume_10 + ask_volume_10 + 1e-10)
    delta_bid_vol_10 = bid_volume_10 - sum(v for _, v in list(local_bids.items())[10:20])
    delta_ask_vol_10 = ask_volume_10 - sum(v for _, v in list(local_asks.items())[10:20])

    return {
        "timestamp": timestamp,
        "spread_5": spread_5,
        "mid_price": mid_price,
        "imbalance_10": imbalance_10,
        "rel_bid_volume_10": rel_bid_volume_10,
        "rel_ask_volume_10": rel_ask_volume_10,
        "bid_volume_10": bid_volume_10,
        "ask_volume_10": ask_volume_10,
        "delta_bid_vol_10": delta_bid_vol_10,
        "delta_ask_vol_10": delta_ask_vol_10
    }

async def orderbook_snapshot_loop(interval=1):
    while True:
        try:
            await fetch_orderbook_snapshot()
        except Exception as e:
            logger.error(f"Error in orderbook_snapshot_loop: {e}", exc_info=True)
        await asyncio.sleep(interval)

async def producer_ws(uri, name, queue):
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
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                logger.info(f"WebSocket {name} connected to {uri}")
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    logger.debug(f"Orderbook WS message: keys={list(data.keys())}, data={data}")
                    data["timestamp"] = pd.Timestamp.now(tz=MSK_TZ).isoformat()
                    await queue.put((name, json.dumps(data)))
        except Exception as e:
            logger.error(f"WebSocket {name} error: {e}")
            await asyncio.sleep(5)

async def consumer_loop(raw_queue):
    global last_update_id, sync_issues_count
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
                    logger.info(f"Klines processed: {message_count}, buffer size: {len(fivesec_buffer)}, unique close={pd.DataFrame(fivesec_buffer)['close'].nunique()}", extra={'source': 'binance_api'})
            
            elif name == "orderbook_diff" and data.get("e") == "depthUpdate" and "b" in data and "a" in data:
                U = data.get("U")
                u = data.get("u")
                if last_update_id is None:
                    logger.warning(f"Diff received before snapshot: U={U}, u={u}, last_update_id={last_update_id}")
                    asyncio.create_task(fetch_orderbook_snapshot())
                    continue
                if u < last_update_id:
                    logger.debug(f"Ignoring outdated orderbook_diff: U={U}, u={u}, last_update_id={last_update_id}")
                    continue
                if U <= last_update_id + 1 <= u:
                    if not local_bids or not local_asks:
                        logger.warning("Empty local_bids or local_asks, fetching snapshot")
                        asyncio.create_task(fetch_orderbook_snapshot())
                        continue
                    best_bid = local_bids.peekitem(0)[0] if local_bids else float(data["b"][0][0])
                    best_ask = local_asks.peekitem(0)[0] if local_asks else float(data["a"][0][0])
                    logger.debug(f"Applying orderbook_diff: best_bid={best_bid}, best_ask={best_ask}")
                    for price, volume in data["b"]:
                        price = float(price)
                        volume = float(volume)
                        if price > 1e6 or price < 0 or abs(price - best_bid) > 300:
                            logger.warning(f"Anomalous bid price in diff: {price}")
                            continue
                        if volume == 0:
                            local_bids.pop(price, None)
                        else:
                            local_bids[price] = volume
                    for price, volume in data["a"]:
                        price = float(price)
                        volume = float(volume)
                        if price > 1e6 or price < 0 or abs(price - best_ask) > 300:
                            logger.warning(f"Anomalous ask price in diff: {price}")
                            continue
                        if volume == 0:
                            local_asks.pop(price, None)
                        else:
                            local_asks[price] = volume

                    last_update_id = u
                    logger.debug(f"Applied orderbook_diff: U={U}, u={u}, last_update_id={last_update_id}")

                    item = calculate_orderbook_features(data["timestamp"])
                    if item is not None:
                        with buffer_lock:
                            orderbook_buffer.append(item)
                else:
                    logger.warning(f"Out-of-sync orderbook_diff: U={U}, u={u}, last_update_id={last_update_id}")
                    sync_issues_count += 1
                    logger.info(f"Sync issues count: {sync_issues_count}", extra={'source': 'binance_api'})
                    asyncio.create_task(fetch_orderbook_snapshot())
                    continue
            else:
                logger.warning(f"Invalid message: name={name}, keys={list(data.keys())}")
        except Exception as e:
            logger.error(f"Consumer error: {e}", exc_info=True)