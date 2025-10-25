# data_handler/binance_api.py
"""
Модуль API Binance. Содержит функции загрузки исторических данных
и работы с WebSocket (producer/consumer).
"""

from logger import pd
import requests
import asyncio
import json
import websockets
import numpy as np
from .config import config, INTERVAL_SECONDS, logger, MSK_TZ
from .buffers import buffer_lock, fivesec_buffer, orderbook_buffer, sample_tail_head
from .indicators import process_timestamp, process_data_for_model
from model import train_fivesec_model
import time
from sortedcontainers import SortedDict

class OrderBookBuffer:
    def __init__(self, depth=500):
        self.bids = SortedDict()  # ascending by price
        self.asks = SortedDict()  # ascending by price
        self.last_update_id = None
        self.depth = depth
        self.sync_issues_count = 0

    def apply_snapshot(self, data):
        self.bids.clear()
        self.asks.clear()
        for price, volume in data.get("bids", []):
            price = round(float(price), 2)
            volume = float(volume)
            if volume > 0:
                self.bids[price] = volume
        for price, volume in data.get("asks", []):
            price = round(float(price), 2)
            volume = float(volume)
            if volume > 0:
                self.asks[price] = volume
        self.last_update_id = data.get("lastUpdateId")
        self.sync_issues_count = 0
        logger.info(f"Order book snapshot applied, last_update_id={self.last_update_id}")

    def apply_diff(self, data):
        U = data.get("U")
        u = data.get("u")
        if self.last_update_id is None:
            logger.debug(f"No snapshot available, ignoring diff: U={U}, u={u}")
            return False
        if u <= self.last_update_id:
            logger.debug(f"Ignoring outdated diff: U={U}, u={u}, last_update_id={self.last_update_id}")
            return False
        if U > self.last_update_id + 1:  # Strict check for gap
            logger.warning(f"Gap detected (out-of-sync): U={U} != {self.last_update_id + 1}, u={u}")
            self.sync_issues_count += 1
            if self.sync_issues_count >= 2:
                logger.info("Gap issues threshold reached, requesting new snapshot")
                self.sync_issues_count = 0
                return False
            return False  # Do not apply if gap
        # Apply only if U == last +1
        for price, volume in data.get("b", []):
            price = round(float(price), 2)
            volume = float(volume)
            if volume == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = volume
        for price, volume in data.get("a", []):
            price = round(float(price), 2)
            volume = float(volume)
            if volume == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = volume
        self.last_update_id = u
        self.sync_issues_count = 0
        logger.debug(f"Diff applied, new last_update_id={self.last_update_id}")
        return True

    def get_features(self, timestamp):
        if not self.bids or not self.asks:
            logger.warning("Orderbook bids or asks are empty, skipping features")
            return None
        
        try:
            # Получаем отсортированные ключи (цены) из SortedDict
            bid_keys = list(self.bids.keys())  # Список цен в порядке возрастания
            ask_keys = list(self.asks.keys())  # Список цен в порядке возрастания
            best_bid = bid_keys[-1] if bid_keys else 0  # Последний ключ — максимальный bid
            best_ask = ask_keys[0] if ask_keys else 0   # Первый ключ — минимальный ask
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0

            # Топ-10 levels: highest для bids (last 10), lowest для asks (first 10)
            bids_sorted = list(self.bids.items())[-10:]  # Последние 10 (высшие bids)
            asks_sorted = list(self.asks.items())[:10]   # Первые 10 (низшие asks)

            bid_volume_10 = sum(vol for _, vol in bids_sorted)
            ask_volume_10 = sum(vol for _, vol in asks_sorted)

            # Расчёт imbalance и relative volumes
            imbalance_10 = (bid_volume_10 - ask_volume_10) / (bid_volume_10 + ask_volume_10 + 1e-10)
            rel_bid_volume_10 = bid_volume_10 / (bid_volume_10 + ask_volume_10 + 1e-10)
            rel_ask_volume_10 = 1 - rel_bid_volume_10

            # Временная дельта: разница с предыдущим состоянием
            self.prev_bid_volume_10 = getattr(self, "prev_bid_volume_10", 0)
            self.prev_ask_volume_10 = getattr(self, "prev_ask_volume_10", 0)
            
            delta_bid_vol_10 = bid_volume_10 - self.prev_bid_volume_10
            delta_ask_vol_10 = ask_volume_10 - self.prev_ask_volume_10
            
            # Обновляем предыдущие значения
            self.prev_bid_volume_10 = bid_volume_10
            self.prev_ask_volume_10 = ask_volume_10

            # Дополнительное логирование для отладки
            logger.debug(f"get_features: best_bid={best_bid}, best_ask={best_ask}, mid_price={mid_price}, len_bids={len(self.bids)}, len_asks={len(self.asks)}")

            return {
                "timestamp": timestamp,
                "mid_price": mid_price,
                "bid_volume_10": bid_volume_10,
                "ask_volume_10": ask_volume_10,
                "imbalance_10": imbalance_10,
                "rel_bid_volume_10": rel_bid_volume_10,
                "rel_ask_volume_10": rel_ask_volume_10,
                "delta_bid_vol_10": delta_bid_vol_10,
                "delta_ask_vol_10": delta_ask_vol_10
            }
        except (IndexError, KeyError) as e:
            logger.error(f"Error accessing orderbook levels: {e}, bids_len={len(self.bids)}, asks_len={len(self.asks)}")
            return None

orderbook = OrderBookBuffer()

async def fetch_fivesec_historical_data():
    try:
        range_ms = 60 * 60 * 1000 * 2
        interval = "1s"
        end_time = int(time.time() * 1000)
        start_time = end_time - range_ms
        limit = 1000
        klines = []
        while start_time < end_time:
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&startTime={start_time}&endTime={end_time}&limit={limit}"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            new_klines = response.json()
            if not new_klines:
                break
            klines.extend(new_klines)
            start_time = new_klines[-1][0] + (INTERVAL_SECONDS[interval] * 1000)
            await asyncio.sleep(0.5)

        if not klines:
            logger.error("No historical data fetched")
            return

        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(MSK_TZ)
        # Преобразуем только числовые столбцы в float
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.set_index("timestamp", inplace=True)
        df = df.sort_index().interpolate(method="linear")

        if df.isna().any().any() or np.any(np.isinf(df.values)) or (df[numeric_cols] < 0).any().any():
            logger.error("Invalid data detected in historical data")
            return

        with buffer_lock:
            fivesec_buffer.clear()
            fivesec_buffer.extend(df.reset_index().to_dict("records"))
        logger.info(f"Buffer updated with {len(fivesec_buffer)} records")

        if len(fivesec_buffer) >= config["data"]["min_records"]:
            df = process_data_for_model(df, interval="5s")
            if df is not None:
                train_fivesec_model(df, use_orderbook=False)
                train_fivesec_model(df, use_orderbook=True)
                logger.info("Initial 5-second models trained")
    except Exception as e:
        logger.error(f"Error fetching 5-second historical data: {e}", exc_info=True)

async def fetch_orderbook_snapshot():
    try:
        url = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=500"  # Увеличен limit
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        orderbook.apply_snapshot(data)
        item = orderbook.get_features(pd.Timestamp.now(tz=MSK_TZ))
        if item is not None:
            with buffer_lock:
                orderbook_buffer.append(item)
        logger.info(f"SNAPSHOT applied: last_update_id={orderbook.last_update_id} bids={len(orderbook.bids)} asks={len(orderbook.asks)} sync_issues={orderbook.sync_issues_count}")
        df_temp = pd.DataFrame(list(orderbook_buffer)) if orderbook_buffer else pd.DataFrame()
        logger.info(f"SNAPSHOT SAMPLE:\n{sample_tail_head(df_temp)}")
    except Exception as e:
        logger.error(f"Error fetching orderbook snapshot: {e}", exc_info=True)

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

async def consumer_loop(raw_queue):
    message_count = 0
    last_resync_time = time.time()  # Для периодического resync
    resync_interval = 60  # Каждые 60 секунд
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
                with buffer_lock:
                    fivesec_buffer.append(item)
                message_count += 1
                if message_count % 100 == 0:
                    logger.info(f"Klines processed: {message_count}, buffer size: {len(fivesec_buffer)}")
            elif name == "orderbook_diff" and data.get("e") == "depthUpdate":
                timestamp = process_timestamp(data["E"])  # Используем event time
                if not orderbook.apply_diff(data):
                    asyncio.create_task(fetch_orderbook_snapshot())
                else:
                    item = orderbook.get_features(timestamp)
                    if item is not None:
                        with buffer_lock:
                            orderbook_buffer.append(item)
                        logger.debug(f"OB APPEND: len={len(orderbook_buffer)} last_ts={item['timestamp']} mid={item['mid_price']:.2f} imb={item['imbalance_10']:.6f}")
                        if len(orderbook_buffer) % 500 == 0:  # Каждые 500 для экономии
                            df_temp = pd.DataFrame(list(orderbook_buffer))
                            logger.info(f"OB BUFFER SAMPLE ({len(orderbook_buffer)}):\n{sample_tail_head(df_temp)}")
                            # Проверка константного mid_price
                            repeated_mid = (df_temp['mid_price'].diff() == 0).astype(int).groupby((df_temp['mid_price'].diff() != 0).cumsum()).sum().max()
                            logger.info(f"OB max constant-mid run={repeated_mid}")
                            if (df_temp['imbalance_10'].abs() > 1).any():
                                logger.warning("OB APPEND: imbalance >1 detected")
                # Периодический resync
                if time.time() - last_resync_time >= resync_interval:
                    logger.info("Periodic orderbook resync triggered")
                    asyncio.create_task(fetch_orderbook_snapshot())
                    last_resync_time = time.time()
            else:
                logger.warning(f"Invalid message: name={name}")
        except Exception as e:
            logger.error(f"Consumer error: {e}", exc_info=True)