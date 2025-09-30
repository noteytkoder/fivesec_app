"""
Пакет data_handler.
Собирает все функции и переменные, чтобы внешние модули могли импортировать как раньше:
from data_handler import fetch_fivesec_historical_data, fivesec_prediction_loop, start_binance_websocket ...
"""


from .config import ROOT_DIR, LOGS_DIR, logger, MSK_TZ, INTERVAL_SECONDS, config
from .buffers import (
    buffer_lock, fivesec_buffer, fivesec_predictions, fivesec_prediction_file_lock, get_current_buffer_df, get_current_orderbook_df, orderbook_buffer
)
from .indicators import (
    process_timestamp, ensure_datetime_index, compute_rsi,
    calculate_indicators, process_data_for_model
)
from .binance_api import fetch_fivesec_historical_data, producer_ws, consumer_loop, fetch_orderbook_snapshot
from .prediction_loop import fivesec_prediction_loop
from .retrain_loop import fivesec_retrain_loop
from .errors_loop import update_fivesec_errors_loop
from .system_control import (
    set_main_loop, start_binance_websocket, stop_system, resume_system
)


cached_mae_10min = None
cached_trend_accuracy_10min = None