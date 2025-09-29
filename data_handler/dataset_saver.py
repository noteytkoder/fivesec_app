import time
from pathlib import Path
import pandas as pd
from .buffers import get_current_buffer_df, get_current_orderbook_df
from .indicators import merge_features
from .config import logger

def append_unique(df: pd.DataFrame, file_path: Path, index_col="timestamp"):
    """
    Дописывает df в CSV без дубликатов по index_col.
    Создаёт файл если его нет.
    """
    if file_path.exists():
        try:
            old = pd.read_csv(file_path, parse_dates=[index_col])
            combined = pd.concat([old, df]).drop_duplicates(subset=[index_col])
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            combined = df.drop_duplicates(subset=[index_col])
    else:
        combined = df.drop_duplicates(subset=[index_col])
    combined.to_csv(file_path, index=True)
    logger.debug(f"Appended {len(df)} rows to {file_path.name}, total {len(combined)}")

def save_current_datasets(base_dir="data/downloads", append=True):
    """
    Сохраняет kline, orderbook, merged.
    Если append=True — добавляет новые строки без дубликатов.
    """
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    # свечи
    kline_df = get_current_buffer_df()
    if kline_df is not None and not kline_df.empty:
        kline_file = Path(base_dir) / "kline_all.csv" if append else Path(base_dir) / f"kline_{ts}.csv"
        append_unique(kline_df.reset_index(), kline_file)

    # стакан
    orderbook_df = get_current_orderbook_df()
    if orderbook_df is not None and not orderbook_df.empty:
        orderbook_file = Path(base_dir) / "orderbook_all.csv" if append else Path(base_dir) / f"orderbook_{ts}.csv"
        append_unique(orderbook_df.reset_index(), orderbook_file)

    # merged
    if kline_df is not None and not kline_df.empty and \
       orderbook_df is not None and not orderbook_df.empty:
        merged_df = merge_features(kline_df, orderbook_df)
        if merged_df is not None and not merged_df.empty:
            merged_file = Path(base_dir) / "merged_all.csv" if append else Path(base_dir) / f"merged_{ts}.csv"
            append_unique(merged_df.reset_index(), merged_file)
