import pandas as pd, numpy as np
import plotly.graph_objects as go
from model.predict import predict_fivesec
from features.indicators import process_data_for_model
from logger import setup_logger
from config_manager import load_config

logger = setup_logger()
config = load_config()

cached_df = None
cached_timestamp = None
cached_pred_df = None
cached_pred_timestamp = None
cached_mae_10min = None

def prepare_data(data_copy, msk_tz):
    # твоя prepare_data
    pass

def prepare_predictions(msk_tz, last_time, time_delta):
    # твоя prepare_predictions
    pass

def create_main_figure(df, show_candles, show_error_band, last_time, error_band_width):
    # твоя create_main_figure
    pass

def create_prediction_figure(pred_df, mse_fivesec, mae_fivesec, pred_count,
                             last_time, time_delta, x_range_pred, y_range):
    # твоя create_prediction_figure
    pass
