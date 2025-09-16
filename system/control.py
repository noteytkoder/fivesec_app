from data.ws_client import producer_ws, consumer_loop
from model.predict import fivesec_prediction_loop, update_fivesec_errors_loop
from model.train import fivesec_retrain_loop

async def _spawn_all_tasks(root_dir):
    # … твой код создания тасков …

async def start_binance_websocket(root_dir):
    # … твой код …

async def _stop_system_async():
    # …

async def _resume_system_async(root_dir):
    # …

def stop_system():
    # …

def resume_system(root_dir):
    # …
