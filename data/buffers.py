from collections import deque
from threading import Lock
import time

from config_manager import load_config
config = load_config()

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
