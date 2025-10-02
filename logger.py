import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(log_dir="logs", default_source="unknown"):
    """Настройка основного логгера с поддержкой source"""
    logger = logging.getLogger("FiveSecAppLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs(log_dir, exist_ok=True)
        handler = RotatingFileHandler(
            os.path.join(log_dir, "fivesec_app.log"),
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(source)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Переопределяем методы log для поддержки default_source
    orig_log = logger.log
    def log_with_source(level, msg, *args, **kwargs):
        extra = kwargs.get('extra', {})
        if 'source' not in extra:
            extra['source'] = default_source
        kwargs['extra'] = extra
        orig_log(level, msg, *args, **kwargs)
    
    logger.log = log_with_source
    logger.debug = lambda msg, *args, **kwargs: log_with_source(logging.DEBUG, msg, *args, **kwargs)
    logger.info = lambda msg, *args, **kwargs: log_with_source(logging.INFO, msg, *args, **kwargs)
    logger.warning = lambda msg, *args, **kwargs: log_with_source(logging.WARNING, msg, *args, **kwargs)
    logger.error = lambda msg, *args, **kwargs: log_with_source(logging.ERROR, msg, *args, **kwargs)
    
    return logger

def setup_predictions_logger(log_dir="logs", default_source="unknown"):
    """Настройка логгера для прогнозов с поддержкой source"""
    logger = logging.getLogger("FiveSecPredictionsLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs(log_dir, exist_ok=True)
        handler = RotatingFileHandler(
            os.path.join(log_dir, "fivesec_predictions.log"),
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        formatter = logging.Formatter("%(source)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    orig_log = logger.log
    def log_with_source(level, msg, *args, **kwargs):
        extra = kwargs.get('extra', {})
        if 'source' not in extra:
            extra['source'] = default_source
        kwargs['extra'] = extra
        orig_log(level, msg, *args, **kwargs)
    
    logger.log = log_with_source
    logger.debug = lambda msg, *args, **kwargs: log_with_source(logging.DEBUG, msg, *args, **kwargs)
    logger.info = lambda msg, *args, **kwargs: log_with_source(logging.INFO, msg, *args, **kwargs)
    logger.warning = lambda msg, *args, **kwargs: log_with_source(logging.WARNING, msg, *args, **kwargs)
    logger.error = lambda msg, *args, **kwargs: log_with_source(logging.ERROR, msg, *args, **kwargs)
    
    return logger