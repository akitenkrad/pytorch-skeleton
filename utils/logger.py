from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging
from logging.handlers import RotatingFileHandler
from logging import getLogger, StreamHandler, Formatter, Logger

def get_json_liner(name) -> Logger:
    """Generate Logger instance

    Args:
        name: (str) name of the logger
    Returns:
        Logger
    """

    # --------------------------------
    # 0. mkdir
    # --------------------------------
    JST = timezone(timedelta(hours=+9), 'JST')
    now =  datetime.now(JST)
    now = datetime(now.year, now.month, now.day, now.hour, now.minute, 0, tzinfo=JST)
    log_dir = Path('./logs/log') / now.strftime('%Y%m%d%H%M%S') / 'json_lines'
    log_dir.mkdir(parents=True, exist_ok=True)

    
    # --------------------------------
    # 1. logger configuration
    # --------------------------------
    logger = getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler_format = Formatter('')

    # --------------------------------
    # 3. log file configuration
    # --------------------------------
    fh = RotatingFileHandler(str(log_dir / name), maxBytes=3145728, backupCount=3000)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(handler_format)
    logger.addHandler(fh)

    # --------------------------------
    # 4. error log file configuration
    # --------------------------------
    er_fh = RotatingFileHandler(str(log_dir / name), maxBytes=3145728, backupCount=3000)
    er_fh.setLevel(logging.ERROR)
    er_fh.setFormatter(handler_format)
    logger.addHandler(er_fh)

    return logger

def get_logger(name, logfile='', silent=False) -> Logger:
    """Generate Logger instance

    Args:
        name: (str) name of the logger
        logfile: (str) logfile name
        silent: (bool) if True, don't write logs into stdout
    Returns:
        Logger
    """

    # --------------------------------
    # 0. mkdir
    # --------------------------------
    JST = timezone(timedelta(hours=+9), 'JST')
    now =  datetime.now(JST)
    now = datetime(now.year, now.month, now.day, now.hour, now.minute, 0, tzinfo=JST)
    log_dir = Path('./logs/log') / now.strftime('%Y%m%d%H%M%S')
    log_dir.mkdir(parents=True, exist_ok=True)

    if logfile == '':
        logfile = name
    
    # --------------------------------
    # 1. logger configuration
    # --------------------------------
    logger = getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler_format = Formatter('%(asctime)s [%(levelname)8s] %(name)15s - %(message)s')

    # --------------------------------
    # 2. handler configuration
    # --------------------------------
    if not silent:
        stream_handler = StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(handler_format)
        logger.addHandler(stream_handler)

    # --------------------------------
    # 3. log file configuration
    # --------------------------------
    fh = RotatingFileHandler(str(log_dir / logfile), maxBytes=3145728, backupCount=3000)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(handler_format)
    logger.addHandler(fh)

    # --------------------------------
    # 4. error log file configuration
    # --------------------------------
    er_fh = RotatingFileHandler(str(log_dir / logfile), maxBytes=3145728, backupCount=3000)
    er_fh.setLevel(logging.ERROR)
    er_fh.setFormatter(handler_format)
    logger.addHandler(er_fh)

    return logger
