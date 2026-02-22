import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

class DotMsFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
            ms = int(record.msecs)
            s = s.replace('%f', f'{ms:03d}')
        else:
            s = ct.strftime("%Y-%m-%d %H:%M:%S")
            s += f".{int(record.msecs):03d}"
        return s

def setup_logger(name: str, log_path: str | Path, level: int = logging.INFO) -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = DotMsFormatter(
        '[%(asctime)s] %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S.%f'
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger