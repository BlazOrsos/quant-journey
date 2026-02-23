import io
import logging
import sys
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

    # Console handler — force UTF-8 so non-ASCII ticker names don't raise
    # UnicodeEncodeError on Windows consoles that default to cp1252.
    if hasattr(sys.stdout, "buffer"):
        utf8_stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    else:
        utf8_stream = sys.stdout
    ch = logging.StreamHandler(utf8_stream)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger