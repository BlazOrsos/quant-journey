import json
import logging
import sys
from pathlib import Path

# Resolve project root (two levels up from this file: src/pipelines/ -> quant-journey/)
ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

from utils.logger import setup_logger
import time


# ---------------------------------------------------------------------------
# STAGE 1 — INIT
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def init() -> tuple[dict, logging.Logger]:
    config_path = ROOT / "config" / "pnd_config.json"
    config = load_config(config_path)

    log_path = ROOT / config["data_paths"]["log_path"] / "pnd_pipeline.log"
    log_level = getattr(logging, config.get("log_level", "INFO").upper(), logging.INFO)

    logger = setup_logger("pnd_pipeline", log_path, level=log_level)
    logger.info("--- PND Pipeline starting ---")
    logger.info(f"Config loaded from: {config_path}")
    logger.info(
        f"Strategy: {config['strategy_name']} | "
        f"Interval: {config['parameters']['candle_interval']} | "
        f"Lookback: {config['parameters']['lookback_days']}d"
    )

    return config, logger


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config, logger = init()

    try:
        while True:
            logger.info("Heartbeat: PND Pipeline is running...")
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("PND Pipeline stopped by user.")
