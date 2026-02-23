import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Resolve project root (two levels up from this file: src/pipelines/ -> quant-journey/)
ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

from utils.logger import setup_logger
from exchanges.binance_vision import BinanceVisionClient
from exchanges.binance_klines import BinanceFuturesKlines
from data.storage import OHLCVStorage
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
    logger.info("========== Crypto Pump & Dump Pipeline starting ==========")
    logger.info("========== Stage 1 ==========")
    logger.info(f"Config loaded from: {config_path}")
    logger.info(
        f"Strategy: {config['strategy_name']} | "
        f"Interval: {config['parameters']['candle_interval']} | "
        f"Lookback: {config['parameters']['lookback_days']}d"
    )

    return config, logger


# ---------------------------------------------------------------------------
# STAGE 2 — FETCH TICKER UNIVERSE
# ---------------------------------------------------------------------------

def fetch_tickers(config: dict, logger: logging.Logger) -> list[str]:
    """Retrieve the full list of USDT-margined perp tickers from Binance Vision."""
    logger.info("========== Stage 2 ==========")
    logger.info("Fetching ticker universe from Binance Vision...")

    client = BinanceVisionClient()
    tickers = client.get_symbols()

    logger.info(f"{len(tickers)} USDT tickers fetched.")
    return tickers


# ---------------------------------------------------------------------------
# STAGE 3 — DATA HYDRATION
# ---------------------------------------------------------------------------

def hydrate_data(
    config: dict,
    tickers: list[str],
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    """Sync local parquet store for every ticker via the three-phase cascade."""
    logger.info("========== Stage 3 ==========")
    logger.info("Starting data hydration …")

    data_dir = ROOT / config["data_paths"]["data_path"]
    interval = config["parameters"]["candle_interval"]
    lookback = config["parameters"]["lookback_days"]

    storage = OHLCVStorage(
        data_dir=data_dir,
        lookback_days=lookback,
        candle_interval=interval,
        logger=logger,
    )
    vision_client = BinanceVisionClient()
    klines_client = BinanceFuturesKlines()

    results = storage.sync_all(tickers, vision_client, klines_client)

    logger.info(
        f"Data hydration complete — {len(results)}/{len(tickers)} tickers ready."
    )
    return results


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config, logger = init()

    tickers = fetch_tickers(config, logger)

    dataframes = hydrate_data(config, tickers, logger)

    try:
        while True:
            logger.info("Heartbeat: PND Pipeline is running...")
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("PND Pipeline stopped by user.")