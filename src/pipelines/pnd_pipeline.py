import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd

# Resolve project root (two levels up from this file: src/pipelines/ -> quant-journey/)
ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

from utils.logger import setup_logger
from exchanges.binance_vision import BinanceVisionClient
from exchanges.binance_klines import BinanceFuturesKlines
from data.storage import OHLCVStorage
from exchanges.binance_websocket import BinanceFuturesKlineStream
from strategies.pnd import PnDSignalManager, SignalAction


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
# STAGE 4 — WEBSOCKET SUBSCRIPTION
# ---------------------------------------------------------------------------

def build_candle_handler(
    storage: OHLCVStorage,
    signal_manager: PnDSignalManager,
    logger: logging.Logger,
) -> Callable[[str, pd.DataFrame], None]:
    """
    Return a closed-candle callback that:
      1. Merges the new candle into local Parquet storage.
      2. Runs feature engineering + signal detection via PnDSignalManager.
      3. Logs any entry/exit signals (execution layer will consume these).
    """

    last_log_time = [0]  # Use a mutable object to allow modification in closure

    def on_closed_candle(ticker: str, candle: pd.DataFrame) -> None:
        now = time.time()
        if now - last_log_time[0] > 10:  # Log at most once every 10 seconds
            logger.info("Received new closed candles from websocket.")
            last_log_time[0] = now

        # --- Step 1: Update local storage ---
        existing = storage.read_ticker(ticker)
        merged = OHLCVStorage._merge(existing, candle)
        storage.write_ticker(ticker, merged)
        logger.debug(f"[{ticker}] Storage updated ({len(merged)} rows).")

        # --- Step 2 & 3: Feature engineering + signal detection ---
        actions = signal_manager.update(ticker, merged)

        for signal in actions:
            if signal.action == SignalAction.ENTER_SHORT:
                logger.info(
                    f"[{ticker}] *** ENTRY SIGNAL *** "
                    f"time={signal.signal_time}, price={signal.entry_price}, "
                    f"ret_z={signal.ret_z:.2f}, rvol={signal.rvol:.2f}"
                )
                # TODO: Stage 7 — pass to execution layer
            elif signal.action == SignalAction.EXIT_SHORT:
                logger.info(
                    f"[{ticker}] *** EXIT SIGNAL *** "
                    f"time={signal.signal_time}, reason={signal.reason}"
                )
                # TODO: Stage 7 — pass to execution layer

    return on_closed_candle


async def heartbeat(logger: logging.Logger, interval_sec: int = 600):
    while True:
        logger.info("Heartbeat: pipeline is running.")
        await asyncio.sleep(interval_sec)

def run_websocket(
    config: dict,
    dataframes: dict[str, pd.DataFrame],
    logger: logging.Logger,
) -> None:
    """Stage 4: subscribe to live kline streams for all hydrated tickers."""
    logger.info("========== Stage 4 ==========")

    tickers = list(dataframes.keys())
    logger.info(
        f"{len(tickers)} tickers with data — subscribing to WebSocket streams …"
    )

    interval = config["parameters"]["candle_interval"]
    lookback = config["parameters"]["lookback_days"]
    data_dir = ROOT / config["data_paths"]["data_path"]

    storage = OHLCVStorage(
        data_dir=data_dir,
        lookback_days=lookback,
        candle_interval=interval,
        logger=logger,
    )

    positions_path = ROOT / "data" / "signals" / "active_positions.json"
    signal_manager = PnDSignalManager(
        config=config, logger=logger, positions_path=positions_path,
    )

    handler = build_candle_handler(storage, signal_manager, logger)

    stream = BinanceFuturesKlineStream(
        tickers=tickers,
        interval=interval,
        on_closed_candle=handler,
        logger=logger,
    )

    async def main():
        heartbeat_task = asyncio.create_task(heartbeat(logger))
        stream_task = asyncio.create_task(stream.run_forever())
        await asyncio.gather(heartbeat_task, stream_task)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        stream.stop()
        logger.info("WebSocket stream stopped by user.")


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config, logger = init()

    tickers = fetch_tickers(config, logger)

    dataframes = hydrate_data(config, tickers, logger)

    run_websocket(config, dataframes, logger)