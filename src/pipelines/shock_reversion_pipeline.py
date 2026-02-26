"""
Shock Reversion Pipeline — main entry point.

Orchestrates all stages of the shock-reversion strategy:

    [1] INIT           — Load config, setup logger, create directories
    [2] TICKER VALID.  — Fetch USDT-margined perp tickers from Binance REST
    [3] DATA HYDRATION — Fetch last 240 × 1m candles per ticker via REST klines
    [4] WEBSOCKET      — Subscribe to live 1m kline streams for all tickers
    [5] FEAT. ENG.     — Compute ret_z, vol_mean on each closed candle
    [6] EXECUTION      — Enter/exit shorts (TODO)

Usage::

    python src/pipelines/shock_reversion_pipeline.py
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Project root (two levels up: src/pipelines/ → quant-journey/)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.logger import setup_logger
from exchanges.binance_klines import BinanceFuturesKlines
from exchanges.binance_websocket import BinanceFuturesKlineStream

from strategies.shock_reversion import ShockReversionSignalManager, SignalAction


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — INIT
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: Path) -> dict:
    """Read and return the JSON configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)


def init() -> tuple[dict, logging.Logger]:
    """
    Stage 1: load configuration, setup logger, ensure directories exist.

    Returns
    -------
    config : dict
        Parsed contents of ``shock_reversion.json``.
    logger : logging.Logger
        Configured rotating logger.
    """
    config_path = ROOT / "config" / "shock_reversion.json"
    config = load_config(config_path)

    # — Logger —
    log_path = ROOT / config["data_paths"]["log_path"] / "shock_reversion_pipeline.log"
    log_level = getattr(
        logging, config.get("log_level", "INFO").upper(), logging.INFO
    )
    logger = setup_logger("shock_reversion", log_path, level=log_level)

    logger.info("=" * 60)
    logger.info("Shock Reversion Pipeline starting")
    logger.info("=" * 60)
    logger.info("Stage 1 — INIT")
    logger.info(f"Config loaded from: {config_path}")
    logger.info(
        f"Strategy: {config['strategy_name']} | "
        f"Interval: {config['parameters']['candle_interval']} | "
        f"Lookback: {config['parameters']['lookback_bars']} bars"
    )

    # — Ensure output directories exist —
    data_dir = ROOT / config["data_paths"]["data_path"]
    signals_dir = ROOT / config["data_paths"]["signals_path"]
    data_dir.mkdir(parents=True, exist_ok=True)
    signals_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data dir:    {data_dir}")
    logger.info(f"Signals dir: {signals_dir}")

    return config, logger


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — TICKER VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"


def fetch_tickers(config: dict, logger: logging.Logger) -> list[str]:
    """
    Stage 2: retrieve USDT-margined perpetual futures tickers from Binance.

    Uses the REST ``/fapi/v1/exchangeInfo`` endpoint (no Vision dependency).

    Returns
    -------
    list[str]
        Sorted ticker symbols, e.g. ``['BTCUSDT', 'ETHUSDT', ...]``.
    """
    logger.info("Stage 2 — TICKER VALIDATION")
    logger.info("Fetching ticker universe from Binance exchangeInfo …")

    resp = requests.get(_EXCHANGE_INFO_URL, timeout=15)
    resp.raise_for_status()
    symbols_info = resp.json().get("symbols", [])

    tickers = sorted(
        s["symbol"]
        for s in symbols_info
        if s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
    )

    logger.info(f"{len(tickers)} USDT perpetual tickers validated.")
    return tickers


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — DATA HYDRATION  (REST klines only — no Vision)
# ═══════════════════════════════════════════════════════════════════════════

def hydrate_data(
    config: dict,
    tickers: list[str],
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    """
    Stage 3: fetch the last 240 × 1m candles for every ticker via REST and
    persist each as a Parquet file.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame for all successfully hydrated
        tickers.
    """
    logger.info("Stage 3 — DATA HYDRATION")
    logger.info("Fetching last 240 × 1m candles per ticker via REST klines …")

    data_dir = ROOT / config["data_paths"]["data_path"]
    data_dir.mkdir(parents=True, exist_ok=True)

    interval = config["parameters"]["candle_interval"]
    lookback = config["parameters"]["lookback_bars"]
    klines_client = BinanceFuturesKlines()

    results: dict[str, pd.DataFrame] = {}
    errors: list[str] = []

    for i, ticker in enumerate(tickers, 1):
        try:
            logger.info(f"Hydrating [{i}/{len(tickers)}]: {ticker}")
            df = klines_client.fetch_klines(
                symbol=ticker,
                interval=interval,
                limit=lookback,
            )
            if df is not None and not df.empty:
                path = data_dir / f"{ticker}.parquet"
                df.to_parquet(path, index=False)
                results[ticker] = df

            if i % 100 == 0 or i == len(tickers):
                logger.info(f"  Hydrated {i}/{len(tickers)} tickers …")

            # Small delay to avoid 429 rate limits
            time.sleep(0.05)

        except Exception as exc:
            errors.append(ticker)
            logger.warning(f"[{ticker}] hydration failed: {exc}")

    logger.info(
        f"Data hydration complete — {len(results)}/{len(tickers)} tickers ready "
        f"({len(errors)} errors)."
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4 — WEBSOCKET SUBSCRIPTION
# ═══════════════════════════════════════════════════════════════════════════

def build_candle_handler(
    config: dict,
    data_dir: Path,
    signal_manager: ShockReversionSignalManager,
    logger: logging.Logger,
) -> Callable[[str, pd.DataFrame], None]:
    """
    Return a closed-candle callback that:
      1. Merges the new 1m candle into local Parquet storage.
      2. Runs feature engineering + signal detection.
      3. Logs any entry/exit signals (execution layer will consume these).
    """
    last_log_time = [0.0]

    def on_closed_candle(ticker: str, candle: pd.DataFrame) -> None:
        now = time.time()
        if now - last_log_time[0] > 60:          # heartbeat every 60s
            logger.info("Receiving closed candles from websocket.")
            last_log_time[0] = now

        # — Step 1: Append to local parquet —
        path = data_dir / f"{ticker}.parquet"
        try:
            existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        except Exception:
            existing = pd.DataFrame()

        merged = pd.concat([existing, candle], ignore_index=True)
        merged.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
        merged.sort_values("open_time", inplace=True)

        # Trim to lookback window + small buffer
        lookback_bars = config["parameters"]["lookback_bars"]
        if len(merged) > lookback_bars + 60:
            merged = merged.iloc[-(lookback_bars + 60) :]

        merged.reset_index(drop=True, inplace=True)
        merged.to_parquet(path, index=False)

        # — Step 2 & 3: Feature engineering + signal detection —
        actions = signal_manager.update(ticker, merged)
        for signal in actions:
            if signal.action == SignalAction.ENTER_SHORT:
                logger.info(
                    f"[{ticker}] *** ENTRY SIGNAL *** "
                    f"time={signal.signal_time}, price={signal.entry_price}, "
                    f"ret_z={signal.ret_z:.2f}, vol_mean={signal.vol_mean:,.0f}"
                )
                # TODO: Stage 6 — pass to execution layer
            elif signal.action == SignalAction.EXIT_SHORT:
                logger.info(
                    f"[{ticker}] *** EXIT SIGNAL *** "
                    f"time={signal.signal_time}, reason={signal.reason}"
                )
                # TODO: Stage 6 — pass to execution layer

    return on_closed_candle


async def heartbeat(logger: logging.Logger, interval_sec: int = 600):
    """Periodic heartbeat so we know the pipeline is alive."""
    while True:
        logger.info("Heartbeat: shock reversion pipeline is running.")
        await asyncio.sleep(interval_sec)


def run_websocket(
    config: dict,
    dataframes: dict[str, pd.DataFrame],
    logger: logging.Logger,
) -> None:
    """Stage 4: subscribe to live 1m kline streams for all hydrated tickers."""
    logger.info("Stage 4 — WEBSOCKET SUBSCRIPTION")

    tickers = list(dataframes.keys())
    interval = config["parameters"]["candle_interval"]
    data_dir = ROOT / config["data_paths"]["data_path"]

    logger.info(
        f"{len(tickers)} tickers with data — subscribing to 1m WebSocket streams …"
    )

    positions_path = ROOT / config["data_paths"]["signals_path"] / "positions.json"
    signal_manager = ShockReversionSignalManager(
        config=config,
        logger=logger,
        positions_path=positions_path,
    )

    handler = build_candle_handler(config, data_dir, signal_manager, logger)

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


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    config, logger = init()

    tickers = fetch_tickers(config, logger)

    dataframes = hydrate_data(config, tickers, logger)

    run_websocket(config, dataframes, logger)
