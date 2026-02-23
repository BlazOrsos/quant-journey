"""
Binance USD-M Futures WebSocket kline stream client.

Subscribes to closed-candle events for a list of tickers via the Binance
Futures combined stream endpoint.  When a kline closes the registered
*on_closed_candle* callback is invoked with the candle as a one-row
DataFrame (same column schema as ``OHLCVStorage``).

Streams are batched into groups of at most ``MAX_STREAMS_PER_CONNECTION``
(Binance hard-limits 200 per connection).  Each batch runs in its own
``asyncio`` task; all tasks share the same event loop.

Usage::

    import asyncio
    from exchanges.binance_websocket import BinanceFuturesKlineStream

    def handle_candle(ticker: str, candle: pd.DataFrame) -> None:
        print(ticker, candle)

    stream = BinanceFuturesKlineStream(
        tickers=["BTCUSDT", "ETHUSDT"],
        interval="4h",
        on_closed_candle=handle_candle,
    )
    asyncio.run(stream.run_forever())
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Optional

import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WS_BASE = "wss://fstream.binance.com/stream?streams="

# Binance hard-limits 200 streams per WebSocket connection.
MAX_STREAMS_PER_CONNECTION = 200

# Seconds to wait before attempting a reconnect.
_RECONNECT_DELAY_SECS = 5

# Type alias for the closed-candle callback.
ClosedCandleCallback = Callable[[str, pd.DataFrame], None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kline_to_row(k: dict) -> pd.DataFrame:
    """Convert a Binance ``k`` kline payload dict to a one-row DataFrame.

    The returned schema matches the ``STORED_COLUMNS`` used by
    ``OHLCVStorage``.
    """
    return pd.DataFrame([{
        "open_time":              pd.to_datetime(k["t"], unit="ms", utc=True),
        "open":                   float(k["o"]),
        "high":                   float(k["h"]),
        "low":                    float(k["l"]),
        "close":                  float(k["c"]),
        "volume":                 float(k["v"]),
        "close_time":             pd.to_datetime(k["T"], unit="ms", utc=True),
        "quote_volume":           float(k["q"]),
        "count":                  int(k["n"]),
        "taker_buy_volume":       float(k["V"]),
        "taker_buy_quote_volume": float(k["Q"]),
    }])


# ---------------------------------------------------------------------------
# Stream client
# ---------------------------------------------------------------------------

class BinanceFuturesKlineStream:
    """
    Subscribe to closed-candle kline events for a list of USDT-perp tickers.

    Parameters
    ----------
    tickers : list[str]
        Symbols to subscribe to, e.g. ``["BTCUSDT", "ETHUSDT"]``.
    interval : str
        Kline interval string, e.g. ``"4h"``.
    on_closed_candle : ClosedCandleCallback
        Signature: ``(ticker: str, candle: pd.DataFrame) -> None``.
        Called on the event-loop thread — must be non-blocking.  If you
        need to run CPU-intensive work, wrap it with
        ``loop.run_in_executor``.
    logger : logging.Logger, optional
        Falls back to a module-level logger.
    """

    def __init__(
        self,
        tickers: list[str],
        interval: str,
        on_closed_candle: ClosedCandleCallback,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.tickers = [t.upper() for t in tickers]
        self.interval = interval
        self.on_closed_candle = on_closed_candle
        self.logger = logger or logging.getLogger(__name__)
        self._stop_event: asyncio.Event | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_forever(self) -> None:
        """Start all stream batches and run until :meth:`stop` is called."""
        self._stop_event = asyncio.Event()
        batches = self._build_batches()
        self.logger.info(
            f"Starting {len(batches)} WebSocket connection(s) for "
            f"{len(self.tickers)} ticker(s) @ {self.interval}."
        )
        tasks = [
            asyncio.create_task(self._run_batch(batch, idx))
            for idx, batch in enumerate(batches)
        ]
        await asyncio.gather(*tasks)

    def stop(self) -> None:
        """Signal all connections to close gracefully."""
        if self._stop_event:
            self._stop_event.set()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_batches(self) -> list[list[str]]:
        """Split tickers into groups bounded by MAX_STREAMS_PER_CONNECTION."""
        n = MAX_STREAMS_PER_CONNECTION
        return [self.tickers[i : i + n] for i in range(0, len(self.tickers), n)]

    def _build_url(self, batch: list[str]) -> str:
        streams = "/".join(f"{t.lower()}@kline_{self.interval}" for t in batch)
        return _WS_BASE + streams

    async def _run_batch(self, batch: list[str], batch_idx: int) -> None:
        """Connect and listen for one batch, reconnecting on error."""
        url = self._build_url(batch)
        while not (self._stop_event and self._stop_event.is_set()):
            try:
                self.logger.info(
                    f"[Batch {batch_idx}] Connecting ({len(batch)} streams) …"
                )
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=10
                ) as ws:
                    self.logger.info(f"[Batch {batch_idx}] Connected.")
                    await self._listen(ws, batch_idx)

            except (ConnectionClosedError, ConnectionClosedOK) as exc:
                if self._stop_event and self._stop_event.is_set():
                    break
                self.logger.warning(
                    f"[Batch {batch_idx}] Connection closed ({exc}). "
                    f"Reconnecting in {_RECONNECT_DELAY_SECS}s …"
                )
                await asyncio.sleep(_RECONNECT_DELAY_SECS)

            except Exception as exc:
                if self._stop_event and self._stop_event.is_set():
                    break
                self.logger.error(
                    f"[Batch {batch_idx}] Unexpected error: {exc}. "
                    f"Reconnecting in {_RECONNECT_DELAY_SECS}s …"
                )
                await asyncio.sleep(_RECONNECT_DELAY_SECS)

        self.logger.info(f"[Batch {batch_idx}] Stream stopped.")

    async def _listen(self, ws, batch_idx: int) -> None:
        """Receive messages and dispatch closed-candle events."""
        async for raw in ws:
            if self._stop_event and self._stop_event.is_set():
                break
            try:
                msg = json.loads(raw)
                data = msg.get("data", {})

                # Filter: only kline events for closed candles.
                if data.get("e") != "kline":
                    continue
                k = data["k"]
                if not k.get("x"):
                    continue

                ticker = data["s"]
                candle = _kline_to_row(k)
                self.on_closed_candle(ticker, candle)

            except Exception as exc:
                self.logger.error(
                    f"[Batch {batch_idx}] Message handling error: {exc}",
                    exc_info=True,
                )
