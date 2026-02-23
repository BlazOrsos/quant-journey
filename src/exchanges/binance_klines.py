"""
Thin client for the Binance USD-M Futures REST klines endpoint.

    GET https://fapi.binance.com/fapi/v1/klines
        ?symbol=BTCUSDT&interval=4h&limit=720
"""

import requests
import pandas as pd

_BASE_URL = "https://fapi.binance.com"
_KLINES_ENDPOINT = "/fapi/v1/klines"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

_REQUEST_TIMEOUT = 10


class BinanceFuturesKlines:
    """
    Fetches OHLCV klines from the Binance USD-M Futures REST API.

    Parameters
    ----------
    base_url : str
        Override the default base URL (useful for testing).
    """

    def __init__(self, base_url: str = _BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch klines for a single symbol.

        Parameters
        ----------
        symbol : str
            Futures ticker, e.g. ``"BTCUSDT"``.
        interval : str
            Binance interval string, e.g. ``"1m"``, ``"4h"``, ``"1d"``.
        limit : int
            Number of candles to return (max 1500, default 500).

        Returns
        -------
        pd.DataFrame
            Columns: open_time, open, high, low, close, volume,
                     close_time, quote_volume, count,
                     taker_buy_volume, taker_buy_quote_volume, ignore.
            ``open_time`` and ``close_time`` are UTC-aware datetimes.
            Numeric columns are cast to float.
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        response = requests.get(
            self._base_url + _KLINES_ENDPOINT,
            params=params,
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        raw = response.json()
        df = pd.DataFrame(raw, columns=KLINE_COLUMNS)

        # Cast timestamps to UTC datetimes
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        # Cast numeric columns to float
        numeric_cols = [
            "open", "high", "low", "close", "volume",
            "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume",
        ]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df.drop(columns=["ignore"], inplace=True)

        return df
