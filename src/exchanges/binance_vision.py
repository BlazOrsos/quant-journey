"""
Client for https://data.binance.vision/ public market data.

Provides three core methods for the pipeline to call:
  - get_symbols()              → list of all available USDT-margined perp tickers
  - fetch_klines_monthly()     → DataFrame of klines from monthly zip files
  - fetch_klines_daily()       → DataFrame of klines from daily zip files

No storage logic lives here — the caller (pipeline) decides what to do with
the returned DataFrames.
"""

import io
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
S3_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DOWNLOAD_BASE = "https://data.binance.vision"
FUTURES_KLINES_PREFIX = "data/futures/um/{sampling}/klines/"  # {sampling} = daily | monthly

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

_DEFAULT_MAX_WORKERS = 8
_REQUEST_TIMEOUT = 30
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 2


# ---------------------------------------------------------------------------
# BinanceVisionClient
# ---------------------------------------------------------------------------
class BinanceVisionClient:
    """
    Thin wrapper around the Binance Vision S3 public data bucket.

    Parameters
    ----------
    max_workers : int
        Number of parallel download threads used when fetching zip files.
    """

    def __init__(self, max_workers: int = _DEFAULT_MAX_WORKERS) -> None:
        self.max_workers = max_workers
        self._session = self._build_session(max_workers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_symbols(self) -> list[str]:
        """
        Return a sorted list of all USDT-margined perpetual futures tickers
        available on Binance Vision.

        Returns
        -------
        list[str]
            e.g. ['1000BONKUSDT', 'ADAUSDT', 'BTCUSDT', ...]
        """
        prefix = FUTURES_KLINES_PREFIX.format(sampling="daily")
        all_prefixes = self._s3_list_common_prefixes(prefix)
        tickers = []
        for p in all_prefixes:
            name = p.rstrip("/").split("/")[-1]
            if name.endswith("USDT"):
                tickers.append(name)
        return sorted(tickers)

    def fetch_klines_monthly(
        self,
        ticker: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Download and merge monthly kline zip files for *ticker*.

        Parameters
        ----------
        ticker : str
            e.g. 'BTCUSDT'
        timeframe : str
            e.g. '4h', '1d', '15m'
        start_date : str
            Inclusive lower bound, 'YYYY-MM' or 'YYYY-MM-DD'.
        end_date : str, optional
            Inclusive upper bound, 'YYYY-MM' or 'YYYY-MM-DD'.
            Defaults to the current month.

        Returns
        -------
        pd.DataFrame
            Columns: open_time, open, high, low, close, volume, close_time,
                     quote_volume, count, taker_buy_volume,
                     taker_buy_quote_volume, ignore.
            open_time / close_time are UTC-aware datetimes.
            Empty DataFrame if no data is available.
        """
        cutoff_start = self._parse_month_cutoff(start_date)
        cutoff_end = self._parse_month_cutoff(end_date) if end_date else None

        prefix = f"{FUTURES_KLINES_PREFIX.format(sampling='monthly')}{ticker}/{timeframe}/"
        keys = self._s3_list_keys(prefix)
        zip_keys = self._filter_monthly_keys(keys, cutoff_start, cutoff_end)

        return self._download_and_merge(zip_keys)

    def fetch_klines_daily(
        self,
        ticker: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Download and merge daily kline zip files for *ticker*.

        Parameters
        ----------
        ticker : str
            e.g. 'BTCUSDT'
        timeframe : str
            e.g. '4h', '1d', '15m'
        start_date : str
            Inclusive lower bound, 'YYYY-MM-DD'.
        end_date : str, optional
            Inclusive upper bound, 'YYYY-MM-DD'.
            Defaults to today (UTC).

        Returns
        -------
        pd.DataFrame
            Same schema as fetch_klines_monthly.
            Empty DataFrame if no data is available.
        """
        cutoff_start = self._parse_day_cutoff(start_date)
        cutoff_end = self._parse_day_cutoff(end_date) if end_date else None

        prefix = f"{FUTURES_KLINES_PREFIX.format(sampling='daily')}{ticker}/{timeframe}/"
        keys = self._s3_list_keys(prefix)
        zip_keys = self._filter_daily_keys(keys, cutoff_start, cutoff_end)

        return self._download_and_merge(zip_keys)

    # ------------------------------------------------------------------
    # S3 listing helpers
    # ------------------------------------------------------------------

    def _s3_list_common_prefixes(self, prefix: str, delimiter: str = "/") -> list[str]:
        """List immediate 'subdirectories' under *prefix* via S3 list API."""
        prefixes: list[str] = []
        marker = ""
        while True:
            url = f"{S3_BASE}?prefix={prefix}&delimiter={delimiter}&marker={marker}"
            resp = self._get_with_retry(url)
            root = ET.fromstring(resp.content)
            ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            for cp in root.findall("s3:CommonPrefixes/s3:Prefix", ns):
                prefixes.append(cp.text)
            if root.findtext("s3:IsTruncated", namespaces=ns) == "true":
                marker = prefixes[-1]
            else:
                break
        return prefixes

    def _s3_list_keys(self, prefix: str) -> list[str]:
        """List all object keys under *prefix* (flat listing)."""
        keys: list[str] = []
        marker = ""
        while True:
            url = f"{S3_BASE}?prefix={prefix}&marker={marker}"
            resp = self._get_with_retry(url)
            root = ET.fromstring(resp.content)
            ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            for key_el in root.findall("s3:Contents/s3:Key", ns):
                keys.append(key_el.text)
            if root.findtext("s3:IsTruncated", namespaces=ns) == "true":
                marker = keys[-1]
            else:
                break
        return keys

    # ------------------------------------------------------------------
    # Key filtering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_day_cutoff(date_str: str) -> datetime:
        """Parse 'YYYY-MM-DD' into a UTC-aware datetime."""
        return datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)

    @staticmethod
    def _parse_month_cutoff(date_str: str) -> datetime:
        """Parse 'YYYY-MM' or 'YYYY-MM-DD' into the first day of that month (UTC)."""
        return datetime.strptime(date_str[:7], "%Y-%m").replace(
            day=1, tzinfo=timezone.utc
        )

    @staticmethod
    def _date_from_daily_key(key: str) -> Optional[datetime]:
        """Extract date from a key like …/BTCUSDT-4h-2025-11-03.zip"""
        m = re.search(r"(\d{4}-\d{2}-\d{2})\.zip$", key)
        if m:
            return datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return None

    @staticmethod
    def _date_from_monthly_key(key: str) -> Optional[datetime]:
        """Extract month from a key like …/BTCUSDT-4h-2025-11.zip"""
        m = re.search(r"(\d{4}-\d{2})\.zip$", key)
        if m:
            return datetime.strptime(m.group(1), "%Y-%m").replace(
                day=1, tzinfo=timezone.utc
            )
        return None

    def _filter_daily_keys(
        self,
        keys: list[str],
        start: datetime,
        end: Optional[datetime],
    ) -> list[str]:
        result = []
        for k in keys:
            if not k.endswith(".zip"):
                continue
            dt = self._date_from_daily_key(k)
            if dt is None:
                continue
            if dt < start:
                continue
            if end is not None and dt > end:
                continue
            result.append(k)
        return result

    def _filter_monthly_keys(
        self,
        keys: list[str],
        start: datetime,
        end: Optional[datetime],
    ) -> list[str]:
        result = []
        for k in keys:
            if not k.endswith(".zip"):
                continue
            dt = self._date_from_monthly_key(k)
            if dt is None:
                continue
            if dt < start:
                continue
            if end is not None and dt > end:
                continue
            result.append(k)
        return result

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _download_zip(self, key: str) -> Optional[pd.DataFrame]:
        """Download a single zip and return the CSV inside as a DataFrame."""
        url = f"{DOWNLOAD_BASE}/{key}"
        resp = self._get_with_retry(url)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return None
            raw = zf.read(csv_names[0])
            df = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLUMNS)
            # Some files include a header row as the first data row — drop it
            if df["open_time"].iloc[0] == "open_time":
                df = df.iloc[1:].reset_index(drop=True)
            return df

    def _download_and_merge(self, zip_keys: list[str]) -> pd.DataFrame:
        """Download *zip_keys* in parallel and return a merged, sorted DataFrame."""
        if not zip_keys:
            return pd.DataFrame(columns=KLINE_COLUMNS)

        frames: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_key = {pool.submit(self._download_zip, k): k for k in zip_keys}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        frames.append(df)
                except Exception as exc:
                    print(f"[BinanceVision] failed to download {key}: {exc}")

        if not frames:
            return pd.DataFrame(columns=KLINE_COLUMNS)

        merged = pd.concat(frames, ignore_index=True)
        merged["open_time"] = pd.to_datetime(pd.to_numeric(merged["open_time"]), unit="ms", utc=True)
        merged["close_time"] = pd.to_datetime(pd.to_numeric(merged["close_time"]), unit="ms", utc=True)
        merged.sort_values("open_time", inplace=True)
        merged.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
        merged.reset_index(drop=True, inplace=True)
        return merged

    # ------------------------------------------------------------------
    # HTTP session
    # ------------------------------------------------------------------

    @staticmethod
    def _build_session(max_workers: int) -> requests.Session:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_workers,
            pool_maxsize=max_workers * 2,
        )
        session.mount("https://", adapter)
        return session

    def _get_with_retry(self, url: str, **kwargs) -> requests.Response:
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                resp = self._session.get(url, timeout=_REQUEST_TIMEOUT, **kwargs)
                resp.raise_for_status()
                return resp
            except (requests.RequestException, requests.HTTPError) as exc:
                if attempt == _RETRY_ATTEMPTS:
                    raise
                print(f"[BinanceVision] attempt {attempt} failed for {url}: {exc}. Retrying...")
                time.sleep(_RETRY_DELAY * attempt)
