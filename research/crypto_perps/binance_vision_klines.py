"""
Download Binance USDT perpetual futures OHLCV data — optimised three-phase cascade.

Why is this faster than the original binance_perp_data.py?
-----------------------------------------------------------
The original downloads one *daily* zip per calendar day per ticker, which can
mean 400–500 HTTP requests per ticker when START_DATE is a year ago.

This version uses a three-phase waterfall:

    Phase 1 — Monthly zips (Binance Vision)
        Covers the bulk of history with one file per month instead of ~30
        daily files.  A single monthly zip replaces ≈30 daily downloads.

    Phase 2 — Daily zips (Binance Vision)
        Fills the gap from the end of the last complete month up to ~2 days
        ago, when Vision publishes files with a lag.

    Phase 3 — REST klines (Binance Futures API)
        Tops up the last 1–2 days that Vision has not yet published.

Result: for a 1-year START_DATE and a ~500-ticker universe the total number
of HTTP requests drops from ~200 000 (all-daily) to ~8 000 (monthly + few
daily + 1 REST per ticker).

Usage
-----
    python research/crypto_perps/binance_perp_data_new.py

Configuration is at the top of this file.  Already-downloaded tickers are
skipped automatically (delete the CSV to force a re-download).
"""

from __future__ import annotations

import io
import math
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

TIMEFRAME  = "1m"           # 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w
START_DATE = "2026-03-01"   # Inclusive lower bound (YYYY-MM-DD)

# Output directory — mirrors the convention from the original script
_BASE = Path(__file__).resolve().parent
OUTPUT_DIR = _BASE / "binance_vision_klines" if TIMEFRAME == "1d" else \
             _BASE / f"binance_vision_klines_{TIMEFRAME}"

MAX_WORKERS   = 8   # Parallel download threads per phase
REQ_TIMEOUT   = 30  # Seconds per HTTP request
RETRY_ATTEMPTS = 3
RETRY_DELAY   = 2   # Seconds between retry attempts

# How many days Vision daily files lag behind the current date
_VISION_DAILY_LAG_DAYS = 2

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

S3_BASE       = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DOWNLOAD_BASE = "https://data.binance.vision"
_DAILY_PREFIX   = "data/futures/um/daily/klines/"
_MONTHLY_PREFIX = "data/futures/um/monthly/klines/"
_REST_BASE      = "https://fapi.binance.com"
_REST_ENDPOINT  = "/fapi/v1/klines"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
NUMERIC_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
]

# Maps interval string → hours (used to estimate REST kline limit)
_INTERVAL_HOURS: dict[str, float] = {
    "1m": 1/60, "3m": 3/60, "5m": 5/60, "15m": 0.25, "30m": 0.5,
    "1h": 1, "2h": 2, "4h": 4, "6h": 6, "8h": 8, "12h": 12,
    "1d": 24, "3d": 72, "1w": 168,
}

# ═══════════════════════════════════════════════════════════════════════════
# HTTP SESSION
# ═══════════════════════════════════════════════════════════════════════════

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(
    pool_connections=MAX_WORKERS,
    pool_maxsize=MAX_WORKERS * 2,
)
_session.mount("https://", _adapter)


def _get(url: str, **kwargs) -> requests.Response:
    """GET with exponential-backoff retry."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = _session.get(url, timeout=REQ_TIMEOUT, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.HTTPError) as exc:
            if attempt == RETRY_ATTEMPTS:
                raise
            print(f"    [retry {attempt}] {url}: {exc}")
            time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError("unreachable")  # pragma: no cover


# ═══════════════════════════════════════════════════════════════════════════
# S3 LISTING
# ═══════════════════════════════════════════════════════════════════════════

def _s3_common_prefixes(prefix: str) -> list[str]:
    """List immediate 'subdirectory' prefixes under *prefix* via S3 XML API."""
    prefixes: list[str] = []
    marker = ""
    while True:
        url = f"{S3_BASE}?prefix={prefix}&delimiter=/&marker={marker}"
        root = ET.fromstring(_get(url).content)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for cp in root.findall("s3:CommonPrefixes/s3:Prefix", ns):
            prefixes.append(cp.text)
        if root.findtext("s3:IsTruncated", namespaces=ns) == "true":
            marker = prefixes[-1]
        else:
            break
    return prefixes


def _s3_keys(prefix: str) -> list[str]:
    """List all object keys under *prefix* (flat)."""
    keys: list[str] = []
    marker = ""
    while True:
        url = f"{S3_BASE}?prefix={prefix}&marker={marker}"
        root = ET.fromstring(_get(url).content)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for el in root.findall("s3:Contents/s3:Key", ns):
            keys.append(el.text)
        if root.findtext("s3:IsTruncated", namespaces=ns) == "true":
            marker = keys[-1]
        else:
            break
    return keys


# ═══════════════════════════════════════════════════════════════════════════
# TICKER UNIVERSE
# ═══════════════════════════════════════════════════════════════════════════

def get_usdt_tickers() -> list[str]:
    """Return all USDT-margined perpetual futures tickers from Binance Vision."""
    all_prefixes = _s3_common_prefixes(_DAILY_PREFIX)
    tickers = [
        p.rstrip("/").split("/")[-1]
        for p in all_prefixes
        if p.rstrip("/").split("/")[-1].endswith("USDT")
    ]
    return sorted(tickers)


# ═══════════════════════════════════════════════════════════════════════════
# ZIP DOWNLOAD HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _download_zip(key: str) -> Optional[pd.DataFrame]:
    """Download one zip from Binance Vision and return its CSV as a DataFrame."""
    url = f"{DOWNLOAD_BASE}/{key}"
    resp = _get(url)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            return None
        raw = zf.read(csv_names[0])
        df = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLUMNS)
        # Some files have a literal header row as the first data row — drop it
        if str(df["open_time"].iloc[0]) == "open_time":
            df = df.iloc[1:].reset_index(drop=True)
        return df


def _download_many(keys: list[str]) -> pd.DataFrame:
    """Download *keys* in parallel and return a merged DataFrame."""
    if not keys:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_key = {pool.submit(_download_zip, k): k for k in keys}
        for future in as_completed(future_to_key):
            try:
                df = future.result()
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as exc:
                print(f"    ✗ {future_to_key[future]}: {exc}")

    if not frames:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    merged = pd.concat(frames, ignore_index=True)
    merged["open_time"]  = pd.to_datetime(pd.to_numeric(merged["open_time"]),  unit="ms", utc=True)
    merged["close_time"] = pd.to_datetime(pd.to_numeric(merged["close_time"]), unit="ms", utc=True)
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# KEY FILTERING
# ═══════════════════════════════════════════════════════════════════════════

def _dt_from_daily_key(key: str) -> Optional[datetime]:
    m = re.search(r"(\d{4}-\d{2}-\d{2})\.zip$", key)
    return datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc) if m else None


def _dt_from_monthly_key(key: str) -> Optional[datetime]:
    m = re.search(r"(\d{4}-\d{2})\.zip$", key)
    return datetime.strptime(m.group(1), "%Y-%m").replace(
        day=1, tzinfo=timezone.utc
    ) if m else None


def _monthly_keys(ticker: str, start: datetime, end: datetime) -> list[str]:
    prefix = f"{_MONTHLY_PREFIX}{ticker}/{TIMEFRAME}/"
    keys = _s3_keys(prefix)
    return [
        k for k in keys
        if k.endswith(".zip")
        and (dt := _dt_from_monthly_key(k)) is not None
        and start <= dt <= end
    ]


def _daily_keys(ticker: str, start: datetime, end: datetime) -> list[str]:
    prefix = f"{_DAILY_PREFIX}{ticker}/{TIMEFRAME}/"
    keys = _s3_keys(prefix)
    return [
        k for k in keys
        if k.endswith(".zip")
        and (dt := _dt_from_daily_key(k)) is not None
        and start <= dt <= end
    ]


# ═══════════════════════════════════════════════════════════════════════════
# REST KLINES (phase 3 top-up)
# ═══════════════════════════════════════════════════════════════════════════

def _rest_klines(ticker: str, limit: int) -> pd.DataFrame:
    """Fetch the most recent *limit* candles via the Binance Futures REST API."""
    params = {"symbol": ticker, "interval": TIMEFRAME, "limit": min(limit, 1500)}
    resp = _get(_REST_BASE + _REST_ENDPOINT, params=params)
    raw = resp.json()
    df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df[NUMERIC_COLS] = df[NUMERIC_COLS].astype(float)
    df.drop(columns=["ignore"], inplace=True, errors="ignore")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# MERGE HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _merge(
    existing: Optional[pd.DataFrame],
    new: pd.DataFrame,
) -> pd.DataFrame:
    if existing is None or existing.empty:
        combined = new
    elif new is None or new.empty:
        combined = existing
    else:
        combined = pd.concat([existing, new], ignore_index=True)
    combined.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
    combined.sort_values("open_time", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


# ═══════════════════════════════════════════════════════════════════════════
# THREE-PHASE SYNC FOR ONE TICKER
# ═══════════════════════════════════════════════════════════════════════════

def sync_ticker(ticker: str) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for *ticker* using the three-phase cascade.

    Phase 1 — monthly zips  →  bulk of the history
    Phase 2 — daily zips    →  gap from end-of-last-month until ~2 days ago
    Phase 3 — REST klines   →  last 1-2 days not yet on Vision
    """
    now     = datetime.now(tz=timezone.utc)
    start   = datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    horizon = now  # inclusive upper bound for the data we want

    # The first day of the current month — last complete month ends the day before
    current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    two_days_ago = now - pd.Timedelta(days=_VISION_DAILY_LAG_DAYS).to_pytimedelta()

    df: Optional[pd.DataFrame] = None

    # ── Phase 1: monthly zips ──────────────────────────────────────────────
    # Only run if the start date is before the current month
    prev_month_end = current_month_start - pd.Timedelta(days=1).to_pytimedelta()
    if start < current_month_start.replace(tzinfo=timezone.utc):
        month_end = prev_month_end.replace(tzinfo=timezone.utc)
        keys = _monthly_keys(ticker, start, month_end)
        if keys:
            print(f"    Phase 1 (monthly): {len(keys)} zip(s)")
            new_df = _download_many(keys)
            df = _merge(df, new_df)
        else:
            print(f"    Phase 1 (monthly): no files found for this range")

    # Determine what our data covers after phase 1
    latest_ts = df["open_time"].max() if df is not None and not df.empty else None

    # ── Phase 2: daily zips ────────────────────────────────────────────────
    # Fill from end of monthly coverage up to ~2 days ago
    daily_start = (
        latest_ts.to_pydatetime() if isinstance(latest_ts, pd.Timestamp) else
        latest_ts if latest_ts is not None else start
    )
    daily_end = two_days_ago
    if daily_start < daily_end:
        # Normalise to calendar day boundaries
        daily_start_day = daily_start.replace(hour=0, minute=0, second=0, microsecond=0)
        keys = _daily_keys(ticker, daily_start_day, daily_end)
        if keys:
            print(f"    Phase 2 (daily):   {len(keys)} zip(s)")
            new_df = _download_many(keys)
            df = _merge(df, new_df)
        else:
            print(f"    Phase 2 (daily):   no files found for this range")
    else:
        print(f"    Phase 2 (daily):   skipped — monthly already covers up to {latest_ts}")

    # Re-check after phase 2
    latest_ts = df["open_time"].max() if df is not None and not df.empty else None

    # ── Phase 3: REST klines top-up ────────────────────────────────────────
    interval_hours = _INTERVAL_HOURS.get(TIMEFRAME, 4)
    if latest_ts is None:
        # Nothing from Vision at all — pull a full REST lookback
        total_hours = (now - start).total_seconds() / 3600
        limit = min(1500, int(total_hours / interval_hours) + 1)
    else:
        gap_hours = (now - (latest_ts.to_pydatetime() if isinstance(latest_ts, pd.Timestamp) else latest_ts)).total_seconds() / 3600
        if gap_hours <= interval_hours:
            print(f"    Phase 3 (REST):    skipped — already up to date")
            return _finalise(df, start, now)
        limit = math.ceil(gap_hours / interval_hours) + 2

    limit = max(1, min(limit, 1500))
    print(f"    Phase 3 (REST):    limit={limit}")
    rest_df = _rest_klines(ticker, limit)
    df = _merge(df, rest_df)

    return _finalise(df, start, now)


def _finalise(
    df: Optional[pd.DataFrame],
    start: datetime,
    end: datetime,
) -> Optional[pd.DataFrame]:
    """Trim to [start, end], cast numeric cols, drop ignore column."""
    if df is None or df.empty:
        return df

    # Drop Binance's padding column if present
    if "ignore" in df.columns:
        df.drop(columns=["ignore"], inplace=True)

    # Ensure numeric columns are float
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Trim to configured window
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    df = df[(df["open_time"] >= start_ts) & (df["open_time"] <= end_ts)].copy()
    df.reset_index(drop=True, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Binance USDT Perp OHLCV — three-phase download")
    print(f"  Timeframe  : {TIMEFRAME}")
    print(f"  Start date : {START_DATE}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print("=" * 60)

    tickers = get_usdt_tickers()
    print(f"\n{len(tickers)} USDT perp tickers found on Binance Vision.\n")

    existing_files = {f.stem for f in OUTPUT_DIR.glob("*.csv")}
    remaining = [t for t in tickers if t not in existing_files]
    skipped = len(tickers) - len(remaining)
    if skipped:
        print(f"Skipping {skipped} already-downloaded tickers, {len(remaining)} remaining.\n")

    successes = 0
    failures  = 0

    for i, ticker in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] {ticker}")
        try:
            df = sync_ticker(ticker)
            if df is not None and not df.empty:
                out_path = OUTPUT_DIR / f"{ticker}.csv"
                df.to_csv(out_path, index=False)
                print(f"    ✓ saved {len(df)} rows → {out_path.name}")
                successes += 1
            else:
                print(f"    ✗ no data obtained")
                failures += 1
        except Exception as exc:
            print(f"    ✗ error: {exc}")
            failures += 1

    print(f"\nDone. {successes} saved, {failures} failed, {skipped} previously downloaded.")


if __name__ == "__main__":
    main()
