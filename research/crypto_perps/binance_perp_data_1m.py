"""
Download Binance USDT perpetual futures 1m OHLCV data from data.binance.vision.

Flow:
  1. List all tickers via the S3 XML API
  2. Filter for USDT-margined tickers
  3. For each ticker, list all .zip files in the 1m/ folder
  4. Filter to only the last 4 months of data
  5. Download, extract, and merge CSVs into one file per ticker
  6. Save to OUTPUT_DIR as {TICKER}.csv
"""

import io
import os
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
S3_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DOWNLOAD_BASE = "https://data.binance.vision"
PREFIX = "data/futures/um/daily/klines/"

OUTPUT_DIR = Path(__file__).resolve().parent / "ohlcv_binance_perp_1m"
LOOKBACK_MONTHS = 4      # only download the last N months of data

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

MAX_WORKERS = 8          # parallel downloads per ticker
REQUEST_TIMEOUT = 30     # seconds per HTTP request
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2          # seconds between retries


# ---------------------------------------------------------------------------
# S3 XML helpers
# ---------------------------------------------------------------------------
def s3_list_common_prefixes(prefix: str, delimiter: str = "/") -> list[str]:
    """List 'subdirectories' under *prefix* using the S3 list API."""
    prefixes: list[str] = []
    marker = ""
    while True:
        url = f"{S3_BASE}?prefix={prefix}&delimiter={delimiter}&marker={marker}"
        resp = _get_with_retry(url)
        root = ET.fromstring(resp.content)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for cp in root.findall("s3:CommonPrefixes/s3:Prefix", ns):
            prefixes.append(cp.text)
        is_truncated = root.findtext("s3:IsTruncated", namespaces=ns)
        if is_truncated == "true":
            # use last prefix as marker for next page
            marker = prefixes[-1]
        else:
            break
    return prefixes


def s3_list_keys(prefix: str) -> list[str]:
    """List all object keys under *prefix* (no delimiter → flat listing)."""
    keys: list[str] = []
    marker = ""
    while True:
        url = f"{S3_BASE}?prefix={prefix}&marker={marker}"
        resp = _get_with_retry(url)
        root = ET.fromstring(resp.content)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for key_el in root.findall("s3:Contents/s3:Key", ns):
            keys.append(key_el.text)
        is_truncated = root.findtext("s3:IsTruncated", namespaces=ns)
        if is_truncated == "true":
            marker = keys[-1]
        else:
            break
    return keys


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=MAX_WORKERS,
    pool_maxsize=MAX_WORKERS * 2,
)
session.mount("https://", adapter)


def _get_with_retry(url: str, **kwargs) -> requests.Response:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.HTTPError) as exc:
            if attempt == RETRY_ATTEMPTS:
                raise
            print(f"  ⚠ attempt {attempt} failed for {url}: {exc}. Retrying...")
            time.sleep(RETRY_DELAY * attempt)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def get_usdt_tickers() -> list[str]:
    """Return sorted list of USDT-margined ticker names."""
    all_prefixes = s3_list_common_prefixes(PREFIX)
    tickers = []
    for p in all_prefixes:
        # e.g. "data/futures/um/daily/klines/BTCUSDT/"
        name = p.rstrip("/").split("/")[-1]
        if name.endswith("USDT"):
            tickers.append(name)
    print(f"Found {len(tickers)} USDT tickers (out of {len(all_prefixes)} total)")
    return sorted(tickers)


def _cutoff_date() -> datetime:
    """Return the cutoff date = today minus LOOKBACK_MONTHS months."""
    today = datetime.now(tz=timezone.utc)
    # Approximate months by 30 days each
    return today - timedelta(days=LOOKBACK_MONTHS * 30)


def _date_from_zip_key(key: str) -> datetime | None:
    """Extract the date from a zip filename like …/BTCUSDT-1m-2025-11-03.zip"""
    m = re.search(r"(\d{4}-\d{2}-\d{2})\.zip$", key)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return None


def list_zip_keys(ticker: str) -> list[str]:
    """List recent .zip file keys for the 1m kline folder of *ticker*."""
    prefix = f"{PREFIX}{ticker}/1m/"
    keys = s3_list_keys(prefix)
    cutoff = _cutoff_date()
    recent = []
    for k in keys:
        if not k.endswith(".zip"):
            continue
        dt = _date_from_zip_key(k)
        if dt is not None and dt >= cutoff:
            recent.append(k)
    return recent


def download_and_extract_zip(key: str) -> pd.DataFrame | None:
    """Download a single zip, extract the CSV inside, return a DataFrame."""
    url = f"{DOWNLOAD_BASE}/{key}"
    resp = _get_with_retry(url)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            return None
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f)
            # Normalise column names in case header is missing
            if df.columns[0] != "open_time":
                df = pd.read_csv(io.BytesIO(zf.read(csv_names[0])),
                                 header=None, names=KLINE_COLUMNS)
            return df


def process_ticker(ticker: str) -> bool:
    """Download recent 1m zips for *ticker*, merge, and save. Returns True on success."""
    out_path = OUTPUT_DIR / f"{ticker}.csv"
    if out_path.exists():
        print(f"  ✓ {ticker} — already exists, skipping")
        return True

    zip_keys = list_zip_keys(ticker)
    if not zip_keys:
        print(f"  ✗ {ticker} — no 1m zip files found in last {LOOKBACK_MONTHS} months")
        return False

    print(f"  ⏳ {ticker} — downloading {len(zip_keys)} files...")

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_key = {pool.submit(download_and_extract_zip, k): k for k in zip_keys}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as exc:
                print(f"    ✗ failed {key}: {exc}")

    if not frames:
        print(f"  ✗ {ticker} — all downloads failed")
        return False

    merged = pd.concat(frames, ignore_index=True)

    # Convert open_time from ms epoch → datetime, sort, deduplicate
    merged["open_time"] = pd.to_datetime(merged["open_time"], unit="ms")
    merged["close_time"] = pd.to_datetime(merged["close_time"], unit="ms")
    merged.sort_values("open_time", inplace=True)
    merged.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    merged.to_csv(out_path, index=False)
    print(f"  ✓ {ticker} — saved {len(merged)} rows → {out_path.name}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    tickers = get_usdt_tickers()

    # Skip tickers that already have a saved CSV
    existing = {f.stem for f in OUTPUT_DIR.glob("*.csv")}
    remaining = [t for t in tickers if t not in existing]
    skipped = len(tickers) - len(remaining)
    if skipped:
        print(f"Skipping {skipped} already-downloaded tickers, {len(remaining)} remaining.\n")
    else:
        print()

    successes = 0
    failures = 0
    for i, ticker in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] {ticker}")
        try:
            ok = process_ticker(ticker)
            if ok:
                successes += 1
            else:
                failures += 1
        except Exception as exc:
            print(f"  ✗ {ticker} — error: {exc}")
            failures += 1

    print(f"\nDone. {successes} succeeded, {failures} failed, {skipped} previously downloaded.")


if __name__ == "__main__":
    main()
