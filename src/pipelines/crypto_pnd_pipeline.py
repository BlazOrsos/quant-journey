"""
Binance Pump Scanner

Downloads recent daily klines for all USDT perpetual futures via the
Binance Futures REST API, computes pump indicators, and prints tickers
where a pump was detected on yesterday's candle.

Why 65 days?  The features use shift(1).rolling(30), which needs 31
rows of returns (32 rows of prices) before the first valid value.
65 days gives comfortable headroom for holidays / missing candles.

Usage:
    python binance_klines.py
"""

import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FAPI_BASE = "https://fapi.binance.com"
S3_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DOWNLOAD_BASE = "https://data.binance.vision"
S3_PREFIX = "data/futures/um/daily/klines/"

# Need ~65 days so the rolling(30) windows are valid on the latest row.
LOOKBACK_DAYS = 65

MAX_WORKERS = 12
REQUEST_TIMEOUT = 15
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1

# Pump thresholds (same as notebook)
RET_Z_THRESH = 5
RVOL_THRESH = 5
COUNT_Z_THRESH = 5
BUY_RATIO_THRESH = 0
VOL_MEAN_THRESH = 5_000_000


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=MAX_WORKERS,
    pool_maxsize=MAX_WORKERS * 2,
)
session.mount("https://", adapter)


def _get_json(url: str, params: dict | None = None) -> dict | list:
    """GET with retries, return parsed JSON."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, requests.HTTPError) as exc:
            if attempt == RETRY_ATTEMPTS:
                raise
            time.sleep(RETRY_DELAY * attempt)


# ---------------------------------------------------------------------------
# S3 discovery
# ---------------------------------------------------------------------------
def _s3_list_common_prefixes(prefix: str, delimiter: str = "/") -> list[str]:
    """List 'subdirectories' under *prefix* using the S3 list API."""
    prefixes: list[str] = []
    marker = ""
    while True:
        url = f"{S3_BASE}?prefix={prefix}&delimiter={delimiter}&marker={marker}"
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for cp in root.findall("s3:CommonPrefixes/s3:Prefix", ns):
            prefixes.append(cp.text)
        is_truncated = root.findtext("s3:IsTruncated", namespaces=ns)
        if is_truncated == "true":
            marker = prefixes[-1]
        else:
            break
    return prefixes


def _has_yesterday_zip(ticker: str, yesterday: pd.Timestamp) -> bool:
    """HEAD-check whether data.binance.vision has yesterday's 1d zip."""
    date_str = yesterday.strftime("%Y-%m-%d")
    url = (
        f"{DOWNLOAD_BASE}/{S3_PREFIX}{ticker}/1d/"
        f"{ticker}-1d-{date_str}.zip"
    )
    try:
        resp = session.head(url, timeout=REQUEST_TIMEOUT)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def get_usdt_perp_symbols(yesterday: pd.Timestamp) -> list[str]:
    """
    Discover all USDT-margined tickers on data.binance.vision, then keep
    only those whose 1d zip for *yesterday* exists (HEAD check).
    """
    all_prefixes = _s3_list_common_prefixes(S3_PREFIX)
    usdt_tickers = []
    for p in all_prefixes:
        name = p.rstrip("/").split("/")[-1]
        if name.endswith("USDT"):
            usdt_tickers.append(name)
    print(f"Found {len(usdt_tickers)} USDT tickers on data.binance.vision")

    # Parallel HEAD requests to filter for tickers with yesterday's data
    print(f"Checking which tickers have data for {yesterday.date()} …")
    fresh: list[str] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_ticker = {
            pool.submit(_has_yesterday_zip, t, yesterday): t
            for t in usdt_tickers
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                if future.result():
                    fresh.append(ticker)
            except Exception:
                pass

    return sorted(fresh)


def fetch_klines(symbol: str, limit: int = LOOKBACK_DAYS) -> pd.DataFrame | None:
    """Fetch daily klines for a single symbol via the REST API."""
    try:
        data = _get_json(
            f"{FAPI_BASE}/fapi/v1/klines",
            params={"symbol": symbol, "interval": "1d", "limit": limit},
        )
    except Exception:
        return None

    if not data:
        return None

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ])

    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                 "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = df[col].astype(float)
    df["count"] = df["count"].astype(int)

    df = df.set_index("date").sort_index()
    return df


# ---------------------------------------------------------------------------
# Feature engineering & pump detection
# ---------------------------------------------------------------------------
def detect_pumps(
    dataframes: dict[str, pd.DataFrame],
    yesterday: pd.Timestamp,
) -> list[dict]:
    """
    Build wide (date × symbol) frames, compute indicators, and return
    a list of pump-signal dicts for *yesterday*.
    """
    # Build wide DataFrames for each needed field
    fields = ["close", "volume", "count", "taker_buy_volume"]
    wide: dict[str, pd.DataFrame] = {}
    for field in fields:
        wide[field] = pd.DataFrame(
            {sym: df[field] for sym, df in dataframes.items()}
        ).sort_index()

    # --- Feature engineering (mirrors the notebook exactly) ----------------
    r = np.log(wide["close"]).diff()

    vol = r.shift(1).rolling(30).std()
    ret_z = r / vol

    vol_mean = wide["volume"].shift(1).rolling(30).mean()
    rvol = wide["volume"] / vol_mean

    cnt_mean = wide["count"].shift(1).rolling(30).mean()
    cnt_std = wide["count"].shift(1).rolling(30).std()
    count_z = (wide["count"] - cnt_mean) / cnt_std

    buy_ratio = wide["taker_buy_volume"] / wide["volume"]
    buy_ratio = buy_ratio.clip(0, 1)

    # Pump definition
    is_pump = (
        (ret_z > RET_Z_THRESH)
        & (rvol > RVOL_THRESH)
        & (count_z > COUNT_Z_THRESH)
        & (buy_ratio > BUY_RATIO_THRESH)
        & (vol_mean > VOL_MEAN_THRESH)
    )

    # --- Check yesterday's row ---------------------------------------------
    if yesterday not in is_pump.index:
        print(f"⚠ Yesterday ({yesterday.date()}) not found in data index.")
        return []

    row = is_pump.loc[yesterday]
    pump_tickers = row[row == True].index.tolist()

    signals = []
    for ticker in pump_tickers:
        signals.append({
            "symbol": ticker,
            "ret_z": ret_z.loc[yesterday, ticker],
            "rvol": rvol.loc[yesterday, ticker],
            "count_z": count_z.loc[yesterday, ticker],
            "buy_ratio": buy_ratio.loc[yesterday, ticker],
            "vol_mean": vol_mean.loc[yesterday, ticker],
            "log_return": r.loc[yesterday, ticker],
        })

    return signals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    yesterday = today - pd.Timedelta(days=1)
    print(f"Today (UTC):     {today.date()}")
    print(f"Signal date:     {yesterday.date()}")
    print(f"Lookback:        {LOOKBACK_DAYS} days\n")

    # 1. Discover USDT perp symbols with yesterday's data on data.binance.vision
    print("Discovering symbols from data.binance.vision …")
    symbols = get_usdt_perp_symbols(yesterday)
    print(f"{len(symbols)} tickers have data for {yesterday.date()}.\n")

    if not symbols:
        print("No tickers with yesterday's data. Exiting.")
        return

    # 2. Download klines via REST API in parallel
    print(f"Downloading {LOOKBACK_DAYS}-day klines (workers={MAX_WORKERS}) …")
    dataframes: dict[str, pd.DataFrame] = {}
    skipped = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_sym = {
            pool.submit(fetch_klines, sym): sym for sym in symbols
        }
        for future in as_completed(future_to_sym):
            sym = future_to_sym[future]
            try:
                df = future.result()
                if df is None or df.empty:
                    print(f"  ✗ {sym}: no data returned")
                    skipped += 1
                    continue
                dataframes[sym] = df
            except Exception as exc:
                print(f"  ✗ {sym}: {exc}")
                skipped += 1

    print(
        f"Loaded {len(dataframes)} tickers "
        f"({skipped} skipped — download failed).\n"
    )

    if not dataframes:
        print("No valid data. Exiting.")
        return

    # 3. Detect pumps
    signals = detect_pumps(dataframes, yesterday)

    # 4. Print results
    if not signals:
        print(f"No pump signals detected on {yesterday.date()}.")
        return

    signals.sort(key=lambda s: s["ret_z"], reverse=True)

    print(f"{'=' * 72}")
    print(f"  PUMP SIGNALS — {yesterday.date()}  ({len(signals)} detected)")
    print(f"{'=' * 72}")
    print(
        f"\n  {'Symbol':<14} {'ret_z':>8} {'rvol':>8} {'count_z':>8}"
        f" {'buy_ratio':>10} {'log_ret':>10}"
    )
    print(f"  {'-' * 66}")
    for s in signals:
        print(
            f"  {s['symbol']:<14} {s['ret_z']:>8.2f} {s['rvol']:>8.2f}"
            f" {s['count_z']:>8.2f} {s['buy_ratio']:>10.3f}"
            f" {s['log_return']:>10.4f}"
        )
    print()


if __name__ == "__main__":
    main()
