"""
Local data persistence for the PND pipeline.

Manages per-ticker Parquet files under ``data/crypto_pnd/`` and provides
an intelligent three-phase data-hydration routine:

    Phase 1 — **Monthly** (Binance Vision monthly zips)
        Backfills complete months that are entirely absent locally.

    Phase 2 — **Daily** (Binance Vision daily zips)
        Fills the gap from the end of monthly coverage up to ~2 days ago.

    Phase 3 — **REST klines** (Binance Futures REST API)
        Tops up the most-recent ~48 h where Vision files aren't yet
        published.

Usage (from the pipeline)::

    from data.storage import OHLCVStorage

    storage = OHLCVStorage(data_dir="data/crypto_pnd", lookback_days=60,
                           candle_interval="4h", logger=logger)
    results = storage.sync_all(tickers, vision_client, klines_client)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import pandas as pd

from exchanges.binance_klines import BinanceFuturesKlines
from exchanges.binance_vision import BinanceVisionClient

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
STORED_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
]

# Binance Vision daily files typically lag by this many days.
_VISION_DAILY_LAG_DAYS = 2

# Maps Binance interval strings to their duration in hours.
_INTERVAL_HOURS: dict[str, float] = {
    "1m": 1 / 60,
    "3m": 3 / 60,
    "5m": 5 / 60,
    "15m": 0.25,
    "30m": 0.5,
    "1h": 1,
    "2h": 2,
    "4h": 4,
    "6h": 6,
    "8h": 8,
    "12h": 12,
    "1d": 24,
    "3d": 72,
    "1w": 168,
}


# ---------------------------------------------------------------------------
# OHLCVStorage
# ---------------------------------------------------------------------------
class OHLCVStorage:
    """
    Manages per-ticker Parquet files and orchestrates three-phase data
    hydration (monthly → daily → REST klines).

    Parameters
    ----------
    data_dir : str | Path
        Directory for parquet files (e.g. ``data/crypto_pnd/``).
    lookback_days : int
        Number of days of history to maintain (default 60).
    candle_interval : str
        Binance kline interval (default ``"4h"``).
    logger : logging.Logger, optional
        Logger instance.  Falls back to a module-level logger.
    """

    def __init__(
        self,
        data_dir: str | Path,
        lookback_days: int = 60,
        candle_interval: str = "4h",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lookback_days = lookback_days
        self.candle_interval = candle_interval
        self.logger = logger or logging.getLogger(__name__)
        self._interval_hours = _INTERVAL_HOURS.get(candle_interval, 4)

    # ------------------------------------------------------------------
    # Read / Write
    # ------------------------------------------------------------------

    def _parquet_path(self, ticker: str) -> Path:
        """Return the path for a ticker's parquet file."""
        return self.data_dir / f"{ticker}.parquet"

    def read_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """Return the stored DataFrame for *ticker*, or ``None`` if absent."""
        path = self._parquet_path(ticker)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if df.empty:
            return None
        return df

    def write_ticker(self, ticker: str, df: pd.DataFrame) -> None:
        """
        Persist *df* as a Parquet file.

        Before writing the data is normalised (consistent dtypes, sorted,
        de-duplicated) and trimmed so that only rows within the lookback
        window (plus a 5-day buffer) are kept.
        """
        if df is None or df.empty:
            return
        df = self._normalize(df)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(
            days=self.lookback_days + 5
        )
        df = df[df["open_time"] >= cutoff].copy()
        df.reset_index(drop=True, inplace=True)
        df.to_parquet(self._parquet_path(ticker), index=False)

    # ------------------------------------------------------------------
    # Three-phase sync
    # ------------------------------------------------------------------

    def sync_ticker(
        self,
        ticker: str,
        vision_client: BinanceVisionClient,
        klines_client: BinanceFuturesKlines,
    ) -> pd.DataFrame:
        """
        Ensure local data for *ticker* covers the full lookback window.

        The three phases run in order; each phase only fires when its
        gap condition is met.  Returns the merged, up-to-date DataFrame.

        Decision logic
        --------------
        1. **Phase 1 (monthly)** — runs when the latest local timestamp is
           before the 1st of the current month (or no local data exists).
        2. **Phase 2 (daily)** — runs when the latest local timestamp is
           more than ``_VISION_DAILY_LAG_DAYS`` behind ``now``.
        3. **Phase 3 (REST klines)** — always runs to top up the most
           recent candles that Vision hasn't published yet.
        """
        now = pd.Timestamp.now(tz="UTC")
        window_start = now - pd.Timedelta(days=self.lookback_days)
        two_days_ago = now - pd.Timedelta(days=_VISION_DAILY_LAG_DAYS)
        current_month_start = now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )

        existing = self.read_ticker(ticker)
        latest_ts = self._latest_timestamp(existing)

        self.logger.debug(
            f"[{ticker}] sync start — latest_ts={latest_ts}, "
            f"window_start={window_start.date()}"
        )

        # Phase 1 — Monthly backfill  ─────────────────────────────────
        if latest_ts is None or latest_ts < current_month_start:
            existing = self._phase_monthly(
                ticker, vision_client, existing,
                window_start, current_month_start,
            )
            latest_ts = self._latest_timestamp(existing)

        # Phase 2 — Daily gap-fill  ───────────────────────────────────
        if latest_ts is not None and latest_ts < two_days_ago:
            existing = self._phase_daily(
                ticker, vision_client, existing,
                latest_ts, two_days_ago,
            )
            latest_ts = self._latest_timestamp(existing)
        elif latest_ts is None:
            # Monthly yielded nothing — try daily for the full range.
            existing = self._phase_daily(
                ticker, vision_client, existing,
                window_start, two_days_ago,
            )
            latest_ts = self._latest_timestamp(existing)

        # Phase 3 — REST klines top-up  ───────────────────────────────
        existing = self._phase_klines(
            ticker, klines_client, existing, latest_ts, now,
        )

        # Persist  ────────────────────────────────────────────────────
        if existing is not None and not existing.empty:
            self.write_ticker(ticker, existing)
        else:
            self.logger.warning(f"[{ticker}] No data obtained after sync.")

        return (
            existing
            if existing is not None
            else pd.DataFrame(columns=STORED_COLUMNS)
        )

    def sync_all(
        self,
        tickers: list[str],
        vision_client: BinanceVisionClient,
        klines_client: BinanceFuturesKlines,
    ) -> dict[str, pd.DataFrame]:
        """
        Sync every ticker in *tickers*.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of ``{ticker: DataFrame}`` for tickers that have data.
        """
        results: dict[str, pd.DataFrame] = {}
        total = len(tickers)
        for idx, ticker in enumerate(tickers, 1):
            self.logger.info(f"[{idx}/{total}] Syncing {ticker} …")
            try:
                df = self.sync_ticker(ticker, vision_client, klines_client)
                if df is not None and not df.empty:
                    results[ticker] = df
                else:
                    self.logger.warning(f"[{ticker}] Skipped — no data.")
            except Exception as exc:
                self.logger.error(f"[{ticker}] Sync failed: {exc}", exc_info=True)

        self.logger.info(
            f"Sync complete: {len(results)}/{total} tickers with data."
        )
        return results

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _phase_monthly(
        self,
        ticker: str,
        vision: BinanceVisionClient,
        existing: Optional[pd.DataFrame],
        window_start: pd.Timestamp,
        current_month_start: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        """Phase 1: fetch complete months from Binance Vision."""
        start_month = window_start.strftime("%Y-%m")
        prev_month_last_day = current_month_start - pd.Timedelta(days=1)
        end_month = prev_month_last_day.strftime("%Y-%m")

        # If the window starts in or after the current month there are
        # no complete months to fetch.
        if pd.Timestamp(start_month + "-01", tz="UTC") > prev_month_last_day:
            self.logger.debug(
                f"[{ticker}] Phase 1 skipped — lookback window is within "
                "the current month."
            )
            return existing

        self.logger.info(
            f"[{ticker}] Phase 1 (monthly): {start_month} -> {end_month}"
        )
        try:
            new_df = vision.fetch_klines_monthly(
                ticker, self.candle_interval, start_month, end_month,
            )
            if new_df is not None and not new_df.empty:
                existing = self._merge(existing, new_df)
                self.logger.info(
                    f"[{ticker}] Phase 1: {len(new_df)} rows fetched."
                )
            else:
                self.logger.info(
                    f"[{ticker}] Phase 1: no monthly data available."
                )
        except Exception as exc:
            self.logger.error(f"[{ticker}] Phase 1 failed: {exc}")

        return existing

    def _phase_daily(
        self,
        ticker: str,
        vision: BinanceVisionClient,
        existing: Optional[pd.DataFrame],
        from_ts: pd.Timestamp,
        to_ts: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        """Phase 2: fetch daily zip files from Binance Vision.

        *from_ts* is typically the latest timestamp already held locally.
        We start the daily download from from_ts's calendar date so that
        any partially-covered day is completed (duplicates are removed
        during merge).
        """
        if from_ts >= to_ts:
            self.logger.debug(f"[{ticker}] Phase 2 skipped — no daily gap.")
            return existing

        start_date = from_ts.normalize().strftime("%Y-%m-%d")
        end_date = to_ts.strftime("%Y-%m-%d")

        self.logger.info(
            f"[{ticker}] Phase 2 (daily): {start_date} -> {end_date}"
        )
        try:
            new_df = vision.fetch_klines_daily(
                ticker, self.candle_interval, start_date, end_date,
            )
            if new_df is not None and not new_df.empty:
                existing = self._merge(existing, new_df)
                self.logger.info(
                    f"[{ticker}] Phase 2: {len(new_df)} rows fetched."
                )
            else:
                self.logger.info(
                    f"[{ticker}] Phase 2: no daily data available."
                )
        except Exception as exc:
            self.logger.error(f"[{ticker}] Phase 2 failed: {exc}")

        return existing

    def _phase_klines(
        self,
        ticker: str,
        klines_client: BinanceFuturesKlines,
        existing: Optional[pd.DataFrame],
        latest_ts: Optional[pd.Timestamp],
        now: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        """Phase 3: top up the most recent candles via the Futures REST API."""
        if latest_ts is None:
            # Nothing from Vision at all — pull a full lookback via REST.
            limit = min(
                1500,
                int(self.lookback_days * 24 / self._interval_hours) + 1,
            )
        else:
            gap_hours = (now - latest_ts).total_seconds() / 3600
            if gap_hours <= self._interval_hours:
                self.logger.debug(
                    f"[{ticker}] Phase 3 skipped — already up to date."
                )
                return existing
            limit = math.ceil(gap_hours / self._interval_hours) + 2

        limit = max(limit, 1)
        limit = min(limit, 1500)

        self.logger.info(f"[{ticker}] Phase 3 (REST klines): limit={limit}")
        try:
            new_df = klines_client.fetch_klines(
                ticker, self.candle_interval, limit=limit,
            )
            if new_df is not None and not new_df.empty:
                existing = self._merge(existing, new_df)
                self.logger.info(
                    f"[{ticker}] Phase 3: {len(new_df)} rows fetched."
                )
            else:
                self.logger.info(
                    f"[{ticker}] Phase 3: no kline data returned."
                )
        except Exception as exc:
            self.logger.error(f"[{ticker}] Phase 3 failed: {exc}")

        return existing

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _latest_timestamp(df: Optional[pd.DataFrame]) -> Optional[pd.Timestamp]:
        """Return the most recent ``open_time`` in *df*, or ``None``."""
        if df is None or df.empty:
            return None
        return df["open_time"].max()

    @staticmethod
    def _merge(
        existing: Optional[pd.DataFrame],
        new: pd.DataFrame,
    ) -> pd.DataFrame:
        """Concatenate, de-duplicate on ``open_time``, and sort."""
        if existing is None or existing.empty:
            return new
        if new is None or new.empty:
            return existing
        combined = pd.concat([existing, new], ignore_index=True)
        combined.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
        combined.sort_values("open_time", inplace=True)
        combined.reset_index(drop=True, inplace=True)
        return combined

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure consistent column set, dtypes, ordering, and de-duplication
        before writing to Parquet.
        """
        df = df.copy()

        # Drop the unused 'ignore' column that Binance Vision includes.
        if "ignore" in df.columns:
            df.drop(columns=["ignore"], inplace=True)

        # Datetime columns → UTC-aware.
        for col in ("open_time", "close_time"):
            if col not in df.columns:
                continue
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
            elif df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize("UTC")

        # Numeric columns → float64.
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Keep only the standard column set (in order).
        present = [c for c in STORED_COLUMNS if c in df.columns]
        df = df[present]

        df.sort_values("open_time", inplace=True)
        df.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
