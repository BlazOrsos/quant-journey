"""
Pump-and-dump feature engineering and signal detection.

Stages 5 & 6 of the PND pipeline.  Operates entirely in memory — reads raw
OHLCV DataFrames from storage, computes rolling features, and emits signal
objects for the execution layer.

Signal flow
-----------
1. ``compute_features(df)`` — takes a single-ticker DataFrame, returns it
   with appended feature columns.
2. ``detect_signals_for_ticker(feat_df, ticker)`` — checks the latest bar(s)
   against pump thresholds, returns entry signals.
3. ``PnDSignalManager`` — orchestrates both steps, tracks active positions
   (persisted to disk), and emits entry/exit signals for the execution layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (defaults — overridable via config)
# ---------------------------------------------------------------------------

# 24h window expressed in number of bars (4h candles → 6 bars = 24h)
DEFAULT_WINDOW_BARS = 6

# Lookback for historical statistics (30 days of 4h bars = 180 bars)
DEFAULT_LOOKBACK_DAYS = 30

# Signal thresholds
DEFAULT_THRESHOLDS = {
    "ret_z_min": 5.0,          # Minimum return z-score
    "rvol_min": 5.0,           # Minimum relative volume
    "count_z_min": 5.0,        # Minimum trade-count z-score
    "buy_ratio_min": 0.0,      # Minimum taker-buy ratio (strict >)
    "vol_mean_min": 5_000_000, # Minimum average 24h volume (liquidity filter)
}

# Exit parameters
DEFAULT_EXIT_PARAMS = {
    "min_hold_bars": 6,        # 24h minimum hold  (6 × 4h bars)
    "max_hold_bars": 12,       # 48h maximum hold (12 × 4h bars)
}

# Maps interval strings to bars-per-day
_BARS_PER_DAY: dict[str, int] = {
    "1h": 24,
    "2h": 12,
    "4h": 6,
    "6h": 4,
    "8h": 3,
    "12h": 2,
    "1d": 1,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class SignalAction(Enum):
    ENTER_SHORT = "enter_short"
    EXIT_SHORT = "exit_short"


@dataclass
class PnDSignal:
    """
    Represents a detected pump signal or an exit instruction.

    The execution layer reads these to place / close orders.
    """
    ticker: str
    action: SignalAction
    signal_time: pd.Timestamp          # open_time of the candle that triggered
    entry_price: Optional[float] = None # close price at signal time (for logging)
    reason: str = ""

    # Feature snapshot at signal time (for logging / auditing)
    ret_z: Optional[float] = None
    rvol: Optional[float] = None
    count_z: Optional[float] = None
    buy_ratio: Optional[float] = None
    r_24h: Optional[float] = None


@dataclass
class ActivePosition:
    """Tracks a (short) position — both open and closed."""
    ticker: str
    entry_time: str               # ISO-8601 string (JSON-friendly)
    entry_bar_index: int          # row index in the ticker's DataFrame at entry
    entry_price: float
    bars_held: int = 0
    min_hold_bars: int = 6
    max_hold_bars: int = 12
    id: str = ""                  # unique key: {ticker}_{entry_time}
    status: str = "open"          # "open" or "closed"
    exit_time: Optional[str] = None   # ISO-8601 string
    exit_price: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.id:
            # Normalise entry_time to a filesystem-safe string for the key
            safe_ts = self.entry_time.replace(" ", "T").replace(":", "-")
            self.id = f"{self.ticker}_{safe_ts}"


# ---------------------------------------------------------------------------
# Position persistence
# ---------------------------------------------------------------------------

class PositionStore:
    """
    Persists active positions to a JSON file so they survive restarts and
    can be read independently by the execution layer.

    File layout (``data/signals/positions.json``)::

        [
          {
            "ticker": "BTCUSDT",
            "entry_time": "2026-02-24T04:00:00+00:00",
            "entry_bar_index": 359,
            "entry_price": 95000.0,
            "bars_held": 2,
            "min_hold_bars": 6,
            "max_hold_bars": 12,
            "id": "BTCUSDT_2026-02-24T04-00-00+00-00",
            "status": "open",
            "exit_time": null,
            "exit_price": null
          },
          ...
        ]
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[ActivePosition]:
        """Read positions from disk.  Returns empty list if file missing."""
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r") as f:
                raw = json.load(f)
            return [ActivePosition(**item) for item in raw]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def save(self, positions: list[ActivePosition]) -> None:
        """Atomically write current positions to disk."""
        data = [asdict(p) for p in positions]
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp.replace(self.path)


# ---------------------------------------------------------------------------
# Feature engineering (single ticker)
# ---------------------------------------------------------------------------

def compute_features(
    df: pd.DataFrame,
    window_bars: int = DEFAULT_WINDOW_BARS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    candle_interval: str = "4h",
) -> pd.DataFrame:
    """
    Compute pump-detection features on a single-ticker OHLCV DataFrame.

    The input DataFrame must have at least the columns:
    ``open_time``, ``close``, ``volume``, ``count``, ``taker_buy_volume``.

    Returns a **copy** with the following columns appended:

    - ``log_ret``           — bar-to-bar log return
    - ``r_24h``             — rolling 24h log return
    - ``vol_24h``           — rolling 24h volume
    - ``count_24h``         — rolling 24h trade count
    - ``taker_buy_vol_24h`` — rolling 24h taker-buy volume
    - ``ret_vol``           — rolling std of 24h returns (shifted)
    - ``ret_z``             — return z-score  (r_24h / ret_vol)
    - ``vol_mean``          — rolling mean of 24h volume (shifted)
    - ``rvol``              — relative volume (vol_24h / vol_mean)
    - ``cnt_mean``          — rolling mean of 24h count (shifted)
    - ``cnt_std``           — rolling std of 24h count (shifted)
    - ``count_z``           — count z-score
    - ``buy_ratio``         — taker buy ratio for the 24h window
    """
    bars_per_day = _BARS_PER_DAY.get(candle_interval, 6)
    lookback_bars = lookback_days * bars_per_day

    feat = df.copy()
    feat.sort_values("open_time", inplace=True)
    feat.reset_index(drop=True, inplace=True)

    close = feat["close"].astype(float)
    volume = feat["volume"].astype(float)
    count = feat["count"].astype(float)
    taker_buy_vol = feat["taker_buy_volume"].astype(float)

    # Bar-to-bar log return
    feat["log_ret"] = np.log(close).diff()

    # 24h aggregated metrics
    feat["r_24h"] = feat["log_ret"].rolling(window_bars).sum()
    feat["vol_24h"] = volume.rolling(window_bars).sum()
    feat["count_24h"] = count.rolling(window_bars).sum()
    feat["taker_buy_vol_24h"] = taker_buy_vol.rolling(window_bars).sum()

    # Historical statistics (shifted by 1 to avoid look-ahead bias)
    feat["ret_vol"] = feat["r_24h"].shift(1).rolling(lookback_bars).std()
    feat["ret_z"] = feat["r_24h"] / feat["ret_vol"]

    feat["vol_mean"] = feat["vol_24h"].shift(1).rolling(lookback_bars).mean()
    feat["rvol"] = feat["vol_24h"] / feat["vol_mean"]

    feat["cnt_mean"] = feat["count_24h"].shift(1).rolling(lookback_bars).mean()
    feat["cnt_std"] = feat["count_24h"].shift(1).rolling(lookback_bars).std()
    feat["count_z"] = (feat["count_24h"] - feat["cnt_mean"]) / feat["cnt_std"]

    # Taker buy ratio
    feat["buy_ratio"] = (feat["taker_buy_vol_24h"] / feat["vol_24h"]).clip(0, 1)

    return feat


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

def is_pump_signal(
    row: pd.Series,
    thresholds: Optional[dict] = None,
) -> bool:
    """
    Evaluate whether a single row (with feature columns) meets pump criteria.

    Returns ``True`` if **all** conditions are met simultaneously.
    """
    t = thresholds or DEFAULT_THRESHOLDS

    return bool(
        pd.notna(row.get("ret_z"))
        and row["ret_z"] > t["ret_z_min"]
        and row["r_24h"] > 0
        and row["log_ret"] > 0
        and row["rvol"] > t["rvol_min"]
        and row["count_z"] > t["count_z_min"]
        and row["buy_ratio"] > t["buy_ratio_min"]
        and row["vol_mean"] > t["vol_mean_min"]
    )


def detect_signals_for_ticker(
    feat_df: pd.DataFrame,
    ticker: str,
    thresholds: Optional[dict] = None,
    only_last_n: int = 1,
) -> list[PnDSignal]:
    """
    Scan the last *only_last_n* rows of a feature-enriched DataFrame for
    pump signals.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Output of ``compute_features()`` for one ticker.
    ticker : str
        Symbol name.
    thresholds : dict, optional
        Override default thresholds.
    only_last_n : int
        How many of the most recent bars to check.  In live mode this is
        typically ``1`` (the just-closed candle).

    Returns
    -------
    list[PnDSignal]
        Possibly empty list of new entry signals.
    """
    signals: list[PnDSignal] = []
    if feat_df is None or feat_df.empty:
        return signals

    tail = feat_df.tail(only_last_n)
    for idx, row in tail.iterrows():
        if is_pump_signal(row, thresholds):
            signals.append(
                PnDSignal(
                    ticker=ticker,
                    action=SignalAction.ENTER_SHORT,
                    signal_time=row["open_time"],
                    entry_price=float(row["close"]),
                    reason="pump_detected",
                    ret_z=float(row["ret_z"]),
                    rvol=float(row["rvol"]),
                    count_z=float(row["count_z"]),
                    buy_ratio=float(row["buy_ratio"]),
                    r_24h=float(row["r_24h"]),
                )
            )
    return signals


# ---------------------------------------------------------------------------
# Signal manager — orchestrates features + signals + exit tracking
# ---------------------------------------------------------------------------

class PnDSignalManager:
    """
    Stateful manager that:

    * Computes features for each ticker.
    * Detects new pump entry signals.
    * Tracks active positions (persisted to JSON) and emits exit signals
      based on the 24h/48h conditional exit rule.
    * Allows multiple concurrent positions per ticker.

    Usage (from pipeline Stage 4 callback or a batch scan)::

        manager = PnDSignalManager(config, logger)

        # On each new closed candle (or batch):
        actions = manager.update(ticker, ohlcv_df)
        # actions is a list of PnDSignal with ENTER_SHORT or EXIT_SHORT
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        logger: Optional[logging.Logger] = None,
        positions_path: Optional[Path] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}

        params = self.config.get("parameters", {})
        strategy = self.config.get("strategy", {})

        self.candle_interval: str = params.get("candle_interval", "4h")
        self.lookback_days: int = strategy.get("lookback_days", DEFAULT_LOOKBACK_DAYS)
        self.window_bars: int = strategy.get("window_bars", DEFAULT_WINDOW_BARS)
        self.thresholds: dict = strategy.get("thresholds", DEFAULT_THRESHOLDS)
        self.exit_params: dict = strategy.get("exit_params", DEFAULT_EXIT_PARAMS)

        # Position persistence
        if positions_path is None:
            positions_path = Path("data") / "signals" / "positions.json"
        self.position_store = PositionStore(positions_path)
        self._all_positions: list[ActivePosition] = self.position_store.load()

        active = [p for p in self._all_positions if p.status == "open"]
        if active:
            self.logger.info(
                f"Restored {len(active)} active positions from disk "
                f"({len(self._all_positions)} total on file)."
            )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @property
    def active_positions(self) -> list[ActivePosition]:
        """Return only open positions."""
        return [p for p in self._all_positions if p.status == "open"]

    def _persist(self) -> None:
        """Flush all positions (open + closed) to disk."""
        self.position_store.save(self._all_positions)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        ticker: str,
        ohlcv_df: pd.DataFrame,
    ) -> list[PnDSignal]:
        """
        Process a new candle for *ticker*.

        1. Compute features over the full local history.
        2. Check the latest bar for a pump entry signal.
        3. Increment ``bars_held`` for any active positions on this ticker
           and evaluate exit conditions.

        Returns a list of ``PnDSignal`` actions (entries + exits).
        """
        actions: list[PnDSignal] = []

        # --- Feature engineering ---
        feat = compute_features(
            ohlcv_df,
            window_bars=self.window_bars,
            lookback_days=self.lookback_days,
            candle_interval=self.candle_interval,
        )

        # --- Entry detection ---
        new_entries = detect_signals_for_ticker(
            feat, ticker, thresholds=self.thresholds, only_last_n=1,
        )
        if new_entries:
            for sig in new_entries:
                self.logger.debug(
                    f"[{ticker}] PUMP SIGNAL — ret_z={sig.ret_z:.2f}, "
                    f"rvol={sig.rvol:.2f}, count_z={sig.count_z:.2f}, "
                    f"buy_ratio={sig.buy_ratio:.3f}, r_24h={sig.r_24h:.4f}"
                )
                # Register the position for exit tracking
                self._all_positions.append(
                    ActivePosition(
                        ticker=ticker,
                        entry_time=str(sig.signal_time),
                        entry_bar_index=len(feat) - 1,
                        entry_price=sig.entry_price,
                        min_hold_bars=self.exit_params.get("min_hold_bars", 6),
                        max_hold_bars=self.exit_params.get("max_hold_bars", 12),
                    )
                )
            actions.extend(new_entries)

        # --- Exit evaluation ---
        exit_signals = self._evaluate_exits(ticker, feat)
        actions.extend(exit_signals)

        # Persist after every update that produced actions
        if actions:
            self._persist()

        return actions

    def update_all(
        self,
        ohlcv_by_ticker: dict[str, pd.DataFrame],
    ) -> list[PnDSignal]:
        """
        Batch-update all tickers (e.g. for a backtest or after a data
        hydration).  Returns combined action list.
        """
        all_actions: list[PnDSignal] = []
        for ticker, df in ohlcv_by_ticker.items():
            actions = self.update(ticker, df)
            all_actions.extend(actions)
        return all_actions

    def get_active_positions(self, ticker: Optional[str] = None) -> list[ActivePosition]:
        """Return open positions, optionally filtered by ticker."""
        positions = self.active_positions
        if ticker is not None:
            positions = [p for p in positions if p.ticker == ticker]
        return positions

    def get_all_positions(self, ticker: Optional[str] = None) -> list[ActivePosition]:
        """Return all positions (open + closed), optionally filtered by ticker."""
        if ticker is None:
            return list(self._all_positions)
        return [p for p in self._all_positions if p.ticker == ticker]

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def _evaluate_exits(
        self,
        ticker: str,
        feat: pd.DataFrame,
    ) -> list[PnDSignal]:
        """
        For every active position on *ticker*, increment bars_held and
        apply the conditional exit rule:

        - After ``min_hold_bars`` (24h): if cumulative return since entry
          is negative (i.e. short is profitable), exit immediately.
        - After ``max_hold_bars`` (48h): exit unconditionally.

        Returns a list of EXIT_SHORT signals.  Exited positions are marked
        as ``status="closed"`` with ``exit_time`` and ``exit_price`` set.
        """
        exits: list[PnDSignal] = []

        latest_row = feat.iloc[-1] if not feat.empty else None

        for pos in self.active_positions:
            if pos.ticker != ticker:
                continue

            pos.bars_held += 1

            should_exit = False
            reason = ""

            if pos.bars_held >= pos.max_hold_bars:
                # Hard exit at 48h
                should_exit = True
                reason = f"max_hold_{pos.max_hold_bars}_bars"
            elif pos.bars_held >= pos.min_hold_bars:
                # After 24h: exit if the short is in profit (price dropped)
                # Compute cumulative log return since entry
                entry_idx = pos.entry_bar_index
                if entry_idx < len(feat) - 1:
                    post_entry = feat.iloc[entry_idx + 1 : len(feat)]
                    cum_ret = post_entry["log_ret"].sum() if not post_entry.empty else 0.0
                    # For a short: we profit when cum_ret < 0 (price dropped)
                    if cum_ret < 0:
                        should_exit = True
                        reason = f"exit_at_{pos.bars_held}_bars_ret={cum_ret:.4f}"

            if should_exit:
                current_price = float(latest_row["close"]) if latest_row is not None else None
                current_time = latest_row["open_time"] if latest_row is not None else None

                # Mark position as closed (stays in file)
                pos.status = "closed"
                pos.exit_time = str(current_time) if current_time is not None else None
                pos.exit_price = current_price

                exits.append(
                    PnDSignal(
                        ticker=ticker,
                        action=SignalAction.EXIT_SHORT,
                        signal_time=current_time,
                        entry_price=current_price,
                        reason=reason,
                    )
                )
                self.logger.debug(
                    f"[{ticker}] EXIT SIGNAL — id={pos.id}, "
                    f"bars_held={pos.bars_held}, reason={reason}"
                )

        return exits
