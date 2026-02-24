"""
Shock reversion feature engineering and signal detection.

Stages 5 & 6 of the shock-reversion pipeline.

Signal flow
-----------
1. ``compute_features(df)`` — takes a single-ticker OHLCV DataFrame,
   returns it with appended feature columns (ret, vol, ret_z, vol_mean).
2. ``is_shock_signal(row, thresholds)`` — checks whether the latest bar
   exceeds the configured ret_z and vol_mean thresholds.
3. ``ShockReversionSignalManager`` — stateful orchestrator that runs
   features + signal detection on each closed candle, tracks active positions
   (persisted to JSON), and emits entry/exit signals for the execution layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants (defaults — overridden by config)
# ---------------------------------------------------------------------------

DEFAULT_LOOKBACK_BARS = 240          # Rolling window: 240 × 1m = 4 hours

DEFAULT_THRESHOLDS = {
    "ret_z_min": 10.0,               # |ret_z| must exceed this
    "vol_mean_min": 5_000_000,       # 240-bar mean volume must exceed this
}

DEFAULT_EXIT_PARAMS = {
    "hold_bars": 30,                 # Fixed 30-minute hold; exit unconditionally
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class SignalAction(Enum):
    ENTER_SHORT = "enter_short"
    EXIT_SHORT  = "exit_short"


@dataclass
class ShockReversionSignal:
    """
    Represents a detected shock signal or an exit instruction emitted to
    the execution layer.
    """
    ticker:      str
    action:      SignalAction
    signal_time: pd.Timestamp
    entry_price: Optional[float] = None   # close at signal candle
    reason:      str = ""

    # Feature snapshot (for logging / auditing)
    ret_z:    Optional[float] = None
    vol_mean: Optional[float] = None


@dataclass
class ActivePosition:
    """Tracks an open short, awaiting its fixed-hold exit."""
    ticker:         str
    entry_time:     str             # ISO-8601 string (JSON-friendly)
    entry_price:    float
    bars_held:      int = 0
    hold_bars:      int = DEFAULT_EXIT_PARAMS["hold_bars"]


# ---------------------------------------------------------------------------
# Position persistence
# ---------------------------------------------------------------------------

class PositionStore:
    """
    Persists active shock-reversion positions to a JSON file so they survive
    restarts and can be consumed independently by the execution layer.

    File layout (``data/crypto_shock_reversion/signals/active_positions.json``)::

        [
          {
            "ticker": "BTCUSDT",
            "entry_time": "2026-02-24T12:00:00+00:00",
            "entry_price": 95000.0,
            "bars_held": 5,
            "hold_bars": 30
          },
          ...
        ]
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[ActivePosition]:
        """Read positions from disk. Returns empty list if file is missing."""
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
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
) -> pd.DataFrame:
    """
    Compute shock-detection features for a single-ticker OHLCV DataFrame.

    Appended columns
    ----------------
    - ``ret``      — bar-to-bar log return:  ln(close_t / close_{t-1})
    - ``vol``      — rolling std of ``ret`` over *lookback_bars*, lagged by 1
                     (realised volatility proxy; avoids look-ahead)
    - ``ret_z``    — return z-score:  ret / vol
    - ``vol_mean`` — rolling mean of ``volume`` over *lookback_bars*, lagged by 1
                     (liquidity filter)

    The input DataFrame must contain at least ``open_time``, ``close``,
    and ``volume`` columns. Returns a sorted copy.
    """
    feat = df.copy()
    feat.sort_values("open_time", inplace=True)
    feat.reset_index(drop=True, inplace=True)

    close  = feat["close"].astype(float)
    volume = feat["volume"].astype(float)

    feat["ret"]      = np.log(close).diff()
    feat["vol"]      = feat["ret"].shift(1).rolling(lookback_bars).std()
    feat["ret_z"]    = feat["ret"] / feat["vol"]
    feat["vol_mean"] = volume.shift(1).rolling(lookback_bars).mean()

    return feat


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

def is_shock_signal(
    row: pd.Series,
    thresholds: Optional[dict] = None,
) -> bool:
    """
    Return ``True`` when the row meets both shock conditions:

    - ``ret_z``    > ``ret_z_min``    (extreme normalised return spike)
    - ``vol_mean`` > ``vol_mean_min`` (sufficient liquidity)
    """
    t = thresholds or DEFAULT_THRESHOLDS

    return bool(
        pd.notna(row.get("ret_z"))
        and pd.notna(row.get("vol_mean"))
        and row["ret_z"]    > t["ret_z_min"]
        and row["vol_mean"] > t["vol_mean_min"]
    )


def detect_signals_for_ticker(
    feat_df: pd.DataFrame,
    ticker: str,
    thresholds: Optional[dict] = None,
    only_last_n: int = 1,
) -> list[ShockReversionSignal]:
    """
    Scan the last *only_last_n* rows of a feature-enriched DataFrame for
    shock entry signals.

    In live mode ``only_last_n=1`` (just the closed candle).
    """
    signals: list[ShockReversionSignal] = []
    if feat_df is None or feat_df.empty:
        return signals

    for _, row in feat_df.tail(only_last_n).iterrows():
        if is_shock_signal(row, thresholds):
            signals.append(
                ShockReversionSignal(
                    ticker=ticker,
                    action=SignalAction.ENTER_SHORT,
                    signal_time=row["open_time"],
                    entry_price=float(row["close"]),
                    reason="shock_detected",
                    ret_z=float(row["ret_z"]),
                    vol_mean=float(row["vol_mean"]),
                )
            )
    return signals


# ---------------------------------------------------------------------------
# Signal manager
# ---------------------------------------------------------------------------

class ShockReversionSignalManager:
    """
    Stateful orchestrator that:

    * Computes features for each ticker on every new closed candle.
    * Detects new shock entry signals.
    * Tracks active positions (persisted to JSON on disk).
    * Emits ``EXIT_SHORT`` after the fixed ``hold_bars`` hold period.

    Usage (from pipeline Stage 4 WebSocket callback)::

        manager = ShockReversionSignalManager(config, logger, positions_path)

        # On each closed candle:
        actions = manager.update(ticker, ohlcv_df)
        # actions is a list of ShockReversionSignal (ENTER_SHORT | EXIT_SHORT)
    """

    def __init__(
        self,
        config:         Optional[dict] = None,
        logger:         Optional[logging.Logger] = None,
        positions_path: Optional[Path] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}

        strategy = self.config.get("strategy", {})

        self.lookback_bars: int  = strategy.get("lookback_bars", DEFAULT_LOOKBACK_BARS)
        self.thresholds:    dict = strategy.get("thresholds",    DEFAULT_THRESHOLDS)
        self.exit_params:   dict = strategy.get("exit_params",   DEFAULT_EXIT_PARAMS)

        if positions_path is None:
            positions_path = (
                Path("data") / "crypto_shock_reversion" / "signals" / "active_positions.json"
            )
        self.position_store = PositionStore(positions_path)
        self.active_positions: list[ActivePosition] = self.position_store.load()

        if self.active_positions:
            self.logger.info(
                f"Restored {len(self.active_positions)} active positions from disk."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        ticker:   str,
        ohlcv_df: pd.DataFrame,
    ) -> list[ShockReversionSignal]:
        """
        Process one new closed candle for *ticker*.

        1. Compute features.
        2. Check the latest bar for a shock entry signal.
        3. Increment ``bars_held`` for any active positions on this ticker
           and emit ``EXIT_SHORT`` once ``hold_bars`` is reached.

        Returns a list of ``ShockReversionSignal`` actions.
        """
        actions: list[ShockReversionSignal] = []

        # --- Feature engineering ---
        feat = compute_features(ohlcv_df, lookback_bars=self.lookback_bars)

        # --- Entry detection ---
        new_entries = detect_signals_for_ticker(
            feat, ticker, thresholds=self.thresholds, only_last_n=1,
        )
        for sig in new_entries:
            self.logger.info(
                f"[{ticker}] SHOCK SIGNAL — ret_z={sig.ret_z:.2f}, "
                f"vol_mean={sig.vol_mean:,.0f}, price={sig.entry_price}"
            )
            self.active_positions.append(
                ActivePosition(
                    ticker=ticker,
                    entry_time=str(sig.signal_time),
                    entry_price=sig.entry_price,
                    hold_bars=self.exit_params.get("hold_bars", DEFAULT_EXIT_PARAMS["hold_bars"]),
                )
            )
        actions.extend(new_entries)

        # --- Exit evaluation (fixed hold) ---
        exits = self._evaluate_exits(ticker, feat)
        actions.extend(exits)

        if actions:
            self.position_store.save(self.active_positions)

        return actions

    def get_active_positions(self, ticker: Optional[str] = None) -> list[ActivePosition]:
        """Return active positions, optionally filtered by ticker."""
        if ticker is None:
            return list(self.active_positions)
        return [p for p in self.active_positions if p.ticker == ticker]

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def _evaluate_exits(
        self,
        ticker: str,
        feat:   pd.DataFrame,
    ) -> list[ShockReversionSignal]:
        """
        Increment ``bars_held`` for every active position on *ticker* and
        emit an ``EXIT_SHORT`` signal once the fixed ``hold_bars`` is reached.
        """
        exits:     list[ShockReversionSignal] = []
        remaining: list[ActivePosition]       = []

        latest_row = feat.iloc[-1] if not feat.empty else None

        for pos in self.active_positions:
            if pos.ticker != ticker:
                remaining.append(pos)
                continue

            pos.bars_held += 1

            if pos.bars_held >= pos.hold_bars:
                current_price = float(latest_row["close"]) if latest_row is not None else None
                current_time  = latest_row["open_time"]    if latest_row is not None else None
                reason        = f"hold_{pos.hold_bars}_bars_expired"

                exits.append(
                    ShockReversionSignal(
                        ticker=ticker,
                        action=SignalAction.EXIT_SHORT,
                        signal_time=current_time,
                        entry_price=current_price,
                        reason=reason,
                    )
                )
                self.logger.info(
                    f"[{ticker}] EXIT SIGNAL — bars_held={pos.bars_held}, reason={reason}"
                )
            else:
                remaining.append(pos)

        self.active_positions = remaining
        return exits
