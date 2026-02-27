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

    # Set by ShockReversionSignalManager to link signal → position
    position_id: Optional[str] = None


@dataclass
class ActivePosition:
    """Tracks a (short) position — both open and closed."""
    ticker:         str
    entry_time:     str             # ISO-8601 string (JSON-friendly)
    entry_bar_index: int            # row index in the ticker's DataFrame at entry
    entry_price:    float
    bars_held:      int = 0
    hold_bars:      int = DEFAULT_EXIT_PARAMS["hold_bars"]
    id:             str = ""       # unique key: {ticker}_{entry_time}
    status:         str = "open"   # "open" or "closed"
    exit_time:      Optional[str] = None   # ISO-8601 string
    exit_price:     Optional[float] = None

    # Trade execution details (filled by pipeline after Binance order)
    order_id:       Optional[str] = None   # Binance entry order ID
    quantity:       Optional[float] = None # base-asset quantity traded
    exit_order_id:  Optional[str] = None   # Binance exit order ID

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
    Persists all shock-reversion positions (open + closed) to a JSON file
    so they survive restarts and can be consumed independently by the
    execution layer.

    File layout (``data/crypto_shock_reversion/signals/positions.json``)::

        [
          {
            "ticker": "BTCUSDT",
            "entry_time": "2026-02-24T12:00:00+00:00",
            "entry_bar_index": 239,
            "entry_price": 95000.0,
            "bars_held": 30,
            "hold_bars": 30,
            "id": "BTCUSDT_2026-02-24T12-00-00+00-00",
            "status": "closed",
            "exit_time": "2026-02-24T12:30:00+00:00",
            "exit_price": 94800.0
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
                    signal_time=row["close_time"],
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

        # Position persistence
        if positions_path is None:
            positions_path = (
                Path("data") / "crypto_shock_reversion" / "signals" / "positions.json"
            )
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
        if new_entries:
            for sig in new_entries:
                self.logger.info(
                    f"[{ticker}] SHOCK SIGNAL — ret_z={sig.ret_z:.2f}, "
                    f"vol_mean={sig.vol_mean:,.0f}, price={sig.entry_price}"
                )
                # Register the position for exit tracking
                pos = ActivePosition(
                    ticker=ticker,
                    entry_time=str(sig.signal_time),
                    entry_bar_index=len(feat) - 1,
                    entry_price=sig.entry_price,
                    hold_bars=self.exit_params.get("hold_bars", DEFAULT_EXIT_PARAMS["hold_bars"]),
                )
                self._all_positions.append(pos)
                sig.position_id = pos.id
            actions.extend(new_entries)

        # --- Exit evaluation (fixed hold) ---
        had_active_positions = any(p.ticker == ticker for p in self.active_positions)
        exits = self._evaluate_exits(ticker, feat)
        actions.extend(exits)

        # Persist if signals were generated OR if bars_held was incremented
        # (active positions have bars_held incremented on every candle even
        # when no exit is triggered, so we must persist those too)
        if actions or had_active_positions:
            self._persist()

        return actions

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

    def get_position_by_id(self, position_id: str) -> Optional[ActivePosition]:
        """Look up a position by its unique ID."""
        for p in self._all_positions:
            if p.id == position_id:
                return p
        return None

    def update_position_trade(
        self,
        position_id: str,
        order_id: Optional[str] = None,
        quantity: Optional[float] = None,
        exit_order_id: Optional[str] = None,
    ) -> None:
        """Update a position with trade execution details and persist."""
        pos = self.get_position_by_id(position_id)
        if pos is None:
            self.logger.warning(f"Position {position_id} not found for trade update.")
            return
        if order_id is not None:
            pos.order_id = order_id
        if quantity is not None:
            pos.quantity = quantity
        if exit_order_id is not None:
            pos.exit_order_id = exit_order_id
        self._persist()

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

        Exited positions are marked as ``status="closed"`` with
        ``exit_time`` and ``exit_price`` set (they remain in the file).
        """
        exits: list[ShockReversionSignal] = []

        latest_row = feat.iloc[-1] if not feat.empty else None

        for pos in self.active_positions:
            if pos.ticker != ticker:
                continue

            pos.bars_held += 1

            if pos.bars_held >= pos.hold_bars:
                current_price = float(latest_row["close"]) if latest_row is not None else None
                current_time  = latest_row["open_time"]    if latest_row is not None else None
                reason        = f"hold_{pos.hold_bars}_bars_expired"

                # Mark position as closed (stays in file)
                pos.status = "closed"
                pos.exit_time = str(current_time) if current_time is not None else None
                pos.exit_price = current_price

                exits.append(
                    ShockReversionSignal(
                        ticker=ticker,
                        action=SignalAction.EXIT_SHORT,
                        signal_time=current_time,
                        entry_price=current_price,
                        reason=reason,
                        position_id=pos.id,
                    )
                )
                self.logger.info(
                    f"[{ticker}] EXIT SIGNAL — id={pos.id}, "
                    f"bars_held={pos.bars_held}, reason={reason}"
                )

        return exits
