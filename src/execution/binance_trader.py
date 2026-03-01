"""
Binance Futures USDT-M execution layer.

Unified trader for both PnD and Shock Reversion pipelines.
Handles opening and closing short positions via market orders.

Usage::

    from execution.binance_trader import BinanceFuturesTrader

    trader = BinanceFuturesTrader(config, logger)

    # Open a short position
    order = trader.open_short("BTCUSDT")

    # Close a short position (buy back the quantity)
    order = trader.close_short("BTCUSDT", quantity=0.001)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import ccxt


class BinanceFuturesTrader:
    """
    Places market short-entry and short-exit orders on Binance USDT-M Futures.

    Reads configuration from the strategy config dict::

        {
            "execution": {
                "position_size_usdt": 100.0,
                "dry_run": true
            },
            "binance": {
                "api_key": "...",
                "api_secret": "..."
            }
        }

    In **dry-run** mode every method still calls public endpoints (market
    metadata, ticker price) to compute realistic quantities, but no orders
    are actually placed.

    Parameters
    ----------
    config : dict
        Strategy config containing ``binance`` and ``execution`` sections.
    logger : logging.Logger, optional
        Pipeline logger.
    """

    def __init__(
        self,
        config: dict,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.config = config

        binance_cfg = config.get("binance", {})
        execution_cfg = config.get("execution", {})

        self.position_size_usdt: float = execution_cfg.get("position_size_usdt", 100.0)
        self.leverage: int = execution_cfg.get("leverage", 1)
        self.dry_run: bool = execution_cfg.get("dry_run", True)

        self.exchange = ccxt.binance({
            "apiKey": binance_cfg.get("api_key", ""),
            "secret": binance_cfg.get("api_secret", ""),
            "options": {"defaultType": "future"},
            "enableRateLimit": True,
        })

        self._markets_loaded = False

        mode = "DRY RUN" if self.dry_run else "LIVE"
        self.logger.info(
            f"BinanceFuturesTrader initialised — mode={mode}, "
            f"position_size=${self.position_size_usdt}, leverage={self.leverage}x"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_markets(self) -> None:
        """Lazily load exchange market metadata (public endpoint)."""
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True
            self.logger.debug("Binance market metadata loaded.")

    def _to_ccxt_symbol(self, ticker: str) -> str:
        """
        Convert a raw Binance ticker (e.g. ``BTCUSDT``) to the ccxt
        unified symbol (e.g. ``BTC/USDT:USDT``).
        """
        self._ensure_markets()
        market = self.exchange.market(ticker)
        return market["symbol"]

    def _set_leverage(self, ccxt_symbol: str) -> None:
        """Set the configured leverage for *ccxt_symbol* on Binance."""
        if self.dry_run:
            self.logger.info(
                f"[DRY RUN] set_leverage {ccxt_symbol}: {self.leverage}x"
            )
            return
        self.exchange.set_leverage(self.leverage, ccxt_symbol)
        self.logger.debug(f"Leverage set to {self.leverage}x for {ccxt_symbol}.")

    def _calculate_quantity(
        self,
        ccxt_symbol: str,
        size_usdt: float,
    ) -> tuple[float, float]:
        """
        Calculate the order quantity in base asset for a given USDT notional.

        *size_usdt* is the gross notional value of the position (matching
        the Binance UI). Leverage controls the margin required but does
        not affect the notional size.

        Returns
        -------
        (quantity, price) : tuple[float, float]
        """
        ticker_data = self.exchange.fetch_ticker(ccxt_symbol)
        price = ticker_data["last"]
        if price is None or price <= 0:
            raise ValueError(f"Invalid price for {ccxt_symbol}: {price}")

        raw_qty = size_usdt / price
        qty_str = self.exchange.amount_to_precision(ccxt_symbol, raw_qty)
        qty = float(qty_str)
        if qty <= 0:
            raise ValueError(
                f"Calculated quantity rounds to 0 for {ccxt_symbol} "
                f"(size=${size_usdt}, price={price})"
            )
        return qty, float(price)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_short(
        self,
        ticker: str,
        size_usdt: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Place a market SELL (short entry) on Binance USDT-M Futures.

        Parameters
        ----------
        ticker : str
            Raw Binance symbol, e.g. ``"BTCUSDT"``.
        size_usdt : float, optional
            Override the default ``position_size_usdt`` from config.

        Returns
        -------
        dict or None
            ccxt order dict on success (key fields: ``id``, ``filled``,
            ``average``, ``status``).  ``None`` on failure.
        """
        size = size_usdt or self.position_size_usdt
        try:
            ccxt_symbol = self._to_ccxt_symbol(ticker)
            self._set_leverage(ccxt_symbol)
            qty, price = self._calculate_quantity(ccxt_symbol, size)

            if self.dry_run:
                order_id = f"DRY_{ticker}_{int(time.time() * 1000)}"
                self.logger.info(
                    f"[DRY RUN] open_short {ticker}: "
                    f"qty={qty}, price≈{price:.4f}, size=${size}, leverage={self.leverage}x"
                )
                return {
                    "id": order_id,
                    "filled": qty,
                    "average": price,
                    "status": "closed",
                }

            order = self.exchange.create_market_sell_order(ccxt_symbol, qty)
            self.logger.info(
                f"[{ticker}] SHORT OPENED — order_id={order['id']}, "
                f"filled={order.get('filled')}, avg_price={order.get('average')}"
            )
            return order

        except Exception as exc:
            self.logger.error(f"[{ticker}] Failed to open short: {exc}")
            return None

    def close_short(
        self,
        ticker: str,
        quantity: float,
    ) -> Optional[dict]:
        """
        Place a market BUY with ``reduceOnly=True`` to close an existing
        short position.

        Parameters
        ----------
        ticker : str
            Raw Binance symbol.
        quantity : float
            Base-asset quantity to buy back (must match the entry quantity).

        Returns
        -------
        dict or None
            ccxt order dict on success, ``None`` on failure.
        """
        try:
            ccxt_symbol = self._to_ccxt_symbol(ticker)

            if self.dry_run:
                order_id = f"DRY_{ticker}_{int(time.time() * 1000)}"
                self.logger.info(
                    f"[DRY RUN] close_short {ticker}: qty={quantity}"
                )
                return {
                    "id": order_id,
                    "filled": quantity,
                    "average": 0,
                    "status": "closed",
                }

            order = self.exchange.create_market_buy_order(
                ccxt_symbol,
                quantity,
                params={"reduceOnly": True},
            )
            self.logger.info(
                f"[{ticker}] SHORT CLOSED — order_id={order['id']}, "
                f"filled={order.get('filled')}, avg_price={order.get('average')}"
            )
            return order

        except Exception as exc:
            self.logger.error(f"[{ticker}] Failed to close short: {exc}")
            return None

    def get_position(self, ticker: str) -> Optional[dict]:
        """
        Query the current net Binance position for *ticker*.

        Useful for manual reconciliation.  Returns the raw ccxt position
        dict, or ``None`` if no open position or on error.
        """
        try:
            ccxt_symbol = self._to_ccxt_symbol(ticker)
            positions = self.exchange.fetch_positions([ccxt_symbol])
            for pos in positions:
                if pos["symbol"] == ccxt_symbol and float(pos.get("contracts", 0)) != 0:
                    return pos
            return None
        except Exception as exc:
            self.logger.error(f"[{ticker}] Failed to fetch position: {exc}")
            return None
