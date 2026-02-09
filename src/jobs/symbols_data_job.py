"""Symbols Data Job

This job fetches and saves the list of available trading symbols (futures) from 
Binance, Bybit, and Hyperliquid exchanges.

The job:
1. Initializes exchange adapters for Binance, Bybit, and Hyperliquid
2. Fetches futures symbols from each exchange
3. Filters out invalid symbols (non-ASCII, date suffixes)
4. Saves symbols to JSON files in data/symbols/

Designed to run every 4 days via scheduled task/cron to keep symbols up to date.
"""

from datetime import datetime
from src.exchanges.binance import BinanceAdapter
from src.exchanges.bybit import BybitAdapter
from src.exchanges.hyperliquid import HyperliquidAdapter


def fetch_futures_symbols():
    """Fetch futures symbols from all supported exchanges.
    
    Returns:
        Dictionary with exchange names as keys and symbol lists as values
    """
    results = {}
    
    # Initialize exchange adapters
    exchanges = {
        'binance': BinanceAdapter(),
        #'bybit': BybitAdapter(),
        'hyperliquid': HyperliquidAdapter()
    }
    
    # Fetch and save symbols for each exchange
    for exchange_name, adapter in exchanges.items():
        try:
            print(f"Fetching futures symbols from {exchange_name}...")
            symbols = adapter.save_symbols('futures')
            results[exchange_name] = symbols
            print(f"  ✓ Saved {len(symbols)} symbols for {exchange_name}")
        except Exception as e:
            print(f"  ✗ Error fetching symbols from {exchange_name}: {str(e)}")
            results[exchange_name] = None
    
    return results


def main():
    """Main entry point for the symbols data job."""
    print("=" * 60)
    print("Symbols Data Job - Futures")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = fetch_futures_symbols()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    total_symbols = 0
    for exchange, symbols in results.items():
        if symbols:
            count = len(symbols)
            total_symbols += count
            print(f"  {exchange:12s}: {count:4d} symbols")
        else:
            print(f"  {exchange:12s}: FAILED")
    
    print(f"\nTotal symbols fetched: {total_symbols}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
