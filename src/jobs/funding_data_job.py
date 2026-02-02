"""Funding Data Job

This job fetches historical funding rate data from crypto futures exchanges
and saves it to CSV files. It checks for existing data and only downloads
missing dates to avoid redundant downloads.

The job:
1. Reads configuration from config/funding_data_job_config.json
2. Loads symbols from the symbols data files
3. For each symbol, checks if data file exists
4. Fetches funding rate history for the last N days
5. Appends/saves to CSV in the data/funding folder

Designed to run daily via scheduled task/cron.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exchanges.binance import BinanceAdapter
from exchanges.bybit import BybitAdapter
from exchanges.hyperliquid import HyperliquidAdapter
from helpers.data_helper import save_df_to_csv, load_df_from_csv


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to config JSON file
        
    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def load_symbols_from_file(symbols_file: str) -> List[str]:
    """Load symbols from JSON file.
    
    Args:
        symbols_file: Path to symbols JSON file
        
    Returns:
        List of symbol strings
    """
    with open(symbols_file, 'r') as f:
        data = json.load(f)
    return data.get('symbols', [])


def get_data_file_path(symbol: str, exchange: str, data_folder: str) -> str:
    """Generate the file path for a symbol's funding data file.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
        exchange: Exchange name (e.g., 'binance')
        data_folder: Base data folder path
        
    Returns:
        Full path to the CSV file for the symbol
    """
    # Clean symbol for filename (replace / and : with _)
    clean_symbol = symbol.replace('/', '_').replace(':', '_')
    return os.path.join(data_folder, f"funding_{exchange}_{clean_symbol}.csv")


def get_existing_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load existing data if file exists.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with existing data or None if file doesn't exist
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        df = load_df_from_csv(
            file_path,
            parse_dates=['timestamp'],
        )
        return df
    except Exception as e:
        print(f"Error loading existing data from {file_path}: {e}")
        return None


def fetch_and_save_funding_data(
    symbol: str,
    exchange_name: str,
    exchange_adapter,
    lookback_days: int,
    data_folder: str
) -> bool:
    """Fetch and save funding data for a single symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
        exchange_name: Name of the exchange
        exchange_adapter: Exchange adapter instance
        lookback_days: Number of days to look back for historical data
        data_folder: Folder to save data files
        
    Returns:
        True if successful, False otherwise
    """
    file_path = get_data_file_path(symbol, exchange_name, data_folder)
    
    # Calculate start timestamp (lookback_days ago)
    start_dt = datetime.now() - timedelta(days=lookback_days)
    start_ts = int(start_dt.timestamp() * 1000)  # Convert to milliseconds
    
    try:
        # Fetch funding rate history
        funding_list = exchange_adapter.fetch_funding(symbol, start_ts)
        
        if not funding_list:
            print(f"  No funding data returned for {symbol}")
            return False
        
        # Convert to DataFrame
        new_data = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(f.ts, unit='ms'),
                'exchange': f.exchange,
                'symbol': f.symbol,
                'funding_rate': f.funding_rate,
                'mark_price': f.mark_price
            }
            for f in funding_list
        ])
        
        # Check for existing data
        existing_df = get_existing_data(file_path)
        
        if existing_df is not None and not existing_df.empty:
            # Merge with existing data
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
            combined_df = pd.concat([existing_df, new_data], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
            combined_df = combined_df.sort_values('timestamp')
            
            print(f"  ✓ {symbol}: {len(new_data)} new records, {len(combined_df)} total")
        else:
            combined_df = new_data.sort_values('timestamp')
            print(f"  ✓ {symbol}: {len(combined_df)} records")
        
        # Save to CSV
        save_df_to_csv(
            combined_df,
            file_path,
            index=False,
            create_dirs=True
        )
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {symbol}: {str(e)}")
        return False


def get_exchange_adapter(exchange_name: str):
    """Get the appropriate exchange adapter.
    
    Args:
        exchange_name: Name of the exchange
        
    Returns:
        Exchange adapter instance
    """
    adapters = {
        'binance': BinanceAdapter,
        'bybit': BybitAdapter,
        'hyperliquid': HyperliquidAdapter
    }
    
    adapter_class = adapters.get(exchange_name.lower())
    if not adapter_class:
        raise ValueError(f"Unsupported exchange: {exchange_name}. Supported: {list(adapters.keys())}")
    
    return adapter_class()


def run_funding_data_job(config_path: str = None) -> None:
    """Main job execution function.
    
    Args:
        config_path: Path to configuration file (defaults to config/funding_data_job_config.json)
    """
    # Determine config path
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "funding_data_job_config.json"
    
    print(f"Loading configuration from {config_path}")
    config = load_config(str(config_path))
    
    # Extract configuration
    exchanges = config.get('exchanges', ['binance'])
    lookback_days = config.get('lookback_days', 4)
    data_folder = config.get('data_folder', 'data/funding')
    
    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    
    if not os.path.isabs(data_folder):
        data_folder = str(project_root / data_folder)
    
    print(f"\n{'='*60}")
    print(f"Funding Data Job")
    print(f"{'='*60}")
    print(f"Exchanges: {', '.join(exchanges)}")
    print(f"Lookback Days: {lookback_days}")
    print(f"Data Folder: {data_folder}")
    print(f"{'='*60}\n")
    
    # Process each exchange
    total_success = 0
    total_symbols = 0
    
    for exchange_name in exchanges:
        print(f"\n{'='*60}")
        print(f"Processing Exchange: {exchange_name.upper()}")
        print(f"{'='*60}\n")
        
        # Build symbols file path
        symbols_file = project_root / 'data' / 'symbols' / f'{exchange_name}_futures_symbols.json'
        
        if not os.path.exists(symbols_file):
            print(f"  ✗ Symbols file not found: {symbols_file}")
            print(f"  Skipping {exchange_name}\n")
            continue
        
        # Load symbols
        print(f"Loading symbols from {symbols_file}")
        symbols = load_symbols_from_file(str(symbols_file))
        print(f"Loaded {len(symbols)} symbols\n")
        
        # Initialize exchange adapter
        try:
            exchange_adapter = get_exchange_adapter(exchange_name)
        except ValueError as e:
            print(f"  ✗ {e}")
            print(f"  Skipping {exchange_name}\n")
            continue
        
        # Process each symbol
        success_count = 0
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] Processing {symbol}...")
            if fetch_and_save_funding_data(
                symbol=symbol,
                exchange_name=exchange_name,
                exchange_adapter=exchange_adapter,
                lookback_days=lookback_days,
                data_folder=data_folder
            ):
                success_count += 1
        
        total_success += success_count
        total_symbols += len(symbols)
        
        print(f"\n{exchange_name.upper()} Complete: {success_count}/{len(symbols)} symbols processed successfully\n")
    
    print(f"\n{'='*60}")
    print(f"Job Complete: {total_success}/{total_symbols} total symbols processed successfully")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_funding_data_job()