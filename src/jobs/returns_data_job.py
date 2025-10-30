"""Returns Data Job

This job calculates returns and log returns from OHLCV raw data and saves them to CSV files.
It reads the raw OHLCV data, computes simple returns and log returns, and saves the results
to the returns data folder.

The job:
1. Reads configuration from config/returns_data_job_config.json
2. For each ticker, loads the OHLCV raw data
3. Calculates simple returns and log returns from Close prices
4. Saves the returns data to CSV in the returns folder

This job should be run after the ohlcv_raw_data_job to ensure the source data is up to date.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def get_ohlcv_file_path(ticker: str, data_folder: str) -> str:
    """Generate the file path for a ticker's OHLCV data file.
    
    Args:
        ticker: Stock ticker symbol
        data_folder: OHLCV data folder path
        
    Returns:
        Full path to the OHLCV CSV file for the ticker
    """
    return os.path.join(data_folder, f"ohlcv_raw_{ticker}.csv")


def get_returns_file_path(ticker: str, data_folder: str) -> str:
    """Generate the file path for a ticker's returns data file.
    
    Args:
        ticker: Stock ticker symbol
        data_folder: Returns data folder path
        
    Returns:
        Full path to the returns CSV file for the ticker
    """
    return os.path.join(data_folder, f"returns_{ticker}.csv")


def load_ohlcv_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load OHLCV data from CSV file.
    
    Args:
        file_path: Path to the OHLCV CSV file
        
    Returns:
        DataFrame with OHLCV data or None if file doesn't exist or error occurs
    """
    if not os.path.exists(file_path):
        print(f"OHLCV file not found: {file_path}")
        return None
    
    try:
        df = load_df_from_csv(
            file_path,
            parse_dates=['Date'],
            index_col='Date'
        )
        return df
    except Exception as e:
        print(f"Error loading OHLCV data from {file_path}: {e}")
        return None


def calculate_returns(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """Calculate simple returns and log returns from price data.
    
    Args:
        df: DataFrame with OHLCV data (must have Date as index)
        price_col: Column name for price data (default: 'Close')
        
    Returns:
        DataFrame with Date, Close, Return, and LogReturn columns
    """
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    
    # Create new DataFrame with Date and Close
    returns_df = pd.DataFrame(index=df.index)
    returns_df['Close'] = df[price_col]
    
    # Calculate simple returns: (P_t / P_{t-1}) - 1
    returns_df['Return'] = returns_df['Close'].pct_change()
    
    # Calculate log returns: ln(P_t / P_{t-1})
    returns_df['LogReturn'] = np.log(returns_df['Close'] / returns_df['Close'].shift(1))
    
    # Drop the first row which will have NaN for returns
    returns_df = returns_df.dropna()
    
    return returns_df


def process_ticker_returns(
    ticker: str,
    ohlcv_folder: str,
    returns_folder: str
) -> bool:
    """Process returns for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        ohlcv_folder: Folder containing OHLCV raw data
        returns_folder: Folder to save returns data
        
    Returns:
        True if successful, False otherwise
    """
    # Load OHLCV data
    ohlcv_path = get_ohlcv_file_path(ticker, ohlcv_folder)
    ohlcv_df = load_ohlcv_data(ohlcv_path)
    
    if ohlcv_df is None or ohlcv_df.empty:
        print(f"No OHLCV data available for {ticker}")
        return False
    
    print(f"Loaded {len(ohlcv_df)} records for {ticker}")
    
    # Calculate returns
    try:
        returns_df = calculate_returns(ohlcv_df)
        print(f"Calculated {len(returns_df)} return records")
    except Exception as e:
        print(f"Error calculating returns for {ticker}: {e}")
        return False
    
    # Save to CSV
    returns_path = get_returns_file_path(ticker, returns_folder)
    try:
        # Reset index to have Date as a column
        returns_df_to_save = returns_df.reset_index()
        
        save_df_to_csv(
            returns_df_to_save,
            returns_path,
            index=False,
            create_dirs=True
        )
        print(f"Successfully saved returns data to {returns_path}")
        return True
    except Exception as e:
        print(f"Error saving returns data for {ticker}: {e}")
        return False


def run_returns_data_job(config_path: str = None) -> None:
    """Main job execution function.
    
    Args:
        config_path: Path to configuration file (defaults to config/returns_data_job_config.json)
    """
    # Determine config path
    if config_path is None:
        # Assume we're running from project root or src/jobs
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "returns_data_job_config.json"
    
    print(f"Loading configuration from {config_path}")
    config = load_config(str(config_path))
    
    # Extract configuration
    tickers = config.get('tickers', [])
    ohlcv_folder = config.get('ohlcv_data_folder', 'data/ohlcv_raw')
    returns_folder = config.get('returns_data_folder', 'data/returns')
    
    # Resolve folder paths
    project_root = Path(__file__).parent.parent.parent
    
    if not os.path.isabs(ohlcv_folder):
        ohlcv_folder = str(project_root / ohlcv_folder)
    
    if not os.path.isabs(returns_folder):
        returns_folder = str(project_root / returns_folder)
    
    print(f"\n{'='*60}")
    print(f"Returns Data Job")
    print(f"{'='*60}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"OHLCV Data Folder: {ohlcv_folder}")
    print(f"Returns Data Folder: {returns_folder}")
    print(f"{'='*60}\n")
    
    # Process each ticker
    success_count = 0
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        if process_ticker_returns(
            ticker=ticker,
            ohlcv_folder=ohlcv_folder,
            returns_folder=returns_folder
        ):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Job Complete: {success_count}/{len(tickers)} tickers processed successfully")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_returns_data_job()
