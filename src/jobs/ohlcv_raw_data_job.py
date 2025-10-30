"""OHLCV Raw Data Download Job

This job downloads historical OHLCV data from Yahoo Finance and saves it to CSV files.
It checks for existing data and only downloads missing dates to avoid redundant downloads.

The job:
1. Reads configuration from config/data_config.json
2. For each ticker, checks if data file exists
3. Identifies missing dates using business day frequency
4. Downloads only missing data
5. Appends/saves to CSV in the data folder

Designed to run daily via scheduled task/cron.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.yahoo_finance import download_ticker_data
from helpers.data_helper import save_df_to_csv, load_df_from_csv, get_last_date_from_csv
from helpers.data_helper import get_last_date_from_csv


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to config JSON file
        
    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def get_data_file_path(ticker: str, data_folder: str) -> str:
    """Generate the file path for a ticker's data file.
    
    Args:
        ticker: Stock ticker symbol
        data_folder: Base data folder path
        
    Returns:
        Full path to the CSV file for the ticker
    """
    return os.path.join(data_folder, f"ohlcv_raw_{ticker}.csv")


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
            parse_dates=['Date'],
            index_col='Date'
        )
        return df
    except Exception as e:
        print(f"Error loading existing data from {file_path}: {e}")
        return None


def get_previous_business_day(reference_date: Optional[datetime] = None) -> str:
    """Get the previous business day (Monday-Friday).
    
    Args:
        reference_date: Reference date (defaults to today)
        
    Returns:
        Previous business day in 'YYYY-MM-DD' format
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    # Go back one day
    prev_day = reference_date - timedelta(days=1)
    
    # If it's Sunday (6) or Saturday (5), go back to Friday
    while prev_day.weekday() >= 5:
        prev_day -= timedelta(days=1)
    
    return prev_day.strftime('%Y-%m-%d')


def download_and_save_ticker_data(
    ticker: str,
    start_date: str,
    end_date: str,
    data_folder: str,
    interval: str = "1d"
) -> bool:
    """Download and save data for a single ticker, handling existing data.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for download (YYYY-MM-DD)
        end_date: End date for download (YYYY-MM-DD)
        data_folder: Folder to save data files
        interval: Data interval (default: '1d')
        
    Returns:
        True if successful, False otherwise
    """
    file_path = get_data_file_path(ticker, data_folder)
    
    # Check for existing data
    existing_df = get_existing_data(file_path)
    
    if existing_df is not None and not existing_df.empty:
        print(f"Found existing data for {ticker} with {len(existing_df)} records")
        
        # Get the last date from existing data
        last_date = get_last_date_from_csv(file_path, date_col='Date')
        
        if last_date is None:
            print(f"Could not determine last date for {ticker}. Downloading full history.")
            download_start = start_date
            download_end = end_date
        else:
            # Download from day after last date to end_date
            next_day = last_date + timedelta(days=1)
            download_start = next_day.strftime('%Y-%m-%d')
            download_end = end_date
            
            # Check if we're already up to date
            if next_day.strftime('%Y-%m-%d') >= end_date:
                print(f"Data for {ticker} is already up to date (last date: {last_date.strftime('%Y-%m-%d')}).")
                return True
            
            print(f"Downloading {ticker} from {download_start} to {download_end}")
    else:
        print(f"No existing data for {ticker}. Downloading full history.")
        download_start = start_date
        download_end = end_date
    
    # Download the data
    new_data = download_ticker_data(
        ticker=ticker,
        start_date=download_start,
        end_date=download_end,
        interval=interval
    )
    
    if new_data is None or new_data.empty:
        print(f"No new data downloaded for {ticker}")
        return False
    
    # Reset index to have Date as a column
    new_data.reset_index(inplace=True)
    
    # Ensure Date column exists and is datetime
    if 'Date' not in new_data.columns:
        print(f"Error: Downloaded data for {ticker} doesn't have Date column")
        return False
    
    new_data['Date'] = pd.to_datetime(new_data['Date'])
    
    # Merge with existing data if it exists
    if existing_df is not None and not existing_df.empty:
        # Reset index of existing data
        existing_df_reset = existing_df.reset_index()
        existing_df_reset['Date'] = pd.to_datetime(existing_df_reset['Date'])
        
        # Concatenate and remove duplicates, keeping the latest data
        combined_df = pd.concat([existing_df_reset, new_data], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
        combined_df = combined_df.sort_values('Date')
        
        print(f"Combined data: {len(combined_df)} total records ({len(new_data)} new)")
    else:
        combined_df = new_data.sort_values('Date')
        print(f"Saved {len(combined_df)} records")
    
    # Save to CSV
    try:
        save_df_to_csv(
            combined_df,
            file_path,
            index=False,
            create_dirs=True
        )
        print(f"Successfully saved data for {ticker} to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving data for {ticker}: {e}")
        return False


def run_ohlcv_raw_datajob(config_path: str = None) -> None:
    """Main job execution function.
    
    Args:
        config_path: Path to configuration file (defaults to config/olhcv_raw_data_job_config.json)
    """
    # Determine config path
    if config_path is None:
        # Assume we're running from project root or src/jobs
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "olhcv_raw_data_job_config.json"
    
    print(f"Loading configuration from {config_path}")
    config = load_config(str(config_path))
    
    # Extract configuration
    tickers = config.get('tickers', [])
    start_date = config.get('start_date', '2015-01-01')
    interval = config.get('interval', '1d')
    data_folder = config.get('data_folder', 'data')
    
    # Resolve data folder path
    if not os.path.isabs(data_folder):
        project_root = Path(__file__).parent.parent.parent
        data_folder = str(project_root / data_folder)
    
    # Calculate end date (previous business day)
    end_date = get_previous_business_day()
    
    print(f"\n{'='*60}")
    print(f"OHLCV Data Download Job")
    print(f"{'='*60}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Interval: {interval}")
    print(f"Data Folder: {data_folder}")
    print(f"{'='*60}\n")
    
    # Process each ticker
    success_count = 0
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        if download_and_save_ticker_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            data_folder=data_folder,
            interval=interval
        ):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Job Complete: {success_count}/{len(tickers)} tickers processed successfully")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_ohlcv_raw_datajob()
