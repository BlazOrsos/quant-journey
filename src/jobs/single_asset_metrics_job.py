"""Single Asset Metrics Job

This job calculates performance metrics for individual assets across multiple timeframes.
It reads returns data, computes metrics (annualized return, volatility, Sharpe, Sortino, 
CAGR, max drawdown) for various lookback periods, and saves the results to CSV files.

The job:
1. Reads configuration from config/single_asset_metrics_job_config.json
2. For each ticker, loads the returns data
3. Calculates metrics for multiple timeframes (1Y, 2Y, 3Y, 5Y, Full Period)
4. Saves the metrics to CSV in the metrics folder

This job should be run after the returns_data_job to ensure returns data is up to date.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.data_helper import save_df_to_csv, load_df_from_csv
from helpers.financial_helper import calculate_all_metrics


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to config JSON file
        
    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def get_returns_file_path(ticker: str, data_folder: str) -> str:
    """Generate the file path for a ticker's returns data file.
    
    Args:
        ticker: Stock ticker symbol
        data_folder: Returns data folder path
        
    Returns:
        Full path to the returns CSV file for the ticker
    """
    return os.path.join(data_folder, f"returns_{ticker}.csv")


def get_metrics_file_path(ticker: str, data_folder: str) -> str:
    """Generate the file path for a ticker's metrics data file.
    
    Args:
        ticker: Stock ticker symbol
        data_folder: Metrics data folder path
        
    Returns:
        Full path to the metrics CSV file for the ticker
    """
    return os.path.join(data_folder, f"metrics_{ticker}.csv")


def load_returns_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load returns data from CSV file.
    
    Args:
        file_path: Path to the returns CSV file
        
    Returns:
        DataFrame with returns data or None if file doesn't exist or error occurs
    """
    if not os.path.exists(file_path):
        print(f"Returns file not found: {file_path}")
        return None
    
    try:
        df = load_df_from_csv(
            file_path,
            parse_dates=['Date'],
            index_col='Date'
        )
        return df
    except Exception as e:
        print(f"Error loading returns data from {file_path}: {e}")
        return None


def get_returns_for_timeframe(
    returns_df: pd.DataFrame,
    years: Optional[int] = None,
    end_date: Optional[datetime] = None
) -> pd.Series:
    """Extract returns for a specific timeframe.
    
    Args:
        returns_df: DataFrame with returns data (Date index, Return column)
        years: Number of years to look back (None for full period)
        end_date: End date for the timeframe (defaults to last date in data)
        
    Returns:
        Series of returns for the specified timeframe
    """
    if end_date is None:
        end_date = returns_df.index.max()
    
    if years is None:
        # Return all data up to end_date
        filtered_df = returns_df[returns_df.index <= end_date]
    else:
        # Calculate start date
        start_date = end_date - timedelta(days=years * 365)
        filtered_df = returns_df[(returns_df.index >= start_date) & (returns_df.index <= end_date)]
    
    return filtered_df['Return']


def calculate_metrics_for_ticker(
    ticker: str,
    returns_folder: str,
    timeframes: List[Dict],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Optional[pd.DataFrame]:
    """Calculate metrics for a ticker across multiple timeframes.
    
    Args:
        ticker: Stock ticker symbol
        returns_folder: Folder containing returns data
        timeframes: List of timeframe configurations (e.g., [{'name': '1Y', 'years': 1}, ...])
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)
        
    Returns:
        DataFrame with metrics for each timeframe, or None if error occurs
    """
    # Load returns data
    returns_path = get_returns_file_path(ticker, returns_folder)
    returns_df = load_returns_data(returns_path)
    
    if returns_df is None or returns_df.empty:
        print(f"No returns data available for {ticker}")
        return None
    
    print(f"Loaded {len(returns_df)} return records for {ticker}")
    
    # Calculate metrics for each timeframe
    metrics_list = []
    
    for tf in timeframes:
        tf_name = tf['name']
        tf_years = tf.get('years', None)
        
        print(f"  Calculating metrics for {tf_name}...")
        
        # Get returns for this timeframe
        returns_series = get_returns_for_timeframe(returns_df, tf_years)
        
        if len(returns_series) == 0:
            print(f"    No data available for {tf_name}")
            continue
        
        # Calculate all metrics
        metrics = calculate_all_metrics(
            returns_series,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year
        )
        
        # Add timeframe name
        metrics['Timeframe'] = tf_name
        metrics_list.append(metrics)
    
    if not metrics_list:
        print(f"No metrics calculated for {ticker}")
        return None
    
    # Create DataFrame with Timeframe as first column
    metrics_df = pd.DataFrame(metrics_list)
    
    # Reorder columns to put Timeframe first
    cols = ['Timeframe'] + [col for col in metrics_df.columns if col != 'Timeframe']
    metrics_df = metrics_df[cols]
    
    return metrics_df


def process_ticker_metrics(
    ticker: str,
    returns_folder: str,
    metrics_folder: str,
    timeframes: List[Dict],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> bool:
    """Process and save metrics for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        returns_folder: Folder containing returns data
        metrics_folder: Folder to save metrics data
        timeframes: List of timeframe configurations
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        True if successful, False otherwise
    """
    # Calculate metrics
    metrics_df = calculate_metrics_for_ticker(
        ticker=ticker,
        returns_folder=returns_folder,
        timeframes=timeframes,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year
    )
    
    if metrics_df is None:
        return False
    
    # Save to CSV
    metrics_path = get_metrics_file_path(ticker, metrics_folder)
    try:
        save_df_to_csv(
            metrics_df,
            metrics_path,
            index=False,
            create_dirs=True
        )
        print(f"Successfully saved metrics to {metrics_path}")
        return True
    except Exception as e:
        print(f"Error saving metrics for {ticker}: {e}")
        return False


def run_single_asset_metrics_job(config_path: str = None) -> None:
    """Main job execution function.
    
    Args:
        config_path: Path to configuration file (defaults to config/single_asset_metrics_job_config.json)
    """
    # Determine config path
    if config_path is None:
        # Assume we're running from project root or src/jobs
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "single_asset_metrics_job_config.json"
    
    print(f"Loading configuration from {config_path}")
    config = load_config(str(config_path))
    
    # Extract configuration
    tickers = config.get('tickers', [])
    returns_folder = config.get('returns_data_folder', 'data/returns')
    metrics_folder = config.get('metrics_data_folder', 'data/metrics')
    timeframes = config.get('timeframes', [
        {'name': '1Y', 'years': 1},
        {'name': '2Y', 'years': 2},
        {'name': '3Y', 'years': 3},
        {'name': '5Y', 'years': 5},
        {'name': 'Full Period', 'years': None}
    ])
    risk_free_rate = config.get('risk_free_rate', 0.0)
    periods_per_year = config.get('periods_per_year', 252)
    
    # Resolve folder paths
    project_root = Path(__file__).parent.parent.parent
    
    if not os.path.isabs(returns_folder):
        returns_folder = str(project_root / returns_folder)
    
    if not os.path.isabs(metrics_folder):
        metrics_folder = str(project_root / metrics_folder)
    
    print(f"\n{'='*60}")
    print(f"Single Asset Metrics Job")
    print(f"{'='*60}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Returns Data Folder: {returns_folder}")
    print(f"Metrics Data Folder: {metrics_folder}")
    print(f"Timeframes: {', '.join([tf['name'] for tf in timeframes])}")
    print(f"Risk-Free Rate: {risk_free_rate}")
    print(f"Periods per Year: {periods_per_year}")
    print(f"{'='*60}\n")
    
    # Process each ticker
    success_count = 0
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        if process_ticker_metrics(
            ticker=ticker,
            returns_folder=returns_folder,
            metrics_folder=metrics_folder,
            timeframes=timeframes,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year
        ):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Job Complete: {success_count}/{len(tickers)} tickers processed successfully")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_single_asset_metrics_job()
 