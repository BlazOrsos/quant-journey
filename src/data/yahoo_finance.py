import yfinance as yf
import pandas as pd
from typing import Optional

def download_ticker_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """
    Download historical price data for a ticker from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '1wk', '1mo', etc.)
    
    Returns:
        DataFrame with OHLCV data including Adj Close, or None if download fails
    """
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        
        if data.empty:
            print(f"No data found for {ticker}")
            return None
        
        # Flatten multi-level columns to remove ticker from column names
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data
    
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None
    
