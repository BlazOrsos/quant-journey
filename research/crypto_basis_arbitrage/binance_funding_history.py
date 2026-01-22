import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path

def get_funding_rate_history(symbol: str, start_time: int, end_time: int, limit: int = 1000) -> list:
    """Fetch funding rate history from Binance API."""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_rates = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_time,
            "limit": limit
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_rates.extend(data)
            
            # Move start time to after the last received timestamp
            current_start = int(data[-1]["fundingTime"]) + 1
            
            # Rate limiting
            time.sleep(0.1)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {symbol}: {e}")
            time.sleep(1)
            continue
            
    return all_rates

def main():
    # Read the list of symbols
    input_file = Path(__file__).parent / "binance_usdt_perps_last_5_years.csv"
    symbols_df = pd.read_csv(input_file)
    
    # Assume the column with symbols is named 'symbol' or is the first column
    if 'symbol' in symbols_df.columns:
        symbols = symbols_df['symbol'].tolist()
    else:
        symbols = symbols_df.iloc[:, 0].tolist()
    
    # Calculate time range (last 5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    start_time = int(start_date.timestamp() * 1000)
    end_time = int(end_date.timestamp() * 1000)
    
    print(f"Fetching funding rates from {start_date.date()} to {end_date.date()}")
    print(f"Total symbols to process: {len(symbols)}")
    
    all_funding_data = []
    
    for i, symbol in enumerate(symbols):
        print(f"Processing {i+1}/{len(symbols)}: {symbol}")
        
        rates = get_funding_rate_history(symbol, start_time, end_time)
        
        for rate in rates:
            # Handle empty strings or missing values for markPrice
            mark_price = rate.get("markPrice", 0)
            mark_price = float(mark_price) if mark_price else 0.0
            
            all_funding_data.append({
                "symbol": symbol,
                "funding_time": pd.to_datetime(rate["fundingTime"], unit="ms"),
                "funding_rate": float(rate["fundingRate"]),
                "mark_price": mark_price
            })
        
        print(f"  Retrieved {len(rates)} funding rate records")
    
    # Create DataFrame and save
    df = pd.DataFrame(all_funding_data)
    
    if not df.empty:
        df = df.sort_values(["symbol", "funding_time"]).reset_index(drop=True)
        
        output_file = Path(__file__).parent / "binance_funding_rate_history.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(df)} total records to {output_file}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Symbols processed: {df['symbol'].nunique()}")
        print(f"  Date range: {df['funding_time'].min()} to {df['funding_time'].max()}")
        print(f"  Total funding rate records: {len(df)}")
    else:
        print("No data retrieved")

if __name__ == "__main__":
    main()