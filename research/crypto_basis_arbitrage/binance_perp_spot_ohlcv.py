import pandas as pd
import ccxt
from datetime import datetime
import time
import os

# Configuration
START_DATE = "2021-01-23"
TIMEFRAME = "1d"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_tickers(filepath):
    """Load tickers from CSV file."""
    df = pd.read_csv(filepath)
    # Assuming the CSV has a column with ticker symbols
    # Adjust column name based on actual CSV structure
    if 'symbol' in df.columns:
        return df['symbol'].tolist()
    elif 'ticker' in df.columns:
        return df['ticker'].tolist()
    else:
        # Use first column if no standard name found
        return df.iloc[:, 0].tolist()

def fetch_ohlcv(exchange, symbol, timeframe, since, limit=1000):
    """Fetch OHLCV data for a symbol."""
    all_ohlcv = []
    since_ts = since
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since_ts = ohlcv[-1][0] + 1  # Move to next candle
            
            # Check if we've reached current time
            if ohlcv[-1][0] >= exchange.milliseconds() - 86400000:
                break
                
            time.sleep(exchange.rateLimit / 1000)  # Respect rate limits
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break
    
    return all_ohlcv

def ohlcv_to_dataframe(ohlcv_data, symbol):
    """Convert OHLCV list to DataFrame."""
    if not ohlcv_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['symbol'] = symbol
    df = df[['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
    return df

def main():
    # Load tickers
    tickers_file = os.path.join(OUTPUT_DIR, "binance_usdt_perps_last_5_years.csv")
    tickers = load_tickers(tickers_file)
    print(f"Loaded {len(tickers)} tickers")
    
    # Initialize exchanges
    spot_exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    perp_exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
        }
    })
    
    # Load markets
    spot_exchange.load_markets()
    perp_exchange.load_markets()
    
    # Convert start date to timestamp
    since = int(datetime.strptime(START_DATE, "%Y-%m-%d").timestamp() * 1000)
    
    spot_data = []
    perp_data = []
    
    for i, ticker in enumerate(tickers):
        # Clean ticker name (remove /USDT:USDT if present)
        base_ticker = ticker.replace('/USDT:USDT', '').replace('/USDT', '').replace(':USDT', '')
        spot_symbol = f"{base_ticker}/USDT"
        perp_symbol = f"{base_ticker}/USDT:USDT"
        
        print(f"[{i+1}/{len(tickers)}] Fetching {base_ticker}...")
        
        # Fetch SPOT data
        if spot_symbol in spot_exchange.markets:
            print(f"  Fetching SPOT {spot_symbol}...")
            ohlcv = fetch_ohlcv(spot_exchange, spot_symbol, TIMEFRAME, since)
            df = ohlcv_to_dataframe(ohlcv, spot_symbol)
            if not df.empty:
                spot_data.append(df)
                print(f"    Got {len(df)} candles")
        else:
            print(f"  SPOT {spot_symbol} not found")
        
        # Fetch PERP data
        if perp_symbol in perp_exchange.markets:
            print(f"  Fetching PERP {perp_symbol}...")
            ohlcv = fetch_ohlcv(perp_exchange, perp_symbol, TIMEFRAME, since)
            df = ohlcv_to_dataframe(ohlcv, perp_symbol)
            if not df.empty:
                perp_data.append(df)
                print(f"    Got {len(df)} candles")
        else:
            print(f"  PERP {perp_symbol} not found")
    
    # Combine and save data
    if spot_data:
        spot_df = pd.concat(spot_data, ignore_index=True)
        spot_output = os.path.join(OUTPUT_DIR, "binance_spot_ohlcv_history.csv")
        spot_df.to_csv(spot_output, index=False)
        print(f"\nSaved SPOT data: {len(spot_df)} rows to {spot_output}")
    
    if perp_data:
        perp_df = pd.concat(perp_data, ignore_index=True)
        perp_output = os.path.join(OUTPUT_DIR, "binance_perp_ohlcv_history.csv")
        perp_df.to_csv(perp_output, index=False)
        print(f"Saved PERP data: {len(perp_df)} rows to {perp_output}")

if __name__ == "__main__":
    main()