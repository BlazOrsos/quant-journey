from src.exchanges.base import ExchangeAdapter
from src.core.models import Candle, Funding
import ccxt
import json
from pathlib import Path

class BinanceAdapter(ExchangeAdapter):
    def fetch_candles(self, symbol, venue_type, timeframe, start_ts, end_ts):
        
        pass

    def fetch_funding(self, symbol, start_ts, end_ts):
        # Implement Binance-specific logic here
        pass

    def fetch_symbols(self, venue_type):       
        exchange = ccxt.binance()
        markets = exchange.load_markets()
        
        symbols = []
        for symbol, market in markets.items():
            if venue_type == 'spot' and market['spot']:
                symbols.append(symbol)
            elif venue_type == 'futures' and market.get('linear') and market.get('quote') == 'USDT':
                symbols.append(symbol)
        return symbols
    
    def save_symbols(self, venue_type):
        symbols = self.fetch_symbols(venue_type)

        # Filter out symbols with non-ASCII characters or date suffixes
        symbols = [s for s in symbols if s.isascii() and not any(c.isdigit() for c in s.split('/')[-1].split(':')[-1])]

        # Create data directory if it doesn't exist
        data_dir = Path(__file__).parent.parent.parent / 'data' / 'symbols'
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save to JSON file
        file_path = data_dir / f'binance_{venue_type}_symbols.json'
        with open(file_path, 'w') as f:
            json.dump({
                'symbols': symbols,
                'venue_type': venue_type
            }, f, indent=2)
        return symbols
    
    def load_symbols(self, venue_type):
        data_dir = Path(__file__).parent.parent.parent / 'data' / 'symbols'
        file_path = data_dir / f'binance_{venue_type}_symbols.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['symbols']
    
if __name__ == "__main__":
    adapter = BinanceAdapter()
    # Test spot symbols
    """
    spot_symbols = adapter.save_symbols('spot')
    print(f"Saved {len(spot_symbols)} spot symbols.")
    # Test futures symbols
    futures_symbols = adapter.save_symbols('futures')
    print(f"Saved {len(futures_symbols)} futures symbols.")
    """
    spot_symbols = adapter.load_symbols('spot')
    print(f"Loaded {len(spot_symbols)} spot symbols.")
    futures_symbols = adapter.load_symbols('futures')
    print(f"Loaded {len(futures_symbols)} futures symbols.")