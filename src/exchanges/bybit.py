from src.exchanges.base import ExchangeAdapter
from src.core.models import Candle, Funding
import ccxt
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

class BybitAdapter(ExchangeAdapter):
    def fetch_funding(self, symbol, start_ts):
        exchange = ccxt.bybit()
        exchange.options['defaultType'] = 'swap'  # Use 'swap' instead of 'future'
        
        # Verify the market is linear/inverse
        markets = exchange.load_markets()
        market = markets.get(symbol)
        
        if not market or not (market.get('linear') or market.get('inverse')):
            raise ValueError(f"{symbol} is not a linear or inverse perpetual contract")
        
        # Fetch funding rate history
        funding_history = exchange.fetch_funding_rate_history(symbol, since=start_ts, limit=200)
    
        if not funding_history:
            return None
        
        # Convert funding history array to Funding objects
        funding_list = []
        for funding_data in funding_history:
            funding_list.append(Funding(
                exchange='bybit',
                symbol=symbol,
                ts=funding_data['timestamp'],
                funding_rate=funding_data['fundingRate']
            ))
        
        return funding_list

    def fetch_symbols(self, venue_type):       
        exchange = ccxt.bybit()
        markets = exchange.load_markets()
        
        symbols = []
        for symbol, market in markets.items():
            if venue_type == 'spot' and market['spot']:
                symbols.append(symbol)
            elif venue_type == 'futures' and market.get('linear') and market.get('quote') == 'USDT':
                symbols.append(symbol)
        return symbols
    
    def fetch_symbols_volume(self, venue_type, min_volume_usd=2000000, days=4):
        exchange = ccxt.bybit()
        
        if venue_type == 'futures':
            exchange.options['defaultType'] = 'future'
        elif venue_type == 'spot':
            exchange.options['defaultType'] = 'spot'
        
        markets = exchange.load_markets()
        
        # Load symbols with active signals
        active_signal_symbols = self._get_active_signal_symbols()
        
        symbols = []
        for symbol, market in markets.items():
            # Filter by market type first
            if venue_type == 'spot' and not market['spot']:
                continue
            elif venue_type == 'futures' and not (market.get('linear') and market.get('quote') == 'USDT'):
                continue
            
            # Always include symbols with active signals
            if symbol in active_signal_symbols:
                symbols.append(symbol)
                continue
            
            try:
                # Fetch last 4 daily candles
                ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=days)
                
                if not ohlcv or len(ohlcv) < days:
                    continue
                
                # Calculate average quote volume in USD
                avg_quote_volume = sum(candle[5] * candle[4] for candle in ohlcv) / len(ohlcv)
                
                if avg_quote_volume >= min_volume_usd:
                    symbols.append(symbol)
            except Exception:
                # Skip symbols that fail to fetch
                continue
        
        return symbols
    
    def save_symbols(self, venue_type):
        symbols = self.fetch_symbols_volume(venue_type)

        # Filter out symbols with non-ASCII characters or date suffixes
        symbols = [s for s in symbols if s.isascii() and not any(c.isdigit() for c in s.split('/')[-1].split(':')[-1])]

        # Create data directory if it doesn't exist
        data_dir = Path(__file__).parent.parent.parent / 'data' / 'symbols'
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save to JSON file
        file_path = data_dir / f'bybit_{venue_type}_symbols.json'
        with open(file_path, 'w') as f:
            json.dump({
                'symbols': symbols,
                'venue_type': venue_type
            }, f, indent=2)
        return symbols

    def _get_active_signal_symbols(self):
        """Get symbols with SIGNAL=1 from the latest basis_arb_signals file"""
        signals_dir = Path(__file__).parent.parent.parent / 'data' / 'signals'
        
        if not signals_dir.exists():
            return set()
        
        # Find the latest basis_arb_signals file
        signal_files = sorted(signals_dir.glob('basis_arb_signals_*.csv'), reverse=True)
        
        if not signal_files:
            return set()
        
        latest_file = signal_files[0]
        
        try:
            df = pd.read_csv(latest_file)
            # Filter rows where SIGNAL == 1 and extract unique symbols
            active_symbols = set(df[df['SIGNAL'] == 1]['SYMBOL'].unique())
            return active_symbols
        except Exception:
            return set()