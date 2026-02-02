from src.exchanges.base import ExchangeAdapter
from src.core.models import Candle, Funding
import ccxt
import json
from pathlib import Path
from datetime import datetime

class HyperliquidAdapter(ExchangeAdapter):
    def fetch_candles(self, symbol, venue_type, timeframe, start_ts):
        exchange = ccxt.hyperliquid()
                
        if venue_type == 'futures':
            exchange.options['defaultType'] = 'future'
        elif venue_type == 'spot':
            exchange.options['defaultType'] = 'spot'
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=start_ts, limit=1)
        
        if not ohlcv:
            return None
        
        candle_data = ohlcv[0]

        return Candle(
            exchange='hyperliquid',
            venue_type=venue_type,
            symbol=symbol,
            ts=candle_data[0],
            open=candle_data[1],
            high=candle_data[2],
            low=candle_data[3],
            close=candle_data[4],
            volume=candle_data[5]
        )

    def fetch_funding(self, symbol, start_ts):
        exchange = ccxt.hyperliquid()
        exchange.options['defaultType'] = 'future'
        
        # Fetch funding rate history
        funding_history = exchange.fetch_funding_rate_history(symbol, since=start_ts, limit=200)
        
        if not funding_history:
            return None
        
        # Convert funding history array to Funding objects
        funding_list = []
        for funding_data in funding_history:
            funding_list.append(Funding(
                exchange='hyperliquid',
                symbol=symbol,
                ts=funding_data['timestamp'],
                funding_rate=funding_data['fundingRate']
            ))
        
        return funding_list

    def fetch_symbols(self, venue_type):       
        exchange = ccxt.hyperliquid()
        markets = exchange.load_markets()
        
        symbols = []
        for symbol, market in markets.items():
            if venue_type == 'spot' and market['spot']:
                symbols.append(symbol)
            elif venue_type == 'futures' and market.get('linear') and market.get('quote') == 'USDC':
                symbols.append(symbol)
        return symbols

    def fetch_symbols_volume(self, venue_type, min_volume_usd=2000000, days=4):
        exchange = ccxt.hyperliquid()
        
        if venue_type == 'futures':
            exchange.options['defaultType'] = 'future'
        elif venue_type == 'spot':
            exchange.options['defaultType'] = 'spot'
        
        markets = exchange.load_markets()
        
        symbols = []
        for symbol, market in markets.items():
            # Filter by market type first
            if venue_type == 'spot' and not market['spot']:
                continue
            elif venue_type == 'futures' and not (market.get('linear') and market.get('quote') == 'USDC'):
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
        file_path = data_dir / f'hyperliquid_{venue_type}_symbols.json'
        with open(file_path, 'w') as f:
            json.dump({
                'symbols': symbols,
                'venue_type': venue_type
            }, f, indent=2)
        return symbols
    
    def load_symbols(self, venue_type):
        data_dir = Path(__file__).parent.parent.parent / 'data' / 'symbols'
        file_path = data_dir / f'hyperliquid_{venue_type}_symbols.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['symbols']