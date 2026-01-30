from datetime import datetime, timezone
from src.exchanges.bybit import BybitAdapter

def test_symbols():
    """Test saving and loading symbols"""
    print("\n=== Testing Symbols ===")
    adapter = BybitAdapter()
    
    # Test spot symbols
    spot_symbols = adapter.save_symbols('spot')
    print(f"✓ Saved {len(spot_symbols)} spot symbols")
    assert len(spot_symbols) > 0, "No spot symbols saved"
    
    # Test futures symbols
    futures_symbols = adapter.save_symbols('futures')
    print(f"✓ Saved {len(futures_symbols)} futures symbols")
    assert len(futures_symbols) > 0, "No futures symbols saved"
    
    # Test loading
    loaded_spot = adapter.load_symbols('spot')
    print(f"✓ Loaded {len(loaded_spot)} spot symbols")
    assert len(loaded_spot) == len(spot_symbols), "Loaded spot symbols don't match saved"
    
    loaded_futures = adapter.load_symbols('futures')
    print(f"✓ Loaded {len(loaded_futures)} futures symbols")
    assert len(loaded_futures) == len(futures_symbols), "Loaded futures symbols don't match saved"

def test_funding():
    """Test fetching funding rates"""
    print("\n=== Testing Funding Rates ===")
    adapter = BybitAdapter()
    
    funding_date = datetime(2026, 1, 27, 0, 0, 0)
    funding_ts = int(funding_date.timestamp() * 1000)
    funding = adapter.fetch_funding('BTC/USDT:USDT', funding_ts)
    
    print(f"✓ Fetched {len(funding)} funding rate entries")
    assert len(funding) > 0, "No funding rates fetched"
    
    # Print first few entries
    for i, f in enumerate(funding[:5]):
        timestamp = datetime.fromtimestamp(f.ts / 1000, tz=timezone.utc)
        print(f"  {timestamp}: {f}")

if __name__ == "__main__":
    try:
        test_symbols()
        test_funding()
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")