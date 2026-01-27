from dataclasses import dataclass
from typing import Optional

@dataclass
class Candle:
    exchange: str
    venue_type: str  # "spot" or "perp"
    symbol: str      # canonical, e.g. "BTC/USDT"
    ts: int          # UTC timestamp (e.g. seconds since epoch, open time)
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class Funding:
    exchange: str
    symbol: str
    ts: int          # UTC timestamp
    funding_rate: float
    mark_price: Optional[float] = None

# Add Order, Fill, Position