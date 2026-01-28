from abc import ABC, abstractmethod
from typing import List
from src.core.models import Candle, Funding

class ExchangeAdapter(ABC):

    # Market Data Methods
    @abstractmethod
    def fetch_candles(self, symbol: str, venue_type: str, timeframe: str, start_ts: int, end_ts: int) -> List[Candle]:
        pass

    @abstractmethod
    def fetch_funding(self, symbol: str, start_ts: int, end_ts: int) -> List[Funding]:
        pass

    @abstractmethod
    def fetch_symbols(self, venue_type: str) -> List[str]:
        pass

    @abstractmethod
    def save_symbols(self, venue_type: str) -> List[str]:
        pass

    @abstractmethod
    def load_symbols(self, venue_type: str) -> List[str]:
        pass

    # Trading Methods