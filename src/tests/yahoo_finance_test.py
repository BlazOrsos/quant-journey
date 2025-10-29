import pytest
from src.data.yahoo_finance import download_ticker_data

def test_download_ticker_data_valid():
	df = download_ticker_data("AAPL", "2023-01-01", "2023-01-10")
	assert df is not None
	assert not df.empty
	assert "Close" in df.columns
	assert "Open" in df.columns
	assert "High" in df.columns
	assert "Low" in df.columns
	assert "Volume" in df.columns

def test_download_ticker_data_invalid_ticker():
	df = download_ticker_data("INVALIDTICKER123", "2023-01-01", "2023-01-10")
	assert df is None

def test_download_ticker_data_empty_range():
	df = download_ticker_data("AAPL", "2023-01-01", "2023-01-01")
	assert df is None or df.empty
