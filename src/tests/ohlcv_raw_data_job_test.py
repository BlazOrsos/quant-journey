"""Tests for OHLCV Raw Data Job

Tests cover:
- Configuration loading
- File path generation
- Date calculations (business days)
- Existing data handling
- Data download and merging logic
- Error handling
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from jobs.ohlcv_raw_data_job import (
    load_config,
    get_data_file_path,
    get_existing_data,
    get_previous_business_day,
    download_and_save_ticker_data,
    run_ohlcv_raw_datajob
)


class TestLoadConfig:
    """Test configuration loading."""
    
    def test_load_valid_config(self, tmp_path):
        """Test loading a valid configuration file."""
        config_data = {
            "tickers": ["SPY", "QQQ"],
            "start_date": "2020-01-01",
            "interval": "1d",
            "data_folder": "data"
        }
        
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        result = load_config(str(config_file))
        
        assert result == config_data
        assert result["tickers"] == ["SPY", "QQQ"]
        assert result["start_date"] == "2020-01-01"
    
    def test_load_config_file_not_found(self):
        """Test loading config with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.json")
    
    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        config_file = tmp_path / "invalid_config.json"
        with open(config_file, 'w') as f:
            f.write("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            load_config(str(config_file))


class TestGetDataFilePath:
    """Test data file path generation."""
    
    def test_get_data_file_path_basic(self):
        """Test basic file path generation."""
        result = get_data_file_path("SPY", "data")
        assert result == os.path.join("data", "ohlcv_raw_SPY.csv")
    
    def test_get_data_file_path_with_absolute_path(self):
        """Test file path generation with absolute path."""
        data_folder = os.path.abspath("test_data")
        result = get_data_file_path("QQQ", data_folder)
        expected = os.path.join(data_folder, "ohlcv_raw_QQQ.csv")
        assert result == expected
    
    def test_get_data_file_path_different_tickers(self):
        """Test that different tickers generate different paths."""
        path1 = get_data_file_path("SPY", "data")
        path2 = get_data_file_path("QQQ", "data")
        assert path1 != path2
        assert "SPY" in path1
        assert "QQQ" in path2


class TestGetExistingData:
    """Test loading existing data files."""
    
    def test_get_existing_data_file_not_exists(self):
        """Test when file doesn't exist."""
        result = get_existing_data("nonexistent_file.csv")
        assert result is None
    
    def test_get_existing_data_valid_file(self, tmp_path):
        """Test loading valid existing data."""
        # Create test CSV
        test_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=5, freq='D'),
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        test_file = tmp_path / "test_data.csv"
        test_data.to_csv(test_file, index=False)
        
        result = get_existing_data(str(test_file))
        
        assert result is not None
        assert len(result) == 5
        assert isinstance(result.index, pd.DatetimeIndex)
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
    
    @patch('jobs.ohlcv_raw_data_job.load_df_from_csv')
    def test_get_existing_data_handles_errors(self, mock_load):
        """Test error handling when loading corrupted data."""
        mock_load.side_effect = Exception("Corrupted file")
        
        # Create a temporary file that exists
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        
        try:
            result = get_existing_data(temp_file)
            assert result is None
        finally:
            os.unlink(temp_file)


class TestGetPreviousBusinessDay:
    """Test business day calculation."""
    
    def test_previous_business_day_from_monday(self):
        """Test getting previous business day from Monday."""
        # Monday, January 8, 2024
        monday = datetime(2024, 1, 8)
        result = get_previous_business_day(monday)
        # Should return Friday, January 5, 2024
        assert result == "2024-01-05"
    
    def test_previous_business_day_from_friday(self):
        """Test getting previous business day from Friday."""
        # Friday, January 5, 2024
        friday = datetime(2024, 1, 5)
        result = get_previous_business_day(friday)
        # Should return Thursday, January 4, 2024
        assert result == "2024-01-04"
    
    def test_previous_business_day_from_saturday(self):
        """Test getting previous business day from Saturday."""
        # Saturday, January 6, 2024
        saturday = datetime(2024, 1, 6)
        result = get_previous_business_day(saturday)
        # Should return Friday, January 5, 2024
        assert result == "2024-01-05"
    
    def test_previous_business_day_from_sunday(self):
        """Test getting previous business day from Sunday."""
        # Sunday, January 7, 2024
        sunday = datetime(2024, 1, 7)
        result = get_previous_business_day(sunday)
        # Should return Friday, January 5, 2024
        assert result == "2024-01-05"
    
    def test_previous_business_day_default_uses_today(self):
        """Test that default parameter uses today."""
        result = get_previous_business_day()
        assert isinstance(result, str)
        assert len(result) == 10  # YYYY-MM-DD format
        # Parse and verify it's a business day
        result_date = datetime.strptime(result, '%Y-%m-%d')
        assert result_date.weekday() < 5  # Monday=0, Friday=4


class TestDownloadAndSaveTickerData:
    """Test the main download and save function."""
    
    @patch('jobs.ohlcv_raw_data_job.download_ticker_data')
    @patch('jobs.ohlcv_raw_data_job.save_df_to_csv')
    @patch('jobs.ohlcv_raw_data_job.get_existing_data')
    def test_download_new_ticker_no_existing_data(
        self, mock_get_existing, mock_save, mock_download, tmp_path
    ):
        """Test downloading data for a ticker with no existing data."""
        # Setup mocks
        mock_get_existing.return_value = None
        
        # Mock downloaded data
        new_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=3, freq='D'),
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200]
        })
        new_data.set_index('Date', inplace=True)
        mock_download.return_value = new_data
        
        # Execute
        result = download_and_save_ticker_data(
            ticker="SPY",
            start_date="2024-01-01",
            end_date="2024-01-31",
            data_folder=str(tmp_path),
            interval="1d"
        )
        
        # Verify
        assert result is True
        mock_download.assert_called_once_with(
            ticker="SPY",
            start_date="2024-01-01",
            end_date="2024-01-31",
            interval="1d"
        )
        assert mock_save.called
    
    @patch('jobs.ohlcv_raw_data_job.download_ticker_data')
    @patch('jobs.ohlcv_raw_data_job.save_df_to_csv')
    @patch('jobs.ohlcv_raw_data_job.get_existing_data')
    @patch('jobs.ohlcv_raw_data_job.get_last_date_from_csv')
    def test_download_incremental_update(
        self, mock_get_last_date, mock_get_existing, mock_save, mock_download, tmp_path
    ):
        """Test downloading only new data when existing data is present."""
        # Setup existing data
        existing_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=3, freq='D'),
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200]
        })
        existing_data.set_index('Date', inplace=True)
        mock_get_existing.return_value = existing_data
        mock_get_last_date.return_value = datetime(2024, 1, 3)
        
        # Mock new downloaded data
        new_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-04', periods=2, freq='D'),
            'Open': [103, 104],
            'High': [108, 109],
            'Low': [98, 99],
            'Close': [105, 106],
            'Volume': [1300, 1400]
        })
        new_data.set_index('Date', inplace=True)
        mock_download.return_value = new_data
        
        # Execute
        result = download_and_save_ticker_data(
            ticker="SPY",
            start_date="2024-01-01",
            end_date="2024-01-31",
            data_folder=str(tmp_path),
            interval="1d"
        )
        
        # Verify
        assert result is True
        # Should download from day after last date
        mock_download.assert_called_once_with(
            ticker="SPY",
            start_date="2024-01-04",
            end_date="2024-01-31",
            interval="1d"
        )
        # Should save combined data
        assert mock_save.called
    
    @patch('jobs.ohlcv_raw_data_job.download_ticker_data')
    @patch('jobs.ohlcv_raw_data_job.get_existing_data')
    @patch('jobs.ohlcv_raw_data_job.get_last_date_from_csv')
    def test_download_already_up_to_date(
        self, mock_get_last_date, mock_get_existing, mock_download, tmp_path
    ):
        """Test when data is already up to date."""
        # Setup existing data with recent last date
        existing_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=3, freq='D'),
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200]
        })
        existing_data.set_index('Date', inplace=True)
        mock_get_existing.return_value = existing_data
        mock_get_last_date.return_value = datetime(2024, 1, 31)
        
        # Execute - end date is before last date
        result = download_and_save_ticker_data(
            ticker="SPY",
            start_date="2024-01-01",
            end_date="2024-01-15",
            data_folder=str(tmp_path),
            interval="1d"
        )
        
        # Verify
        assert result is True
        # Should not download anything
        mock_download.assert_not_called()
    
    @patch('jobs.ohlcv_raw_data_job.download_ticker_data')
    @patch('jobs.ohlcv_raw_data_job.get_existing_data')
    def test_download_returns_none(self, mock_get_existing, mock_download, tmp_path):
        """Test handling when download returns no data."""
        mock_get_existing.return_value = None
        mock_download.return_value = None
        
        result = download_and_save_ticker_data(
            ticker="INVALID",
            start_date="2024-01-01",
            end_date="2024-01-31",
            data_folder=str(tmp_path),
            interval="1d"
        )
        
        assert result is False
    
    @patch('jobs.ohlcv_raw_data_job.download_ticker_data')
    @patch('jobs.ohlcv_raw_data_job.get_existing_data')
    def test_download_returns_empty_dataframe(
        self, mock_get_existing, mock_download, tmp_path
    ):
        """Test handling when download returns empty DataFrame."""
        mock_get_existing.return_value = None
        mock_download.return_value = pd.DataFrame()
        
        result = download_and_save_ticker_data(
            ticker="INVALID",
            start_date="2024-01-01",
            end_date="2024-01-31",
            data_folder=str(tmp_path),
            interval="1d"
        )
        
        assert result is False
    
    @patch('jobs.ohlcv_raw_data_job.download_ticker_data')
    @patch('jobs.ohlcv_raw_data_job.save_df_to_csv')
    @patch('jobs.ohlcv_raw_data_job.get_existing_data')
    def test_download_handles_save_error(
        self, mock_get_existing, mock_save, mock_download, tmp_path
    ):
        """Test error handling when save fails."""
        mock_get_existing.return_value = None
        
        new_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=3, freq='D'),
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200]
        })
        new_data.set_index('Date', inplace=True)
        mock_download.return_value = new_data
        
        # Mock save to raise error
        mock_save.side_effect = Exception("Permission denied")
        
        result = download_and_save_ticker_data(
            ticker="SPY",
            start_date="2024-01-01",
            end_date="2024-01-31",
            data_folder=str(tmp_path),
            interval="1d"
        )
        
        assert result is False


class TestRunJob:
    """Test the main job execution function."""
    
    @patch('jobs.ohlcv_raw_data_job.download_and_save_ticker_data')
    @patch('jobs.ohlcv_raw_data_job.get_previous_business_day')
    def test_run_job_with_config(
        self, mock_get_prev_day, mock_download_save, tmp_path
    ):
        """Test running the job with a configuration file."""
        # Create config file
        config_data = {
            "tickers": ["SPY", "QQQ"],
            "start_date": "2020-01-01",
            "interval": "1d",
            "data_folder": str(tmp_path)
        }
        
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Mock
        mock_get_prev_day.return_value = "2024-01-31"
        mock_download_save.return_value = True
        
        # Execute
        run_ohlcv_raw_datajob(str(config_file))
        
        # Verify
        assert mock_download_save.call_count == 2
        mock_download_save.assert_any_call(
            ticker="SPY",
            start_date="2020-01-01",
            end_date="2024-01-31",
            data_folder=str(tmp_path),
            interval="1d"
        )
        mock_download_save.assert_any_call(
            ticker="QQQ",
            start_date="2020-01-01",
            end_date="2024-01-31",
            data_folder=str(tmp_path),
            interval="1d"
        )
    
    @patch('jobs.ohlcv_raw_data_job.download_and_save_ticker_data')
    @patch('jobs.ohlcv_raw_data_job.get_previous_business_day')
    def test_run_job_partial_success(
        self, mock_get_prev_day, mock_download_save, tmp_path
    ):
        """Test job continues when some tickers fail."""
        # Create config
        config_data = {
            "tickers": ["SPY", "INVALID", "QQQ"],
            "start_date": "2020-01-01",
            "interval": "1d",
            "data_folder": str(tmp_path)
        }
        
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Mock - middle ticker fails
        mock_get_prev_day.return_value = "2024-01-31"
        mock_download_save.side_effect = [True, False, True]
        
        # Execute
        run_ohlcv_raw_datajob(str(config_file))
        
        # Verify all tickers were attempted
        assert mock_download_save.call_count == 3
