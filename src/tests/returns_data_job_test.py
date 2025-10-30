"""Tests for Returns Data Job

Tests cover:
- Configuration loading
- File path generation
- OHLCV data loading
- Returns calculation (simple and log returns)
- Data processing and saving
- Error handling
- End-to-end job execution
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from jobs.returns_data_job import (
    load_config,
    get_ohlcv_file_path,
    get_returns_file_path,
    load_ohlcv_data,
    calculate_returns,
    process_ticker_returns,
    run_returns_data_job
)


class TestLoadConfig:
    """Test configuration loading."""
    
    def test_load_valid_config(self, tmp_path):
        """Test loading a valid configuration file."""
        config_data = {
            "tickers": ["SPY", "GLD"],
            "ohlcv_data_folder": "data/ohlcv_raw",
            "returns_data_folder": "data/returns"
        }
        
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        result = load_config(str(config_file))
        
        assert result == config_data
        assert result["tickers"] == ["SPY", "GLD"]
        assert result["ohlcv_data_folder"] == "data/ohlcv_raw"
        assert result["returns_data_folder"] == "data/returns"
    
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


class TestFilePathGeneration:
    """Test file path generation functions."""
    
    def test_get_ohlcv_file_path_basic(self):
        """Test OHLCV file path generation."""
        result = get_ohlcv_file_path("SPY", "data/ohlcv_raw")
        expected = os.path.join("data/ohlcv_raw", "ohlcv_raw_SPY.csv")
        assert result == expected
    
    def test_get_ohlcv_file_path_different_tickers(self):
        """Test that different tickers generate different OHLCV paths."""
        path1 = get_ohlcv_file_path("SPY", "data")
        path2 = get_ohlcv_file_path("GLD", "data")
        assert path1 != path2
        assert "SPY" in path1
        assert "GLD" in path2
    
    def test_get_returns_file_path_basic(self):
        """Test returns file path generation."""
        result = get_returns_file_path("SPY", "data/returns")
        expected = os.path.join("data/returns", "returns_SPY.csv")
        assert result == expected
    
    def test_get_returns_file_path_different_tickers(self):
        """Test that different tickers generate different returns paths."""
        path1 = get_returns_file_path("SPY", "data")
        path2 = get_returns_file_path("GLD", "data")
        assert path1 != path2
        assert "SPY" in path1
        assert "GLD" in path2
    
    def test_file_paths_with_absolute_path(self):
        """Test file path generation with absolute paths."""
        abs_path = os.path.abspath("test_data")
        ohlcv_path = get_ohlcv_file_path("SPY", abs_path)
        returns_path = get_returns_file_path("SPY", abs_path)
        
        assert abs_path in ohlcv_path
        assert abs_path in returns_path


class TestLoadOhlcvData:
    """Test OHLCV data loading functionality."""
    
    def test_load_ohlcv_data_file_not_exists(self):
        """Test loading OHLCV data when file doesn't exist."""
        result = load_ohlcv_data("nonexistent_file.csv")
        assert result is None
    
    def test_load_ohlcv_data_valid_file(self, tmp_path):
        """Test loading valid OHLCV data."""
        # Create test OHLCV CSV
        test_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=5, freq='D'),
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        test_file = tmp_path / "ohlcv_test.csv"
        test_data.to_csv(test_file, index=False)
        
        result = load_ohlcv_data(str(test_file))
        
        assert result is not None
        assert len(result) == 5
        assert isinstance(result.index, pd.DatetimeIndex)
        assert 'Close' in result.columns
        assert 'Open' in result.columns
        assert 'High' in result.columns
        assert 'Low' in result.columns
        assert 'Volume' in result.columns
    
    @patch('jobs.returns_data_job.load_df_from_csv')
    def test_load_ohlcv_data_handles_errors(self, mock_load):
        """Test error handling when loading corrupted OHLCV data."""
        mock_load.side_effect = Exception("Corrupted file")
        
        # Create a temporary file that exists
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            result = load_ohlcv_data(temp_file)
            assert result is None
        finally:
            os.unlink(temp_file)
    
    def test_load_ohlcv_data_with_index(self, tmp_path):
        """Test that loaded data has Date as index."""
        test_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=3, freq='D'),
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [95.0, 96.0, 97.0],
            'Close': [102.0, 103.0, 104.0],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        test_file = tmp_path / "ohlcv_indexed.csv"
        test_data.to_csv(test_file, index=False)
        
        result = load_ohlcv_data(str(test_file))
        
        assert result.index.name == 'Date'
        assert isinstance(result.index, pd.DatetimeIndex)


class TestCalculateReturns:
    """Test returns calculation functionality."""
    
    def test_calculate_returns_basic(self):
        """Test basic returns calculation."""
        # Create sample price data
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [100.0, 110.0, 105.0, 115.0, 120.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)
        df.index.name = 'Date'
        
        result = calculate_returns(df)
        
        # Verify structure
        assert len(result) == 4  # First row dropped due to NaN
        assert 'Close' in result.columns
        assert 'Return' in result.columns
        assert 'LogReturn' in result.columns
        assert isinstance(result.index, pd.DatetimeIndex)
        
        # Verify calculations
        # Day 2: (110 / 100) - 1 = 0.1
        assert result.iloc[0]['Return'] == pytest.approx(0.1, abs=1e-6)
        # Day 2: ln(110 / 100) = ln(1.1)
        assert result.iloc[0]['LogReturn'] == pytest.approx(np.log(1.1), abs=1e-6)
        
        # Day 3: (105 / 110) - 1 ≈ -0.0454545
        assert result.iloc[1]['Return'] == pytest.approx(-0.0454545, abs=1e-6)
        # Day 3: ln(105 / 110)
        assert result.iloc[1]['LogReturn'] == pytest.approx(np.log(105/110), abs=1e-6)
    
    def test_calculate_returns_no_nan_in_output(self):
        """Test that output has no NaN values."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 10)
        }, index=dates)
        
        result = calculate_returns(df)
        
        # Should have no NaN values
        assert not result['Return'].isna().any()
        assert not result['LogReturn'].isna().any()
    
    def test_calculate_returns_missing_price_column(self):
        """Test error handling when price column is missing."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0]
        }, index=dates)
        
        with pytest.raises(ValueError, match="Price column 'Close' not found"):
            calculate_returns(df)
    
    def test_calculate_returns_mathematical_properties(self):
        """Test mathematical properties of returns."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        # Generate random prices
        prices = np.random.uniform(50, 150, 100)
        df = pd.DataFrame({'Close': prices}, index=dates)
        
        result = calculate_returns(df)
        
        # Property 1: For small returns, simple return ≈ log return
        small_returns_mask = result['Return'].abs() < 0.05
        if small_returns_mask.any():
            simple_small = result.loc[small_returns_mask, 'Return']
            log_small = result.loc[small_returns_mask, 'LogReturn']
            # Should be very close for small returns
            assert np.allclose(simple_small, log_small, atol=0.01)
        
        # Property 2: Log return should be less than simple return for positive returns
        positive_mask = result['Return'] > 0
        if positive_mask.any():
            assert (result.loc[positive_mask, 'LogReturn'] < 
                    result.loc[positive_mask, 'Return']).all()
    
    def test_calculate_returns_preserves_index(self):
        """Test that date index is preserved in output."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({'Close': [100.0, 110.0, 105.0, 115.0, 120.0]}, index=dates)
        df.index.name = 'Date'
        
        result = calculate_returns(df)
        
        # Index should be DatetimeIndex (minus first row)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == 'Date'
        assert len(result) == len(df) - 1


class TestProcessTickerReturns:
    """Test processing returns for a single ticker."""
    
    @patch('jobs.returns_data_job.save_df_to_csv')
    @patch('jobs.returns_data_job.load_ohlcv_data')
    def test_process_ticker_returns_success(self, mock_load, mock_save, tmp_path):
        """Test successful processing of ticker returns."""
        # Mock OHLCV data
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        ohlcv_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [100.0, 110.0, 105.0, 115.0, 120.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)
        ohlcv_data.index.name = 'Date'
        mock_load.return_value = ohlcv_data
        
        # Execute
        result = process_ticker_returns(
            ticker="SPY",
            ohlcv_folder=str(tmp_path / "ohlcv"),
            returns_folder=str(tmp_path / "returns")
        )
        
        # Verify
        assert result is True
        assert mock_load.called
        assert mock_save.called
        
        # Check that save was called with correct structure
        saved_df = mock_save.call_args[0][0]
        assert 'Date' in saved_df.columns
        assert 'Close' in saved_df.columns
        assert 'Return' in saved_df.columns
        assert 'LogReturn' in saved_df.columns
        assert len(saved_df) == 4  # 5 prices -> 4 returns
    
    @patch('jobs.returns_data_job.load_ohlcv_data')
    def test_process_ticker_returns_no_ohlcv_data(self, mock_load, tmp_path):
        """Test processing when OHLCV data doesn't exist."""
        mock_load.return_value = None
        
        result = process_ticker_returns(
            ticker="INVALID",
            ohlcv_folder=str(tmp_path / "ohlcv"),
            returns_folder=str(tmp_path / "returns")
        )
        
        assert result is False
    
    @patch('jobs.returns_data_job.load_ohlcv_data')
    def test_process_ticker_returns_empty_ohlcv_data(self, mock_load, tmp_path):
        """Test processing when OHLCV data is empty."""
        mock_load.return_value = pd.DataFrame()
        
        result = process_ticker_returns(
            ticker="INVALID",
            ohlcv_folder=str(tmp_path / "ohlcv"),
            returns_folder=str(tmp_path / "returns")
        )
        
        assert result is False
    
    @patch('jobs.returns_data_job.save_df_to_csv')
    @patch('jobs.returns_data_job.load_ohlcv_data')
    def test_process_ticker_returns_calculation_error(self, mock_load, mock_save, tmp_path):
        """Test error handling during returns calculation."""
        # Mock OHLCV data without Close column
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        ohlcv_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0]
        }, index=dates)
        mock_load.return_value = ohlcv_data
        
        result = process_ticker_returns(
            ticker="SPY",
            ohlcv_folder=str(tmp_path / "ohlcv"),
            returns_folder=str(tmp_path / "returns")
        )
        
        assert result is False
        assert not mock_save.called
    
    @patch('jobs.returns_data_job.save_df_to_csv')
    @patch('jobs.returns_data_job.load_ohlcv_data')
    def test_process_ticker_returns_save_error(self, mock_load, mock_save, tmp_path):
        """Test error handling when save fails."""
        # Mock valid OHLCV data
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        ohlcv_data = pd.DataFrame({
            'Close': [100.0, 110.0, 105.0, 115.0, 120.0]
        }, index=dates)
        mock_load.return_value = ohlcv_data
        
        # Mock save to raise error
        mock_save.side_effect = Exception("Permission denied")
        
        result = process_ticker_returns(
            ticker="SPY",
            ohlcv_folder=str(tmp_path / "ohlcv"),
            returns_folder=str(tmp_path / "returns")
        )
        
        assert result is False
    
    def test_process_ticker_returns_integration(self, tmp_path):
        """Integration test with real files."""
        # Create OHLCV data file
        ohlcv_folder = tmp_path / "ohlcv"
        returns_folder = tmp_path / "returns"
        ohlcv_folder.mkdir()
        
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        ohlcv_data = pd.DataFrame({
            'Date': dates,
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [100.0, 110.0, 105.0, 115.0, 120.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        ohlcv_file = ohlcv_folder / "ohlcv_raw_SPY.csv"
        ohlcv_data.to_csv(ohlcv_file, index=False)
        
        # Process returns
        result = process_ticker_returns(
            ticker="SPY",
            ohlcv_folder=str(ohlcv_folder),
            returns_folder=str(returns_folder)
        )
        
        assert result is True
        
        # Verify output file exists and has correct structure
        returns_file = returns_folder / "returns_SPY.csv"
        assert returns_file.exists()
        
        # Load and verify
        returns_data = pd.read_csv(returns_file, parse_dates=['Date'])
        assert len(returns_data) == 4
        assert list(returns_data.columns) == ['Date', 'Close', 'Return', 'LogReturn']
        assert returns_data.iloc[0]['Return'] == pytest.approx(0.1, abs=1e-6)


class TestRunReturnsDataJob:
    """Test the main job execution function."""
    
    @patch('jobs.returns_data_job.process_ticker_returns')
    def test_run_job_with_config(self, mock_process, tmp_path):
        """Test running the job with a configuration file."""
        # Create config file
        config_data = {
            "tickers": ["SPY", "GLD"],
            "ohlcv_data_folder": str(tmp_path / "ohlcv"),
            "returns_data_folder": str(tmp_path / "returns")
        }
        
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Mock
        mock_process.return_value = True
        
        # Execute
        run_returns_data_job(str(config_file))
        
        # Verify
        assert mock_process.call_count == 2
        mock_process.assert_any_call(
            ticker="SPY",
            ohlcv_folder=str(tmp_path / "ohlcv"),
            returns_folder=str(tmp_path / "returns")
        )
        mock_process.assert_any_call(
            ticker="GLD",
            ohlcv_folder=str(tmp_path / "ohlcv"),
            returns_folder=str(tmp_path / "returns")
        )
    
    @patch('jobs.returns_data_job.process_ticker_returns')
    def test_run_job_partial_success(self, mock_process, tmp_path):
        """Test job continues when some tickers fail."""
        # Create config
        config_data = {
            "tickers": ["SPY", "INVALID", "GLD"],
            "ohlcv_data_folder": str(tmp_path / "ohlcv"),
            "returns_data_folder": str(tmp_path / "returns")
        }
        
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Mock - middle ticker fails
        mock_process.side_effect = [True, False, True]
        
        # Execute
        run_returns_data_job(str(config_file))
        
        # Verify all tickers were attempted
        assert mock_process.call_count == 3
    
    @patch('jobs.returns_data_job.process_ticker_returns')
    def test_run_job_default_config_path(self, mock_process, tmp_path):
        """Test job uses default config path when none provided."""
        # Create config in expected location
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "config"
        
        # Only run this test if config directory exists
        if config_dir.exists():
            default_config = config_dir / "returns_data_job_config.json"
            if default_config.exists():
                mock_process.return_value = True
                
                # Execute with no config path
                run_returns_data_job()
                
                # Should have been called for each ticker in default config
                assert mock_process.called
    
    @patch('jobs.returns_data_job.process_ticker_returns')
    def test_run_job_relative_paths(self, mock_process, tmp_path):
        """Test job handles relative paths correctly."""
        # Create config with relative paths
        config_data = {
            "tickers": ["SPY"],
            "ohlcv_data_folder": "data/ohlcv_raw",
            "returns_data_folder": "data/returns"
        }
        
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        mock_process.return_value = True
        
        # Execute
        run_returns_data_job(str(config_file))
        
        # Verify paths were converted to absolute
        call_args = mock_process.call_args[1]
        assert os.path.isabs(call_args['ohlcv_folder'])
        assert os.path.isabs(call_args['returns_folder'])
    
    def test_run_job_integration(self, tmp_path):
        """End-to-end integration test."""
        # Setup directories
        ohlcv_folder = tmp_path / "ohlcv"
        returns_folder = tmp_path / "returns"
        ohlcv_folder.mkdir()
        
        # Create OHLCV data for two tickers
        for ticker in ["SPY", "GLD"]:
            dates = pd.date_range('2024-01-01', periods=10, freq='D')
            ohlcv_data = pd.DataFrame({
                'Date': dates,
                'Open': np.random.uniform(100, 200, 10),
                'High': np.random.uniform(200, 300, 10),
                'Low': np.random.uniform(50, 100, 10),
                'Close': np.random.uniform(100, 200, 10),
                'Volume': np.random.randint(1000000, 2000000, 10)
            })
            ohlcv_file = ohlcv_folder / f"ohlcv_raw_{ticker}.csv"
            ohlcv_data.to_csv(ohlcv_file, index=False)
        
        # Create config
        config_data = {
            "tickers": ["SPY", "GLD"],
            "ohlcv_data_folder": str(ohlcv_folder),
            "returns_data_folder": str(returns_folder)
        }
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Execute job
        run_returns_data_job(str(config_file))
        
        # Verify outputs
        for ticker in ["SPY", "GLD"]:
            returns_file = returns_folder / f"returns_{ticker}.csv"
            assert returns_file.exists()
            
            # Load and verify structure
            returns_data = pd.read_csv(returns_file, parse_dates=['Date'])
            assert len(returns_data) == 9  # 10 prices -> 9 returns
            assert list(returns_data.columns) == ['Date', 'Close', 'Return', 'LogReturn']
            
            # Verify no NaN values
            assert not returns_data['Return'].isna().any()
            assert not returns_data['LogReturn'].isna().any()
