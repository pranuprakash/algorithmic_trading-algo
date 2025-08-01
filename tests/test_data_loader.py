"""
Unit tests for the data_loader module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader, CustomCSVData, PredictionsData


class TestDataLoader:
    """Test cases for the DataLoader class."""
    
    @pytest.fixture
    def data_loader(self):
        """Create a DataLoader instance for testing."""
        return DataLoader(data_directory="test_data")
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        return pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(150, 250, 100),
            'Low': np.random.uniform(50, 150, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        })
    
    @pytest.fixture
    def sample_ticker_data(self):
        """Create sample ticker data for testing."""
        return pd.DataFrame({
            'Ticker': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        })
    
    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup is handled by pytest
    
    def test_init(self, data_loader):
        """Test DataLoader initialization."""
        assert data_loader.data_directory == "test_data"
    
    def test_load_tickers_with_ticker_column(self, data_loader, temp_directory, sample_ticker_data):
        """Test loading tickers with 'Ticker' column."""
        # Create temporary CSV file
        csv_path = os.path.join(temp_directory, "tickers.csv")
        sample_ticker_data.to_csv(csv_path, index=False)
        
        # Update data loader to use temp directory
        data_loader.data_directory = temp_directory
        
        tickers = data_loader.load_tickers("tickers.csv")
        assert tickers == ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    def test_load_tickers_with_symbol_column(self, data_loader, temp_directory):
        """Test loading tickers with 'Symbol' column."""
        symbol_data = pd.DataFrame({
            'Symbol': ['NFLX', 'META', 'NVDA']
        })
        
        csv_path = os.path.join(temp_directory, "symbols.csv")
        symbol_data.to_csv(csv_path, index=False)
        
        data_loader.data_directory = temp_directory
        tickers = data_loader.load_tickers("symbols.csv")
        assert tickers == ['NFLX', 'META', 'NVDA']
    
    def test_load_tickers_first_column_fallback(self, data_loader, temp_directory):
        """Test loading tickers using first column as fallback."""
        other_data = pd.DataFrame({
            'StockCode': ['IBM', 'ORCL', 'CRM'],
            'Name': ['IBM Corp', 'Oracle', 'Salesforce']
        })
        
        csv_path = os.path.join(temp_directory, "other.csv")
        other_data.to_csv(csv_path, index=False)
        
        data_loader.data_directory = temp_directory
        tickers = data_loader.load_tickers("other.csv")
        assert tickers == ['IBM', 'ORCL', 'CRM']
    
    @patch('yfinance.download')
    def test_fetch_yahoo_data_success(self, mock_download, data_loader, sample_csv_data):
        """Test successful Yahoo Finance data fetching."""
        mock_download.return_value = sample_csv_data
        
        result = data_loader.fetch_yahoo_data("AAPL")
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        mock_download.assert_called_once_with("AAPL", start="2000-01-01", end="2021-11-12")
    
    @patch('yfinance.download')
    def test_fetch_yahoo_data_empty_result(self, mock_download, data_loader):
        """Test Yahoo Finance data fetching with empty result."""
        mock_download.return_value = pd.DataFrame()
        
        result = data_loader.fetch_yahoo_data("INVALID")
        assert result is None
    
    @patch('yfinance.download')
    def test_fetch_yahoo_data_exception(self, mock_download, data_loader):
        """Test Yahoo Finance data fetching with exception."""
        mock_download.side_effect = Exception("Network error")
        
        result = data_loader.fetch_yahoo_data("AAPL")
        assert result is None
    
    def test_load_csv_data_success(self, data_loader, temp_directory, sample_csv_data):
        """Test successful CSV data loading."""
        csv_path = os.path.join(temp_directory, "test_data.csv")
        sample_csv_data.to_csv(csv_path, index=False)
        
        data_loader.data_directory = temp_directory
        result = data_loader.load_csv_data("test_data.csv")
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
    
    def test_load_csv_data_file_not_found(self, data_loader, temp_directory):
        """Test CSV data loading with non-existent file."""
        data_loader.data_directory = temp_directory
        result = data_loader.load_csv_data("nonexistent.csv")
        assert result is None
    
    def test_validate_data_valid(self, data_loader, sample_csv_data):
        """Test data validation with valid OHLCV data."""
        assert data_loader.validate_data(sample_csv_data) == True
    
    def test_validate_data_missing_columns(self, data_loader):
        """Test data validation with missing required columns."""
        invalid_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Price': np.random.uniform(100, 200, 10)
        })
        assert data_loader.validate_data(invalid_data) == False
    
    def test_validate_data_none(self, data_loader):
        """Test data validation with None input."""
        assert data_loader.validate_data(None) == False
    
    def test_validate_data_empty(self, data_loader):
        """Test data validation with empty DataFrame."""
        assert data_loader.validate_data(pd.DataFrame()) == False
    
    def test_validate_data_case_insensitive(self, data_loader):
        """Test data validation with lowercase column names."""
        lowercase_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 10),
            'high': np.random.uniform(150, 250, 10),
            'low': np.random.uniform(50, 150, 10),
            'close': np.random.uniform(100, 200, 10),
            'volume': np.random.randint(1000, 10000, 10)
        })
        assert data_loader.validate_data(lowercase_data) == True
    
    def test_get_stock_universe_with_limit(self, data_loader, temp_directory, sample_ticker_data):
        """Test getting stock universe with limit."""
        csv_path = os.path.join(temp_directory, "tickers.csv")
        sample_ticker_data.to_csv(csv_path, index=False)
        
        data_loader.data_directory = temp_directory
        universe = data_loader.get_stock_universe("tickers.csv", max_stocks=3)
        
        assert len(universe) == 3
        assert universe == ['AAPL', 'GOOGL', 'MSFT']
    
    def test_get_stock_universe_no_limit(self, data_loader, temp_directory, sample_ticker_data):
        """Test getting stock universe without limit."""
        csv_path = os.path.join(temp_directory, "tickers.csv")
        sample_ticker_data.to_csv(csv_path, index=False)
        
        data_loader.data_directory = temp_directory
        universe = data_loader.get_stock_universe("tickers.csv")
        
        assert len(universe) == 5
        assert universe == ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']


class TestCustomCSVData:
    """Test cases for the CustomCSVData class."""
    
    @pytest.fixture
    def sample_csv_file(self, temp_directory):
        """Create a sample CSV file for testing."""
        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=50),
            'Open': np.random.uniform(100, 200, 50),
            'High': np.random.uniform(150, 250, 50),
            'Low': np.random.uniform(50, 150, 50),
            'Close': np.random.uniform(100, 200, 50),
            'Volume': np.random.randint(1000, 10000, 50)
        })
        
        csv_path = os.path.join(temp_directory, "AAPL.csv")
        data.to_csv(csv_path, index=False)
        return csv_path
    
    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory for testing."""
        return tempfile.mkdtemp()
    
    def test_construct_file_path(self, temp_directory):
        """Test file path construction."""
        from data_loader import CustomCSVData
        
        # Create a mock instance without calling __init__ to avoid file loading
        csv_data = object.__new__(CustomCSVData)
        
        # Mock the params manually
        class MockParams:
            csv_directory = temp_directory
            stock_name = "AAPL"
        
        csv_data.p = MockParams()
        
        expected_path = os.path.join(temp_directory, "AAPL.csv")
        assert csv_data._construct_file_path() == expected_path


class TestPredictionsData:
    """Test cases for the PredictionsData class."""
    
    def test_predictions_data_creation(self):
        """Test PredictionsData can be instantiated."""
        # Create sample data with predictions
        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Close': np.random.uniform(100, 200, 10),
            'predictions': np.random.choice([0, 1], 10)
        })
        
        pred_data = PredictionsData(dataname=data)
        assert hasattr(pred_data, 'lines')
        assert hasattr(pred_data.lines, 'predictions')
    
    def test_predictions_data_params(self):
        """Test PredictionsData parameters."""
        assert hasattr(PredictionsData, 'params')
        # Check that predictions parameter exists
        assert hasattr(PredictionsData.params, 'predictions')
        assert hasattr(PredictionsData.params, 'prediction_proba')


# Integration tests
class TestDataLoaderIntegration:
    """Integration tests for DataLoader with real-like scenarios."""
    
    @pytest.fixture
    def integration_setup(self):
        """Set up integration test environment."""
        temp_dir = tempfile.mkdtemp()
        
        # Create sample market data files
        market_files = {
            'DGS2.csv': pd.DataFrame({
                'Date': pd.date_range('2020-01-01', periods=30),
                'DGS2': np.random.uniform(1, 3, 30)
            }),
            'equitypc.csv': pd.DataFrame({
                'Date': pd.date_range('2020-01-01', periods=30),
                'NYGRS_eq': np.random.uniform(50, 150, 30)
            }),
            'vixpc.csv': pd.DataFrame({
                'Date': pd.date_range('2020-01-01', periods=30),
                'VIX': np.random.uniform(10, 80, 30)
            })
        }
        
        for filename, data in market_files.items():
            data.to_csv(os.path.join(temp_dir, filename), index=False)
        
        yield temp_dir
    
    def test_load_market_data_integration(self, integration_setup):
        """Test loading multiple market data files."""
        data_loader = DataLoader(data_directory=integration_setup)
        market_data = data_loader.load_market_data()
        
        assert isinstance(market_data, dict)
        assert 'dgs2' in market_data
        assert 'equity_pc' in market_data
        assert 'vix_pc' in market_data
        
        # Check that data was loaded correctly
        for key, data in market_data.items():
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 30


if __name__ == "__main__":
    pytest.main([__file__])