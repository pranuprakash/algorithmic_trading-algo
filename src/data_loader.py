"""
Data loading and preprocessing module for the algorithmic trading system.
"""

import pandas as pd
import yfinance as yf
import backtrader as bt
import os
from typing import Optional, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class CustomCSVData(bt.feeds.GenericCSVData):
    """Custom CSV data loader for Backtrader with enhanced functionality."""
    
    params = (
        ('stock_name', None),
        ('csv_directory', "data/stock_dfs"),
        ('data_type', None)
    )

    def __init__(self):
        super().__init__()
        self.data_frame = self._load_csv()

    def _construct_file_path(self) -> str:
        """Construct the file path for the CSV file."""
        return os.path.join(self.p.csv_directory, f"{self.p.stock_name}.csv")

    def _load_csv(self) -> pd.DataFrame:
        """Load data from a CSV file."""
        file_path = self._construct_file_path()
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.lower()
        
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)

        if self.p.data_type == 'hour':
            # Implement specific operations for hourly data
            pass

        return data

    def _prepare_data(self) -> pd.DataFrame:
        """Prepare data for Backtrader feed."""
        if self.p.data_type is not None:
            return self.data_frame[['high', 'low', 'open', 'close', 'volume']]
        return self.data_frame


class PredictionsData(bt.feeds.PandasData):
    """Custom data feed for Backtrader that includes model predictions."""
    
    lines = ('predictions', 'prediction_proba')
    params = (
        ('predictions', -1),
        ('prediction_proba', -1),
    )


class DataLoader:
    """Main data loading class for the trading system."""
    
    def __init__(self, data_directory: str = "data"):
        self.data_directory = data_directory
        
    def load_tickers(self, ticker_file: str = "tickers.csv") -> list:
        """Load list of tickers from CSV file."""
        ticker_path = os.path.join(self.data_directory, ticker_file)
        ticker_df = pd.read_csv(ticker_path)
        
        # Handle different column names
        if 'Ticker' in ticker_df.columns:
            return list(ticker_df["Ticker"])
        elif 'Symbol' in ticker_df.columns:
            return list(ticker_df["Symbol"])
        else:
            # Assume first column contains tickers
            return list(ticker_df.iloc[:, 0])
    
    def fetch_yahoo_data(self, ticker: str, start_date: str = "2000-01-01", 
                        end_date: str = "2021-11-12") -> Optional[pd.DataFrame]:
        """Fetch stock data from Yahoo Finance."""
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                print(f"Warning: No data found for ticker {ticker}")
                return None
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def load_csv_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from CSV file in the data directory."""
        try:
            file_path = os.path.join(self.data_directory, filename)
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            return None
    
    def load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load various market indicators and indices."""
        market_files = {
            'dgs2': 'DGS2.csv',
            'equity_pc': 'equitypc.csv',
            'index_pc': 'indexpc.csv',
            'total_pc': 'totalpc.csv',
            'vix_pc': 'vixpc.csv'
        }
        
        market_data = {}
        for key, filename in market_files.items():
            data = self.load_csv_data(filename)
            if data is not None:
                market_data[key] = data
        
        return market_data
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that the data has the required columns for trading."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if data is None or data.empty:
            return False
        
        # Check if all required columns exist (case insensitive)
        data_columns = [col.lower() for col in data.columns]
        required_lower = [col.lower() for col in required_columns]
        
        return all(req_col in data_columns for req_col in required_lower)
    
    def get_stock_universe(self, ticker_file: str = "tickers_nasd.csv", 
                          max_stocks: Optional[int] = None) -> list:
        """Get a universe of stocks for analysis."""
        tickers = self.load_tickers(ticker_file)
        
        if max_stocks:
            return tickers[:max_stocks]
        
        return tickers