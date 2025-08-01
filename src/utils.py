"""
Utility functions for the algorithmic trading system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class Logger:
    """Custom logger for the trading system."""
    
    def __init__(self, name: str = "trading_system", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)


class PerformanceMetrics:
    """Calculate various performance metrics for trading strategies."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if returns.std() == 0:
            return 0
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (uses downside deviation instead of total volatility)."""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf if excess_returns.mean() > 0 else 0
        
        downside_deviation = downside_returns.std()
        return excess_returns.mean() / downside_deviation * np.sqrt(252)
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        max_dd = PerformanceMetrics.calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0
        
        return annual_return / abs(max_dd)
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        return drawdowns.min()
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk."""
        var = PerformanceMetrics.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio."""
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return 0
        
        return active_returns.mean() / tracking_error * np.sqrt(252)
    
    @staticmethod
    def calculate_omega_ratio(returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio."""
        excess_returns = returns - threshold
        positive_returns = excess_returns[excess_returns > 0].sum()
        negative_returns = abs(excess_returns[excess_returns < 0].sum())
        
        if negative_returns == 0:
            return np.inf if positive_returns > 0 else 1
        
        return positive_returns / negative_returns


class DataValidator:
    """Data validation utilities."""
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality and return summary."""
        quality_report = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'date_range': None,
            'suspicious_values': {}
        }
        
        # Check date range if datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            quality_report['date_range'] = {
                'start': df.index.min(),
                'end': df.index.max(),
                'total_days': (df.index.max() - df.index.min()).days
            }
        
        # Check for suspicious values in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                q1, q3 = col_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                quality_report['suspicious_values'][col] = {
                    'outliers': outliers,
                    'negative_values': (col_data < 0).sum() if col in ['Volume', 'Open', 'High', 'Low', 'Close'] else 0,
                    'zero_values': (col_data == 0).sum()
                }
        
        return quality_report
    
    @staticmethod
    def clean_data(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """Clean data using specified method."""
        df_clean = df.copy()
        
        if method == 'forward_fill':
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        elif method == 'interpolate':
            df_clean = df_clean.interpolate()
        elif method == 'drop':
            df_clean = df_clean.dropna()
        
        # Remove duplicates
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        return df_clean
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> List[str]:
        """Validate OHLCV data for inconsistencies."""
        errors = []
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return errors
        
        # Check for logical inconsistencies
        if (df['High'] < df['Low']).any():
            errors.append("High prices lower than Low prices detected")
        
        if (df['High'] < df['Open']).any():
            errors.append("High prices lower than Open prices detected")
        
        if (df['High'] < df['Close']).any():
            errors.append("High prices lower than Close prices detected")
        
        if (df['Low'] > df['Open']).any():
            errors.append("Low prices higher than Open prices detected")
        
        if (df['Low'] > df['Close']).any():
            errors.append("Low prices higher than Close prices detected")
        
        if (df['Volume'] < 0).any():
            errors.append("Negative volume values detected")
        
        return errors


class Visualizer:
    """Visualization utilities for the trading system."""
    
    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_price_data(self, data: pd.DataFrame, title: str = "Price Data", 
                       figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot OHLCV price data."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Price plot
        ax1.plot(data.index, data['Close'], label='Close', linewidth=1)
        if 'ma_20' in data.columns:
            ax1.plot(data.index, data['ma_20'], label='MA 20', alpha=0.7)
        if 'ma_50' in data.columns:
            ax1.plot(data.index, data['ma_50'], label='MA 50', alpha=0.7)
        
        ax1.set_title(title)
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume plot
        ax2.bar(data.index, data['Volume'], alpha=0.6, width=1)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20,
                              figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot feature importance."""
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_equity_curve(self, returns: pd.Series, benchmark: Optional[pd.Series] = None,
                         title: str = "Equity Curve", figsize: Tuple[int, int] = (12, 6)) -> None:
        """Plot equity curve."""
        equity = (1 + returns).cumprod()
        
        plt.figure(figsize=figsize)
        plt.plot(equity.index, equity.values, label='Strategy', linewidth=2)
        
        if benchmark is not None:
            benchmark_equity = (1 + benchmark).cumprod()
            plt.plot(benchmark_equity.index, benchmark_equity.values, 
                    label='Benchmark', alpha=0.7)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown(self, returns: pd.Series, figsize: Tuple[int, int] = (12, 4)) -> None:
        """Plot drawdown chart."""
        equity = (1 + returns).cumprod()
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        
        plt.figure(figsize=figsize)
        plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.6, color='red')
        plt.title('Drawdown Chart')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_returns(self, returns: pd.Series, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot monthly returns heatmap."""
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns.index = monthly_returns.index.to_period('M')
        
        # Create pivot table for heatmap
        monthly_table = monthly_returns.groupby([monthly_returns.index.year, 
                                               monthly_returns.index.month]).first().unstack()
        
        plt.figure(figsize=figsize)
        sns.heatmap(monthly_table, annot=True, fmt='.2%', cmap='RdYlGn', center=0)
        plt.title('Monthly Returns Heatmap')
        plt.ylabel('Year')
        plt.xlabel('Month')
        plt.tight_layout()
        plt.show()


class FileManager:
    """File management utilities."""
    
    @staticmethod
    def ensure_directory(directory: str) -> None:
        """Ensure directory exists, create if not."""
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    @staticmethod
    def save_results(results: Dict[str, Any], filename: str, directory: str = "results") -> None:
        """Save results to file."""
        FileManager.ensure_directory(directory)
        filepath = os.path.join(directory, filename)
        
        if filename.endswith('.csv'):
            if isinstance(results, pd.DataFrame):
                results.to_csv(filepath)
            else:
                pd.DataFrame([results]).to_csv(filepath)
        elif filename.endswith('.json'):
            import json
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        else:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
    
    @staticmethod
    def load_results(filename: str, directory: str = "results") -> Any:
        """Load results from file."""
        filepath = os.path.join(directory, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found")
        
        if filename.endswith('.csv'):
            return pd.read_csv(filepath, index_col=0)
        elif filename.endswith('.json'):
            import json
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            import pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)


class ConfigManager:
    """Configuration management utilities."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            import json
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        default_config = {
            'data': {
                'start_date': '2000-01-01',
                'end_date': '2021-12-31',
                'data_directory': 'data',
                'tickers_file': 'tickers.csv'
            },
            'features': {
                'ma_periods': [5, 10, 20, 50, 200],
                'volatility_periods': [5, 10, 20, 60],
                'volume_periods': [1, 5, 21, 63, 126, 252]
            },
            'models': {
                'random_forest': {
                    'n_estimators': 150,
                    'max_depth': 10,
                    'min_samples_split': 4,
                    'min_samples_leaf': 2
                },
                'xgboost': {
                    'max_depth': 6,
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'gamma': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                }
            },
            'backtesting': {
                'initial_capital': 10000,
                'commission': 0.001,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10,
                'risk_per_trade': 0.02
            }
        }
        
        self.save_config(default_config)
        return default_config
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Save configuration to file."""
        if config is None:
            config = self.config
        
        import json
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config()


def format_number(num: float, format_type: str = 'currency') -> str:
    """Format numbers for display."""
    if format_type == 'currency':
        return f"${num:,.2f}"
    elif format_type == 'percentage':
        return f"{num:.2%}"
    elif format_type == 'decimal':
        return f"{num:.4f}"
    else:
        return str(num)


def calculate_portfolio_stats(returns: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
    """Calculate comprehensive portfolio statistics."""
    stats = {
        'Total Return': (1 + returns).prod() - 1,
        'Annual Return': (1 + returns).prod() ** (252 / len(returns)) - 1,
        'Volatility': returns.std() * np.sqrt(252),
        'Sharpe Ratio': PerformanceMetrics.calculate_sharpe_ratio(returns),
        'Sortino Ratio': PerformanceMetrics.calculate_sortino_ratio(returns),
        'Calmar Ratio': PerformanceMetrics.calculate_calmar_ratio(returns),
        'Max Drawdown': PerformanceMetrics.calculate_max_drawdown(returns),
        'VaR (5%)': PerformanceMetrics.calculate_var(returns),
        'CVaR (5%)': PerformanceMetrics.calculate_cvar(returns),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis(),
        'Best Day': returns.max(),
        'Worst Day': returns.min(),
        'Positive Days': (returns > 0).sum() / len(returns),
        'Average Win': returns[returns > 0].mean() if (returns > 0).any() else 0,
        'Average Loss': returns[returns < 0].mean() if (returns < 0).any() else 0
    }
    
    if benchmark is not None:
        stats['Beta'] = returns.cov(benchmark) / benchmark.var() if benchmark.var() > 0 else 0
        stats['Alpha'] = stats['Annual Return'] - (0.02 + stats['Beta'] * (benchmark.mean() * 252 - 0.02))
        stats['Information Ratio'] = PerformanceMetrics.calculate_information_ratio(returns, benchmark)
    
    return stats