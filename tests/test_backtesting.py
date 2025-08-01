"""
Unit tests for the backtesting module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock, call
import tempfile

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtesting import MLStrategy, RiskManager, BacktestEngine


class TestMLStrategy:
    """Test cases for the MLStrategy class."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for backtesting."""
        np.random.seed(42)
        n_days = 100
        
        # Generate realistic price data
        base_price = 100
        prices = [base_price]
        
        for _ in range(n_days - 1):
            change = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))
        
        data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_days),
            'predictions': np.random.choice([0, 1], n_days, p=[0.6, 0.4]),
            'prediction_proba': np.random.uniform(0.4, 0.9, n_days)
        }, index=pd.date_range('2020-01-01', periods=n_days))
        
        # Ensure price relationships
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        return data
    
    def test_strategy_initialization(self):
        """Test MLStrategy initialization with default parameters."""
        # Note: This is a conceptual test - actual Backtrader testing requires more setup
        
        # Check that expected parameters exist by accessing them directly
        expected_params = [
            'predictions_col', 'prob_threshold', 'stop_loss_pct', 
            'take_profit_pct', 'position_size_method', 'risk_per_trade'
        ]
        
        # In backtrader, params are accessible as class attributes
        for param in expected_params:
            assert hasattr(MLStrategy.params, param)
    
    def test_strategy_parameters_values(self):
        """Test default parameter values."""
        # Access parameter values using backtrader's parameter access method
        assert MLStrategy.params.prob_threshold == 0.6
        assert MLStrategy.params.stop_loss_pct == 0.05
        assert MLStrategy.params.take_profit_pct == 0.10
        assert MLStrategy.params.risk_per_trade == 0.02
        assert MLStrategy.params.position_size_method == 'fixed'


class TestRiskManager:
    """Test cases for the RiskManager class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return series for testing."""
        np.random.seed(42)
        return pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns for one year
    
    def test_calculate_var(self, sample_returns):
        """Test Value at Risk calculation."""
        var_5 = RiskManager.calculate_var(sample_returns, confidence_level=0.05)
        var_1 = RiskManager.calculate_var(sample_returns, confidence_level=0.01)
        
        # VaR should be negative for losses
        assert var_5 < 0
        assert var_1 < 0
        
        # 1% VaR should be more extreme than 5% VaR
        assert var_1 < var_5
    
    def test_calculate_cvar(self, sample_returns):
        """Test Conditional Value at Risk calculation."""
        cvar_5 = RiskManager.calculate_cvar(sample_returns, confidence_level=0.05)
        var_5 = RiskManager.calculate_var(sample_returns, confidence_level=0.05)
        
        # CVaR should be more extreme than VaR
        assert cvar_5 <= var_5
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Create equity curve with known drawdown
        equity = pd.Series([100, 110, 105, 95, 90, 100, 120])
        
        max_dd = RiskManager.calculate_max_drawdown(equity)
        
        # Expected max drawdown from 110 to 90 = -18.18%
        expected_dd = (90 - 110) / 110
        assert abs(max_dd - expected_dd) < 1e-6
    
    def test_calculate_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = RiskManager.calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)
        
        # Sharpe ratio should be a finite number
        assert np.isfinite(sharpe)
        
        # Test with zero volatility (edge case)
        zero_vol_returns = pd.Series([0.001] * 252)
        sharpe_zero_vol = RiskManager.calculate_sharpe_ratio(zero_vol_returns)
        assert sharpe_zero_vol == 0


class TestBacktestEngine:
    """Test cases for the BacktestEngine class."""
    
    @pytest.fixture
    def backtest_engine(self):
        """Create a BacktestEngine instance for testing."""
        return BacktestEngine(initial_capital=10000, commission=0.001)
    
    @pytest.fixture
    def sample_backtest_data(self):
        """Create sample data for backtesting."""
        np.random.seed(42)
        n_days = 50
        
        data = pd.DataFrame({
            'Open': np.random.uniform(95, 105, n_days),
            'High': np.random.uniform(100, 110, n_days),
            'Low': np.random.uniform(90, 100, n_days),
            'Close': np.random.uniform(95, 105, n_days),
            'Volume': np.random.randint(1000000, 5000000, n_days)
        }, index=pd.date_range('2020-01-01', periods=n_days))
        
        # Ensure price relationships
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        return data
    
    def test_backtest_engine_initialization(self, backtest_engine):
        """Test BacktestEngine initialization."""
        assert backtest_engine.initial_capital == 10000
        assert backtest_engine.commission == 0.001
        assert backtest_engine.results == {}
    
    def test_prepare_data_with_predictions(self, backtest_engine, sample_backtest_data):
        """Test data preparation for backtesting."""
        predictions = np.random.choice([0, 1], len(sample_backtest_data))
        prediction_proba = np.random.uniform(0.3, 0.8, (len(sample_backtest_data), 2))
        
        prepared_data = backtest_engine.prepare_data_with_predictions(
            sample_backtest_data, predictions, prediction_proba
        )
        
        # Check that predictions were added
        assert 'predictions' in prepared_data.columns
        assert 'prediction_proba' in prepared_data.columns
        assert len(prepared_data) == len(sample_backtest_data)
        
        # Check that all original columns are preserved
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            assert col in prepared_data.columns
    
    def test_prepare_data_predictions_length_mismatch(self, backtest_engine, sample_backtest_data):
        """Test data preparation with mismatched prediction length."""
        predictions = np.random.choice([0, 1], len(sample_backtest_data) - 10)
        
        prepared_data = backtest_engine.prepare_data_with_predictions(
            sample_backtest_data, predictions
        )
        
        # Should truncate to shorter length
        assert len(prepared_data) == len(predictions)
    
    def test_prepare_data_missing_required_columns(self, backtest_engine):
        """Test data preparation with missing required columns."""
        incomplete_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'Close': [101, 102, 103]
            # Missing High, Low, Volume
        })
        predictions = [0, 1, 0]
        
        with pytest.raises(ValueError, match="Required column .* not found"):
            backtest_engine.prepare_data_with_predictions(incomplete_data, predictions)
    
    def test_prepare_data_with_lowercase_columns(self, backtest_engine):
        """Test data preparation with lowercase column names."""
        lowercase_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })
        predictions = [0, 1, 0]
        
        prepared_data = backtest_engine.prepare_data_with_predictions(lowercase_data, predictions)
        
        # Should work with lowercase columns
        assert 'Open' in prepared_data.columns
        assert 'Close' in prepared_data.columns
        assert len(prepared_data) == 3
    
    @patch('backtrader.Cerebro')
    def test_run_backtest_basic(self, mock_cerebro_class, backtest_engine, sample_backtest_data):
        """Test basic backtest execution."""
        # Mock Cerebro and its components
        mock_cerebro = MagicMock()
        mock_cerebro_class.return_value = mock_cerebro
        
        # Mock broker
        mock_broker = MagicMock()
        mock_broker.getvalue.return_value = 11000  # 10% return
        mock_cerebro.broker = mock_broker
        
        # Mock strategy results
        mock_strategy = MagicMock()
        mock_strategy.analyzers.time_return.get_analysis.return_value = {
            '2020-01-01': 0.01, '2020-01-02': -0.005
        }
        mock_strategy.analyzers.drawdown.get_analysis.return_value = {
            'max': {'drawdown': 0.05}
        }
        mock_strategy.analyzers.trades.get_analysis.return_value = {
            'total': {'total': 10},
            'won': {'total': 6},
            'lost': {'total': 4}
        }
        
        mock_cerebro.run.return_value = [mock_strategy]
        
        # Prepare test data
        predictions = np.random.choice([0, 1], len(sample_backtest_data))
        bt_data = backtest_engine.prepare_data_with_predictions(sample_backtest_data, predictions)
        
        # Run backtest
        results = backtest_engine.run_backtest(bt_data)
        
        # Check results structure
        assert 'initial_capital' in results
        assert 'final_value' in results
        assert 'total_return' in results
        assert 'max_drawdown' in results
        assert 'total_trades' in results
        assert 'win_rate' in results
        
        # Check values
        assert results['initial_capital'] == 10000
        assert results['final_value'] == 11000
        assert results['total_return'] == 0.1
        assert results['win_rate'] == 0.6
    
    def test_aggregate_walk_forward_results(self, backtest_engine):
        """Test aggregation of walk-forward results."""
        # Create sample period results
        period_results = [
            {
                'total_return': 0.05,
                'max_drawdown': -0.02,
                'total_trades': 5,
                'winning_trades': 3,
                'time_returns': {'2020-01-01': 0.01, '2020-01-02': 0.02}
            },
            {
                'total_return': 0.03,
                'max_drawdown': -0.01,
                'total_trades': 3,
                'winning_trades': 2,
                'time_returns': {'2020-01-03': 0.01, '2020-01-04': 0.01}
            }
        ]
        
        aggregated = backtest_engine._aggregate_walk_forward_results(period_results)
        
        # Check aggregated results
        assert 'total_return' in aggregated
        assert 'max_drawdown' in aggregated
        assert 'total_trades' in aggregated
        assert 'winning_trades' in aggregated
        assert 'win_rate' in aggregated
        
        # Check calculations
        expected_total_return = (1.05 * 1.03) - 1
        assert abs(aggregated['total_return'] - expected_total_return) < 1e-6
        assert aggregated['max_drawdown'] == -0.02  # Worst drawdown
        assert aggregated['total_trades'] == 8
        assert aggregated['winning_trades'] == 5
        assert aggregated['win_rate'] == 5/8
    
    @patch('quantstats.reports.html')
    def test_generate_performance_report(self, mock_qs_html, backtest_engine):
        """Test performance report generation."""
        # Create sample results
        results = {
            'time_returns': {
                '2020-01-01': 0.01,
                '2020-01-02': -0.005,
                '2020-01-03': 0.02
            }
        }
        
        returns_series = backtest_engine.generate_performance_report(
            results, 'test_report.html'
        )
        
        # Check that QuantStats was called
        mock_qs_html.assert_called_once()
        
        # Check returns series
        assert isinstance(returns_series, pd.Series)
        assert len(returns_series) == 3
    
    def test_generate_performance_report_no_returns(self, backtest_engine):
        """Test performance report generation with no returns data."""
        results = {}  # No time_returns
        
        returns_series = backtest_engine.generate_performance_report(results)
        
        # Should return empty series
        assert isinstance(returns_series, pd.Series)
        assert len(returns_series) == 0
    
    def test_compare_strategies(self, backtest_engine):
        """Test strategy comparison."""
        strategies_data = {
            'Strategy A': {
                'total_return': 0.15,
                'max_drawdown': -0.08,
                'win_rate': 0.65,
                'total_trades': 25,
                'final_value': 11500
            },
            'Strategy B': {
                'total_return': 0.12,
                'max_drawdown': -0.05,
                'win_rate': 0.70,
                'total_trades': 20,
                'final_value': 11200
            }
        }
        
        comparison_df = backtest_engine.compare_strategies(strategies_data)
        
        # Check comparison table structure
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'Strategy' in comparison_df.columns
        assert 'Total Return' in comparison_df.columns
        assert 'Win Rate' in comparison_df.columns
        
        # Check values formatting
        assert '15.00%' in comparison_df['Total Return'].values
        assert '65.00%' in comparison_df['Win Rate'].values
    
    def test_optimize_strategy_parameters(self, backtest_engine, sample_backtest_data):
        """Test strategy parameter optimization."""
        # This is a simplified test since full optimization requires Backtrader integration
        param_grid = {
            'stop_loss_pct': [0.02, 0.05],
            'take_profit_pct': [0.05, 0.10]
        }
        
        # Mock the run_backtest method to avoid full Backtrader execution
        original_run_backtest = backtest_engine.run_backtest
        
        def mock_run_backtest(data, strategy_params=None):
            # Return different results based on parameters
            if strategy_params and strategy_params.get('stop_loss_pct') == 0.02:
                return {'total_return': 0.15}
            else:
                return {'total_return': 0.10}
        
        backtest_engine.run_backtest = mock_run_backtest
        
        try:
            # Prepare test data
            predictions = np.random.choice([0, 1], len(sample_backtest_data))
            bt_data = backtest_engine.prepare_data_with_predictions(sample_backtest_data, predictions)
            
            # Run optimization
            optimization_results = backtest_engine.optimize_strategy_parameters(bt_data, param_grid)
            
            # Check results
            assert 'best_params' in optimization_results
            assert 'best_return' in optimization_results
            assert 'all_results' in optimization_results
            
            # Best parameters should give highest return
            assert optimization_results['best_return'] == 0.15
            assert optimization_results['best_params']['stop_loss_pct'] == 0.02
            
        finally:
            # Restore original method
            backtest_engine.run_backtest = original_run_backtest


class TestBacktestingIntegration:
    """Integration tests for backtesting components."""
    
    @pytest.fixture
    def integration_data(self):
        """Create comprehensive test data for integration testing."""
        np.random.seed(42)
        n_days = 200
        
        # Generate correlated price series
        base_price = 100
        prices = [base_price]
        
        for _ in range(n_days - 1):
            change = np.random.normal(0.0005, 0.015)  # Slight positive drift
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))
        
        data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, n_days)
        }, index=pd.date_range('2020-01-01', periods=n_days))
        
        # Ensure price relationships
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        # Create predictions with some signal
        # Higher probability of buy signal when price is trending up
        price_momentum = data['Close'].pct_change(5).fillna(0)
        predictions = (price_momentum > 0).astype(int)
        prediction_proba = np.random.uniform(0.4, 0.8, n_days)
        
        return data, predictions, prediction_proba
    
    def test_end_to_end_backtest_workflow(self, integration_data):
        """Test complete end-to-end backtesting workflow."""
        data, predictions, prediction_proba = integration_data
        
        # Initialize backtesting engine
        engine = BacktestEngine(initial_capital=10000, commission=0.001)
        
        # Prepare data
        bt_data = engine.prepare_data_with_predictions(data, predictions, prediction_proba)
        
        # Verify data preparation
        assert len(bt_data) == len(data)
        assert 'predictions' in bt_data.columns
        assert 'prediction_proba' in bt_data.columns
        
        # Check data quality
        assert not bt_data.isna().any().any()
        assert (bt_data['High'] >= bt_data['Low']).all()
        assert (bt_data['High'] >= bt_data['Open']).all()
        assert (bt_data['High'] >= bt_data['Close']).all()
        assert (bt_data['Low'] <= bt_data['Open']).all()
        assert (bt_data['Low'] <= bt_data['Close']).all()
    
    def test_risk_metrics_calculation_workflow(self):
        """Test complete risk metrics calculation workflow."""
        # Create sample returns with known characteristics
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # One year of daily returns
        
        # Calculate all risk metrics
        var_5 = RiskManager.calculate_var(returns, 0.05)
        cvar_5 = RiskManager.calculate_cvar(returns, 0.05)
        sharpe = RiskManager.calculate_sharpe_ratio(returns)
        
        # Create equity curve and calculate max drawdown
        equity_curve = (1 + returns).cumprod() * 10000
        max_dd = RiskManager.calculate_max_drawdown(returns)
        
        # Verify all metrics are reasonable
        assert var_5 < 0  # VaR should be negative
        assert cvar_5 <= var_5  # CVaR should be worse than VaR
        assert np.isfinite(sharpe)  # Sharpe should be finite
        assert max_dd <= 0  # Max drawdown should be negative or zero
        
        # Test relationships
        assert cvar_5 < var_5  # CVaR should be more extreme than VaR


class TestBacktestingEdgeCases:
    """Test edge cases and error conditions for backtesting."""
    
    def test_empty_predictions(self):
        """Test backtesting with empty predictions."""
        engine = BacktestEngine()
        empty_data = pd.DataFrame()
        empty_predictions = np.array([])
        
        # Should handle empty data gracefully
        with pytest.raises(ValueError):
            engine.prepare_data_with_predictions(empty_data, empty_predictions)
    
    def test_all_zero_returns(self):
        """Test risk calculations with zero returns."""
        zero_returns = pd.Series([0.0] * 100)
        
        var = RiskManager.calculate_var(zero_returns)
        cvar = RiskManager.calculate_cvar(zero_returns)
        max_dd = RiskManager.calculate_max_drawdown(zero_returns)
        sharpe = RiskManager.calculate_sharpe_ratio(zero_returns)
        
        assert var == 0.0
        assert cvar == 0.0
        assert max_dd == 0.0
        assert sharpe == 0.0  # No volatility case
    
    def test_extreme_returns(self):
        """Test risk calculations with extreme returns."""
        extreme_returns = pd.Series([-0.5, 0.8, -0.3, 0.6, -0.4])
        
        var = RiskManager.calculate_var(extreme_returns)
        cvar = RiskManager.calculate_cvar(extreme_returns)
        max_dd = RiskManager.calculate_max_drawdown(extreme_returns)
        
        # Should handle extreme values without crashing
        assert np.isfinite(var)
        assert np.isfinite(cvar)
        assert np.isfinite(max_dd)


if __name__ == "__main__":
    pytest.main([__file__])