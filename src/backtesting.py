"""
Backtesting and risk management module for the algorithmic trading system.
"""

import backtrader as bt
import pandas as pd
import numpy as np
import quantstats as qs
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class MLStrategy(bt.Strategy):
    """
    Machine Learning-based trading strategy with risk management.
    """
    
    params = (
        ('predictions_col', 'predictions'),
        ('prob_threshold', 0.6),
        ('stop_loss_pct', 0.05),
        ('take_profit_pct', 0.10),
        ('position_size_method', 'fixed'),  # 'fixed', 'kelly', 'volatility'
        ('risk_per_trade', 0.02),  # 2% risk per trade
        ('max_positions', 1),
        ('min_hold_period', 1),
        ('max_hold_period', 20),
    )
    
    def __init__(self):
        self.predictions = self.datas[0].predictions
        self.prediction_proba = getattr(self.datas[0], 'prediction_proba', None)
        
        # Risk management
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.hold_period = 0
        
        # Performance tracking
        self.trade_count = 0
        self.winning_trades = 0
        self.total_return = 0
        
        # Portfolio metrics
        self.equity_curve = []
        self.drawdowns = []
        
    def next(self):
        current_prediction = self.predictions[0]
        current_price = self.data.close[0]
        
        # Update hold period
        if self.position:
            self.hold_period += 1
        
        # Exit conditions
        if self.position:
            if self._should_exit(current_price, current_prediction):
                self.close()
                self._record_trade_exit(current_price)
                return
        
        # Entry conditions
        if not self.position and self._should_enter(current_prediction):
            position_size = self._calculate_position_size(current_price)
            if position_size > 0:
                self.buy(size=position_size)
                self._record_trade_entry(current_price)
        
        # Update portfolio metrics
        self._update_portfolio_metrics()
    
    def _should_enter(self, prediction: float) -> bool:
        """Determine if we should enter a trade."""
        if pd.isna(prediction):
            return False
        
        # Basic signal
        if prediction != 1:
            return False
        
        # Probability threshold check (if available)
        if self.prediction_proba is not None:
            prob = self.prediction_proba[0]
            if pd.isna(prob) or prob < self.params.prob_threshold:
                return False
        
        # Additional filters can be added here
        return True
    
    def _should_exit(self, current_price: float, prediction: float) -> bool:
        """Determine if we should exit a trade."""
        # Stop loss
        if self.stop_loss_price and current_price <= self.stop_loss_price:
            return True
        
        # Take profit
        if self.take_profit_price and current_price >= self.take_profit_price:
            return True
        
        # Signal reversal
        if prediction == 0:
            return True
        
        # Maximum hold period
        if self.hold_period >= self.params.max_hold_period:
            return True
        
        # Minimum hold period not met
        if self.hold_period < self.params.min_hold_period:
            return False
        
        return False
    
    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on selected method."""
        available_cash = self.broker.get_cash()
        
        if self.params.position_size_method == 'fixed':
            # Fixed percentage of portfolio
            position_value = available_cash * self.params.risk_per_trade
            return position_value / price
        
        elif self.params.position_size_method == 'volatility':
            # Volatility-based sizing
            returns = pd.Series([self.data.close[i] / self.data.close[i-1] - 1 
                               for i in range(-20, 0) if i >= -len(self.data)])
            volatility = returns.std() if len(returns) > 1 else 0.02
            
            if volatility > 0:
                position_value = (available_cash * self.params.risk_per_trade) / volatility
                return position_value / price
            else:
                return 0
        
        elif self.params.position_size_method == 'kelly':
            # Simplified Kelly criterion (requires historical win rate and avg return)
            # This is a placeholder - in practice, you'd calculate these from historical data
            win_rate = 0.55  # Would be calculated from historical performance
            avg_win = 0.05   # Average winning trade return
            avg_loss = 0.03  # Average losing trade return
            
            if win_rate > 0 and avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                position_value = available_cash * kelly_fraction
                return position_value / price
            else:
                return 0
        
        return 0
    
    def _record_trade_entry(self, price: float) -> None:
        """Record trade entry details."""
        self.entry_price = price
        self.stop_loss_price = price * (1 - self.params.stop_loss_pct)
        self.take_profit_price = price * (1 + self.params.take_profit_pct)
        self.hold_period = 0
        self.trade_count += 1
    
    def _record_trade_exit(self, price: float) -> None:
        """Record trade exit details."""
        if self.entry_price:
            trade_return = (price - self.entry_price) / self.entry_price
            self.total_return += trade_return
            
            if trade_return > 0:
                self.winning_trades += 1
        
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.hold_period = 0
    
    def _update_portfolio_metrics(self) -> None:
        """Update portfolio performance metrics."""
        portfolio_value = self.broker.get_value()
        self.equity_curve.append(portfolio_value)
        
        # Calculate drawdown
        if len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            current_dd = (portfolio_value - peak) / peak
            self.drawdowns.append(current_dd)


class RiskManager:
    """Risk management utilities for the trading system."""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)."""
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        var = RiskManager.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = equity_curve.expanding().max()
        
        # Handle edge case where peak values are zero
        mask = peak != 0
        if not mask.any():
            return 0.0  # No drawdown if equity curve is all zeros
        
        drawdown = pd.Series(index=equity_curve.index, dtype=float)
        drawdown[mask] = (equity_curve[mask] - peak[mask]) / peak[mask]
        drawdown[~mask] = 0.0  # Set drawdown to 0 where peak is 0
        
        return drawdown.min()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        std_dev = excess_returns.std()
        mean_excess = excess_returns.mean()
        
        # Handle edge case of zero volatility or when returns are constant
        if std_dev == 0 or np.isnan(std_dev) or np.isclose(std_dev, 0, atol=1e-8):
            return 0.0
        
        return mean_excess / std_dev * np.sqrt(252)


class BacktestEngine:
    """
    Main backtesting engine with advanced features.
    """
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
        
    def prepare_data_with_predictions(self, data: pd.DataFrame, predictions: np.ndarray,
                                    prediction_proba: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Prepare data with predictions for backtesting."""
        bt_data = data.copy()
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in bt_data.columns:
                # Try lowercase version
                if col.lower() in bt_data.columns:
                    bt_data[col] = bt_data[col.lower()]
                else:
                    raise ValueError(f"Required column {col} not found in data")
        
        # Add predictions
        if len(predictions) != len(bt_data):
            min_len = min(len(predictions), len(bt_data))
            bt_data = bt_data.iloc[:min_len]
            predictions = predictions[:min_len]
        
        bt_data['predictions'] = predictions
        
        if prediction_proba is not None:
            if len(prediction_proba.shape) > 1:
                # Take probability of positive class
                bt_data['prediction_proba'] = prediction_proba[:, 1]
            else:
                bt_data['prediction_proba'] = prediction_proba
        
        return bt_data
    
    def run_backtest(self, data: pd.DataFrame, strategy_class=MLStrategy,
                    strategy_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a backtest with the specified strategy."""
        cerebro = bt.Cerebro()
        
        # Add strategy
        if strategy_params:
            cerebro.addstrategy(strategy_class, **strategy_params)
        else:
            cerebro.addstrategy(strategy_class)
        
        # Create custom data feed
        class PredictionsData(bt.feeds.PandasData):
            lines = ('predictions', 'prediction_proba')
            params = (
                ('predictions', -1),
                ('prediction_proba', -1),
            )
        
        # Add data
        data_feed = PredictionsData(dataname=data)
        cerebro.adddata(data_feed)
        
        # Set broker parameters
        cerebro.broker.set_cash(self.initial_capital)
        cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Run backtest
        results = cerebro.run()
        strategy_instance = results[0]
        
        # Extract results
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Get analyzer results
        time_returns = strategy_instance.analyzers.time_return.get_analysis()
        drawdown_info = strategy_instance.analyzers.drawdown.get_analysis()
        trade_info = strategy_instance.analyzers.trades.get_analysis()
        
        results_dict = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': drawdown_info.get('max', {}).get('drawdown', 0),
            'total_trades': trade_info.get('total', {}).get('total', 0),
            'winning_trades': trade_info.get('won', {}).get('total', 0),
            'losing_trades': trade_info.get('lost', {}).get('total', 0),
            'win_rate': (trade_info.get('won', {}).get('total', 0) / 
                        max(1, trade_info.get('total', {}).get('total', 1))),
            'time_returns': time_returns,
            'strategy_instance': strategy_instance
        }
        
        return results_dict
    
    def run_walk_forward_backtest(self, data: pd.DataFrame, predictions: np.ndarray,
                                model_trainer, window_size: int = 252,
                                rebalance_freq: int = 21) -> Dict[str, Any]:
        """Run walk-forward backtesting with model retraining."""
        results = []
        
        for start in range(window_size, len(data) - rebalance_freq, rebalance_freq):
            # Training period
            train_end = start
            train_start = max(0, train_end - window_size)
            
            # Testing period
            test_start = start
            test_end = min(len(data), start + rebalance_freq)
            
            # Get test data
            test_data = data.iloc[test_start:test_end]
            test_predictions = predictions[test_start:test_end]
            
            # Prepare data for backtesting
            bt_data = self.prepare_data_with_predictions(test_data, test_predictions)
            
            # Run backtest for this period
            period_results = self.run_backtest(bt_data)
            period_results['period_start'] = test_data.index[0]
            period_results['period_end'] = test_data.index[-1]
            
            results.append(period_results)
        
        return self._aggregate_walk_forward_results(results)
    
    def _aggregate_walk_forward_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from walk-forward backtesting."""
        total_return = 1.0
        max_drawdown = 0.0
        total_trades = 0
        winning_trades = 0
        
        all_returns = []
        
        for result in results:
            total_return *= (1 + result['total_return'])
            max_drawdown = min(max_drawdown, result['max_drawdown'])
            total_trades += result['total_trades']
            winning_trades += result['winning_trades']
            
            # Combine time returns
            for date, ret in result['time_returns'].items():
                all_returns.append(ret)
        
        aggregated = {
            'total_return': total_return - 1,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': winning_trades / max(1, total_trades),
            'all_returns': all_returns,
            'period_results': results
        }
        
        return aggregated
    
    def generate_performance_report(self, results: Dict[str, Any], 
                                  output_file: Optional[str] = None) -> pd.Series:
        """Generate comprehensive performance report."""
        if 'time_returns' not in results:
            print("No time returns data available for report generation")
            return pd.Series()
        
        returns_series = pd.Series(results['time_returns'])
        returns_series.index = pd.to_datetime(returns_series.index)
        
        if output_file:
            # Generate QuantStats HTML report
            qs.reports.html(returns_series, output=output_file, title="Trading Strategy Performance")
            print(f"Performance report saved to {output_file}")
        
        return returns_series
    
    def compare_strategies(self, strategies_data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Compare performance of multiple strategies."""
        comparison_data = []
        
        for strategy_name, results in strategies_data.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{results['total_return']:.2%}",
                'Max Drawdown': f"{results['max_drawdown']:.2%}",
                'Win Rate': f"{results['win_rate']:.2%}",
                'Total Trades': results['total_trades'],
                'Final Value': f"${results['final_value']:,.2f}"
            })
        
        return pd.DataFrame(comparison_data)
    
    def optimize_strategy_parameters(self, data: pd.DataFrame, 
                                   param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search."""
        best_return = -np.inf
        best_params = {}
        results = {}
        
        # Generate all parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))
            
            try:
                # Run backtest with these parameters
                result = self.run_backtest(data, strategy_params=params)
                
                # Use total return as optimization metric
                if result['total_return'] > best_return:
                    best_return = result['total_return']
                    best_params = params.copy()
                
                results[str(params)] = result
                
            except Exception as e:
                print(f"Error with parameters {params}: {str(e)}")
                continue
        
        return {
            'best_params': best_params,
            'best_return': best_return,
            'all_results': results
        }