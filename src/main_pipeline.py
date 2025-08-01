"""
Main pipeline for the enhanced algorithmic trading system.
This script demonstrates the complete workflow including SHAP analysis,
RFE, Bayesian optimization, and advanced backtesting.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import warnings

# Import our custom modules
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model import ModelTrainer
from backtesting import BacktestEngine, MLStrategy
from utils import Logger, PerformanceMetrics, Visualizer, ConfigManager

warnings.filterwarnings('ignore')


class TradingPipeline:
    """
    Complete trading pipeline with advanced features.
    """
    
    def __init__(self, config_file: str = "config.json"):
        self.config = ConfigManager(config_file)
        self.logger = Logger("trading_pipeline", "logs/pipeline.log")
        
        # Initialize components
        self.data_loader = DataLoader(self.config.get('data.data_directory', 'data'))
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.backtest_engine = BacktestEngine(
            initial_capital=self.config.get('backtesting.initial_capital', 10000),
            commission=self.config.get('backtesting.commission', 0.001)
        )
        self.visualizer = Visualizer()
        
        # Storage for results
        self.results = {}
        self.feature_importance_results = {}
        
    def load_and_prepare_data(self, ticker: str = "AAPL", 
                            start_date: str = "2018-01-01", 
                            end_date: str = "2021-12-31") -> tuple:
        """Load and prepare data for a single stock."""
        self.logger.info(f"Loading data for {ticker}")
        
        # Load data
        raw_data = self.data_loader.fetch_yahoo_data(ticker, start_date, end_date)
        if raw_data is None or raw_data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Validate data
        if not self.data_loader.validate_data(raw_data):
            raise ValueError(f"Invalid data format for ticker {ticker}")
        
        # Feature engineering
        self.logger.info("Creating features and labels")
        features, labels = self.feature_engineer.process_stock_data(raw_data)
        
        # Split data for time series (80% train, 20% test)
        split_index = int(len(features) * 0.8)
        
        X_train = features.iloc[:split_index]
        X_test = features.iloc[split_index:]
        y_train = labels.iloc[:split_index]
        y_test = labels.iloc[split_index:]
        
        self.logger.info(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test, raw_data
    
    def analyze_feature_importance_with_shap(self, model, X_sample: pd.DataFrame, 
                                           feature_name: str = "default") -> dict:
        """Analyze feature importance using SHAP."""
        self.logger.info(f"Running SHAP analysis for {feature_name}")
        
        try:
            # Get SHAP importance
            shap_importance = self.feature_engineer.get_shap_importance(model, X_sample)
            
            # Create SHAP explainer for detailed analysis
            explainer = shap.Explainer(model, X_sample.sample(min(100, len(X_sample))))
            shap_values = explainer(X_sample.sample(min(500, len(X_sample))))
            
            # Save SHAP results
            self.feature_importance_results[feature_name] = {
                'importance_df': shap_importance,
                'shap_values': shap_values,
                'explainer': explainer
            }
            
            # Create SHAP plots
            self._create_shap_plots(shap_values, feature_name)
            
            return shap_importance
            
        except Exception as e:
            self.logger.error(f"SHAP analysis failed: {str(e)}")
            return pd.DataFrame()
    
    def _create_shap_plots(self, shap_values, feature_name: str):
        """Create and save SHAP visualization plots."""
        try:
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, show=False)
            plt.title(f'SHAP Summary Plot - {feature_name}')
            plt.tight_layout()
            plt.savefig(f'reports/shap_summary_{feature_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance - {feature_name}')
            plt.tight_layout()
            plt.savefig(f'reports/shap_importance_{feature_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"SHAP plots saved for {feature_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create SHAP plots: {str(e)}")
    
    def run_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            n_features: int = 20) -> tuple:
        """Run Recursive Feature Elimination."""
        self.logger.info("Running Recursive Feature Elimination")
        
        # Use Random Forest for feature selection
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Apply RFE
        X_selected, selected_features = self.feature_engineer.select_features_rfe(
            X_train, y_train, rf_selector, n_features
        )
        
        self.logger.info(f"Selected {len(selected_features)} features using RFE")
        self.logger.info(f"Selected features: {selected_features}")
        
        return X_selected, selected_features
    
    def train_models_with_optimization(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Train models with Bayesian optimization."""
        self.logger.info("Training models with Bayesian optimization")
        
        model_results = {}
        
        # Train Random Forest with optimization
        self.logger.info("Training Random Forest with Bayesian optimization...")
        rf_model = self.model_trainer.train_single_model(
            'random_forest', X_train, y_train, 
            use_bayesian_opt=True, max_evals=30
        )
        model_results['random_forest'] = rf_model
        
        # Train XGBoost with optimization
        self.logger.info("Training XGBoost with Bayesian optimization...")
        xgb_model = self.model_trainer.train_single_model(
            'xgboost', X_train, y_train, 
            use_bayesian_opt=True, max_evals=30
        )
        model_results['xgboost'] = xgb_model
        
        # Create stacking ensemble
        self.logger.info("Creating stacking ensemble...")
        stacking_model = self.model_trainer.create_stacking_model(X_train, y_train)
        model_results['stacking'] = stacking_model
        
        return model_results
    
    def evaluate_models(self, models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Evaluate all models and return comparison."""
        self.logger.info("Evaluating models")
        
        evaluation_results = []
        
        for model_name, model in models.items():
            metrics = self.model_trainer.evaluate_model(model, X_test, y_test)
            metrics['model'] = model_name
            evaluation_results.append(metrics)
        
        results_df = pd.DataFrame(evaluation_results)
        self.logger.info(f"Model evaluation completed")
        
        return results_df
    
    def run_backtesting(self, raw_data: pd.DataFrame, model, model_name: str) -> dict:
        """Run backtesting with the trained model."""
        self.logger.info(f"Running backtesting for {model_name}")
        
        # Prepare data for backtesting
        features, labels = self.feature_engineer.process_stock_data(raw_data)
        predictions = model.predict(features)
        
        # Get prediction probabilities if available
        try:
            prediction_proba = model.predict_proba(features)
        except:
            prediction_proba = None
        
        # Prepare backtesting data
        bt_data = self.backtest_engine.prepare_data_with_predictions(
            raw_data.iloc[len(raw_data)-len(predictions):], 
            predictions, 
            prediction_proba
        )
        
        # Run backtest with risk management
        strategy_params = {
            'prob_threshold': 0.6,
            'stop_loss_pct': self.config.get('backtesting.stop_loss_pct', 0.05),
            'take_profit_pct': self.config.get('backtesting.take_profit_pct', 0.10),
            'risk_per_trade': self.config.get('backtesting.risk_per_trade', 0.02),
            'position_size_method': 'volatility'
        }
        
        results = self.backtest_engine.run_backtest(bt_data, MLStrategy, strategy_params)
        
        # Generate performance report
        returns_series = self.backtest_engine.generate_performance_report(
            results, f'reports/quantstats_{model_name}_enhanced.html'
        )
        
        return results
    
    def run_walk_forward_analysis(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Run walk-forward optimization analysis."""
        self.logger.info("Running walk-forward optimization")
        
        # Run walk-forward for Random Forest
        rf_accuracies, rf_models = self.model_trainer.walk_forward_optimization(
            X, y, 'random_forest', window_size=252, step_size=21
        )
        
        # Run walk-forward for XGBoost
        xgb_accuracies, xgb_models = self.model_trainer.walk_forward_optimization(
            X, y, 'xgboost', window_size=252, step_size=21
        )
        
        walk_forward_results = {
            'random_forest': {
                'accuracies': rf_accuracies,
                'mean_accuracy': np.mean(rf_accuracies),
                'std_accuracy': np.std(rf_accuracies)
            },
            'xgboost': {
                'accuracies': xgb_accuracies,
                'mean_accuracy': np.mean(xgb_accuracies),
                'std_accuracy': np.std(xgb_accuracies)
            }
        }
        
        self.logger.info("Walk-forward optimization completed")
        return walk_forward_results
    
    def create_comprehensive_report(self, ticker: str, results: dict):
        """Create a comprehensive analysis report."""
        self.logger.info("Creating comprehensive report")
        
        # Create summary report
        report = f"""
# Algorithmic Trading Analysis Report
## Ticker: {ticker}
## Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}

### Model Performance Summary
"""
        
        if 'model_evaluation' in results:
            eval_df = results['model_evaluation']
            report += eval_df.to_string(index=False)
            report += "\n\n"
        
        if 'walk_forward' in results:
            wf_results = results['walk_forward']
            report += "### Walk-Forward Analysis Results\n"
            for model_name, wf_data in wf_results.items():
                report += f"**{model_name.title()}:**\n"
                report += f"- Mean Accuracy: {wf_data['mean_accuracy']:.4f}\n"
                report += f"- Std Accuracy: {wf_data['std_accuracy']:.4f}\n"
            report += "\n"
        
        if 'backtesting' in results:
            bt_results = results['backtesting']
            for model_name, bt_data in bt_results.items():
                report += f"### Backtesting Results - {model_name.title()}\n"
                report += f"- Total Return: {bt_data['total_return']:.2%}\n"
                report += f"- Max Drawdown: {bt_data['max_drawdown']:.2%}\n"
                report += f"- Win Rate: {bt_data['win_rate']:.2%}\n"
                report += f"- Total Trades: {bt_data['total_trades']}\n\n"
        
        # Save report
        with open(f'reports/{ticker}_comprehensive_report.md', 'w') as f:
            f.write(report)
        
        self.logger.info(f"Comprehensive report saved for {ticker}")
    
    def run_complete_pipeline(self, ticker: str = "AAPL", 
                            start_date: str = "2018-01-01", 
                            end_date: str = "2021-12-31"):
        """Run the complete enhanced trading pipeline."""
        self.logger.info(f"Starting complete pipeline for {ticker}")
        
        try:
            # 1. Load and prepare data
            X_train, X_test, y_train, y_test, raw_data = self.load_and_prepare_data(
                ticker, start_date, end_date
            )
            
            # 2. Feature selection using RFE
            X_train_selected, selected_features = self.run_feature_selection(X_train, y_train)
            X_test_selected = X_test[selected_features]
            
            # 3. Train models with Bayesian optimization
            models = self.train_models_with_optimization(X_train_selected, y_train)
            
            # 4. SHAP analysis for best model
            best_model = models['stacking']  # Use stacking as the best model
            shap_importance = self.analyze_feature_importance_with_shap(
                best_model, X_train_selected.sample(min(1000, len(X_train_selected))), 
                f"{ticker}_stacking"
            )
            
            # 5. Model evaluation
            model_evaluation = self.evaluate_models(models, X_test_selected, y_test)
            
            # 6. Walk-forward analysis
            walk_forward_results = self.run_walk_forward_analysis(
                pd.concat([X_train_selected, X_test_selected]), 
                pd.concat([y_train, y_test])
            )
            
            # 7. Backtesting
            backtesting_results = {}
            for model_name, model in models.items():
                bt_results = self.run_backtesting(raw_data, model, f"{ticker}_{model_name}")
                backtesting_results[model_name] = bt_results
            
            # 8. Compile all results
            self.results[ticker] = {
                'model_evaluation': model_evaluation,
                'shap_importance': shap_importance,
                'walk_forward': walk_forward_results,
                'backtesting': backtesting_results,
                'selected_features': selected_features
            }
            
            # 9. Create comprehensive report
            self.create_comprehensive_report(ticker, self.results[ticker])
            
            self.logger.info(f"Pipeline completed successfully for {ticker}")
            return self.results[ticker]
            
        except Exception as e:
            self.logger.error(f"Pipeline failed for {ticker}: {str(e)}")
            raise e


def main():
    """Main function to run the enhanced trading pipeline."""
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Initialize pipeline
    pipeline = TradingPipeline()
    
    # Example: Run pipeline for a single stock
    ticker = "AAPL"
    results = pipeline.run_complete_pipeline(ticker)
    
    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETED FOR {ticker}")
    print(f"{'='*50}")
    
    # Print summary results
    if 'model_evaluation' in results:
        print("\nModel Performance:")
        print(results['model_evaluation'])
    
    if 'shap_importance' in results and not results['shap_importance'].empty:
        print(f"\nTop 10 Most Important Features (SHAP):")
        print(results['shap_importance'].head(10))
    
    print(f"\nDetailed reports saved in 'reports/' directory")
    print(f"Logs saved in 'logs/' directory")


if __name__ == "__main__":
    main()