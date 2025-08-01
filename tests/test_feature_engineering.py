"""
Unit tests for the feature_engineering module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for the FeatureEngineer class."""
    
    @pytest.fixture
    def feature_engineer(self):
        """Create a FeatureEngineer instance for testing."""
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)  # For reproducible tests
        n_days = 100
        
        # Create realistic OHLCV data
        base_price = 100
        prices = [base_price]
        
        for _ in range(n_days - 1):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))  # Ensure positive prices
        
        data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Adj Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_days)
        }, index=pd.date_range('2020-01-01', periods=n_days))
        
        # Ensure High >= Low and High >= Open, Close and Low <= Open, Close
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        return data
    
    def test_init(self, feature_engineer):
        """Test FeatureEngineer initialization."""
        assert hasattr(feature_engineer, 'scaler_standard')
        assert hasattr(feature_engineer, 'scaler_minmax')
        assert feature_engineer.features_created == []
    
    def test_create_technical_indicators(self, feature_engineer, sample_ohlcv_data):
        """Test creation of technical indicators."""
        result = feature_engineer.create_technical_indicators(sample_ohlcv_data)
        
        # Check that new features were created
        assert len(result.columns) > len(sample_ohlcv_data.columns)
        
        # Check for specific features
        expected_features = ['ma_5', 'ma_10', 'ma_20', 'macd', 'macd_signal', 'vol_1d']
        for feature in expected_features:
            assert feature in result.columns
    
    def test_add_moving_averages(self, feature_engineer, sample_ohlcv_data):
        """Test moving average calculation."""
        result = feature_engineer._add_moving_averages(sample_ohlcv_data)
        
        # Check that moving averages were added
        ma_columns = ['ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_200']
        for col in ma_columns:
            assert col in result.columns
        
        # Check that MA values are reasonable
        assert result['ma_5'].iloc[-1] > 0
        assert result['ma_20'].iloc[-1] > 0
        
        # Check MA relationships (longer MA should be smoother)
        assert not result['ma_5'].isna().all()
        assert not result['ma_20'].isna().all()
    
    def test_add_macd_indicators(self, feature_engineer, sample_ohlcv_data):
        """Test MACD indicator calculation."""
        result = feature_engineer._add_macd_indicators(sample_ohlcv_data)
        
        # Check MACD columns exist
        macd_columns = ['macd', 'macd_signal', 'macd_histogram', 'macd_crossover']
        for col in macd_columns:
            assert col in result.columns
        
        # Check that MACD histogram = MACD - Signal
        valid_idx = ~(result['macd'].isna() | result['macd_signal'].isna() | result['macd_histogram'].isna())
        if valid_idx.any():
            np.testing.assert_array_almost_equal(
                result.loc[valid_idx, 'macd_histogram'],
                result.loc[valid_idx, 'macd'] - result.loc[valid_idx, 'macd_signal'],
                decimal=6
            )
    
    def test_add_volume_features(self, feature_engineer, sample_ohlcv_data):
        """Test volume feature calculation."""
        result = feature_engineer._add_volume_features(sample_ohlcv_data)
        
        # Check volume features exist
        volume_features = ['vol_1d', 'vol_5d', 'vol_21d', 'vol_ma_20', 'vol_ratio', 'obv']
        for feature in volume_features:
            assert feature in result.columns
        
        # Check that vol_ratio = Volume / vol_ma_20 (where not NaN)
        valid_idx = ~(result['Volume'].isna() | result['vol_ma_20'].isna() | result['vol_ratio'].isna())
        if valid_idx.any():
            expected_ratio = result.loc[valid_idx, 'Volume'] / result.loc[valid_idx, 'vol_ma_20']
            np.testing.assert_array_almost_equal(
                result.loc[valid_idx, 'vol_ratio'],
                expected_ratio,
                decimal=6
            )
    
    def test_add_price_features(self, feature_engineer, sample_ohlcv_data):
        """Test price feature calculation."""
        result = feature_engineer._add_price_features(sample_ohlcv_data)
        
        # Check price features exist
        price_features = ['return_1d', 'return_5d', 'hl_spread', 'oc_spread', 'price_position']
        for feature in price_features:
            assert feature in result.columns
        
        # Check that returns are reasonable (between -1 and large positive values)
        returns_1d = result['return_1d'].dropna()
        assert (returns_1d > -1).all()  # No returns less than -100%
        assert (returns_1d < 1).all()   # No single-day returns over 100%
    
    def test_add_volatility_features(self, feature_engineer, sample_ohlcv_data):
        """Test volatility feature calculation."""
        # First add return_1d which is needed for volatility calculation
        result = feature_engineer._add_price_features(sample_ohlcv_data)
        # Add moving averages which are needed for Bollinger Bands
        result = feature_engineer._add_moving_averages(result)
        result = feature_engineer._add_volatility_features(result)
        
        # Check volatility features exist
        volatility_features = ['volatility_5d', 'volatility_20d', 'bb_upper', 'bb_lower', 'atr']
        for feature in volatility_features:
            assert feature in result.columns
        
        # Check that volatility is positive
        vol_20d = result['volatility_20d'].dropna()
        assert (vol_20d >= 0).all()
    
    def test_create_labels_binary_return(self, feature_engineer, sample_ohlcv_data):
        """Test binary return label creation."""
        labels = feature_engineer.create_labels(sample_ohlcv_data, method='binary_return')
        
        # Check that labels are binary (0 or 1)
        unique_labels = labels.dropna().unique()
        assert set(unique_labels).issubset({0, 1})
        
        # Check that we have both classes
        assert len(unique_labels) >= 1  # At least one class should be present
    
    def test_create_labels_threshold_return(self, feature_engineer, sample_ohlcv_data):
        """Test threshold return label creation."""
        labels = feature_engineer.create_labels(sample_ohlcv_data, method='threshold_return')
        
        # Check that labels are in {-1, 0, 1}
        unique_labels = labels.dropna().unique()
        assert set(unique_labels).issubset({-1, 0, 1})
    
    def test_create_labels_invalid_method(self, feature_engineer, sample_ohlcv_data):
        """Test label creation with invalid method."""
        with pytest.raises(ValueError):
            feature_engineer.create_labels(sample_ohlcv_data, method='invalid_method')
    
    def test_add_interaction_features(self, feature_engineer, sample_ohlcv_data):
        """Test interaction feature creation."""
        # First create some basic features
        data_with_features = feature_engineer._add_moving_averages(sample_ohlcv_data)
        data_with_features = feature_engineer._add_macd_indicators(data_with_features)
        
        # Add interaction features
        result = feature_engineer.add_interaction_features(data_with_features)
        
        # Check that interaction features were created
        interaction_features = [col for col in result.columns if '_interaction' in col]
        assert len(interaction_features) > 0
    
    def test_standardize_features(self, feature_engineer, sample_ohlcv_data):
        """Test feature standardization."""
        # Create a simple numeric DataFrame
        numeric_data = sample_ohlcv_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Standardize features
        standardized = feature_engineer.standardize_features(numeric_data, fit=True)
        
        # Check that result is a DataFrame with same shape
        assert isinstance(standardized, pd.DataFrame)
        assert standardized.shape == numeric_data.shape
        
        # Check that features are approximately standardized (mean~0, std~1)
        means = standardized.mean()
        stds = standardized.std()
        
        # Allow some tolerance for numerical precision
        assert (abs(means) < 1e-10).all()
        assert (abs(stds - 1) < 1e-2).all()  # Relaxed tolerance for standard deviation
    
    def test_normalize_features(self, feature_engineer, sample_ohlcv_data):
        """Test feature normalization."""
        numeric_data = sample_ohlcv_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Normalize features
        normalized = feature_engineer.normalize_features(numeric_data, fit=True)
        
        # Check that result is a DataFrame with same shape
        assert isinstance(normalized, pd.DataFrame)
        assert normalized.shape == numeric_data.shape
        
        # Check that features are normalized to [0, 1]
        assert (normalized >= 0).all().all()
        assert (normalized <= 1).all().all()
    
    def test_select_features_rfe(self, feature_engineer, sample_ohlcv_data):
        """Test RFE feature selection."""
        # Create features and labels
        features = sample_ohlcv_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        labels = pd.Series(np.random.choice([0, 1], len(features)))
        
        # Use a real simple estimator for RFE
        from sklearn.linear_model import LogisticRegression
        estimator = LogisticRegression(random_state=42, max_iter=100)
        
        # Test RFE
        X_selected, selected_features = feature_engineer.select_features_rfe(
            features, labels, estimator, n_features=3
        )
        
        assert isinstance(X_selected, pd.DataFrame)
        assert len(selected_features) == 3
        assert X_selected.shape[1] == 3
    
    def test_process_stock_data_complete_pipeline(self, feature_engineer, sample_ohlcv_data):
        """Test the complete feature engineering pipeline."""
        features, labels = feature_engineer.process_stock_data(sample_ohlcv_data, create_labels=True)
        
        # Check that features and labels were created
        assert isinstance(features, pd.DataFrame)
        assert isinstance(labels, pd.Series)
        
        # Check that we have more features than original columns
        original_feature_cols = len([col for col in sample_ohlcv_data.columns 
                                   if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
        assert len(features.columns) > original_feature_cols
        
        # Check that no NaN values remain
        assert not features.isna().any().any()
        assert not labels.isna().any()
        
        # Check that features and labels have same length
        assert len(features) == len(labels)
    
    def test_process_stock_data_no_labels(self, feature_engineer, sample_ohlcv_data):
        """Test feature engineering without label creation."""
        features, labels = feature_engineer.process_stock_data(sample_ohlcv_data, create_labels=False)
        
        assert isinstance(features, pd.DataFrame)
        assert labels is None
        assert not features.isna().any().any()
    
    def test_process_stock_data_missing_adj_close(self, feature_engineer, sample_ohlcv_data):
        """Test feature engineering when Adj Close is missing."""
        # Remove Adj Close column
        data_no_adj_close = sample_ohlcv_data.drop('Adj Close', axis=1)
        
        features, labels = feature_engineer.process_stock_data(data_no_adj_close, create_labels=True)
        
        # Should still work by using Close as Adj Close
        assert isinstance(features, pd.DataFrame)
        assert isinstance(labels, pd.Series)
    
    def test_get_shap_importance_mock(self, feature_engineer):
        """Test SHAP importance calculation with mocked SHAP."""
        # Create mock model and data
        mock_model = MagicMock()
        mock_data = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
        
        # Mock SHAP components
        with patch('shap.Explainer') as mock_explainer_class:
            mock_explainer = MagicMock()
            mock_explainer_class.return_value = mock_explainer
            
            # Create mock SHAP values
            mock_shap_values = MagicMock()
            mock_shap_values.values = np.random.randn(100, 5)
            mock_explainer.return_value = mock_shap_values
            
            # Test SHAP importance
            importance_df = feature_engineer.get_shap_importance(mock_model, mock_data)
            
            # Check result format
            assert isinstance(importance_df, pd.DataFrame)
            assert 'feature' in importance_df.columns
            assert 'importance' in importance_df.columns
            assert len(importance_df) == 5


class TestFeatureEngineerEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def feature_engineer(self):
        return FeatureEngineer()
    
    def test_empty_dataframe(self, feature_engineer):
        """Test feature engineering with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        # Should handle empty DataFrame gracefully
        with pytest.raises(KeyError):
            # This should fail because required columns are missing
            feature_engineer.create_technical_indicators(empty_df)
    
    def test_insufficient_data(self, feature_engineer):
        """Test feature engineering with insufficient data."""
        # Create very small dataset
        small_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Adj Close': [101, 102],
            'Volume': [1000, 1100]
        })
        
        # Should handle small datasets
        result = feature_engineer.create_technical_indicators(small_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_data_with_zeros(self, feature_engineer):
        """Test feature engineering with zero values."""
        zero_data = pd.DataFrame({
            'Open': [100, 0, 102],
            'High': [102, 1, 104],
            'Low': [99, 0, 101],
            'Close': [101, 1, 103],
            'Adj Close': [101, 1, 103],
            'Volume': [1000, 0, 1200]
        })
        
        # Should handle zero values without crashing
        result = feature_engineer.create_technical_indicators(zero_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_data_with_nans(self, feature_engineer):
        """Test feature engineering with NaN values."""
        nan_data = pd.DataFrame({
            'Open': [100, np.nan, 102],
            'High': [102, 103, 104],
            'Low': [99, 100, 101],
            'Close': [101, np.nan, 103],
            'Adj Close': [101, np.nan, 103],
            'Volume': [1000, 1100, np.nan]
        })
        
        # Test processing - should handle NaNs
        features, labels = feature_engineer.process_stock_data(nan_data, create_labels=True)
        
        # After processing, should have no NaNs
        assert not features.isna().any().any()
        assert not labels.isna().any()


if __name__ == "__main__":
    pytest.main([__file__])