"""
Feature engineering module for the algorithmic trading system.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
import shap
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering class for creating and processing trading features.
    """
    
    def __init__(self):
        self.scaler_standard = StandardScaler()
        self.scaler_minmax = MinMaxScaler()
        self.features_created = []
        
    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis indicators."""
        df = data.copy()
        
        # Ensure we have the required columns
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        
        # Moving averages
        df = self._add_moving_averages(df)
        
        # MACD indicators
        df = self._add_macd_indicators(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features."""
        periods = [5, 10, 20, 50, 200]
        
        for period in periods:
            df[f"ma_{period}"] = df["Adj Close"].rolling(window=period).mean()
            df[f"ma_{period}_ratio"] = df["Adj Close"] / df[f"ma_{period}"]
        
        # Moving average crossovers
        df['ma_5_10_cross'] = (df['ma_5'] > df['ma_10']).astype(int)
        df['ma_10_20_cross'] = (df['ma_10'] > df['ma_20']).astype(int)
        df['ma_20_50_cross'] = (df['ma_20'] > df['ma_50']).astype(int)
        
        self.features_created.extend([f"ma_{p}" for p in periods] + 
                                   [f"ma_{p}_ratio" for p in periods] +
                                   ['ma_5_10_cross', 'ma_10_20_cross', 'ma_20_50_cross'])
        
        return df
    
    def _add_macd_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD technical indicators."""
        # Calculate MACD
        ema_12 = df['Adj Close'].ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = df['Adj Close'].ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = macd - signal
        df['macd_crossover'] = (macd > signal).astype(int)
        
        self.features_created.extend(['macd', 'macd_signal', 'macd_histogram', 'macd_crossover'])
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume changes over different periods
        periods = [1, 5, 21, 63, 126, 252]
        
        for period in periods:
            df[f'vol_{period}d'] = np.log(df['Volume']).diff(period)
            if period > 1:
                df[f'vol_std_{period}d'] = df['vol_1d'].rolling(period).std()
        
        # Volume moving average
        df['vol_ma_20'] = df['Volume'].rolling(20).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_ma_20']
        
        # On-balance volume
        df['obv'] = (df['Volume'] * np.where(df['Adj Close'].diff() > 0, 1, -1)).cumsum()
        
        self.features_created.extend([f'vol_{p}d' for p in periods] + 
                                   [f'vol_std_{p}d' for p in periods[1:]] +
                                   ['vol_ma_20', 'vol_ratio', 'obv'])
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Price returns over different periods
        periods = [1, 5, 10, 21, 63]
        
        for period in periods:
            df[f'return_{period}d'] = df['Adj Close'].pct_change(period)
        
        # High-Low spreads
        df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
        df['oc_spread'] = (df['Close'] - df['Open']) / df['Open']
        
        # Price position within daily range
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Gap detection
        df['gap_up'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) > 0.02).astype(int)
        df['gap_down'] = ((df['Close'].shift(1) - df['Open']) / df['Close'].shift(1) > 0.02).astype(int)
        
        self.features_created.extend([f'return_{p}d' for p in periods] + 
                                   ['hl_spread', 'oc_spread', 'price_position', 'gap_up', 'gap_down'])
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        # Rolling volatility
        periods = [5, 10, 20, 60]
        
        for period in periods:
            df[f'volatility_{period}d'] = df['return_1d'].rolling(period).std()
        
        # Bollinger Bands
        df['bb_upper'] = df['ma_20'] + (2 * df['volatility_20d'])
        df['bb_lower'] = df['ma_20'] - (2 * df['volatility_20d'])
        df['bb_position'] = (df['Adj Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        df['tr'] = np.maximum(df['High'] - df['Low'], 
                             np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                       abs(df['Low'] - df['Close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()
        
        self.features_created.extend([f'volatility_{p}d' for p in periods] + 
                                   ['bb_upper', 'bb_lower', 'bb_position', 'tr', 'atr'])
        
        return df
    
    def create_labels(self, df: pd.DataFrame, method: str = 'binary_return', 
                     lookahead: int = 1) -> pd.Series:
        """Create target labels for prediction."""
        if method == 'binary_return':
            # Binary classification: 1 if price goes up, 0 if down
            future_return = df['Adj Close'].shift(-lookahead) / df['Adj Close'] - 1
            labels = (future_return > 0).astype(int)
        elif method == 'threshold_return':
            # Classification with threshold
            future_return = df['Adj Close'].shift(-lookahead) / df['Adj Close'] - 1
            labels = pd.Series(np.where(future_return > 0.01, 1, 
                                      np.where(future_return < -0.01, -1, 0)), 
                             index=df.index)
        else:
            raise ValueError(f"Unknown labeling method: {method}")
        
        return labels
    
    def add_interaction_features(self, df: pd.DataFrame, 
                               feature_pairs: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
        """Add interaction features between existing features."""
        if feature_pairs is None:
            # Default important feature pairs
            feature_pairs = [
                ('ma_5', 'ma_20'),
                ('macd', 'macd_signal'),
                ('vol_ratio', 'return_1d'),
                ('volatility_20d', 'return_5d')
            ]
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df[f'{feat1}_{feat2}_interaction'] = df[feat1] * df[feat2]
                self.features_created.append(f'{feat1}_{feat2}_interaction')
        
        return df
    
    def standardize_features(self, df: pd.DataFrame, 
                           fit: bool = True) -> pd.DataFrame:
        """Standardize features using StandardScaler."""
        if fit:
            scaled_data = self.scaler_standard.fit_transform(df)
        else:
            scaled_data = self.scaler_standard.transform(df)
        
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    def normalize_features(self, df: pd.DataFrame, 
                         fit: bool = True) -> pd.DataFrame:
        """Normalize features using MinMaxScaler."""
        if fit:
            scaled_data = self.scaler_minmax.fit_transform(df)
        else:
            scaled_data = self.scaler_minmax.transform(df)
        
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    def select_features_rfe(self, X: pd.DataFrame, y: pd.Series, 
                          estimator, n_features: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using Recursive Feature Elimination."""
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = rfe.fit_transform(X, y)
        selected_features = X.columns[rfe.support_].tolist()
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def get_shap_importance(self, model, X: pd.DataFrame, 
                          max_samples: int = 1000) -> pd.DataFrame:
        """Calculate SHAP feature importance."""
        # Sample data if too large
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=42)
        else:
            X_sample = X
        
        # Create SHAP explainer
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        
        # Calculate mean absolute SHAP values
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values.values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def process_stock_data(self, data: pd.DataFrame, 
                         create_labels: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Complete feature engineering pipeline for a single stock."""
        df = data.copy()
        
        # Fill missing values
        df = df.fillna(method="bfill").fillna(method="ffill")
        
        # Create all features
        df = self.create_technical_indicators(df)
        df = self.add_interaction_features(df)
        
        # Remove 'Close' if both 'Close' and 'Adj Close' exist (after feature creation)
        if 'Close' in df.columns and 'Adj Close' in df.columns:
            df = df.drop('Close', axis=1)
        
        # Create labels if requested
        labels = None
        if create_labels:
            labels = self.create_labels(df)
            # Remove rows where labels are NaN
            valid_idx = ~labels.isna()
            df = df[valid_idx]
            labels = labels[valid_idx]
        
        # Clean up data - remove rows with any NaN values
        df = df.dropna()
        if labels is not None:
            labels = labels.loc[df.index]
        
        # Remove original OHLCV columns for modeling (keep only engineered features)
        columns_to_remove = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
        df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
        
        return df, labels