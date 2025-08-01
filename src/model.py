"""
Model architecture and training module for the algorithmic trading system.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from typing import Dict, Any, Tuple, List, Optional
import joblib
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Model training and evaluation class with advanced features.
    """
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.feature_importance = {}
        
    def create_base_models(self) -> Dict[str, Any]:
        """Create base models with default parameters."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                max_depth=6,
                n_estimators=200,
                learning_rate=0.05,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        }
        
        return models
    
    def get_hyperparameter_space(self, model_name: str) -> Dict[str, Any]:
        """Define hyperparameter search space for Bayesian optimization."""
        spaces = {
            'random_forest': {
                'n_estimators': hp.choice('n_estimators', [100, 150, 200, 300]),
                'max_depth': hp.choice('max_depth', [5, 10, 15, 20, None]),
                'min_samples_split': hp.choice('min_samples_split', [2, 4, 6, 8]),
                'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4, 6]),
                'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
            },
            'xgboost': {
                'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8]),
                'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400]),
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
                'gamma': hp.uniform('gamma', 0, 0.5),
                'subsample': hp.uniform('subsample', 0.6, 1.0),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
                'reg_alpha': hp.uniform('reg_alpha', 0, 1),
                'reg_lambda': hp.uniform('reg_lambda', 0, 1)
            }
        }
        
        return spaces.get(model_name, {})
    
    def objective_function(self, params: Dict[str, Any], model_name: str, 
                          X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Objective function for Bayesian optimization."""
        try:
            if model_name == 'random_forest':
                model = RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,
                    **params
                )
            elif model_name == 'xgboost':
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    random_state=42,
                    n_jobs=-1,
                    **params
                )
            else:
                return {'loss': 1, 'status': STATUS_OK}
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            
            return {'loss': -np.mean(scores), 'status': STATUS_OK}
        
        except Exception as e:
            return {'loss': 1, 'status': STATUS_OK}
    
    def bayesian_optimization(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                            max_evals: int = 100) -> Dict[str, Any]:
        """Perform Bayesian optimization for hyperparameter tuning."""
        space = self.get_hyperparameter_space(model_name)
        
        if not space:
            print(f"No hyperparameter space defined for {model_name}")
            return {}
        
        trials = Trials()
        
        def objective(params):
            return self.objective_function(params, model_name, X, y)
        
        best_params = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            verbose=False
        )
        
        self.best_params[model_name] = best_params
        return best_params
    
    def train_single_model(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                          use_bayesian_opt: bool = False, max_evals: int = 50) -> Any:
        """Train a single model with optional Bayesian optimization."""
        if use_bayesian_opt:
            print(f"Performing Bayesian optimization for {model_name}...")
            best_params = self.bayesian_optimization(model_name, X, y, max_evals)
        else:
            best_params = {}
        
        # Create model with optimized parameters
        if model_name == 'random_forest':
            model = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                **best_params
            )
        elif model_name == 'xgboost':
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1,
                **best_params
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Train the model
        model.fit(X, y)
        self.models[model_name] = model
        
        # Store feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_name] = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return model
    
    def create_stacking_model(self, X: pd.DataFrame, y: pd.Series,
                            base_models: Optional[List[str]] = None) -> StackingClassifier:
        """Create and train a stacking ensemble model."""
        if base_models is None:
            base_models = ['random_forest', 'xgboost']
        
        # Ensure base models are trained
        estimators = []
        for model_name in base_models:
            if model_name not in self.models:
                print(f"Training {model_name} for stacking...")
                self.train_single_model(model_name, X, y)
            estimators.append((model_name, self.models[model_name]))
        
        # Create stacking classifier
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=TimeSeriesSplit(n_splits=3)
        )
        
        # Train stacking model
        stacking_model.fit(X, y)
        self.models['stacking'] = stacking_model
        
        return stacking_model
    
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance with multiple metrics."""
        predictions = model.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted'),
            'recall': recall_score(y, predictions, average='weighted'),
            'f1_score': f1_score(y, predictions, average='weighted')
        }
        
        return metrics
    
    def walk_forward_optimization(self, X: pd.DataFrame, y: pd.Series, 
                                model_name: str, window_size: int = 252,
                                step_size: int = 21) -> Tuple[List[float], List[Any]]:
        """Perform walk-forward optimization."""
        accuracies = []
        models = []
        
        for start in range(0, len(X) - window_size, step_size):
            end = start + window_size
            
            # Training data
            X_train = X.iloc[start:end]
            y_train = y.iloc[start:end]
            
            # Test data (next period)
            X_test = X.iloc[end:min(end + step_size, len(X))]
            y_test = y.iloc[end:min(end + step_size, len(y))]
            
            if len(X_test) == 0:
                break
            
            # Train model on training window
            model = self.train_single_model(model_name, X_train, y_train)
            models.append(model)
            
            # Evaluate on test period
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)
        
        return accuracies, models
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get a summary of feature importance across all models."""
        if not self.feature_importance:
            print("No feature importance data available. Train models first.")
            return pd.DataFrame()
        
        # Combine importance from all models
        all_importance = pd.DataFrame()
        
        for model_name, importance_df in self.feature_importance.items():
            importance_df = importance_df.copy()
            importance_df['model'] = model_name
            all_importance = pd.concat([all_importance, importance_df], ignore_index=True)
        
        # Calculate average importance across models
        avg_importance = all_importance.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)
        
        return avg_importance
    
    def save_models(self, filepath: str) -> None:
        """Save trained models to disk."""
        model_data = {
            'models': self.models,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str) -> None:
        """Load trained models from disk."""
        model_data = joblib.load(filepath)
        self.models = model_data.get('models', {})
        self.best_params = model_data.get('best_params', {})
        self.feature_importance = model_data.get('feature_importance', {})
        print(f"Models loaded from {filepath}")
    
    def predict_single_stock(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions for a single stock using specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        return self.models[model_name].predict(X)
    
    def predict_proba_single_stock(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities for a single stock."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        return self.models[model_name].predict_proba(X)
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get a summary of all trained models."""
        summary_data = []
        
        for model_name, model in self.models.items():
            summary_data.append({
                'model_name': model_name,
                'model_type': type(model).__name__,
                'parameters': str(model.get_params()) if hasattr(model, 'get_params') else 'N/A'
            })
        
        return pd.DataFrame(summary_data)