"""
Model Training Module - Optimized for FlightRank 2025 Competition
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from typing import Optional

class ModelTrainer:
    """High-performance model training with optimized hyperparameters"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
        # Optimized LightGBM parameters for ranking
        self.lgb_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [3, 5, 10],
            'boosting_type': 'gbdt',
            'num_leaves': 63,  # Increased for better performance
            'learning_rate': 0.05,  # Lower for better convergence
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'random_state': random_state,
            'force_col_wise': True,  # For faster training
            'num_threads': -1
        }
        
        # Optimized CatBoost parameters
        self.cb_params = {
            'loss_function': 'RMSE',
            'custom_metric': ['NDCG:top=3'],
            'iterations': 2000,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 3,
            'random_seed': random_state,
            'verbose': False,
            'early_stopping_rounds': 200,
            'thread_count': -1,
            'task_type': 'CPU'
        }
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series, group_train: np.ndarray, 
                      X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, 
                      group_val: Optional[np.ndarray] = None, verbose: bool = True) -> lgb.Booster:
        """Train optimized LightGBM ranking model"""
        
        # Create training dataset
        train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        # Add validation set if provided
        if X_val is not None and y_val is not None and group_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Adjust verbosity
        params = self.lgb_params.copy()
        if not verbose:
            params['verbose'] = -1
        
        # Train model with early stopping
        callbacks = [lgb.early_stopping(200)]
        if verbose:
            callbacks.append(lgb.log_evaluation(100))
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=3000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        if verbose:
            print(f"✓ LightGBM trained with {model.best_iteration} iterations")
            print(f"✓ Best score: {model.best_score['train']['ndcg@3']:.6f}")
        
        return model
    
    def train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series, 
                      verbose: bool = True) -> cb.CatBoost:
        """Train optimized CatBoost ranking model"""
        
        # CatBoost uses group_id instead of group boundaries
        params = self.cb_params.copy()
        if not verbose:
            params['verbose'] = False
        
        model = cb.CatBoost(params)
        
        model.fit(
            X_train, y_train,
            group_id=groups_train,
            eval_set=[(X_train, y_train)],
            plot=False,
            verbose=verbose
        )
        
        if verbose:
            print(f"✓ CatBoost trained with {model.get_best_iteration()} iterations")
        
        return model
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series, group_train: np.ndarray, 
                      groups_train: pd.Series) -> dict:
        """Train ensemble of models"""
        
        print("Training ensemble models...")
        
        # Train LightGBM
        lgb_model = self.train_lightgbm(X_train, y_train, group_train, verbose=False)
        
        # Train CatBoost
        cb_model = self.train_catboost(X_train, y_train, groups_train, verbose=False)
        
        models = {
            'lightgbm': lgb_model,
            'catboost': cb_model
        }
        
        print("✓ Ensemble models trained successfully")
        
        return models
    
    def predict_ensemble(self, models: dict, X_test: pd.DataFrame, weights: Optional[dict] = None) -> np.ndarray:
        """Generate ensemble predictions"""
        
        if weights is None:
            weights = {'lightgbm': 0.7, 'catboost': 0.3}  # LightGBM usually better for ranking
        
        predictions = np.zeros(len(X_test))
        total_weight = sum(weights.values())
        
        for model_name, model in models.items():
            if model_name == 'lightgbm':
                pred = model.predict(X_test, num_iteration=model.best_iteration)
            else:  # catboost
                pred = model.predict(X_test)
            
            weight = weights.get(model_name, 1.0) / total_weight
            predictions += pred * weight
        
        return predictions
