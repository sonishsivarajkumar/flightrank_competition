"""
Hyperparameter Optimization Script for FlightRank 2025
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import optuna
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from evaluation import Evaluator
from data_loader import DataLoader

class HyperparameterOptimizer:
    """Optimize model hyperparameters using Optuna"""
    
    def __init__(self, data_path: str = ".", random_state: int = 42):
        self.data_path = data_path
        self.random_state = random_state
        self.evaluator = Evaluator()
        
        # Load and prepare data once
        self._prepare_data()
    
    def _prepare_data(self):
        """Load and prepare data for optimization"""
        print("Preparing data for hyperparameter optimization...")
        
        data_loader = DataLoader(self.data_path)
        train_df, _, _ = data_loader.load_data()
        
        feature_engineer = FeatureEngineer()
        train_fe = feature_engineer.engineer_features(train_df)
        
        # Prepare features
        exclude_cols = ['Id', 'ranker_id', 'selected']
        self.y = train_fe['selected']
        self.groups = train_fe['ranker_id']
        
        feature_cols = [col for col in train_fe.columns if col not in exclude_cols]
        self.X = train_fe[feature_cols].copy()
        
        # Handle categorical and missing values quickly
        for col in self.X.select_dtypes(include=['object', 'category']).columns:
            if self.X[col].nunique() > 1000:
                freq_map = self.X[col].value_counts()
                self.X[f'{col}_freq'] = self.X[col].map(freq_map).fillna(0)
                self.X.drop(col, axis=1, inplace=True)
            else:
                self.X[col] = pd.Categorical(self.X[col]).codes
        
        # Handle datetime
        for col in self.X.select_dtypes(include=['datetime64']).columns:
            self.X[col] = self.X[col].astype('int64') // 10**9
        
        # Fill missing values
        self.X = self.X.fillna(self.X.median())
        self.X = self.X.select_dtypes(include=[np.number])
        
        print(f"✓ Data prepared: {self.X.shape}")
    
    def objective_lightgbm(self, trial: optuna.Trial) -> float:
        """Objective function for LightGBM hyperparameter optimization"""
        
        # Suggest hyperparameters
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [3],
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
            'verbose': -1,
            'random_state': self.random_state,
            'force_col_wise': True,
            'num_threads': -1
        }
        
        # Cross-validation
        group_kfold = GroupKFold(n_splits=3)
        cv_scores = []
        
        for train_idx, val_idx in group_kfold.split(self.X, self.y, self.groups):
            # Split data
            X_train_fold = self.X.iloc[train_idx]
            y_train_fold = self.y.iloc[train_idx]
            groups_train_fold = self.groups.iloc[train_idx]
            
            X_val_fold = self.X.iloc[val_idx]
            y_val_fold = self.y.iloc[val_idx]
            groups_val_fold = self.groups.iloc[val_idx]
            
            # Prepare group arrays
            group_train_fold = groups_train_fold.value_counts().sort_index().values
            
            # Train model
            import lightgbm as lgb
            train_data = lgb.Dataset(X_train_fold, label=y_train_fold, group=group_train_fold)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(100)]
            )
            
            # Predict and evaluate
            val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
            
            val_df = pd.DataFrame({
                'ranker_id': groups_val_fold,
                'selected': y_val_fold,
                'score': val_pred
            })
            
            val_df['rank'] = self.evaluator.scores_to_ranks(val_df, 'score', 'ranker_id')
            hitrate = self.evaluator.calculate_hitrate3(val_df, 'rank', 'selected', 'ranker_id', min_group_size=10)
            cv_scores.append(hitrate)
        
        return np.mean(cv_scores)
    
    def optimize_lightgbm(self, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters"""
        
        print(f"Starting LightGBM hyperparameter optimization with {n_trials} trials...")
        
        study = optuna.create_study(direction='maximize', 
                                   sampler=optuna.samplers.TPESampler(seed=self.random_state))
        
        study.optimize(self.objective_lightgbm, n_trials=n_trials, show_progress_bar=True)
        
        print(f"✓ Optimization completed!")
        print(f"✓ Best score: {study.best_value:.6f}")
        print(f"✓ Best params: {study.best_params}")
        
        return {
            'best_score': study.best_value,
            'best_params': study.best_params,
            'study': study
        }

def main():
    """Run hyperparameter optimization"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for FlightRank 2025')
    parser.add_argument('--data_path', type=str, default='.', help='Path to data directory')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--output', type=str, default='best_params.json', help='Output file for best parameters')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(data_path=args.data_path)
    
    # Run optimization
    results = optimizer.optimize_lightgbm(n_trials=args.n_trials)
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump({
            'best_score': results['best_score'],
            'best_params': results['best_params']
        }, f, indent=2)
    
    print(f"✓ Best parameters saved to {args.output}")

if __name__ == "__main__":
    main()
