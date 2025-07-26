"""
FlightRank 2025 Competition - Main Training and Prediction Script
Optimized Python implementation for maximum performance
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import gc
import time
from typing import Tuple, List, Dict, Optional
import argparse

warnings.filterwarnings('ignore')

# Import custom modules
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from evaluation import Evaluator
from data_loader import DataLoader

class FlightRankingSolution:
    """Main class for FlightRank 2025 competition solution"""
    
    def __init__(self, data_path: str = ".", random_state: int = 42):
        self.data_path = data_path
        self.random_state = random_state
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(random_state=random_state)
        self.evaluator = Evaluator()
        self.data_loader = DataLoader(data_path)
        
        # Performance tracking
        self.timing = {}
        
    def run_full_pipeline(self, use_cross_validation: bool = True, 
                         ensemble_models: bool = True) -> Dict:
        """Run the complete training and prediction pipeline"""
        
        print("="*60)
        print("FLIGHTRANK 2025 - OPTIMIZED PIPELINE")
        print("="*60)
        
        # 1. Load data
        start_time = time.time()
        train_df, test_df, sample_submission = self.data_loader.load_data()
        self.timing['data_loading'] = time.time() - start_time
        print(f"✓ Data loaded in {self.timing['data_loading']:.2f}s")
        
        # 2. Feature engineering
        start_time = time.time()
        train_fe = self.feature_engineer.engineer_features(train_df)
        test_fe = self.feature_engineer.engineer_features(test_df)
        self.timing['feature_engineering'] = time.time() - start_time
        print(f"✓ Features engineered in {self.timing['feature_engineering']:.2f}s")
        
        # 3. Prepare data for modeling
        start_time = time.time()
        X_train, y_train, groups_train = self._prepare_features(train_fe, 'selected')
        X_test, _, groups_test = self._prepare_features(test_fe)
        
        # Align features
        common_features = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_features]
        X_test = X_test[common_features]
        
        self.timing['data_preparation'] = time.time() - start_time
        print(f"✓ Data prepared in {self.timing['data_preparation']:.2f}s")
        print(f"✓ Feature count: {len(common_features)}")
        
        # 4. Cross-validation (optional)
        cv_scores = {}
        if use_cross_validation:
            start_time = time.time()
            cv_scores = self._run_cross_validation(X_train, y_train, groups_train)
            self.timing['cross_validation'] = time.time() - start_time
            print(f"✓ Cross-validation completed in {self.timing['cross_validation']:.2f}s")
        
        # 5. Train final models
        start_time = time.time()
        group_train = groups_train.value_counts().sort_index().values
        
        if ensemble_models:
            # Train multiple models for ensemble
            lgb_model = self.model_trainer.train_lightgbm(X_train, y_train, group_train)
            cb_model = self.model_trainer.train_catboost(X_train, y_train, groups_train)
            
            # Ensemble predictions
            lgb_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
            cb_pred = cb_model.predict(X_test)
            
            # Weighted ensemble (LightGBM typically performs better on ranking)
            test_predictions = 0.7 * lgb_pred + 0.3 * cb_pred
            
            print("✓ Ensemble models trained (LightGBM + CatBoost)")
        else:
            # Single LightGBM model
            lgb_model = self.model_trainer.train_lightgbm(X_train, y_train, group_train)
            test_predictions = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
            print("✓ LightGBM model trained")
        
        self.timing['model_training'] = time.time() - start_time
        print(f"✓ Model training completed in {self.timing['model_training']:.2f}s")
        
        # 6. Create submission
        start_time = time.time()
        submission = self._create_submission(test_fe, test_predictions)
        self.timing['submission_creation'] = time.time() - start_time
        print(f"✓ Submission created in {self.timing['submission_creation']:.2f}s")
        
        # 7. Summary
        total_time = sum(self.timing.values())
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETED IN {total_time:.2f}s")
        print(f"{'='*60}")
        
        results = {
            'cv_scores': cv_scores,
            'submission': submission,
            'timing': self.timing,
            'feature_count': len(common_features)
        }
        
        return results
    
    def _prepare_features(self, df: pd.DataFrame, target_col: str = 'selected') -> Tuple[pd.DataFrame, Optional[pd.Series], pd.Series]:
        """Optimized feature preparation"""
        
        # Exclude non-feature columns
        exclude_cols = ['Id', 'ranker_id', target_col]
        if target_col in df.columns:
            target = df[target_col]
        else:
            target = None
        
        groups = df['ranker_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        features_df = df[feature_cols].copy()
        
        # Fast categorical encoding
        categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if features_df[col].nunique() > 1000:
                # High cardinality - frequency encoding
                freq_map = features_df[col].value_counts()
                features_df[f'{col}_freq'] = features_df[col].map(freq_map).fillna(0)
                features_df.drop(col, axis=1, inplace=True)
            else:
                # Low/medium cardinality - label encoding
                le = LabelEncoder()
                features_df[col] = le.fit_transform(features_df[col].fillna('missing'))
        
        # Handle datetime columns
        datetime_cols = features_df.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            features_df[col] = features_df[col].astype('int64') // 10**9
        
        # Fill missing values with median (fast)
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if features_df[col].isnull().sum() > 0:
                features_df[col].fillna(features_df[col].median(), inplace=True)
        
        # Ensure all features are numeric
        features_df = features_df.select_dtypes(include=[np.number])
        
        return features_df, target, groups
    
    def _run_cross_validation(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 3) -> Dict:
        """Optimized cross-validation"""
        
        group_kfold = GroupKFold(n_splits=n_splits)
        cv_scores_all = []
        cv_scores_large = []
        
        print(f"\nRunning {n_splits}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups)):
            print(f"Fold {fold + 1}/{n_splits}", end=" - ")
            
            # Split data
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            groups_train_fold = groups.iloc[train_idx]
            
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            groups_val_fold = groups.iloc[val_idx]
            
            # Train model
            group_train_fold = groups_train_fold.value_counts().sort_index().values
            model = self.model_trainer.train_lightgbm(X_train_fold, y_train_fold, group_train_fold, verbose=False)
            
            # Predict and evaluate
            val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
            
            val_df = pd.DataFrame({
                'ranker_id': groups_val_fold,
                'selected': y_val_fold,
                'score': val_pred
            })
            
            val_df['rank'] = self.evaluator.scores_to_ranks(val_df, 'score', 'ranker_id')
            
            hitrate_all = self.evaluator.calculate_hitrate3(val_df, 'rank', 'selected', 'ranker_id', min_group_size=0)
            hitrate_large = self.evaluator.calculate_hitrate3(val_df, 'rank', 'selected', 'ranker_id', min_group_size=10)
            
            cv_scores_all.append(hitrate_all)
            cv_scores_large.append(hitrate_large)
            
            print(f"HitRate@3: {hitrate_all:.4f} (all), {hitrate_large:.4f} (>10)")
        
        results = {
            'hitrate_all_mean': np.mean(cv_scores_all),
            'hitrate_all_std': np.std(cv_scores_all),
            'hitrate_large_mean': np.mean(cv_scores_large),
            'hitrate_large_std': np.std(cv_scores_large),
            'individual_scores': {
                'all': cv_scores_all,
                'large': cv_scores_large
            }
        }
        
        print(f"\nCV Results:")
        print(f"HitRate@3 (all): {results['hitrate_all_mean']:.4f} ± {results['hitrate_all_std']:.4f}")
        print(f"HitRate@3 (>10): {results['hitrate_large_mean']:.4f} ± {results['hitrate_large_std']:.4f}")
        
        return results
    
    def _create_submission(self, test_df: pd.DataFrame, predictions: np.ndarray, output_path: str = "submission.csv") -> pd.DataFrame:
        """Create optimized submission file"""
        
        submission = pd.DataFrame({
            'Id': test_df['Id'],
            'ranker_id': test_df['ranker_id'],
            'score': predictions
        })
        
        # Fast ranking within groups
        submission['selected'] = submission.groupby('ranker_id')['score'].rank(method='dense', ascending=False).astype(int)
        
        final_submission = submission[['Id', 'ranker_id', 'selected']]
        
        # Quick validation
        rank_check = submission.groupby('ranker_id')['selected'].apply(
            lambda x: len(set(x)) == len(x) and min(x) == 1 and max(x) == len(x)
        ).all()
        
        print(f"Submission validation: {'✓ PASSED' if rank_check else '✗ FAILED'}")
        
        # Save submission
        final_submission.to_csv(output_path, index=False)
        print(f"Submission saved to: {output_path}")
        
        return final_submission


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='FlightRank 2025 Competition Solution')
    parser.add_argument('--data_path', type=str, default='.', help='Path to data directory')
    parser.add_argument('--no_cv', action='store_true', help='Skip cross-validation')
    parser.add_argument('--single_model', action='store_true', help='Use single model instead of ensemble')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output submission file')
    
    args = parser.parse_args()
    
    # Initialize solution
    solution = FlightRankingSolution(data_path=args.data_path)
    
    # Run pipeline
    results = solution.run_full_pipeline(
        use_cross_validation=not args.no_cv,
        ensemble_models=not args.single_model
    )
    
    # Save submission with custom filename
    if args.output != 'submission.csv':
        results['submission'].to_csv(args.output, index=False)
        print(f"Submission saved to: {args.output}")
    
    print("\n🎯 Pipeline completed successfully!")
    
    if results['cv_scores']:
        expected_score = results['cv_scores']['hitrate_large_mean']
        if expected_score >= 0.7:
            print(f"🏆 Excellent! CV score {expected_score:.4f} qualifies for bonus prize!")
        elif expected_score >= 0.5:
            print(f"🥈 Great! CV score {expected_score:.4f} is competitive!")
        else:
            print(f"📈 CV score {expected_score:.4f} - room for improvement!")


if __name__ == "__main__":
    main()
