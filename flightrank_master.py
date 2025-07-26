"""
FlightRank 2025: Aeroclub RecSys Cup - Master Solution File
A comprehensive solution for personalized flight recommendation ranking

This master file combines all components into a single executable solution:
- JSON to Parquet conversion
- Feature engineering
- Model training (LightGBM)
- Cross-validation
- Submission generation
- Performance testing

Usage:
    python flightrank_master.py --mode [convert|train|predict|test]
    
Examples:
    python flightrank_master.py --mode convert  # Convert JSON files to parquet
    python flightrank_master.py --mode train    # Train model and generate submission
    python flightrank_master.py --mode test     # Run performance tests
"""

import pandas as pd
import numpy as np
import json
import os
import gc
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings for the FlightRank solution"""
    
    # Data paths
    DATA_DIR = Path(".")
    TRAIN_PATH = DATA_DIR / "train.parquet"
    TEST_PATH = DATA_DIR / "test.parquet"
    SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.parquet"
    JSON_DIR = DATA_DIR / "json_samples"
    
    # Model parameters
    LGBM_PARAMS = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [3],
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'random_state': 42,
        'force_col_wise': True,
        'num_threads': -1
    }
    
    # Cross-validation settings
    CV_FOLDS = 3
    EARLY_STOPPING_ROUNDS = 200
    NUM_BOOST_ROUND = 3000

# ============================================================================
# JSON TO PARQUET CONVERTER
# ============================================================================

class JSONToParquetConverter:
    """Convert JSON files to parquet format for the competition"""
    
    def __init__(self, json_dir: str = "json_samples", output_dir: str = "."):
        self.json_dir = Path(json_dir)
        self.output_dir = Path(output_dir)
        
    def parse_single_json(self, json_file: Path) -> List[Dict]:
        """Parse a single JSON file and extract flight options"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                return []
            
            ranker_id = data.get('ranker_id', json_file.stem)
            request_date = data.get('routeData', {}).get('requestDate') if data.get('routeData') else None
            search_route = data.get('routeData', {}).get('searchRoute') if data.get('routeData') else None
            
            # Personal data
            personal = data.get('personalData', {})
            profile_id = personal.get('profileId') if personal else None
            sex = personal.get('sex') if personal else None
            nationality = personal.get('nationality') if personal else None
            company_id = personal.get('companyID') if personal else None
            is_vip = personal.get('isVip', False) if personal else False
            by_self = personal.get('bySelf', True) if personal else True
            is_access_3d = personal.get('isAccess3D', False) if personal else False
            
            # Extract flight options
            flight_options = []
            flight_data = data.get('data', {}).get('$values', [])
            
            for idx, flight_option in enumerate(flight_data):
                legs = flight_option.get('legs', [])
                pricings = flight_option.get('pricings', [])
                
                if not pricings:
                    continue
                    
                for pricing_idx, pricing in enumerate(pricings):
                    row = {
                        'Id': f"{ranker_id}_{idx}_{pricing_idx}",
                        'ranker_id': ranker_id,
                        'profileId': profile_id,
                        'companyID': company_id,
                        'sex': sex,
                        'nationality': nationality,
                        'isVip': is_vip,
                        'bySelf': by_self,
                        'isAccess3D': is_access_3d,
                        'searchRoute': search_route,
                        'requestDate': request_date,
                        'totalPrice': pricing.get('totalPrice'),
                        'taxes': pricing.get('taxes'),
                        'selected': 1 if pricing_idx == 0 else 0  # Assume first pricing is selected
                    }
                    
                    # Extract corporate tariff code
                    row['corporateTariffCode'] = pricing.get('corporateTariffCode')
                    row['frequentFlyer'] = None
                    
                    # Extract pricing info
                    pricing_info = pricing.get('pricingInfo', [])
                    if pricing_info:
                        first_pricing = pricing_info[0]
                        row['pricingInfo_isAccessTP'] = first_pricing.get('isAccessTP', False)
                        row['pricingInfo_passengerCount'] = first_pricing.get('passengerCount', 1)
                    
                    # Extract legs information
                    for leg_idx, leg in enumerate(legs[:2]):
                        leg_prefix = f'legs{leg_idx}'
                        
                        row[f'{leg_prefix}_departureAt'] = leg.get('departureAt')
                        row[f'{leg_prefix}_arrivalAt'] = leg.get('arrivalAt')
                        row[f'{leg_prefix}_duration'] = leg.get('duration')
                        
                        # Extract segments
                        segments = leg.get('segments', [])
                        for seg_idx, segment in enumerate(segments[:3]):  # Max 3 segments
                            seg_prefix = f'{leg_prefix}_segments{seg_idx}'
                            
                            row[f'{seg_prefix}_duration'] = segment.get('duration')
                            row[f'{seg_prefix}_marketingCarrier_code'] = segment.get('marketingCarrier', {}).get('code')
                            row[f'{seg_prefix}_cabinClass'] = segment.get('cabinClass')
                            row[f'{seg_prefix}_seatsAvailable'] = segment.get('seatsAvailable')
                    
                    flight_options.append(row)
            
            return flight_options
            
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
            return []
    
    def convert_json_to_parquet(self, max_files: Optional[int] = None, train_test_split: float = 0.7):
        """Convert JSON files to train/test parquet files"""
        
        print(f"Converting JSON files from {self.json_dir}...")
        
        json_files = list(self.json_dir.glob("*.json"))
        if max_files:
            json_files = json_files[:max_files]
        
        print(f"Processing {len(json_files)} JSON files...")
        
        all_data = []
        processed = 0
        
        for json_file in json_files:
            flight_options = self.parse_single_json(json_file)
            all_data.extend(flight_options)
            
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{len(json_files)} files...")
        
        if not all_data:
            print("No data extracted from JSON files!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        print(f"Created DataFrame with {len(df)} rows")
        
        # Split by ranker_id
        unique_rankers = df['ranker_id'].unique()
        np.random.shuffle(unique_rankers)
        
        split_idx = int(len(unique_rankers) * train_test_split)
        train_rankers = unique_rankers[:split_idx]
        test_rankers = unique_rankers[split_idx:]
        
        train_df = df[df['ranker_id'].isin(train_rankers)]
        test_df = df[df['ranker_id'].isin(test_rankers)]
        
        # Remove 'selected' column from test set
        test_df_clean = test_df.drop('selected', axis=1)
        
        # Save to parquet
        train_path = self.output_dir / "train.parquet"
        test_path = self.output_dir / "test.parquet"
        
        train_df.to_parquet(train_path, index=False)
        test_df_clean.to_parquet(test_path, index=False)
        
        print(f"Saved train.parquet: {len(train_df)} rows, {train_df['ranker_id'].nunique()} sessions")
        print(f"Saved test.parquet: {len(test_df_clean)} rows, {test_df_clean['ranker_id'].nunique()} sessions")
        
        return train_df, test_df_clean

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Comprehensive feature engineering for flight data"""
    
    def __init__(self):
        self.label_encoders = {}
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create robust features from the flight data"""
        print(f"Engineering features... Input shape: {df.shape}")
        
        df = df.copy()
        
        # Basic numeric features
        df = self._add_basic_features(df)
        
        # Time features
        df = self._add_time_features(df)
        
        # Categorical features
        df = self._add_categorical_features(df)
        
        # Route features
        df = self._add_route_features(df)
        
        # Clean up and select features
        df = self._clean_features(df)
        
        print(f"Created features, output shape: {df.shape}")
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic numeric features"""
        
        # Price features
        if 'totalPrice' in df.columns:
            df['price_log'] = np.log1p(df['totalPrice'].fillna(0))
            
        if 'taxes' in df.columns and 'totalPrice' in df.columns:
            df['tax_ratio'] = df['taxes'].fillna(0) / (df['totalPrice'].fillna(1) + 1e-6)
        
        # Boolean features
        bool_cols = ['isVip', 'bySelf', 'isAccess3D', 'pricingInfo_isAccessTP']
        for col in bool_cols:
            if col in df.columns:
                df[f'{col}_flag'] = df[col].fillna(False).astype(int)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        if 'requestDate' in df.columns:
            try:
                request_dt = pd.to_datetime(df['requestDate'], errors='coerce')
                df['request_hour'] = request_dt.dt.hour.fillna(12)
                df['request_day_of_week'] = request_dt.dt.dayofweek.fillna(0)
                df['is_weekend'] = (df['request_day_of_week'] >= 5).astype(int)
                df['is_business_hours'] = ((df['request_hour'] >= 9) & (df['request_hour'] <= 17)).astype(int)
            except:
                df['request_hour'] = 12
                df['request_day_of_week'] = 0
                df['is_weekend'] = 0
                df['is_business_hours'] = 1
        
        return df
    
    def _add_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add categorical features with label encoding"""
        
        cat_cols = ['corporateTariffCode', 'profileId', 'companyID', 'nationality']
        
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna('missing').astype(str)
                
                if col not in self.label_encoders:
                    unique_vals = df[col].unique()
                    if 'missing' not in unique_vals:
                        unique_vals = np.append(unique_vals, 'missing')
                    
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(unique_vals)
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                else:
                    known_cats = set(self.label_encoders[col].classes_)
                    unknown_mask = ~df[col].isin(known_cats)
                    df.loc[unknown_mask, col] = 'missing'
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _add_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add route-based features"""
        
        # Count segments
        segment_cols = [col for col in df.columns if 'segments' in col and 'duration' in col]
        if segment_cols:
            df['total_segments'] = (df[segment_cols].notna()).sum(axis=1)
            df['has_connections'] = (df['total_segments'] > 1).astype(int)
        
        # Route complexity from searchRoute
        if 'searchRoute' in df.columns:
            df['route_length'] = df['searchRoute'].fillna('').astype(str).str.len()
            df['is_round_trip'] = df['searchRoute'].fillna('').astype(str).str.contains('/').astype(int)
        
        # Carrier features
        carrier_cols = [col for col in df.columns if 'Carrier_code' in col]
        if carrier_cols:
            carrier_data = df[carrier_cols].fillna('missing')
            df['unique_carriers'] = carrier_data.nunique(axis=1)
            
            if len(carrier_cols) > 0:
                first_carrier_col = carrier_cols[0]
                df['primary_carrier'] = df[first_carrier_col].fillna('missing').astype(str)
                
                if 'primary_carrier' not in self.label_encoders:
                    unique_carriers = df['primary_carrier'].unique()
                    if 'missing' not in unique_carriers:
                        unique_carriers = np.append(unique_carriers, 'missing')
                    
                    self.label_encoders['primary_carrier'] = LabelEncoder()
                    self.label_encoders['primary_carrier'].fit(unique_carriers)
                    df['primary_carrier_encoded'] = self.label_encoders['primary_carrier'].transform(df['primary_carrier'])
                else:
                    known_carriers = set(self.label_encoders['primary_carrier'].classes_)
                    unknown_mask = ~df['primary_carrier'].isin(known_carriers)
                    df.loc[unknown_mask, 'primary_carrier'] = 'missing'
                    df['primary_carrier_encoded'] = self.label_encoders['primary_carrier'].transform(df['primary_carrier'])
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and select final features"""
        
        # Select numeric features for model
        numeric_features = [
            'totalPrice', 'taxes', 'price_log', 'tax_ratio',
            'request_hour', 'request_day_of_week', 'is_weekend', 'is_business_hours',
            'total_segments', 'has_connections', 'route_length', 'is_round_trip',
            'unique_carriers'
        ]
        
        flag_features = [col for col in df.columns if col.endswith('_flag')]
        encoded_features = [col for col in df.columns if col.endswith('_encoded')]
        
        all_features = numeric_features + flag_features + encoded_features
        
        # Keep only features that exist and are numeric
        feature_cols = []
        for col in all_features:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    feature_cols.append(col)
                except:
                    pass
        
        # Keep essential columns
        essential_cols = ['Id', 'ranker_id']
        if 'selected' in df.columns:
            essential_cols.append('selected')
        
        final_cols = essential_cols + feature_cols
        final_cols = [col for col in final_cols if col in df.columns]
        
        return df[final_cols].copy()

# ============================================================================
# MODEL TRAINER
# ============================================================================

class ModelTrainer:
    """High-performance model training with LightGBM"""
    
    def __init__(self, params: Dict = None):
        self.params = params or Config.LGBM_PARAMS.copy()
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series, group_train: np.ndarray, 
                      X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, 
                      group_val: Optional[np.ndarray] = None, verbose: bool = True) -> lgb.Booster:
        """Train LightGBM ranking model"""
        
        train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None and group_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        callbacks = [lgb.early_stopping(Config.EARLY_STOPPING_ROUNDS)]
        if verbose:
            callbacks.append(lgb.log_evaluation(100))
        
        model = lgb.train(
            self.params,
            train_data,
            num_boost_round=Config.NUM_BOOST_ROUND,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        if verbose:
            print(f"LightGBM trained with {model.best_iteration} iterations")
        
        return model

# ============================================================================
# EVALUATOR
# ============================================================================

class Evaluator:
    """Evaluation utilities for ranking metrics"""
    
    @staticmethod
    def hitrate_at_k(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, k: int = 3) -> float:
        """Calculate HitRate@k metric"""
        
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'group': groups
        })
        
        hits = 0
        total_groups = 0
        
        for group_id in df['group'].unique():
            group_data = df[df['group'] == group_id].copy()
            
            # Skip groups with <= 10 options (competition rules)
            if len(group_data) <= 10:
                continue
                
            # Sort by prediction and get top k
            group_data = group_data.sort_values('y_pred', ascending=False)
            top_k = group_data.head(k)
            
            # Check if any of top k has y_true = 1
            if top_k['y_true'].sum() > 0:
                hits += 1
            
            total_groups += 1
        
        return hits / total_groups if total_groups > 0 else 0
    
    @staticmethod
    def scores_to_ranks(df: pd.DataFrame, score_col: str, group_col: str = 'ranker_id') -> pd.Series:
        """Convert scores to ranks within each group"""
        return df.groupby(group_col)[score_col].rank(method='dense', ascending=False)

# ============================================================================
# MAIN SOLUTION CLASS
# ============================================================================

class FlightRankSolution:
    """Main solution class that orchestrates the entire pipeline"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.evaluator = Evaluator()
        self.model = None
    
    def convert_json_data(self, max_files: Optional[int] = None):
        """Convert JSON files to parquet format"""
        print("STEP 1: Converting JSON data to parquet format...")
        
        converter = JSONToParquetConverter(
            json_dir=Config.JSON_DIR,
            output_dir=Config.DATA_DIR
        )
        
        train_df, test_df = converter.convert_json_to_parquet(max_files=max_files)
        print("JSON conversion completed!")
        
        return train_df, test_df
    
    def load_data(self):
        """Load training and test data"""
        print("STEP 1: Loading data...")
        
        if not Config.TRAIN_PATH.exists() or not Config.TEST_PATH.exists():
            print("Parquet files not found. Please run with --mode convert first.")
            return None, None
        
        train_df = pd.read_parquet(Config.TRAIN_PATH)
        test_df = pd.read_parquet(Config.TEST_PATH)
        
        print(f"Train: {train_df.shape}")
        print(f"Test: {test_df.shape}")
        
        return train_df, test_df
    
    def engineer_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Engineer features for training and test data"""
        print("\nSTEP 2: Engineering features...")
        
        # Train features (fit encoder)
        train_fe = self.feature_engineer.engineer_features(train_df)
        
        # Test features (transform only)
        test_fe = self.feature_engineer.engineer_features(test_df)
        
        print(f"Train features: {train_fe.shape}")
        print(f"Test features: {test_fe.shape}")
        
        return train_fe, test_fe
    
    def cross_validate(self, train_fe: pd.DataFrame):
        """Perform cross-validation"""
        print("\nSTEP 3: Cross-validation...")
        
        feature_cols = [col for col in train_fe.columns if col not in ['Id', 'ranker_id', 'selected']]
        
        X = train_fe[feature_cols]
        y = train_fe['selected']
        groups = train_fe['ranker_id']
        
        print(f"Features: {len(feature_cols)}")
        print(f"Groups: {groups.nunique()}")
        
        gkf = GroupKFold(n_splits=Config.CV_FOLDS)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            groups_tr = groups.iloc[train_idx]
            groups_val = groups.iloc[val_idx]
            
            # Create group boundaries for LightGBM
            group_boundaries_tr = groups_tr.value_counts().sort_index().values
            group_boundaries_val = groups_val.value_counts().sort_index().values
            
            # Train model
            model = self.model_trainer.train_lightgbm(
                X_tr, y_tr, group_boundaries_tr,
                X_val, y_val, group_boundaries_val,
                verbose=False
            )
            
            # Evaluate
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            score = self.evaluator.hitrate_at_k(y_val.values, y_pred, groups_val.values, k=3)
            cv_scores.append(score)
            
            print(f"Fold {fold+1}: HitRate@3 = {score:.4f}")
        
        print(f"\nCV HitRate@3: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return cv_scores
    
    def train_final_model(self, train_fe: pd.DataFrame):
        """Train the final model on all training data"""
        print("\nSTEP 4: Training final model...")
        
        feature_cols = [col for col in train_fe.columns if col not in ['Id', 'ranker_id', 'selected']]
        
        X = train_fe[feature_cols]
        y = train_fe['selected']
        groups = train_fe['ranker_id']
        
        group_boundaries = groups.value_counts().sort_index().values
        
        self.model = self.model_trainer.train_lightgbm(
            X, y, group_boundaries, verbose=True
        )
        
        return self.model
    
    def generate_submission(self, test_fe: pd.DataFrame):
        """Generate submission file"""
        print("\nSTEP 5: Generating submission...")
        
        if self.model is None:
            print("No model found. Please train model first.")
            return None
        
        feature_cols = [col for col in test_fe.columns if col not in ['Id', 'ranker_id']]
        X_test = test_fe[feature_cols]
        
        # Make predictions
        predictions = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        
        # Create submission DataFrame
        submission = test_fe[['Id', 'ranker_id']].copy()
        submission['score'] = predictions
        
        # Convert scores to ranks
        submission['selected'] = self.evaluator.scores_to_ranks(submission, 'score', 'ranker_id')
        
        # Final submission format
        final_submission = submission[['Id', 'ranker_id', 'selected']].copy()
        
        # Save submission
        submission_path = Config.DATA_DIR / "submission.csv"
        final_submission.to_csv(submission_path, index=False)
        
        print(f"Submission saved to {submission_path}")
        print(f"Submission shape: {final_submission.shape}")
        
        return final_submission
    
    def run_full_pipeline(self, max_files: Optional[int] = None):
        """Run the complete solution pipeline"""
        
        print("=" * 60)
        print("FLIGHTRANK 2025 - COMPLETE SOLUTION PIPELINE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load or convert data
        train_df, test_df = self.load_data()
        if train_df is None:
            print("Converting JSON data first...")
            train_df, test_df = self.convert_json_data(max_files=max_files)
        
        # Feature engineering
        train_fe, test_fe = self.engineer_features(train_df, test_df)
        
        # Cross-validation
        cv_scores = self.cross_validate(train_fe)
        
        # Train final model
        final_model = self.train_final_model(train_fe)
        
        # Generate submission
        submission = self.generate_submission(test_fe)
        
        total_time = time.time() - start_time
        print(f"\nPipeline completed in {total_time:.1f} seconds")
        print(f"Final CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return submission
    
    def run_performance_test(self):
        """Run performance testing"""
        print("=" * 60)
        print("FLIGHTRANK 2025 - PERFORMANCE TESTING")
        print("=" * 60)
        
        # Create synthetic data for testing
        print("Creating synthetic test data...")
        
        np.random.seed(42)
        n_sessions = 1000
        avg_options = 25
        
        data = []
        current_id = 1
        
        for session_id in range(n_sessions):
            n_options = max(1, np.random.poisson(avg_options))
            selected_idx = np.random.randint(n_options)
            
            for option_idx in range(n_options):
                row = {
                    'Id': current_id,
                    'ranker_id': f'session_{session_id}',
                    'selected': 1 if option_idx == selected_idx else 0,
                    'totalPrice': np.random.uniform(100, 2000),
                    'legs0_duration': np.random.uniform(1, 20),
                    'legs1_duration': np.random.uniform(1, 20) if np.random.random() > 0.3 else None,
                    'isVip': np.random.choice([0, 1], p=[0.9, 0.1]),
                    'bySelf': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'pricingInfo_isAccessTP': np.random.choice([0, 1], p=[0.6, 0.4]),
                    'requestDate': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(365)),
                }
                data.append(row)
                current_id += 1
        
        df = pd.DataFrame(data)
        print(f"Created synthetic data: {len(df)} rows, {df['ranker_id'].nunique()} sessions")
        
        # Test feature engineering performance
        print("\nTesting feature engineering...")
        start_time = time.time()
        df_fe = self.feature_engineer.engineer_features(df)
        fe_time = time.time() - start_time
        
        print(f"Feature engineering: {fe_time:.2f}s ({len(df)/fe_time:,.0f} rows/sec)")
        
        # Test model training performance
        print("\nTesting model training...")
        feature_cols = [col for col in df_fe.columns if col not in ['Id', 'ranker_id', 'selected']]
        X = df_fe[feature_cols]
        y = df_fe['selected']
        groups = df_fe['ranker_id']
        
        # Handle categorical variables
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = pd.Categorical(X[col]).codes
        X = X.fillna(X.median())
        X = X.select_dtypes(include=[np.number])
        
        group_array = groups.value_counts().sort_index().values
        
        start_time = time.time()
        model = self.model_trainer.train_lightgbm(X, y, group_array, verbose=False)
        train_time = time.time() - start_time
        
        print(f"Model training: {train_time:.2f}s ({len(X)/train_time:,.0f} rows/sec)")
        
        # Test prediction performance
        start_time = time.time()
        predictions = model.predict(X, num_iteration=model.best_iteration)
        pred_time = time.time() - start_time
        
        print(f"Model prediction: {pred_time:.2f}s ({len(X)/pred_time:,.0f} rows/sec)")
        
        total_time = fe_time + train_time + pred_time
        print(f"\nTotal pipeline time: {total_time:.2f}s")
        print(f"Overall throughput: {len(df)/total_time:,.0f} rows/sec")
        
        # Performance assessment
        if len(df)/total_time > 10000:
            print("EXCELLENT: High-performance pipeline")
        elif len(df)/total_time > 1000:
            print("GOOD: Efficient pipeline")
        else:
            print("MODERATE: Pipeline could be optimized")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point with command line interface"""
    
    parser = argparse.ArgumentParser(description='FlightRank 2025 Master Solution')
    parser.add_argument('--mode', choices=['convert', 'train', 'test', 'full'], 
                       default='full', help='Operation mode')
    parser.add_argument('--max-files', type=int, help='Maximum JSON files to process')
    
    args = parser.parse_args()
    
    solution = FlightRankSolution()
    
    if args.mode == 'convert':
        solution.convert_json_data(max_files=args.max_files)
    
    elif args.mode == 'train':
        solution.run_full_pipeline(max_files=args.max_files)
    
    elif args.mode == 'test':
        solution.run_performance_test()
    
    elif args.mode == 'full':
        solution.run_full_pipeline(max_files=args.max_files)
        print("\n" + "="*40)
        print("Running performance tests...")
        solution.run_performance_test()

if __name__ == "__main__":
    main()
