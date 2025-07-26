"""
Performance Testing Script for FlightRank 2025 Solution
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from evaluation import Evaluator

class PerformanceTester:
    """Test and benchmark solution performance"""
    
    def __init__(self):
        self.results = {}
    
    def test_feature_engineering_performance(self, df: pd.DataFrame, iterations: int = 3) -> Dict:
        """Test feature engineering performance"""
        
        print("Testing feature engineering performance...")
        
        feature_engineer = FeatureEngineer()
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            df_fe = feature_engineer.engineer_features(df)
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.2f}s")
        
        result = {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'original_features': len(df.columns),
            'engineered_features': len(df_fe.columns),
            'new_features': len(df_fe.columns) - len(df.columns),
            'throughput_rows_per_sec': len(df) / np.mean(times)
        }
        
        print(f"✓ Average time: {result['avg_time']:.2f}s")
        print(f"✓ Throughput: {result['throughput_rows_per_sec']:,.0f} rows/sec")
        print(f"✓ Features: {result['original_features']} → {result['engineered_features']} (+{result['new_features']})")
        
        return result
    
    def test_model_training_performance(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray) -> Dict:
        """Test model training performance"""
        
        print("Testing model training performance...")
        
        trainer = ModelTrainer()
        
        # Test LightGBM
        start_time = time.time()
        lgb_model = trainer.train_lightgbm(X, y, groups, verbose=False)
        lgb_time = time.time() - start_time
        
        # Test prediction speed
        start_time = time.time()
        lgb_pred = lgb_model.predict(X, num_iteration=lgb_model.best_iteration)
        lgb_pred_time = time.time() - start_time
        
        result = {
            'lgb_training_time': lgb_time,
            'lgb_prediction_time': lgb_pred_time,
            'lgb_best_iteration': lgb_model.best_iteration,
            'training_throughput': len(X) / lgb_time,
            'prediction_throughput': len(X) / lgb_pred_time
        }
        
        print(f"✓ LightGBM training: {lgb_time:.2f}s ({lgb_model.best_iteration} iterations)")
        print(f"✓ LightGBM prediction: {lgb_pred_time:.2f}s")
        print(f"✓ Training throughput: {result['training_throughput']:,.0f} rows/sec")
        print(f"✓ Prediction throughput: {result['prediction_throughput']:,.0f} rows/sec")
        
        return result
    
    def test_evaluation_performance(self, df: pd.DataFrame, iterations: int = 10) -> Dict:
        """Test evaluation performance"""
        
        print("Testing evaluation performance...")
        
        evaluator = Evaluator()
        
        # Add dummy predictions
        df = df.copy()
        df['score'] = np.random.random(len(df))
        df['rank'] = evaluator.scores_to_ranks(df, 'score', 'ranker_id')
        
        # Test HitRate@3 calculation
        times = []
        for i in range(iterations):
            start_time = time.time()
            hitrate = evaluator.calculate_hitrate3(df, 'rank', 'selected', 'ranker_id')
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        result = {
            'avg_evaluation_time': np.mean(times),
            'min_evaluation_time': np.min(times),
            'max_evaluation_time': np.max(times),
            'evaluation_throughput': len(df) / np.mean(times),
            'sample_hitrate': hitrate
        }
        
        print(f"✓ Average evaluation time: {result['avg_evaluation_time']:.4f}s")
        print(f"✓ Evaluation throughput: {result['evaluation_throughput']:,.0f} rows/sec")
        
        return result
    
    def create_synthetic_data(self, n_sessions: int = 1000, avg_options_per_session: int = 20) -> pd.DataFrame:
        """Create synthetic data for performance testing"""
        
        print(f"Creating synthetic data: {n_sessions} sessions, ~{avg_options_per_session} options each...")
        
        np.random.seed(42)
        
        data = []
        current_id = 1
        
        for session_id in range(n_sessions):
            n_options = np.random.poisson(avg_options_per_session)
            n_options = max(1, n_options)  # At least 1 option
            
            # Select one random option as chosen
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
                    'legs0_departureAt': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(365)),
                    'frequentFlyer': f'airline_{np.random.randint(1, 20)}',
                }
                
                # Add some segment data
                for leg in [0, 1]:
                    for seg in range(2):  # 2 segments max for synthetic data
                        if leg == 1 and row['legs1_duration'] is None:
                            continue
                        if seg == 1 and np.random.random() > 0.3:  # 30% have connections
                            continue
                            
                        row[f'legs{leg}_segments{seg}_duration'] = np.random.uniform(0.5, 10)
                        row[f'legs{leg}_segments{seg}_marketingCarrier_code'] = f'airline_{np.random.randint(1, 20)}'
                        row[f'legs{leg}_segments{seg}_cabinClass'] = np.random.choice([1.0, 2.0, 3.0, 4.0], p=[0.6, 0.2, 0.15, 0.05])
                        row[f'legs{leg}_segments{seg}_seatsAvailable'] = np.random.randint(0, 50)
                
                data.append(row)
                current_id += 1
        
        df = pd.DataFrame(data)
        print(f"✓ Created {len(df)} rows, {df['ranker_id'].nunique()} sessions")
        return df
    
    def run_full_performance_test(self):
        """Run comprehensive performance test"""
        
        print("=" * 60)
        print("FLIGHTRANK 2025 - PERFORMANCE TESTING")
        print("=" * 60)
        
        # Create synthetic data
        df = self.create_synthetic_data(n_sessions=1000, avg_options_per_session=25)
        
        # Test 1: Feature Engineering
        print(f"\n📊 TEST 1: FEATURE ENGINEERING")
        print("-" * 40)
        fe_results = self.test_feature_engineering_performance(df)
        
        # Prepare data for model testing
        feature_engineer = FeatureEngineer()
        df_fe = feature_engineer.engineer_features(df)
        
        # Quick data preparation
        exclude_cols = ['Id', 'ranker_id', 'selected']
        X = df_fe[[col for col in df_fe.columns if col not in exclude_cols]]
        y = df_fe['selected']
        groups = df_fe['ranker_id']
        
        # Handle categorical variables quickly
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = pd.Categorical(X[col]).codes
        X = X.fillna(X.median())
        X = X.select_dtypes(include=[np.number])
        
        group_array = groups.value_counts().sort_index().values
        
        # Test 2: Model Training
        print(f"\n🤖 TEST 2: MODEL TRAINING")
        print("-" * 40)
        model_results = self.test_model_training_performance(X, y, group_array)
        
        # Test 3: Evaluation
        print(f"\n📈 TEST 3: EVALUATION")
        print("-" * 40)
        eval_results = self.test_evaluation_performance(df_fe)
        
        # Summary
        print(f"\n📋 PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Feature Engineering: {fe_results['avg_time']:.2f}s ({fe_results['throughput_rows_per_sec']:,.0f} rows/sec)")
        print(f"Model Training: {model_results['lgb_training_time']:.2f}s ({model_results['training_throughput']:,.0f} rows/sec)")
        print(f"Model Prediction: {model_results['lgb_prediction_time']:.2f}s ({model_results['prediction_throughput']:,.0f} rows/sec)")
        print(f"Evaluation: {eval_results['avg_evaluation_time']:.4f}s ({eval_results['evaluation_throughput']:,.0f} rows/sec)")
        
        total_pipeline_time = fe_results['avg_time'] + model_results['lgb_training_time'] + model_results['lgb_prediction_time']
        print(f"\nTotal Pipeline Time: {total_pipeline_time:.2f}s")
        print(f"Overall Throughput: {len(df)/total_pipeline_time:,.0f} rows/sec")
        
        # Performance classification
        if fe_results['throughput_rows_per_sec'] > 10000:
            print("🚀 EXCELLENT: High-performance feature engineering")
        elif fe_results['throughput_rows_per_sec'] > 1000:
            print("✅ GOOD: Efficient feature engineering")
        else:
            print("⚠️  MODERATE: Feature engineering could be optimized")
        
        print("\n🎯 Performance testing completed!")

def main():
    """Run performance tests"""
    tester = PerformanceTester()
    tester.run_full_performance_test()

if __name__ == "__main__":
    main()
