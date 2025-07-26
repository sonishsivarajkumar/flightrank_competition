"""
Production FlightRank 2025 Solution
Full dataset processing for Kaggle competition
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import warnings
import time
import argparse
warnings.filterwarnings('ignore')

from simple_features import create_simple_features
from json_to_parquet_converter import JSONToParquetConverter

def hitrate_at_k(y_true, y_pred, groups, k=3):
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
        
        # Skip groups with <= 10 options (as per competition rules)
        if len(group_data) <= 10:
            continue
            
        # Sort by prediction (descending) and get top k
        group_data = group_data.sort_values('y_pred', ascending=False)
        top_k = group_data.head(k)
        
        # Check if any of top k has y_true = 1
        if top_k['y_true'].sum() > 0:
            hits += 1
        
        total_groups += 1
    
    return hits / total_groups if total_groups > 0 else 0

def run_production_solution(use_full_data=False, convert_json=False):
    """Run the production FlightRank solution"""
    
    print("🚀 FLIGHTRANK 2025 - PRODUCTION SOLUTION")
    print("=" * 60)
    
    start_time = time.time()
    
    # Convert JSON if needed
    if convert_json:
        print("🔄 Converting JSON files to parquet...")
        converter = JSONToParquetConverter()
        max_files = None if use_full_data else 5000  # Use 5000 files for testing
        converter.run_conversion(max_files=max_files)
        print("✅ JSON conversion completed!")
    
    # 1. Load data
    print("\n📊 Loading data...")
    train_df = pd.read_parquet('train.parquet')
    test_df = pd.read_parquet('test.parquet')
    sample_submission = pd.read_parquet('sample_submission.parquet')
    
    print(f"✓ Train: {train_df.shape}")
    print(f"✓ Test: {test_df.shape}")
    print(f"✓ Train sessions: {train_df['ranker_id'].nunique():,}")
    print(f"✓ Test sessions: {test_df['ranker_id'].nunique():,}")
    
    # 2. Create features
    print("\n🔧 Creating features...")
    train_fe, test_fe = create_simple_features(train_df, test_df)
    
    print(f"✓ Train features: {train_fe.shape}")
    print(f"✓ Test features: {test_fe.shape}")
    
    # 3. Prepare for modeling
    feature_cols = [col for col in train_fe.columns if col not in ['Id', 'ranker_id', 'selected']]
    
    X_train = train_fe[feature_cols]
    y_train = train_fe['selected']
    groups_train = train_fe['ranker_id']
    
    X_test = test_fe[feature_cols]
    groups_test = test_fe['ranker_id']
    
    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Train groups: {groups_train.nunique():,}")
    
    # 4. Cross-validation
    print("\n📈 Cross-validation...")
    
    gkf = GroupKFold(n_splits=5)
    cv_scores = []
    
    # LightGBM parameters
    params = {
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
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups_train)):
        print(f"  Training fold {fold+1}/5...")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        groups_tr = groups_train.iloc[train_idx]
        groups_val = groups_train.iloc[val_idx]
        
        # Create group boundaries for LightGBM
        group_boundaries_tr = groups_tr.value_counts().sort_index().values
        group_boundaries_val = groups_val.value_counts().sort_index().values
        
        # Train LightGBM
        train_data = lgb.Dataset(X_tr, label=y_tr, group=group_boundaries_tr)
        val_data = lgb.Dataset(X_val, label=y_val, group=group_boundaries_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Predict and evaluate
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        score = hitrate_at_k(y_val.values, y_pred, groups_val.values, k=3)
        cv_scores.append(score)
        
        print(f"    ✓ Fold {fold+1}: HitRate@3 = {score:.4f}")
    
    print(f"\n📊 CV HitRate@3: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # 5. Train final model
    print("\n🎯 Training final model...")
    
    group_boundaries_final = groups_train.value_counts().sort_index().values
    train_data_final = lgb.Dataset(X_train, label=y_train, group=group_boundaries_final)
    
    final_model = lgb.train(
        params,
        train_data_final,
        num_boost_round=1500,
        callbacks=[lgb.log_evaluation(100)]
    )
    
    # 6. Make predictions
    print("\n🔮 Making predictions...")
    
    test_predictions = final_model.predict(X_test, num_iteration=final_model.best_iteration)
    
    # Create submission
    submission = test_fe[['Id', 'ranker_id']].copy()
    submission['prediction'] = test_predictions
    
    # Rank within each group
    submission['selected'] = submission.groupby('ranker_id')['prediction'].rank(method='dense', ascending=False).astype(int)
    
    # Prepare final submission
    final_submission = submission[['Id', 'ranker_id', 'selected']].copy()
    
    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    submission_filename = f'submission_{timestamp}.csv'
    final_submission.to_csv(submission_filename, index=False)
    final_submission.to_csv('submission.csv', index=False)  # Also save as default name
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    total_time = time.time() - start_time
    
    # Display results
    print("\n" + "=" * 60)
    print("🎯 PRODUCTION RESULTS SUMMARY")
    print("=" * 60)
    print(f"📊 CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"📁 Features Used: {len(feature_cols)}")
    print(f"📄 Submission: {submission_filename}")
    print(f"🎲 Predicted {len(final_submission):,} flight rankings")
    print(f"⏱️  Total Time: {total_time:.1f} seconds")
    
    print(f"\n🔝 Top 10 Feature Importance:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']}")
    
    print(f"\n📊 Submission Statistics:")
    print(f"    Groups: {final_submission['ranker_id'].nunique():,}")
    print(f"    Min rank: {final_submission['selected'].min()}")
    print(f"    Max rank: {final_submission['selected'].max()}")
    print(f"    Avg group size: {final_submission.groupby('ranker_id').size().mean():.1f}")
    
    return final_submission, cv_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FlightRank 2025 Production Solution')
    parser.add_argument('--full-data', action='store_true', help='Use full dataset (all JSON files)')
    parser.add_argument('--convert-json', action='store_true', help='Convert JSON files to parquet first')
    
    args = parser.parse_args()
    
    submission, cv_scores = run_production_solution(
        use_full_data=args.full_data, 
        convert_json=args.convert_json
    )
    
    print("\n✅ Production run completed! Ready for Kaggle submission.")
    
    if args.full_data:
        print("🏆 This was a FULL DATASET run - submit to Kaggle!")
    else:
        print("🧪 This was a TEST run - use --full-data for competition submission.")
