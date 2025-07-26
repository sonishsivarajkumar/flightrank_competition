"""
Data Loading Module - Optimized for FlightRank 2025 Competition
"""

import pandas as pd
import os
from typing import Tuple

class DataLoader:
    """Efficient data loading and basic validation"""
    
    def __init__(self, data_path: str = "."):
        self.data_path = data_path
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and validate competition data files"""
        
        # Define file paths
        train_path = os.path.join(self.data_path, "train.parquet")
        test_path = os.path.join(self.data_path, "test.parquet")
        sample_path = os.path.join(self.data_path, "sample_submission.parquet")
        
        # Check file existence
        missing_files = []
        for path, name in [(train_path, "train.parquet"), (test_path, "test.parquet"), (sample_path, "sample_submission.parquet")]:
            if not os.path.exists(path):
                missing_files.append(name)
        
        if missing_files:
            raise FileNotFoundError(f"Missing files: {missing_files}. Please download from Kaggle.")
        
        # Load data with optimized settings
        print("Loading data files...")
        
        # Use efficient dtypes where possible
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        sample_submission = pd.read_parquet(sample_path)
        
        # Basic validation
        self._validate_data(train_df, test_df, sample_submission)
        
        print(f"✓ Train: {train_df.shape}")
        print(f"✓ Test: {test_df.shape}")
        print(f"✓ Sample submission: {sample_submission.shape}")
        
        return train_df, test_df, sample_submission
    
    def _validate_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, sample_submission: pd.DataFrame):
        """Validate data integrity"""
        
        # Check target variable
        if 'selected' not in train_df.columns:
            raise ValueError("Target column 'selected' not found in training data")
        
        if 'selected' in test_df.columns:
            print("⚠️ Warning: Target column found in test data")
        
        # Check essential columns
        essential_cols = ['Id', 'ranker_id']
        for col in essential_cols:
            if col not in train_df.columns:
                raise ValueError(f"Essential column '{col}' not found in training data")
            if col not in test_df.columns:
                raise ValueError(f"Essential column '{col}' not found in test data")
        
        # Validate target variable
        selection_rate = train_df['selected'].mean()
        if selection_rate < 0.001 or selection_rate > 0.1:
            print(f"⚠️ Warning: Unusual selection rate: {selection_rate:.4f}")
        
        # Check session structure
        sessions_with_selection = train_df.groupby('ranker_id')['selected'].sum()
        invalid_sessions = (sessions_with_selection != 1).sum()
        if invalid_sessions > 0:
            print(f"⚠️ Warning: {invalid_sessions} sessions don't have exactly 1 selection")
        
        print("✓ Data validation passed")
