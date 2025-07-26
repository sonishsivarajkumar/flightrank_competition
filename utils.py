"""
Utility functions for FlightRank 2025 competition
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import gc
from config import logger

def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of pandas DataFrame by optimizing dtypes
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        logger.info(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
                   f'({(start_mem - end_mem) / start_mem * 100:.1f}% reduction)')
    
    return df

def calculate_hitrate3(df: pd.DataFrame, prediction_col: str = 'rank', 
                       target_col: str = 'selected', group_col: str = 'ranker_id', 
                       min_group_size: int = 0) -> float:
    """
    Calculate HitRate@3 metric for ranking predictions
    """
    # Filter groups by minimum size if specified
    if min_group_size > 0:
        group_sizes = df.groupby(group_col).size()
        valid_groups = group_sizes[group_sizes > min_group_size].index
        df = df[df[group_col].isin(valid_groups)]
    
    # Find the rank of the selected flight in each group
    selected_rows = df[df[target_col] == 1]
    selected_ranks = selected_rows[prediction_col]
    
    # Count how many are in top-3
    hits = (selected_ranks <= 3).sum()
    total = len(selected_ranks)
    
    return hits / total if total > 0 else 0.0

def scores_to_ranks(df: pd.DataFrame, score_col: str, 
                   group_col: str = 'ranker_id') -> pd.Series:
    """Convert scores to ranks within each group (1 = best score)"""
    return df.groupby(group_col)[score_col].rank(method='dense', ascending=False)

def validate_submission(submission: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """
    Validate submission format
    """
    logger.info("Validating submission format...")
    
    # Check row count
    if len(submission) != len(test_df):
        logger.error(f"Row count mismatch: {len(submission)} vs {len(test_df)}")
        return False
    
    # Check columns
    required_cols = ['Id', 'ranker_id', 'selected']
    if not all(col in submission.columns for col in required_cols):
        logger.error(f"Missing columns: {set(required_cols) - set(submission.columns)}")
        return False
    
    # Check rank validity within groups
    rank_check = submission.groupby('ranker_id')['selected'].apply(
        lambda x: sorted(x) == list(range(1, len(x) + 1))
    )
    valid_rankings = rank_check.all()
    
    if not valid_rankings:
        invalid_groups = rank_check[~rank_check].index
        logger.error(f"Invalid rankings in {len(invalid_groups)} groups")
        return False
    
    # Check for duplicate ranks within groups
    duplicate_ranks = submission.groupby('ranker_id')['selected'].apply(
        lambda x: len(x) != len(set(x))
    ).any()
    
    if duplicate_ranks:
        logger.error("Found duplicate ranks within groups")
        return False
    
    logger.info("Submission validation passed!")
    return True

def target_encode_feature(df: pd.DataFrame, feature_col: str, target_col: str, 
                         smoothing: float = 1.0) -> pd.Series:
    """
    Apply target encoding with smoothing to reduce overfitting
    """
    # Calculate global mean
    global_mean = df[target_col].mean()
    
    # Calculate category means and counts
    category_stats = df.groupby(feature_col)[target_col].agg(['mean', 'count'])
    
    # Apply smoothing
    smoothed_means = (
        category_stats['mean'] * category_stats['count'] + 
        global_mean * smoothing
    ) / (category_stats['count'] + smoothing)
    
    # Map back to original data
    return df[feature_col].map(smoothed_means).fillna(global_mean)

def create_group_array(groups: pd.Series) -> np.ndarray:
    """Create group array for LightGBM from group series"""
    return groups.value_counts().sort_index().values

def safe_divide(numerator: pd.Series, denominator: pd.Series, 
                fill_value: float = 0.0) -> pd.Series:
    """Safe division that handles division by zero"""
    return np.where(denominator != 0, numerator / denominator, fill_value)

def parallel_apply(df: pd.DataFrame, func, n_jobs: int = -1, **kwargs):
    """Apply function in parallel using multiprocessing"""
    try:
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=n_jobs, progress_bar=True)
        return df.parallel_apply(func, **kwargs)
    except ImportError:
        logger.warning("pandarallel not available, using regular apply")
        return df.apply(func, **kwargs)

class MemoryManager:
    """Context manager for memory cleanup"""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        
def log_memory_usage():
    """Log current memory usage"""
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"Current memory usage: {memory_mb:.1f} MB")
