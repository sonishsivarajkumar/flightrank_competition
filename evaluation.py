"""
Evaluation Module - Optimized for FlightRank 2025 Competition
"""

import pandas as pd
import numpy as np
from typing import Optional

class Evaluator:
    """High-performance evaluation and ranking utilities"""
    
    def __init__(self):
        pass
    
    def calculate_hitrate3(self, df: pd.DataFrame, prediction_col: str = 'rank', 
                          target_col: str = 'selected', group_col: str = 'ranker_id', 
                          min_group_size: int = 0) -> float:
        """
        Calculate HitRate@3 metric efficiently
        
        Args:
            df: DataFrame with predictions and targets
            prediction_col: Column with rank predictions (1 = best)
            target_col: Column with binary target (1 = selected)
            group_col: Column with group identifiers
            min_group_size: Minimum group size to include in evaluation
        
        Returns:
            HitRate@3 score (0 to 1)
        """
        
        # Filter by group size if specified
        if min_group_size > 0:
            group_sizes = df.groupby(group_col).size()
            valid_groups = group_sizes[group_sizes > min_group_size].index
            df = df[df[group_col].isin(valid_groups)]
        
        if len(df) == 0:
            return 0.0
        
        # Find selected items and their ranks
        selected_mask = df[target_col] == 1
        selected_ranks = df.loc[selected_mask, prediction_col]
        
        # Count hits in top-3
        hits = (selected_ranks <= 3).sum()
        total = len(selected_ranks)
        
        return hits / total if total > 0 else 0.0
    
    def scores_to_ranks(self, df: pd.DataFrame, score_col: str, 
                       group_col: str = 'ranker_id') -> pd.Series:
        """Convert scores to ranks within each group efficiently (1 = best score)"""
        return df.groupby(group_col)[score_col].rank(method='dense', ascending=False)
    
    def calculate_ndcg(self, df: pd.DataFrame, prediction_col: str = 'rank', 
                      target_col: str = 'selected', group_col: str = 'ranker_id', 
                      k: int = 10) -> float:
        """Calculate NDCG@k metric"""
        
        def dcg_at_k(r, k):
            """Calculate DCG@k"""
            r = np.asfarray(r)[:k]
            if r.size:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            return 0.0
        
        def ndcg_at_k(r, k):
            """Calculate NDCG@k"""
            dcg_max = dcg_at_k(sorted(r, reverse=True), k)
            if not dcg_max:
                return 0.0
            return dcg_at_k(r, k) / dcg_max
        
        ndcg_scores = []
        
        for group_id in df[group_col].unique():
            group_df = df[df[group_col] == group_id].copy()
            group_df = group_df.sort_values(prediction_col)
            
            relevance = group_df[target_col].values
            ndcg_score = ndcg_at_k(relevance, k)
            ndcg_scores.append(ndcg_score)
        
        return np.mean(ndcg_scores)
    
    def validate_submission(self, df: pd.DataFrame, id_col: str = 'Id', 
                           group_col: str = 'ranker_id', rank_col: str = 'selected') -> dict:
        """Validate submission format thoroughly"""
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required columns
        required_cols = [id_col, group_col, rank_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Missing columns: {missing_cols}")
            return validation_results
        
        # Check for duplicates in Id
        if df[id_col].duplicated().any():
            validation_results['valid'] = False
            validation_results['errors'].append("Duplicate IDs found")
        
        # Check rank validity within each group
        group_issues = []
        for group_id in df[group_col].unique():
            group_df = df[df[group_col] == group_id]
            ranks = sorted(group_df[rank_col].values)
            expected_ranks = list(range(1, len(group_df) + 1))
            
            if ranks != expected_ranks:
                group_issues.append(group_id)
        
        if group_issues:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Invalid rankings in {len(group_issues)} groups")
            if len(group_issues) <= 5:
                validation_results['errors'].append(f"Sample invalid groups: {group_issues}")
        
        # Check for non-integer ranks
        if not df[rank_col].dtype.kind in 'iu':  # integer types
            try:
                df[rank_col] = df[rank_col].astype(int)
            except:
                validation_results['valid'] = False
                validation_results['errors'].append("Non-integer rank values found")
        
        # Check for negative or zero ranks
        if (df[rank_col] <= 0).any():
            validation_results['valid'] = False
            validation_results['errors'].append("Ranks must be positive integers")
        
        # Statistics
        validation_results['stats'] = {
            'total_rows': len(df),
            'total_groups': df[group_col].nunique(),
            'avg_group_size': len(df) / df[group_col].nunique(),
            'min_rank': df[rank_col].min(),
            'max_rank': df[rank_col].max()
        }
        
        return validation_results
    
    def analyze_predictions(self, df: pd.DataFrame, score_col: str, 
                           group_col: str = 'ranker_id') -> dict:
        """Analyze prediction quality and distribution"""
        
        analysis = {}
        
        # Score distribution
        analysis['score_stats'] = {
            'min': df[score_col].min(),
            'max': df[score_col].max(),
            'mean': df[score_col].mean(),
            'std': df[score_col].std(),
            'median': df[score_col].median()
        }
        
        # Group-wise analysis
        group_score_ranges = df.groupby(group_col)[score_col].apply(lambda x: x.max() - x.min())
        analysis['group_score_separation'] = {
            'mean_range': group_score_ranges.mean(),
            'min_range': group_score_ranges.min(),
            'max_range': group_score_ranges.max(),
            'groups_with_no_separation': (group_score_ranges == 0).sum()
        }
        
        # Rank distribution after conversion
        df_temp = df.copy()
        df_temp['rank'] = self.scores_to_ranks(df_temp, score_col, group_col)
        rank_dist = df_temp['rank'].value_counts().sort_index()
        analysis['rank_distribution'] = rank_dist.to_dict()
        
        return analysis
