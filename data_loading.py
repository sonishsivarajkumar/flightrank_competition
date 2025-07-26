"""
Data loading and preprocessing for FlightRank 2025 competition
"""
import pandas as pd
import numpy as np
from pathlib import Path
import tarfile
import json
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import *
from utils import reduce_memory_usage, MemoryManager, log_memory_usage

class DataLoader:
    """Handle data loading and initial preprocessing"""
    
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.sample_submission = None
    
    def load_main_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load main parquet files"""
        logger.info("Loading main datasets...")
        
        # Check if files exist
        for file_path, name in [(TRAIN_PATH, "train"), (TEST_PATH, "test"), 
                               (SAMPLE_SUBMISSION_PATH, "sample_submission")]:
            if not file_path.exists():
                raise FileNotFoundError(f"{name}.parquet not found at {file_path}")
        
        # Load data
        with MemoryManager():
            self.train_df = pd.read_parquet(TRAIN_PATH)
            self.test_df = pd.read_parquet(TEST_PATH)
            self.sample_submission = pd.read_parquet(SAMPLE_SUBMISSION_PATH)
        
        logger.info(f"Train shape: {self.train_df.shape}")
        logger.info(f"Test shape: {self.test_df.shape}")
        logger.info(f"Sample submission shape: {self.sample_submission.shape}")
        
        # Memory optimization
        if USE_REDUCED_MEMORY:
            self.train_df = reduce_memory_usage(self.train_df)
            self.test_df = reduce_memory_usage(self.test_df)
        
        log_memory_usage()
        return self.train_df, self.test_df, self.sample_submission
    
    def extract_json_data(self, limit_files: Optional[int] = None) -> pd.DataFrame:
        """
        Extract and process JSON raw data
        Warning: This requires ~50GB of disk space
        """
        if not JSONS_RAW_PATH.exists():
            logger.warning(f"JSON raw data not found at {JSONS_RAW_PATH}")
            return pd.DataFrame()
        
        logger.info("Extracting JSON raw data...")
        logger.warning("This will require significant disk space (~50GB)")
        
        # First, rename the file to .tar.gz for proper extraction
        gz_path = JSONS_RAW_PATH.parent / "jsons_raw.tar.gz"
        if not gz_path.exists():
            import shutil
            shutil.copy2(JSONS_RAW_PATH, gz_path)
        
        json_data = []
        file_count = 0
        
        try:
            with tarfile.open(gz_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith('.json'):
                        if limit_files and file_count >= limit_files:
                            break
                        
                        try:
                            f = tar.extractfile(member)
                            if f:
                                data = json.load(f)
                                # Extract ranker_id from filename
                                ranker_id = Path(member.name).stem
                                data['ranker_id'] = ranker_id
                                json_data.append(data)
                                file_count += 1
                                
                                if file_count % 1000 == 0:
                                    logger.info(f"Processed {file_count} JSON files")
                        
                        except Exception as e:
                            logger.warning(f"Failed to process {member.name}: {e}")
                            continue
        
        except Exception as e:
            logger.error(f"Failed to extract JSON data: {e}")
            return pd.DataFrame()
        
        if json_data:
            json_df = pd.json_normalize(json_data)
            logger.info(f"Extracted {len(json_df)} records from JSON files")
            return json_df
        else:
            logger.warning("No JSON data extracted")
            return pd.DataFrame()
    
    def get_basic_stats(self) -> dict:
        """Get basic statistics about the data"""
        if self.train_df is None:
            raise ValueError("Data not loaded. Call load_main_data() first.")
        
        stats = {
            'train_rows': len(self.train_df),
            'test_rows': len(self.test_df),
            'train_sessions': self.train_df['ranker_id'].nunique(),
            'test_sessions': self.test_df['ranker_id'].nunique(),
            'train_columns': len(self.train_df.columns),
            'test_columns': len(self.test_df.columns),
            'target_rate': self.train_df['selected'].mean(),
            'sessions_per_user': self.train_df.groupby('profileId')['ranker_id'].nunique().mean()
        }
        
        # Session size analysis
        session_sizes = self.train_df.groupby('ranker_id').size()
        stats.update({
            'avg_session_size': session_sizes.mean(),
            'median_session_size': session_sizes.median(),
            'max_session_size': session_sizes.max(),
            'large_sessions_count': (session_sizes > MIN_GROUP_SIZE_FOR_EVAL).sum(),
            'large_sessions_pct': (session_sizes > MIN_GROUP_SIZE_FOR_EVAL).mean() * 100
        })
        
        return stats
    
    def analyze_missing_values(self) -> pd.DataFrame:
        """Analyze missing values in the dataset"""
        if self.train_df is None:
            raise ValueError("Data not loaded. Call load_main_data() first.")
        
        missing_analysis = []
        for col in self.train_df.columns:
            missing_count = self.train_df[col].isnull().sum()
            missing_pct = (missing_count / len(self.train_df)) * 100
            dtype = str(self.train_df[col].dtype)
            unique_count = self.train_df[col].nunique()
            
            missing_analysis.append({
                'column': col,
                'missing_count': missing_count,
                'missing_pct': missing_pct,
                'dtype': dtype,
                'unique_count': unique_count
            })
        
        missing_df = pd.DataFrame(missing_analysis)
        missing_df = missing_df.sort_values('missing_pct', ascending=False)
        
        return missing_df
    
    def analyze_target_correlation(self) -> pd.DataFrame:
        """Analyze correlation between features and target"""
        if self.train_df is None:
            raise ValueError("Data not loaded. Call load_main_data() first.")
        
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'selected']
        
        correlations = []
        for col in numeric_cols:
            if self.train_df[col].notna().sum() > 100:  # Skip if too many missing values
                corr = self.train_df[col].corr(self.train_df['selected'])
                correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'abs_correlation': abs(corr) if not pd.isna(corr) else 0
                })
        
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('abs_correlation', ascending=False)
        
        return corr_df

def main():
    """Example usage"""
    loader = DataLoader()
    
    # Load main data
    train_df, test_df, sample_submission = loader.load_main_data()
    
    # Get basic statistics
    stats = loader.get_basic_stats()
    logger.info("Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Analyze missing values
    missing_df = loader.analyze_missing_values()
    logger.info("\nTop 10 columns with missing values:")
    logger.info(missing_df.head(10)[['column', 'missing_pct', 'unique_count']].to_string())
    
    # Analyze target correlation
    corr_df = loader.analyze_target_correlation()
    logger.info("\nTop 10 features correlated with target:")
    logger.info(corr_df.head(10)[['feature', 'correlation']].to_string())

if __name__ == "__main__":
    main()
