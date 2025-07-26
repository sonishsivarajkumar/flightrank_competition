"""
Configuration file for FlightRank 2025 competition
"""
import os
from pathlib import Path

# Data paths
DATA_DIR = Path(".")
TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.parquet"
JSONS_RAW_PATH = DATA_DIR / "jsons_raw.tar.kaggle"

# Output paths
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

# Model parameters
LGBM_PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 7,
    'min_data_in_leaf': 50,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'random_state': 42,
    'num_boost_round': 2000,
    'early_stopping_rounds': 200
}

CATBOOST_PARAMS = {
    'loss_function': 'RMSE',
    'custom_metric': ['NDCG:top=3'],
    'iterations': 2000,
    'learning_rate': 0.1,
    'depth': 8,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'random_seed': 42,
    'verbose': 100,
    'early_stopping_rounds': 200,
    'task_type': 'CPU'
}

# Cross-validation settings
CV_FOLDS = 5
RANDOM_STATE = 42

# Feature engineering settings
MIN_GROUP_SIZE_FOR_EVAL = 10
TARGET_ENCODING_SMOOTHING = 10.0

# Memory optimization
CHUNK_SIZE = 100000
USE_REDUCED_MEMORY = True

# Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'flightrank.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
