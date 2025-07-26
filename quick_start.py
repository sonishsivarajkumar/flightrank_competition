"""
Quick Start Script for FlightRank 2025 Competition
Simple execution with default settings for best performance
"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import FlightRankingSolution

def quick_run():
    """Quick execution with optimal settings"""
    
    print("🚀 FLIGHTRANK 2025 - QUICK START")
    print("=" * 50)
    
    # Check if data files exist
    required_files = ['train.parquet', 'test.parquet', 'sample_submission.parquet']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📥 Please download the competition data from:")
        print("   https://www.kaggle.com/competitions/aeroclub-recsys-2025/data")
        print("\n   Place the files in the current directory and run again.")
        return
    
    print("✅ All data files found!")
    
    # Initialize and run solution
    solution = FlightRankingSolution(data_path=".")
    
    # Run with optimized settings
    results = solution.run_full_pipeline(
        use_cross_validation=True,    # Enable CV for score estimation
        ensemble_models=True          # Use ensemble for best performance
    )
    
    # Display results
    print("\n" + "=" * 50)
    print("🎯 RESULTS SUMMARY")
    print("=" * 50)
    
    if results['cv_scores']:
        cv_score = results['cv_scores']['hitrate_large_mean']
        print(f"📊 Cross-validation HitRate@3: {cv_score:.4f}")
        
        if cv_score >= 0.7:
            print("🏆 EXCELLENT! Score qualifies for bonus prize!")
        elif cv_score >= 0.5:
            print("🥈 GREAT! Competitive score!")
        elif cv_score >= 0.3:
            print("📈 GOOD! Above baseline performance!")
        else:
            print("📉 Room for improvement - consider feature engineering")
    
    print(f"⚡ Total execution time: {sum(results['timing'].values()):.1f}s")
    print(f"🔧 Features used: {results['feature_count']}")
    print(f"📁 Submission saved as: submission.csv")
    
    # Quick validation
    submission = pd.read_csv("submission.csv")
    print(f"📋 Submission rows: {len(submission):,}")
    print(f"🏢 Unique sessions: {submission['ranker_id'].nunique():,}")
    
    print("\n✅ Ready for Kaggle submission!")
    print("💡 To improve further, try:")
    print("   - Feature engineering with domain knowledge")
    print("   - Hyperparameter optimization")
    print("   - External data (airports, airlines)")
    print("   - Advanced ensemble methods")

if __name__ == "__main__":
    quick_run()
