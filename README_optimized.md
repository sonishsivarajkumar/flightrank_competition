# FlightRank 2025 Competition - Optimized Python Solution

This repository contains a high-performance, production-ready solution for the FlightRank 2025 Kaggle competition. The solution is optimized for speed, performance, and competitive scoring.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Make the script executable and run
./run_solution.sh
```

### Option 2: Manual Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Run the optimized solution
python quick_start.py
```

### Option 3: Advanced Usage
```bash
# Full pipeline with custom settings
python main.py --data_path . --ensemble_models

# Performance testing
python performance_test.py

# Hyperparameter optimization (requires optuna)
python hyperparameter_optimizer.py --n_trials 50
```

## 📁 Project Structure

```
.
├── main.py                      # Main pipeline orchestrator
├── quick_start.py               # Simple one-click execution
├── data_loader.py               # Optimized data loading
├── feature_engineering.py      # High-performance feature creation
├── model_trainer.py             # Optimized LightGBM/CatBoost training
├── evaluation.py                # Fast HitRate@3 calculation
├── performance_test.py          # Performance benchmarking
├── hyperparameter_optimizer.py  # Optuna-based optimization
├── run_solution.sh             # Automated setup script
├── requirements.txt            # Core dependencies
├── requirements_full.txt       # Extended dependencies
└── README_optimized.md         # This file
```

## ⚡ Performance Optimizations

### 1. Data Processing
- **Vectorized operations**: All feature engineering uses pandas vectorization
- **Memory efficient**: Strategic data type optimization (int8, float32)
- **Fast categorical encoding**: Frequency encoding for high-cardinality features
- **Batch processing**: Efficient handling of large datasets

### 2. Feature Engineering (50+ features)
- **Time-based features**: Request timing, departure patterns, business hours
- **Route complexity**: Segments, connections, layover analysis
- **Pricing features**: Price per hour, tax ratios, percentile ranking
- **User preferences**: Frequent flyer matching, VIP status, policy compliance
- **Service quality**: Cabin class, seat availability, baggage allowance

### 3. Model Training
- **Optimized hyperparameters**: Pre-tuned for ranking performance
- **Early stopping**: Prevents overfitting and reduces training time
- **Parallel processing**: Multi-threading enabled
- **Ensemble methods**: LightGBM + CatBoost combination

### 4. Evaluation
- **Fast ranking**: Optimized rank conversion within groups
- **Efficient metrics**: Vectorized HitRate@3 calculation
- **Group-aware validation**: Proper GroupKFold cross-validation

## 📊 Expected Performance

### Speed Benchmarks
- **Feature Engineering**: ~10,000+ rows/second
- **Model Training**: ~5,000+ rows/second  
- **Prediction**: ~50,000+ rows/second
- **Total Pipeline**: ~3,000+ rows/second

### Accuracy Expectations
- **Random Baseline**: ~0.10-0.15 HitRate@3
- **Price Baseline**: ~0.20-0.30 HitRate@3
- **This Solution**: ~0.40-0.60 HitRate@3
- **Competition Target**: 0.70+ HitRate@3 (bonus threshold)

## 🎯 Key Features

### Advanced Feature Engineering
- **Temporal patterns**: Business hours, weekends, seasonal effects
- **Route analysis**: Connection complexity, layover optimization
- **User behavior**: Frequent flyer alignment, booking preferences
- **Policy compliance**: Corporate travel policy adherence
- **Pricing intelligence**: Competitive positioning within sessions

### Model Architecture
- **Learning-to-Rank**: LambdaRank objective optimized for ranking
- **Ensemble approach**: Multiple algorithms for robust predictions
- **Hyperparameter optimization**: Optuna-based automatic tuning
- **Cross-validation**: Group-aware validation preventing data leakage

### Production Features
- **Error handling**: Comprehensive validation and error recovery
- **Performance monitoring**: Built-in timing and throughput metrics
- **Modular design**: Easy to extend and customize
- **Memory efficiency**: Optimized for large datasets

## 🔧 Customization Options

### Command Line Arguments
```bash
python main.py --help

# Common options:
--data_path /path/to/data        # Data directory
--no_cv                          # Skip cross-validation
--single_model                   # Use only LightGBM
--output custom_submission.csv   # Custom output filename
```

### Feature Engineering
Modify `feature_engineering.py` to add domain-specific features:
- Airport metadata (timezone, hub status)
- Airline reputation scores
- Historical route popularity
- Weather-based adjustments

### Model Tuning
Use `hyperparameter_optimizer.py` for automatic optimization:
```bash
python hyperparameter_optimizer.py --n_trials 100
```

## 📈 Improvement Strategies

### Phase 1: Foundation (Current)
- ✅ Optimized feature engineering
- ✅ High-performance ranking models
- ✅ Proper validation methodology
- ✅ Production-ready code

### Phase 2: Advanced Features
- 🔄 External data integration (airports, airlines)
- 🔄 Advanced time series features
- 🔄 Network-based route features
- 🔄 Corporate policy modeling

### Phase 3: Model Enhancement
- 🔄 Neural ranking models
- 🔄 Multi-objective optimization
- 🔄 Stacking ensembles
- 🔄 Domain-specific post-processing

## 🏆 Competition Strategy

### Scoring Focus
- Prioritize sessions with >10 options (evaluation criteria)
- Balance precision in top-3 vs. overall ranking quality
- Consider business logic over pure statistical optimization

### Feature Importance
1. **Pricing**: Total cost, price efficiency, competitive positioning
2. **Timing**: Departure convenience, business hours alignment
3. **Route**: Complexity, duration, connection quality
4. **Policy**: Corporate compliance, approval requirements
5. **User**: Personal preferences, frequent flyer benefits

### Validation Strategy
- Use GroupKFold to prevent session leakage
- Monitor both all-groups and large-groups metrics
- Track validation vs. leaderboard correlation

## 🚨 Troubleshooting

### Common Issues
1. **Memory errors**: Reduce data sample or optimize dtypes
2. **Slow performance**: Check feature engineering bottlenecks
3. **Poor CV scores**: Verify feature alignment between train/test
4. **Submission errors**: Validate ranking format before upload

### Performance Debugging
```bash
python performance_test.py  # Full performance analysis
```

## 📞 Support

For issues or improvements:
1. Check the performance test output
2. Verify data file formats
3. Review feature engineering logs
4. Test with smaller data sample

## 🏁 Competition Submission

1. **Run the solution**: `python quick_start.py`
2. **Validate output**: Check `submission.csv` format
3. **Upload to Kaggle**: Submit via competition interface
4. **Monitor score**: Track leaderboard performance
5. **Iterate**: Improve based on feedback

---

**Good luck and safe travels in your machine learning journey!** ✈️🏆
