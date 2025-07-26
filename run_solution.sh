#!/bin/bash

# FlightRank 2025 - Quick Setup and Execution Script

echo "🚀 FlightRank 2025 Competition - Setup and Execution"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Check if data files exist
echo ""
echo "📂 Checking for competition data files..."
missing_files=()

for file in "train.parquet" "test.parquet" "sample_submission.parquet"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    else
        echo "✅ Found: $file"
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo ""
    echo "❌ Missing required data files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "📥 Please download the competition data from:"
    echo "   https://www.kaggle.com/competitions/aeroclub-recsys-2025/data"
    echo ""
    echo "   Place the files in this directory and run the script again."
    exit 1
fi

echo "✅ All required data files found!"

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
python3 -m pip install -r requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies. Please check your Python environment."
    exit 1
fi

echo "✅ Dependencies installed successfully!"

# Run performance test (optional)
read -p "🧪 Run performance test first? (y/N): " run_test
if [[ $run_test =~ ^[Yy]$ ]]; then
    echo ""
    echo "🧪 Running performance test..."
    python3 performance_test.py
    echo ""
fi

# Run main solution
echo "🚀 Running FlightRank 2025 solution..."
echo ""

python3 quick_start.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Solution completed successfully!"
    echo ""
    echo "📁 Your submission file 'submission.csv' is ready for upload to Kaggle!"
    echo ""
    echo "💡 Next steps:"
    echo "   1. Upload submission.csv to Kaggle competition page"
    echo "   2. Check your score on the leaderboard"
    echo "   3. Iterate and improve your features/models"
    echo ""
    echo "🔧 For advanced optimization:"
    echo "   python3 main.py --help"
    echo ""
else
    echo "❌ Solution failed. Please check the error messages above."
    exit 1
fi
