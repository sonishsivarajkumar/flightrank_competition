#!/usr/bin/env python3
"""
FlightRank 2025 Master Solution - Quick Usage Examples

This script demonstrates different ways to use the master solution file.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print the description"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Demonstrate usage of the master solution"""
    
    master_file = "flightrank_master.py"
    
    if not Path(master_file).exists():
        print(f"Error: {master_file} not found!")
        sys.exit(1)
    
    print("FlightRank 2025 Master Solution - Usage Examples")
    print("=" * 60)
    
    # Check if JSON data exists
    json_dir = Path("json_samples")
    if json_dir.exists() and list(json_dir.glob("*.json")):
        print(f"Found JSON data directory: {json_dir}")
        print(f"Number of JSON files: {len(list(json_dir.glob('*.json')))}")
        
        # Example 1: Convert JSON to Parquet (limited files for demo)
        run_command(
            f"python {master_file} --mode convert --max-files 100",
            "Convert first 100 JSON files to parquet format"
        )
        
        # Example 2: Run full pipeline with limited data
        run_command(
            f"python {master_file} --mode train --max-files 100",
            "Run full training pipeline with limited data"
        )
    
    else:
        print("No JSON data found. Checking for existing parquet files...")
        
        if Path("train.parquet").exists() and Path("test.parquet").exists():
            print("Found existing parquet files!")
            
            # Example 3: Train with existing parquet files
            run_command(
                f"python {master_file} --mode train",
                "Train model using existing parquet files"
            )
        
        else:
            print("No data files found. Creating synthetic demo...")
            
            # Example 4: Run performance test (uses synthetic data)
            run_command(
                f"python {master_file} --mode test",
                "Run performance test with synthetic data"
            )
    
    # Example 5: Full pipeline (convert + train + test)
    if json_dir.exists():
        run_command(
            f"python {master_file} --mode full --max-files 50",
            "Run complete pipeline: convert + train + performance test"
        )
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES COMPLETED")
    print("="*60)
    
    print("\nAvailable modes:")
    print("  --mode convert   : Convert JSON files to parquet")
    print("  --mode train     : Train model and generate submission")
    print("  --mode test      : Run performance tests")
    print("  --mode full      : Run complete pipeline + tests")
    print("\nOptional arguments:")
    print("  --max-files N    : Limit number of JSON files to process")
    
    print("\nOutput files generated:")
    output_files = ["train.parquet", "test.parquet", "submission.csv"]
    for filename in output_files:
        if Path(filename).exists():
            size = Path(filename).stat().st_size / 1024  # KB
            print(f"  ✓ {filename} ({size:.1f} KB)")
        else:
            print(f"  - {filename} (not generated)")

if __name__ == "__main__":
    main()
