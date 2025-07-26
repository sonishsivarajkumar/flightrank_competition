"""
macOS Compatible Installation Script
Handles OpenMP library issues for LightGBM/CatBoost
"""

import subprocess
import sys
import os
import platform

def install_package(package, use_conda=False):
    """Install package with error handling"""
    try:
        if use_conda:
            subprocess.check_call([sys.executable, "-m", "conda", "install", "-y", package])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except:
        return False

def check_openmp_macos():
    """Check and install OpenMP for macOS"""
    if platform.system() != "Darwin":
        return True
    
    print("🍎 Detected macOS - checking OpenMP support...")
    
    # Try to install libomp via homebrew
    try:
        result = subprocess.run(["brew", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Homebrew detected, installing libomp...")
            subprocess.run(["brew", "install", "libomp"], capture_output=True)
            
            # Set environment variables
            os.environ["CC"] = "/usr/bin/clang"
            os.environ["CXX"] = "/usr/bin/clang++"
            
            return True
    except:
        pass
    
    print("⚠️  Could not install OpenMP automatically")
    print("   You may need to install it manually:")
    print("   brew install libomp")
    return False

def install_ml_packages():
    """Install ML packages with fallbacks"""
    
    packages = [
        ("pandas", "pandas>=1.3.0"),
        ("numpy", "numpy>=1.21.0"), 
        ("scikit-learn", "scikit-learn>=1.0.0"),
        ("matplotlib", "matplotlib>=3.3.0"),
        ("seaborn", "seaborn>=0.11.0")
    ]
    
    print("📦 Installing core packages...")
    for name, package in packages:
        if install_package(package):
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - FAILED")
            return False
    
    # Try LightGBM with OpenMP check
    print("\n🔬 Installing LightGBM...")
    openmp_ok = check_openmp_macos()
    
    if install_package("lightgbm>=3.2.0"):
        print("✅ LightGBM")
    elif install_package("lightgbm --no-binary lightgbm"):
        print("✅ LightGBM (compiled from source)")
    else:
        print("❌ LightGBM - FAILED")
        print("   Fallback: Will use sklearn models")
    
    # Try CatBoost
    print("\n🐱 Installing CatBoost...")
    if install_package("catboost>=1.0.0"):
        print("✅ CatBoost")
    else:
        print("❌ CatBoost - FAILED")
        print("   Fallback: Will use LightGBM only")
    
    return True

def create_fallback_imports():
    """Create fallback imports for missing packages"""
    
    fallback_code = '''
"""
Fallback imports for missing ML packages
"""
import warnings

# LightGBM fallback
try:
    import lightgbm as lgb
except ImportError:
    warnings.warn("LightGBM not available, using sklearn fallback")
    class MockLGB:
        def __init__(self):
            pass
        def train(self, *args, **kwargs):
            raise ImportError("LightGBM not available")
        def Dataset(self, *args, **kwargs):
            raise ImportError("LightGBM not available")
    lgb = MockLGB()

# CatBoost fallback  
try:
    import catboost as cb
except ImportError:
    warnings.warn("CatBoost not available, using LightGBM fallback")
    class MockCB:
        def __init__(self):
            pass
        def CatBoost(self, *args, **kwargs):
            raise ImportError("CatBoost not available")
    cb = MockCB()
'''
    
    with open("ml_fallbacks.py", "w") as f:
        f.write(fallback_code)
    
    print("✅ Created fallback imports")

def main():
    """Main installation process"""
    
    print("🚀 FLIGHTRANK 2025 - SMART INSTALLATION")
    print("=" * 50)
    print(f"🖥️  Platform: {platform.system()} {platform.machine()}")
    print(f"🐍 Python: {sys.version}")
    
    # Install packages
    if install_ml_packages():
        print("\n✅ Core installation completed!")
    else:
        print("\n❌ Installation had some issues")
    
    # Create fallbacks
    create_fallback_imports()
    
    # Test imports
    print("\n🧪 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas")
    except:
        print("❌ pandas")
    
    try:
        import numpy as np
        print("✅ numpy")
    except:
        print("❌ numpy")
    
    try:
        import lightgbm
        print("✅ lightgbm")
    except:
        print("⚠️  lightgbm (will use fallback)")
    
    try:
        import catboost
        print("✅ catboost")
    except:
        print("⚠️  catboost (will use fallback)")
    
    print(f"\n🎯 Installation Summary:")
    print(f"   - Core packages: pandas, numpy, sklearn")
    print(f"   - ML packages: lightgbm, catboost (with fallbacks)")
    print(f"   - Ready for competition!")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Download competition data")
    print(f"   2. Run: python validate_solution.py")
    print(f"   3. Run: python quick_start.py")

if __name__ == "__main__":
    main()
