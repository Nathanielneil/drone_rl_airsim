#!/usr/bin/env python3
"""
Test script to verify installation
"""
import sys
import importlib
from pathlib import Path

def test_imports():
    """Test importing key dependencies"""
    print("🧪 Testing imports...")
    
    required_packages = [
        "torch", "torchvision", "numpy", "pandas", "cv2", 
        "yaml", "matplotlib", "tqdm", "airsim"
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == "cv2":
                importlib.import_module("cv2")
            elif package == "yaml":
                importlib.import_module("yaml")
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            failed_imports.append(package)
    
    return failed_imports

def test_cuda():
    """Test CUDA availability"""
    print("\n🎮 Testing CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.device_count()} device(s)")
            print(f"  📱 Current device: {torch.cuda.get_device_name()}")
        else:
            print("  ⚠️  CUDA not available, using CPU")
        return True
    except Exception as e:
        print(f"  ❌ CUDA test failed: {e}")
        return False

def test_directory_structure():
    """Test directory structure"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "src", "src/algorithms", "src/environments", "src/core", "src/utils",
        "config", "config/algorithms", "config/environments",
        "experiments", "experiments/scripts", "scripts", "tests", "docs"
    ]
    
    missing_dirs = []
    project_root = Path(__file__).parent.parent
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (missing)")
            missing_dirs.append(dir_name)
    
    return missing_dirs

def test_config_files():
    """Test configuration files"""
    print("\n⚙️  Testing configuration files...")
    
    required_configs = [
        "config/default.yaml",
        "config/algorithms/sac.yaml",
        "config/algorithms/ppo.yaml",
        "config/environments/airsim.yaml"
    ]
    
    missing_configs = []
    project_root = Path(__file__).parent.parent
    
    for config_file in required_configs:
        config_path = project_root / config_file
        if config_path.exists():
            print(f"  ✅ {config_file}")
        else:
            print(f"  ❌ {config_file} (missing)")
            missing_configs.append(config_file)
    
    return missing_configs

def main():
    """Run all tests"""
    print("🚁 Drone RL AirSim Installation Test")
    print("====================================")
    
    # Test imports
    failed_imports = test_imports()
    
    # Test CUDA
    cuda_available = test_cuda()
    
    # Test directory structure
    missing_dirs = test_directory_structure()
    
    # Test config files
    missing_configs = test_config_files()
    
    # Summary
    print("\n📊 Test Summary")
    print("===============")
    
    all_passed = True
    
    if failed_imports:
        print(f"❌ Failed imports: {', '.join(failed_imports)}")
        all_passed = False
    else:
        print("✅ All imports successful")
    
    if missing_dirs:
        print(f"❌ Missing directories: {', '.join(missing_dirs)}")
        all_passed = False
    else:
        print("✅ Directory structure complete")
    
    if missing_configs:
        print(f"❌ Missing configs: {', '.join(missing_configs)}")
        all_passed = False
    else:
        print("✅ Configuration files complete")
    
    if cuda_available:
        print("✅ PyTorch working correctly")
    else:
        print("⚠️  PyTorch issues detected")
    
    if all_passed:
        print("\n🎉 All tests passed! Installation is ready.")
        print("\nNext steps:")
        print("1. Make sure AirSim is installed and running")
        print("2. Try training: python experiments/scripts/train.py --algorithm sac")
        return True
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        print("Run the installation script again or install missing packages.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)