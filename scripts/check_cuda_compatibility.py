#!/usr/bin/env python3
"""
CUDA兼容性检查脚本
专为Windows 10 + AirSim 1.8.1 + UE4.7.2 + CUDA 12.1环境设计
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    
    if 3.8 <= version.major + version.minor/10 <= 3.11:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - 兼容")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - 不兼容")
        print(f"   推荐: Python 3.8-3.11")
        return False

def check_nvidia_driver():
    """检查NVIDIA驱动"""
    print("\n🎮 检查NVIDIA驱动和CUDA...")
    
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, check=True)
        output = result.stdout
        
        # 提取CUDA版本
        lines = output.split('\n')
        for line in lines:
            if 'CUDA Version:' in line:
                cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                print(f"   ✅ NVIDIA驱动已安装")
                print(f"   📊 驱动支持的最高CUDA版本: {cuda_version}")
                
                # 检查是否支持CUDA 12.1
                if cuda_version >= "12.1":
                    print(f"   ✅ 支持CUDA 12.1")
                    return True, cuda_version
                else:
                    print(f"   ⚠️  驱动版本较旧，可能不支持CUDA 12.1")
                    return False, cuda_version
                    
        return False, None
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ❌ NVIDIA驱动未安装或nvidia-smi不可用")
        print("   建议: 安装最新NVIDIA驱动")
        return False, None

def check_pytorch():
    """检查PyTorch安装和CUDA支持"""
    print("\n🔥 检查PyTorch...")
    
    try:
        import torch
        print(f"   ✅ PyTorch已安装: {torch.__version__}")
        
        # 检查CUDA支持
        if torch.cuda.is_available():
            print(f"   ✅ CUDA支持: 可用")
            print(f"   🎯 GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   📱 GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # 检查CUDA版本匹配
            cuda_version = torch.version.cuda
            print(f"   🔧 PyTorch CUDA版本: {cuda_version}")
            
            if cuda_version and cuda_version.startswith("12.1"):
                print(f"   ✅ CUDA版本匹配 (12.1)")
                return True, "cuda"
            else:
                print(f"   ⚠️  CUDA版本不匹配，期望12.1，实际{cuda_version}")
                return True, "cuda_mismatch"
        else:
            print(f"   ⚠️  CUDA支持: 不可用 (CPU版本)")
            return True, "cpu"
            
    except ImportError:
        print("   ❌ PyTorch未安装")
        return False, None

def check_airsim():
    """检查AirSim安装"""
    print("\n🚁 检查AirSim...")
    
    try:
        import airsim
        
        # 尝试获取版本信息
        if hasattr(airsim, '__version__'):
            version = airsim.__version__
        else:
            version = "版本信息不可用"
            
        print(f"   ✅ AirSim已安装: {version}")
        
        # 检查关键模块
        try:
            client = airsim.MultirotorClient()
            print(f"   ✅ MultirotorClient可用")
            return True
        except Exception as e:
            print(f"   ⚠️  AirSim客户端创建警告: {e}")
            print(f"   (这在AirSim环境未运行时是正常的)")
            return True
            
    except ImportError:
        print("   ❌ AirSim未安装")
        return False

def check_key_dependencies():
    """检查关键依赖"""
    print("\n📦 检查关键依赖...")
    
    dependencies = {
        'numpy': '数值计算',
        'opencv-python': '计算机视觉',
        'gymnasium': '强化学习环境',
        'stable_baselines3': '强化学习算法',
        'tensorboard': '训练监控',
        'matplotlib': '数据可视化',
        'yaml': 'YAML配置',
    }
    
    missing = []
    for pkg, desc in dependencies.items():
        try:
            if pkg == 'opencv-python':
                import cv2
                print(f"   ✅ {pkg} ({desc})")
            elif pkg == 'yaml':
                import yaml
                print(f"   ✅ {pkg} ({desc})")
            else:
                importlib.import_module(pkg)
                print(f"   ✅ {pkg} ({desc})")
        except ImportError:
            print(f"   ❌ {pkg} ({desc}) - 未安装")
            missing.append(pkg)
    
    return len(missing) == 0, missing

def check_windows_specific():
    """检查Windows特定配置"""
    print("\n🖥️  检查Windows环境...")
    
    if sys.platform != "win32":
        print("   ⚠️  非Windows系统")
        return False
    
    print("   ✅ Windows系统")
    
    # 检查AirSim配置目录
    airsim_dir = Path.home() / "Documents" / "AirSim"
    settings_file = airsim_dir / "settings.json"
    
    if airsim_dir.exists():
        print(f"   ✅ AirSim配置目录存在")
        if settings_file.exists():
            print(f"   ✅ AirSim配置文件存在")
        else:
            print(f"   ⚠️  AirSim配置文件不存在")
            print(f"   建议: 复制config/airsim_settings.json到该位置")
    else:
        print(f"   ⚠️  AirSim配置目录不存在")
    
    return True

def generate_recommendations(results):
    """生成建议"""
    print("\n" + "="*60)
    print("🔧 安装建议")
    print("="*60)
    
    python_ok, driver_ok, pytorch_ok, airsim_ok, deps_ok = results
    
    if not python_ok:
        print("\n1. 更新Python:")
        print("   - 从 https://python.org 下载Python 3.9")
        print("   - 安装时勾选 'Add Python to PATH'")
    
    if not driver_ok[0]:
        print("\n2. 安装NVIDIA驱动:")
        print("   - 访问 https://www.nvidia.com/drivers")
        print("   - 下载最新Game Ready驱动")
        print("   - 重启电脑后验证: nvidia-smi")
    
    if not pytorch_ok[0] or pytorch_ok[1] == "cpu":
        print("\n3. 安装CUDA版PyTorch:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121")
    
    if not airsim_ok:
        print("\n4. 安装AirSim:")
        print("   pip install airsim==1.8.1")
    
    if not deps_ok[0]:
        print("\n5. 安装依赖包:")
        print("   pip install -r requirements.txt")
        
    print("\n6. 推荐的完整安装命令:")
    print("   scripts\\install_windows_cuda121.bat")

def main():
    """主函数"""
    print("🚁 Drone RL AirSim - CUDA 兼容性检查")
    print("专为 Windows 10 + AirSim 1.8.1 + UE4.7.2 + CUDA 12.1")
    print("="*60)
    
    # 执行检查
    python_ok = check_python_version()
    driver_ok = check_nvidia_driver()
    pytorch_ok = check_pytorch()
    airsim_ok = check_airsim()
    deps_ok = check_key_dependencies()
    windows_ok = check_windows_specific()
    
    # 汇总结果
    print("\n" + "="*60)
    print("📊 兼容性检查结果")
    print("="*60)
    
    results = [python_ok, driver_ok, pytorch_ok, airsim_ok, deps_ok]
    status_symbols = ["✅", "❌"]
    
    print(f"Python版本:     {status_symbols[not python_ok]} {'兼容' if python_ok else '需要更新'}")
    print(f"NVIDIA驱动:     {status_symbols[not driver_ok[0]]} {'正常' if driver_ok[0] else '需要安装'}")
    print(f"PyTorch:        {status_symbols[not pytorch_ok[0]]} {pytorch_ok[1] if pytorch_ok[0] else '需要安装'}")
    print(f"AirSim:         {status_symbols[not airsim_ok]} {'正常' if airsim_ok else '需要安装'}")
    print(f"依赖包:         {status_symbols[not deps_ok[0]]} {'完整' if deps_ok[0] else f'缺少{len(deps_ok[1])}个'}")
    print(f"Windows环境:    {status_symbols[not windows_ok]} {'正常' if windows_ok else '有问题'}")
    
    # 整体评估
    all_critical_ok = python_ok and pytorch_ok[0] and airsim_ok and deps_ok[0]
    
    if all_critical_ok and driver_ok[0] and pytorch_ok[1] == "cuda":
        print(f"\n🎉 完美! 环境配置完全兼容CUDA 12.1训练")
        print(f"可以运行: python scripts\\train_sac_gpu.py")
    elif all_critical_ok:
        print(f"\n⚠️  基本功能可用，但建议配置GPU加速获得更好性能")
    else:
        print(f"\n❌ 环境需要修复才能正常使用")
    
    # 生成建议
    generate_recommendations(results)
    
    return all_critical_ok

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  检查被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 检查过程出错: {e}")
        sys.exit(1)