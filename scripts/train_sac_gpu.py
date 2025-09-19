#!/usr/bin/env python3
"""
GPU加速的SAC训练脚本
使用CUDA进行加速训练
"""
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.scripts.train import main
import sys

def check_gpu():
    """检查GPU可用性"""
    if torch.cuda.is_available():
        print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"🔧 CUDA版本: {torch.version.cuda}")
        return True
    else:
        print("⚠️  未检测到可用GPU，将使用CPU训练")
        return False

if __name__ == "__main__":
    print("🚀 启动GPU加速SAC训练")
    print("=" * 50)
    
    # 检查GPU
    has_gpu = check_gpu()
    
    # 设置训练参数
    if len(sys.argv) == 1:
        # 默认使用GPU优化配置
        if has_gpu:
            sys.argv.extend([
                "--algorithm", "sac",
                "--config", "config/gpu_optimized.yaml",
                "--experiment-name", "sac_gpu_training"
            ])
        else:
            # 降级到CPU配置
            sys.argv.extend([
                "--algorithm", "sac", 
                "--experiment-name", "sac_cpu_training"
            ])
    elif "--algorithm" not in " ".join(sys.argv):
        sys.argv.extend(["--algorithm", "sac"])
    
    print(f"🎯 训练配置: {' '.join(sys.argv[1:])}")
    print("=" * 50)
    
    # 开始训练
    main()