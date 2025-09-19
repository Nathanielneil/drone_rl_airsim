# 🚀 Modern Setup Guide (Windows 10 + CUDA 12.1)

> **完整的现代化安装和优化指南，专为 Windows 10 + AirSim 1.8.1 + CUDA 12.1 优化**

## 📋 系统要求

### 硬件要求
- **操作系统**: Windows 10 (1909+) 或 Windows 11
- **GPU**: NVIDIA RTX 系列 (推荐 RTX 3080+, 8GB+ VRAM)
- **内存**: 16GB+ RAM (推荐 32GB)
- **存储**: 100GB+ 可用空间 (推荐 NVMe SSD)
- **CPU**: 8核+ 处理器 (推荐 Intel i7-10700K+ 或 AMD Ryzen 7 3700X+)

### 软件要求
- **CUDA**: 12.1 + cuDNN 8.9+
- **Python**: 3.9-3.11 (推荐 3.11)
- **Visual Studio**: 2019/2022 with C++ Build Tools
- **AirSim**: 1.8.1
- **Unreal Engine**: 4.27.2

## 🔧 自动化安装

### 方法1: 一键安装 (推荐)
```batch
# 下载项目
git clone https://github.com/your-repo/drone_rl_airsim.git
cd drone_rl_airsim

# 运行自动化安装脚本
scripts\install_windows_cuda121.bat
```

### 方法2: 手动安装
```bash
# 安装现代化依赖
pip install -r requirements_windows_cuda121.txt

# 验证安装
python scripts/check_cuda_compatibility.py
```

## 🔍 环境验证

### CUDA 验证
```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU设备: {torch.cuda.get_device_name()}")
print(f"GPU数量: {torch.cuda.device_count()}")
```

### AirSim 连接测试
```python
from src.environments.airsim_env.modern_airsim_env import ModernAirSimEnv

# 创建环境实例
env = ModernAirSimEnv()
print("✅ AirSim 环境创建成功")

# 测试重置
obs, info = env.reset()
print(f"✅ 环境重置成功，观察空间: {obs['image'].shape}")
env.close()
```

## 🏃‍♂️ 快速开始训练

### 基础训练
```bash
# 使用默认设置开始训练
python scripts/modern_train.py

# 快速测试 (10K步)
python scripts/modern_train.py --total-timesteps 10000 --experiment-name quick_test
```

### 高级训练配置
```bash
# 自定义训练参数
python scripts/modern_train.py \
    --algorithm sac \
    --total-timesteps 1000000 \
    --batch-size 512 \
    --learning-rate 3e-4 \
    --device cuda \
    --experiment-name high_performance_sac \
    --tensorboard-log experiments/logs

# 启用详细日志
python scripts/modern_train.py --log-level DEBUG
```

## 📊 性能监控

### 实时监控
```bash
# TensorBoard 可视化
tensorboard --logdir experiments/logs --port 6006

# GPU 性能监控
nvidia-smi -l 1

# 性能分析脚本
python -c "
from src.utils.performance.gpu_manager import get_performance_monitor
monitor = get_performance_monitor()
monitor.start_monitoring()
import time; time.sleep(5)
print(monitor.get_detailed_stats())
monitor.stop_monitoring()
"
```

### 性能优化检查
```python
# 运行性能优化
from src.utils.performance.gpu_manager import optimize_for_training
optimizations = optimize_for_training()
print("优化结果:", optimizations)
```

## ⚙️ 配置管理

### 创建自定义配置
```python
from src.core.config_manager import ConfigManager

# 创建配置管理器
config_manager = ConfigManager()

# 生成硬件优化配置
config = config_manager.create_experiment_config(
    algorithm="sac",
    experiment_name="my_experiment",
    custom_config={
        "training.total_timesteps": 500000,
        "training.batch_size": 256,
        "gpu.mixed_precision": True,
        "environment.max_episode_steps": 2000
    }
)

# 保存配置
config_manager.save_config(config, "config/my_experiment.yaml")
```

### 使用配置文件
```bash
# 使用自定义配置训练
python scripts/modern_train.py --config config/my_experiment.yaml
```

## 🛠 故障排除

### 常见问题

#### 1. CUDA 版本不匹配
```bash
# 检查 CUDA 版本
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# 重新安装匹配的 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 2. AirSim 连接失败
```python
# 检查 AirSim 状态
import airsim
client = airsim.MultirotorClient()
try:
    client.confirmConnection()
    print("✅ AirSim 连接成功")
except:
    print("❌ AirSim 连接失败")
    print("请确保 AirSim 环境正在运行")
```

#### 3. 内存不足
```python
# GPU 内存优化
import torch
torch.cuda.empty_cache()

# 降低 batch size
python scripts/modern_train.py --batch-size 128
```

#### 4. 性能问题
```bash
# Windows 性能优化
powercfg /setactive SCHEME_MIN  # 高性能电源方案

# 进程优先级设置 (在脚本中自动执行)
# 详见 src/utils/performance/gpu_manager.py WindowsOptimizer
```

## 📈 性能基准

### 预期性能指标

| GPU型号 | 推荐Batch Size | 训练速度 (steps/s) | 内存使用 | 推荐设置 |
|---------|---------------|------------------|----------|----------|
| RTX 4090 | 1024 | 800-1000 | ~20GB | 全功能 |
| RTX 3080 | 512 | 400-600 | ~8GB | 混合精度 |
| RTX 3070 | 256 | 200-400 | ~6GB | 混合精度 |
| RTX 3060 | 128 | 100-200 | ~4GB | CPU辅助 |

### 性能测试
```bash
# 运行性能基准测试
python scripts/benchmark_performance.py

# 与预期性能比较
python scripts/check_cuda_compatibility.py --benchmark
```

## 🔄 高级功能

### 分布式训练 (实验性)
```bash
# 多GPU训练 (如果可用)
python scripts/modern_train.py --device cuda --multi-gpu
```

### 自动超参数调优
```python
from src.core.config_manager import ConfigManager

config_manager = ConfigManager()
# 硬件感知的配置优化
optimized_config = config_manager.get_optimized_config_for_hardware(base_config)
```

### 继续训练
```bash
# 从检查点继续训练
python scripts/modern_train.py --resume models/experiment_name/checkpoint_50000.pth
```

## 📝 日志和报告

### 训练日志位置
- **TensorBoard**: `experiments/logs/`
- **训练日志**: `training.log`
- **性能报告**: `experiments/results/experiment_name/`
- **模型保存**: `models/experiment_name/`

### 生成详细报告
```python
# 训练完成后自动生成报告
# 报告包含：性能指标、GPU使用率、训练曲线等
# 位置：experiments/results/experiment_name/training_report.json
```

## 🆘 获取帮助

### 社区支持
- **GitHub Issues**: 报告问题和功能请求
- **文档**: 查看完整 API 文档
- **示例**: 参考 `examples/` 目录

### 诊断信息收集
```bash
# 生成诊断报告
python scripts/generate_diagnostic_report.py

# 输出系统信息、GPU状态、依赖版本等
```

---

## 🎯 最佳实践

1. **始终使用现代化训练脚本** (`scripts/modern_train.py`)
2. **启用混合精度训练** 以提高性能
3. **定期监控GPU温度** 避免过热
4. **使用硬件适配配置** 获得最佳性能
5. **保持AirSim和依赖更新** 确保兼容性

开始享受高效的现代化无人机强化学习训练吧！🚁✨