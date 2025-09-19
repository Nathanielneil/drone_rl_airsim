# 🚀 快速开始指南

> **在新的Windows电脑上5分钟快速部署现代化无人机强化学习环境**

## ⚡ 5分钟快速部署

### 📋 前置要求
- Windows 10/11 (64位)
- NVIDIA GPU (推荐RTX系列)
- 已安装CUDA 12.1 + cuDNN 8.9+
- 已安装Python 3.9-3.11

### 🚀 一键部署
```batch
# 1. 克隆项目
git clone https://github.com/Nathanielneil/drone_rl_airsim.git
cd drone_rl_airsim

# 2. 运行自动安装脚本
scripts\install_windows_cuda121.bat

# 3. 完成！🎉
```

## 🎮 立即开始训练

### 启动训练
```batch
# 激活环境
venv\Scripts\activate.bat

# 现代化SAC训练 (推荐)
python scripts\modern_train.py

# 快速测试 (1000步)
python scripts\modern_train.py --total-timesteps 1000 --experiment-name quick_test
```

### 监控训练
```batch
# 启动TensorBoard监控
tensorboard --logdir experiments\logs

# 浏览器访问: http://localhost:6006
```

## 🔧 如果遇到问题

### CUDA问题
```batch
# 检查CUDA
nvidia-smi
nvcc --version

# 重装PyTorch
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

### AirSim连接问题
1. 确保AirSim环境正在运行
2. 检查端口41451是否开放
3. 重启AirSim环境

### 内存不足
```batch
# 降低batch size
python scripts\modern_train.py --batch-size 128
```

## 📚 详细文档

- **完整部署**: [WINDOWS_DEPLOYMENT.md](WINDOWS_DEPLOYMENT.md)
- **现代化设置**: [MODERN_SETUP.md](MODERN_SETUP.md)
- **更新日志**: [CHANGELOG.md](CHANGELOG.md)

## 🆘 获取帮助

遇到问题？
1. 查看 [故障排除指南](WINDOWS_DEPLOYMENT.md#故障排除)
2. 在 [GitHub Issues](https://github.com/Nathanielneil/drone_rl_airsim/issues) 提问
3. 查看详细文档

---

🎯 **目标**: 让您在5分钟内开始现代化无人机强化学习训练！