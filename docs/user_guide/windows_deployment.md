# Windows部署指南 - Drone RL AirSim

本文档详细介绍如何在Windows电脑上部署和运行重构后的Drone RL AirSim项目。

## 目录
- [系统要求](#系统要求)
- [环境准备](#环境准备)
- [项目部署](#项目部署)
- [AirSim环境配置](#AirSim环境配置)
- [验证安装](#验证安装)
- [运行训练](#运行训练)
- [常见问题](#常见问题)
- [性能优化](#性能优化)

## 系统要求

### 硬件要求
- **操作系统**: Windows 10/11 (64位)
- **内存**: 最低8GB，推荐16GB以上
- **存储**: 至少20GB可用空间
- **显卡**: 
  - 训练推荐: NVIDIA GTX 1060 6GB 或更高
  - 仅推理: 集成显卡即可
- **处理器**: Intel i5-8400 或 AMD Ryzen 5 2600 及以上

### 软件要求
- **Python**: 3.8 - 3.11 (推荐3.9)
- **Git**: 用于克隆项目
- **Visual Studio**: 2019/2022 (可选，用于C++编译)
- **NVIDIA驱动**: 如使用GPU训练

## 环境准备

### 1. 安装Python

1. 前往 [Python官网](https://www.python.org/downloads/) 下载Python
2. 选择Python 3.9.x版本
3. 安装时**务必勾选**"Add Python to PATH"
4. 验证安装：
   ```cmd
   python --version
   pip --version
   ```

### 2. 安装Git

1. 下载 [Git for Windows](https://git-scm.com/download/win)
2. 使用默认设置安装
3. 验证安装：
   ```cmd
   git --version
   ```

### 3. 安装CUDA (可选，用于GPU加速)

如果有NVIDIA显卡且需要GPU训练：

1. 检查显卡驱动版本：
   ```cmd
   nvidia-smi
   ```

2. 下载对应的CUDA Toolkit (推荐11.8版本)：
   - 访问 [NVIDIA CUDA官网](https://developer.nvidia.com/cuda-downloads)
   - 选择Windows x86_64版本
   - 按照向导安装

## 项目部署

### 1. 克隆项目

打开命令提示符或PowerShell：

```cmd
# 克隆项目到本地
git clone https://github.com/Nathanielneil/drone_rl_airsim.git

# 进入项目目录
cd drone_rl_airsim
```

### 2. 创建虚拟环境

```cmd
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate

# 升级pip
python -m pip install --upgrade pip
```

### 3. 安装依赖

#### 基础安装 (仅推理)
```cmd
pip install -r requirements/base.txt
```

#### 完整安装 (包含训练)
```cmd
# 安装基础依赖
pip install -r requirements/base.txt

# 安装训练依赖
pip install -r requirements/training.txt

# 安装评估依赖 (可选)
pip install -r requirements/evaluation.txt
```

#### GPU支持 (NVIDIA显卡)
```cmd
# 卸载CPU版本PyTorch
pip uninstall torch torchvision torchaudio

# 安装CUDA版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. 验证PyTorch安装

```cmd
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

## AirSim环境配置

### 1. 下载AirSim预编译环境

#### 方案A: 使用预编译环境 (推荐)

1. 访问 [AirSim Releases](https://github.com/microsoft/AirSim/releases)
2. 下载适合的环境，推荐：
   - **Blocks**: 基础环境，适合快速测试
   - **SimpleMaze**: 迷宫环境，适合避障训练
   - **LandscapeMountains**: 大型户外环境

3. 解压到合适位置，如：
   ```
   C:\AirSim\Blocks\
   C:\AirSim\SimpleMaze\
   ```

#### 方案B: 自建环境 (高级用户)

1. 安装Unreal Engine 4.27
2. 克隆AirSim源码：
   ```cmd
   git clone https://github.com/Microsoft/AirSim.git
   ```
3. 按照官方文档编译

### 2. 配置AirSim设置

创建AirSim配置文件：

```cmd
# 创建AirSim配置目录
mkdir %USERPROFILE%\Documents\AirSim

# 复制项目配置文件
copy config\airsim_settings.json %USERPROFILE%\Documents\AirSim\settings.json
```

或手动创建 `%USERPROFILE%\Documents\AirSim\settings.json`：

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockType": "SteppableClock",
  "Vehicles": {
    "SimpleFlight": {
      "VehicleType": "SimpleFlight",
      "AutoCreate": true,
      "PawnBP": "class'/AirSim/VehicleAdv/Vehicle/VehiclePawn.VehiclePawn_C'",
      "Cameras": {
        "front_center": {
          "CameraType": "SceneCapture",
          "ImageType": "DepthVis",
          "FOV_Degrees": 90,
          "AutoExposureSpeed": 100,
          "MotionBlurAmount": 0
        }
      }
    }
  }
}
```

## 验证安装

### 1. 运行安装测试

```cmd
# 激活虚拟环境
venv\Scripts\activate

# 运行测试脚本
python scripts\test_installation.py
```

测试脚本会检查：
- Python环境和依赖包
- PyTorch和CUDA状态
- 项目目录结构
- 配置文件完整性

### 2. 测试AirSim连接

1. 启动AirSim环境：
   ```cmd
   # 进入AirSim环境目录，例如
   cd C:\AirSim\Blocks
   
   # 启动环境
   Blocks.exe
   ```

2. 在新的命令窗口测试连接：
   ```cmd
   # 激活虚拟环境
   cd drone_rl_airsim
   venv\Scripts\activate
   
   # 测试连接
   python -c "import airsim; client = airsim.MultirotorClient(); client.confirmConnection(); print('AirSim连接成功!')"
   ```

## 运行训练

### 1. 快速开始

启动AirSim环境后，在项目目录运行：

```cmd
# 激活虚拟环境
venv\Scripts\activate

# 快速训练SAC算法
python scripts\train_sac.py

# 或快速训练PPO算法  
python scripts\train_ppo.py
```

### 2. 自定义训练

```cmd
# 使用默认配置训练SAC
python experiments\scripts\train.py --algorithm sac

# 使用自定义配置
python experiments\scripts\train.py --algorithm ppo --config config\algorithms\ppo.yaml

# 指定实验名称
python experiments\scripts\train.py --algorithm sac --experiment-name my_sac_experiment
```

### 3. 监控训练

训练日志会保存在：
- **TensorBoard日志**: `experiments\logs\`
- **模型检查点**: `models\`
- **训练输出**: `experiments\results\`

启动TensorBoard监控：
```cmd
tensorboard --logdir experiments\logs
```

然后在浏览器访问 `http://localhost:6006`

## 常见问题

### 1. Python相关

**问题**: "python不是内部或外部命令"
**解决**: 
- 重新安装Python，确保勾选"Add Python to PATH"
- 或手动添加Python到系统PATH

**问题**: 虚拟环境激活失败
**解决**:
```cmd
# 使用PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 重新激活
venv\Scripts\activate
```

### 2. 依赖安装问题

**问题**: pip安装超时
**解决**:
```cmd
# 使用国内镜像源
pip install -r requirements/base.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

**问题**: PyTorch CUDA版本不匹配
**解决**:
```cmd
# 检查CUDA版本
nvidia-smi

# 安装对应版本的PyTorch
# 访问 https://pytorch.org/ 获取正确的安装命令
```

### 3. AirSim连接问题

**问题**: "Connection refused"
**解决**:
- 确保AirSim环境已启动
- 检查防火墙设置
- 确认端口41451未被占用

**问题**: 性能低下
**解决**:
- 降低AirSim环境的图形设置
- 关闭不必要的后台程序
- 考虑使用更简单的环境

### 4. 训练问题

**问题**: 内存不足
**解决**:
- 减小batch_size
- 减小replay buffer大小
- 使用CPU训练

**问题**: 训练不收敛
**解决**:
- 检查环境配置
- 调整学习率
- 增加训练步数

## 性能优化

### 1. 硬件优化

**GPU使用**:
```cmd
# 检查GPU使用情况
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

**内存优化**:
- 关闭不必要的程序
- 设置合适的batch_size
- 使用混合精度训练 (高级)

### 2. 配置优化

**训练配置** (`config/algorithms/sac.yaml`):
```yaml
# 针对性能优化的配置示例
sac:
  batch_size: 128        # 降低batch_size减少内存使用
  learning_starts: 5000  # 减少初始步数加快开始
  train_freq: 4          # 降低训练频率提升速度
```

**环境配置** (`config/environments/airsim.yaml`):
```yaml
# 性能优化的环境配置
sensors:
  camera:
    width: 64      # 降低分辨率提升性能
    height: 64
    
episode:
  max_steps: 500   # 减少episode长度
```

### 3. 监控性能

```cmd
# 监控系统资源使用
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, 内存: {psutil.virtual_memory().percent}%')"

# 监控GPU使用 (如果有)
nvidia-smi
```

## 部署检查清单

在新Windows电脑上部署时，按以下清单检查：

- [ ] Python 3.8-3.11 已安装并添加到PATH
- [ ] Git 已安装
- [ ] 项目已克隆到本地
- [ ] 虚拟环境已创建并激活
- [ ] 依赖包已安装 (base/training/evaluation)
- [ ] PyTorch版本正确 (CPU或GPU)
- [ ] AirSim环境已下载并可启动
- [ ] AirSim配置文件已设置
- [ ] 安装测试通过
- [ ] AirSim连接测试成功
- [ ] 可以运行快速训练测试

## 技术支持

如遇到问题，请按以下顺序解决：

1. **查看本文档**的常见问题部分
2. **运行诊断**：`python scripts\test_installation.py`
3. **检查日志**：查看训练输出和错误信息
4. **搜索Issues**：在GitHub项目中搜索相似问题
5. **创建Issue**：提供详细的错误信息和系统配置

## 更新项目

定期更新项目获取最新功能：

```cmd
# 拉取最新代码
git pull origin master

# 更新依赖包
pip install -r requirements/base.txt --upgrade
```

---

**注意**: 本文档基于项目重构后的最新架构编写。如有问题或建议，请在项目仓库中创建Issue。