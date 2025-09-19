# 🖥️ Windows 完整部署指南

> **在全新的Windows电脑上从零开始部署现代化无人机强化学习环境**

## 📋 第一步：系统要求检查

### 硬件要求验证
```powershell
# 在PowerShell中运行以下命令检查系统信息
systeminfo | findstr /B /C:"OS Name" /C:"Total Physical Memory"
wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors
wmic path win32_VideoController get Name,AdapterRAM
```

**最低要求**:
- Windows 10 (版本1909+) 或 Windows 11
- NVIDIA GPU (GTX 1660+ 推荐，RTX系列最佳)
- 16GB+ RAM (推荐 32GB)
- 100GB+ 可用存储空间
- 8核+ CPU

## 🔧 第二步：安装核心依赖

### 2.1 安装 Visual Studio Build Tools
```powershell
# 方法1：使用Chocolatey (推荐)
# 首先安装Chocolatey
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# 安装Visual Studio Build Tools
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools"
```

**手动安装方法**:
1. 下载 [Visual Studio Installer](https://visualstudio.microsoft.com/downloads/)
2. 选择 "Build Tools for Visual Studio 2022"
3. 勾选 "C++ build tools" 工作负荷
4. 安装并重启

### 2.2 安装 CUDA 12.1 + cuDNN
```powershell
# 检查是否已安装CUDA
nvcc --version
nvidia-smi
```

**CUDA 12.1 安装**:
1. 下载 [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)
2. 选择 Windows → x86_64 → 10/11 → exe (network)
3. 运行安装程序，选择"自定义安装"
4. 确保勾选：
   - CUDA Toolkit 12.1
   - CUDA Documentation
   - CUDA Samples
   - CUDA Driver (如果需要更新)

**cuDNN 安装**:
1. 下载 [cuDNN 8.9 for CUDA 12.1](https://developer.nvidia.com/cudnn)
2. 解压到临时文件夹
3. 复制文件到CUDA安装目录：
```powershell
# 通常CUDA安装在这里
cd "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"

# 复制cuDNN文件 (需要管理员权限)
# bin目录下的dll文件
# include目录下的头文件  
# lib目录下的库文件
```

### 2.3 验证CUDA安装
```powershell
# 检查CUDA版本
nvcc --version

# 检查GPU状态
nvidia-smi

# 编译CUDA示例测试
cd "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v12.1\1_Utilities\deviceQuery"
# 如果有Visual Studio，可以编译运行测试
```

### 2.4 安装 Python 3.11
```powershell
# 使用Chocolatey安装Python
choco install python --version=3.11.7

# 或者手动下载安装
# https://www.python.org/downloads/release/python-3117/
```

**重要**：安装时勾选 "Add Python to PATH"

### 2.5 验证Python安装
```powershell
python --version
pip --version

# 升级pip
python -m pip install --upgrade pip
```

## 🚀 第三步：下载和设置项目

### 3.1 克隆项目
```powershell
# 如果没有git，先安装
choco install git

# 克隆项目
cd C:\
git clone https://github.com/Nathanielneil/drone_rl_airsim.git
cd drone_rl_airsim
```

### 3.2 创建Python虚拟环境 (推荐)
```powershell
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\activate

# 验证虚拟环境
where python
```

### 3.3 安装Python依赖
```powershell
# 确保在虚拟环境中
# 安装现代化依赖包
pip install -r requirements_windows_cuda121.txt

# 如果上述文件不存在，使用以下命令：
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install gymnasium==0.29.1
pip install omegaconf==2.3.0
pip install airsim==1.8.1
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install matplotlib==3.7.2
pip install tensorboard==2.14.1
pip install stable-baselines3==2.1.0
pip install pynvml==11.5.0
pip install psutil==5.9.5
pip install gputil==1.4.0
```

### 3.4 验证PyTorch CUDA支持
```python
# 创建测试脚本 test_cuda.py
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'cuDNN版本: {torch.backends.cudnn.version()}')
if torch.cuda.is_available():
    print(f'GPU设备: {torch.cuda.get_device_name()}')
    print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"
```

## 🎮 第四步：安装和配置AirSim

### 4.1 下载AirSim二进制文件
```powershell
# 创建AirSim目录
mkdir C:\AirSim
cd C:\AirSim

# 下载AirSim 1.8.1 Blocks环境 (示例)
# 需要从GitHub Releases下载对应的Windows二进制文件
# https://github.com/Microsoft/AirSim/releases/tag/v1.8.1-windows
```

**手动下载步骤**:
1. 访问 [AirSim Releases](https://github.com/Microsoft/AirSim/releases)
2. 下载 `AirSim-1.8.1-Windows.zip` 或相应的环境包
3. 解压到 `C:\AirSim\`

### 4.2 配置AirSim设置
```powershell
# 创建AirSim配置目录
mkdir "$env:USERPROFILE\Documents\AirSim"

# 创建settings.json文件
New-Item -Path "$env:USERPROFILE\Documents\AirSim\settings.json" -ItemType File -Force
```

**settings.json 内容**:
```json
{
    "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
    "SettingsVersion": 1.2,
    "SimMode": "Multirotor",
    "ClockSpeed": 1.0,
    "Vehicles": {
        "SimpleFlight": {
            "VehicleType": "SimpleFlight",
            "DefaultVehicleState": "Armed",
            "EnableCollisionPassthrough": false,
            "EnableCollisions": true,
            "AllowAPIAlways": true,
            "RC": {
                "RemoteControlID": 0,
                "AllowAPIWhenDisconnected": true
            }
        }
    },
    "CameraDefaults": {
        "CaptureSettings": [
            {
                "ImageType": 0,
                "Width": 256,
                "Height": 144,
                "FOV_Degrees": 90,
                "AutoExposureSpeed": 100,
                "MotionBlurAmount": 0
            },
            {
                "ImageType": 3,
                "Width": 256, 
                "Height": 144,
                "FOV_Degrees": 90,
                "AutoExposureSpeed": 100,
                "MotionBlurAmount": 0
            }
        ]
    },
    "ApiServerPort": 41451,
    "LogMessagesVisible": true
}
```

### 4.3 测试AirSim连接
```powershell
# 启动AirSim环境 (双击exe文件或命令行)
cd C:\AirSim
# .\Blocks.exe (或对应的环境executable)

# 在另一个PowerShell窗口测试连接
python -c "
import airsim
try:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print('✅ AirSim连接成功!')
    print(f'API版本: {client.getApiVersion()}')
except Exception as e:
    print(f'❌ AirSim连接失败: {e}')
"
```

## 🔍 第五步：环境验证和测试

### 5.1 运行完整环境检查
```powershell
# 在项目目录中运行
cd C:\drone_rl_airsim
python scripts\check_cuda_compatibility.py
```

### 5.2 运行现代化环境测试
```powershell
# 测试现代化环境接口
python -c "
from src.environments.airsim_env.modern_airsim_env import ModernAirSimEnv
print('创建现代化AirSim环境...')
env = ModernAirSimEnv()
print('环境创建成功!')
obs, info = env.reset()
print(f'观察空间: {obs['image'].shape}, {obs['state'].shape}')
env.close()
print('✅ 环境测试完成!')
"
```

### 5.3 运行快速训练测试
```powershell
# 快速训练测试 (1000步)
python scripts\modern_train.py --total-timesteps 1000 --experiment-name test_deployment
```

## 🚀 第六步：开始实际训练

### 6.1 基础训练
```powershell
# 使用默认设置开始训练
python scripts\modern_train.py

# 或者指定参数
python scripts\modern_train.py --algorithm sac --total-timesteps 100000 --experiment-name my_first_training
```

### 6.2 监控训练
```powershell
# 在新的PowerShell窗口启动TensorBoard
tensorboard --logdir experiments\logs --port 6006

# 浏览器访问 http://localhost:6006
```

### 6.3 GPU监控
```powershell
# 实时GPU监控
nvidia-smi -l 1

# 或者使用任务管理器的性能标签页监控GPU使用率
```

## 🛠️ 故障排除

### 常见问题解决

#### 1. CUDA相关错误
```powershell
# 重新安装PyTorch CUDA版本
pip uninstall torch torchvision torchaudio
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 2. AirSim连接失败
```powershell
# 检查防火墙设置
# 确保端口41451开放

# 重启AirSim环境
# 检查settings.json配置

# 测试网络连接
telnet 127.0.0.1 41451
```

#### 3. 内存不足
```powershell
# 降低batch size
python scripts\modern_train.py --batch-size 128

# 或者修改配置文件中的batch_size参数
```

#### 4. Visual Studio Build Tools错误
```powershell
# 重新安装Build Tools
choco uninstall visualstudio2022buildtools
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools"
```

### 性能优化检查
```powershell
# 运行性能优化
python -c "
from src.utils.performance.gpu_manager import optimize_for_training
result = optimize_for_training()
print('优化结果:', result)
"
```

## 📋 部署检查清单

完成以下所有步骤后，您的环境就准备就绪了：

- [ ] ✅ 安装Visual Studio Build Tools 2022
- [ ] ✅ 安装CUDA 12.1 + cuDNN 8.9
- [ ] ✅ 安装Python 3.11
- [ ] ✅ 克隆项目代码
- [ ] ✅ 创建并激活虚拟环境
- [ ] ✅ 安装Python依赖包
- [ ] ✅ 验证PyTorch CUDA支持
- [ ] ✅ 下载并配置AirSim 1.8.1
- [ ] ✅ 测试AirSim连接
- [ ] ✅ 运行环境验证脚本
- [ ] ✅ 完成快速训练测试
- [ ] ✅ 设置TensorBoard监控

## 🎯 下一步

环境部署完成后，您可以：

1. **开始正式训练**: `python scripts\modern_train.py`
2. **查看训练监控**: 访问 http://localhost:6006
3. **调整配置参数**: 编辑 `config\default.yaml`
4. **尝试不同算法**: 使用 `--algorithm` 参数
5. **性能调优**: 根据硬件配置优化参数

🎉 **恭喜！您已成功在Windows上部署了现代化的无人机强化学习环境！**