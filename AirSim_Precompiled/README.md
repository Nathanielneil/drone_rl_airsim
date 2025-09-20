# AirSim 预编译仿真环境下载指南

> **本目录用于存放AirSim预编译的仿真环境。为了减少仓库大小，请手动下载对应平台的环境。**

## 🎮 Windows 预编译环境下载

### 📍 官方下载地址

**AirSim 1.8.1 官方Release页面**:
- 🔗 **主下载页面**: [AirSim v1.8.1 Releases](https://github.com/Microsoft/AirSim/releases/tag/v1.8.1-windows)

### 🏗️ 推荐的仿真环境

| 环境名称 | 文件大小 | 适用场景 | 下载链接 |
|----------|----------|----------|----------|
| **Blocks** | ~1.5GB | 🔰 入门训练、算法测试 | [AirSimNH.zip](https://github.com/Microsoft/AirSim/releases/download/v1.8.1-windows/AirSimNH.zip) |
| **Neighborhood** | ~2.2GB | 🏘️ 城市环境、复杂导航 | [Neighborhood.zip](https://github.com/Microsoft/AirSim/releases/download/v1.8.1-windows/Neighborhood.zip) |
| **LandscapeMountains** | ~3.1GB | 🏔️ 地形导航、高度挑战 | [LandscapeMountains.zip](https://github.com/Microsoft/AirSim/releases/download/v1.8.1-windows/LandscapeMountains.zip) |
| **Building99** | ~2.8GB | 🏢 室内导航、精确控制 | [Building99.zip](https://github.com/Microsoft/AirSim/releases/download/v1.8.1-windows/Building99.zip) |

### 🚀 快速下载 (推荐用于入门)

```powershell
# 下载Blocks环境 (最适合入门和测试)
Invoke-WebRequest -Uri "https://github.com/Microsoft/AirSim/releases/download/v1.8.1-windows/AirSimNH.zip" -OutFile "AirSimNH.zip"

# 解压到当前目录
Expand-Archive -Path "AirSimNH.zip" -DestinationPath "./AirSim_Precompiled/"
```

### 📥 下载和安装步骤

1. **选择环境**: 根据训练需求选择合适的仿真环境
2. **下载**: 点击对应链接下载zip文件
3. **解压**: 解压到 `AirSim_Precompiled/` 目录下
4. **运行**: 双击解压后的 `.exe` 文件启动仿真

### 🎯 推荐配置

#### 🔰 初学者 - Blocks环境
```
✅ 简单的几何结构
✅ 快速加载
✅ 低系统要求
✅ 适合算法开发和测试
```

#### 🏘️ 进阶 - Neighborhood环境  
```
✅ 真实城市场景
✅ 复杂的导航挑战
✅ 动态障碍物
✅ 适合高级算法训练
```

## 🔧 配置AirSim设置

下载并运行仿真环境后，需要配置AirSim设置：

### Windows设置文件位置
```
%USERPROFILE%\Documents\AirSim\settings.json
```

### 推荐的settings.json配置
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
                "FOV_Degrees": 90
            },
            {
                "ImageType": 3,
                "Width": 256, 
                "Height": 144,
                "FOV_Degrees": 90
            }
        ]
    },
    "ApiServerPort": 41451,
    "LogMessagesVisible": true
}
```

## 🧪 测试连接

下载并配置完成后，测试AirSim连接：

```python
import airsim

# 创建客户端连接
client = airsim.MultirotorClient()

# 确认连接
try:
    client.confirmConnection()
    print("✅ AirSim连接成功!")
    print(f"API版本: {client.getApiVersion()}")
except:
    print("❌ AirSim连接失败，请检查仿真环境是否运行")
```

## 🎯 与训练脚本集成

配置完成后，使用现代化训练脚本：

```powershell
# 激活conda环境
conda activate drone_rl

# 运行现代化训练
python scripts\modern_train.py --total-timesteps 10000 --experiment-name airsim_test

# 启动训练监控
tensorboard --logdir experiments\logs
```

## 🔗 相关资源

- 📖 **AirSim官方文档**: [https://microsoft.github.io/AirSim/](https://microsoft.github.io/AirSim/)
- 🛠️ **设置指南**: [https://microsoft.github.io/AirSim/settings/](https://microsoft.github.io/AirSim/settings/)
- 🐍 **Python API**: [https://microsoft.github.io/AirSim/apis/](https://microsoft.github.io/AirSim/apis/)
- 🎮 **所有Release**: [https://github.com/Microsoft/AirSim/releases](https://github.com/Microsoft/AirSim/releases)

## ⚠️ 注意事项

1. **系统要求**: Windows 10/11, DirectX 11支持的显卡
2. **杀毒软件**: 可能需要将AirSim添加到杀毒软件白名单
3. **防火墙**: 确保端口41451开放
4. **性能**: 建议8GB+内存，独立显卡

---

🎉 **开始您的无人机强化学习之旅吧！**