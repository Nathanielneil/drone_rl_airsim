# 故障排除指南 - Drone RL AirSim

本文档提供了在Windows系统上运行Drone RL AirSim时可能遇到的问题及其解决方案。

## 快速诊断

首先运行项目自带的诊断工具：

```cmd
# 激活虚拟环境
venv\Scripts\activate

# 运行完整诊断
python scripts\test_installation.py

# 检查具体组件
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import airsim; print('AirSim导入成功')"
```

## 安装相关问题

### Python环境问题

#### 问题: "python不是内部或外部命令"

**原因**: Python未添加到系统PATH或未正确安装

**解决方案**:
1. **重新安装Python**:
   - 从 [python.org](https://python.org) 下载官方安装包
   - 安装时**必须勾选** "Add Python to PATH"
   - 选择 "Customize installation" → 勾选所有选项

2. **手动添加PATH**:
   ```cmd
   # 找到Python安装路径，通常在:
   # C:\Users\{用户名}\AppData\Local\Programs\Python\Python39\
   # C:\Python39\
   
   # 添加到系统PATH (需要管理员权限)
   # 方法1: 通过命令行 (临时)
   set PATH=%PATH%;C:\Python39;C:\Python39\Scripts
   
   # 方法2: 通过系统设置 (永久)
   # 右键"此电脑" → 属性 → 高级系统设置 → 环境变量 → 编辑PATH
   ```

3. **验证修复**:
   ```cmd
   python --version
   pip --version
   ```

#### 问题: 虚拟环境激活失败

**错误信息**: "无法加载文件 venv\Scripts\Activate.ps1，因为在此系统上禁止运行脚本"

**解决方案**:
```cmd
# 方法1: 修改PowerShell执行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 方法2: 使用批处理文件激活
venv\Scripts\activate.bat

# 方法3: 使用命令提示符而非PowerShell
cmd
venv\Scripts\activate
```

### 依赖包安装问题

#### 问题: pip安装超时或失败

**错误信息**: "Read timeout" 或 "Connection broken"

**解决方案**:
```cmd
# 使用国内镜像源
pip install -r requirements/base.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 或使用其他镜像源
pip install -r requirements/base.txt -i https://mirrors.aliyun.com/pypi/simple/

# 增加超时时间
pip install -r requirements/base.txt --timeout 300

# 升级pip版本
python -m pip install --upgrade pip
```

#### 问题: 特定包安装失败

**常见失败包及解决方案**:

1. **PyTorch安装失败**:
   ```cmd
   # 直接从官网安装
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # 或CPU版本
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **AirSim安装失败**:
   ```cmd
   # 尝试不同版本
   pip install airsim==1.8.1
   
   # 或从源码安装
   pip install git+https://github.com/Microsoft/AirSim.git
   ```

3. **OpenCV安装失败**:
   ```cmd
   # 安装无头版本
   pip install opencv-python-headless
   
   # 或完整版本
   pip install opencv-contrib-python
   ```

### CUDA和GPU问题

#### 问题: CUDA版本不匹配

**检查CUDA版本**:
```cmd
# 检查驱动支持的CUDA版本
nvidia-smi

# 检查PyTorch CUDA版本
python -c "import torch; print(torch.version.cuda)"
```

**解决方案**:
```cmd
# 卸载现有PyTorch
pip uninstall torch torchvision torchaudio

# 根据CUDA版本安装对应PyTorch
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU版本 (无GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 问题: "CUDA out of memory"

**解决方案**:
```yaml
# 在配置文件中减少批次大小
# config/algorithms/sac.yaml
sac:
  batch_size: 64        # 从256降到64
  buffer_size: 100000   # 减少缓存大小

# 或强制使用CPU
training:
  device: "cpu"
```

## AirSim连接问题

### 连接失败

#### 问题: "Connection refused" 或 "ConnectionError"

**检查清单**:
1. **AirSim环境是否启动**:
   ```cmd
   # 确保AirSim可执行文件正在运行
   # 例如: C:\AirSim\Blocks\Blocks.exe
   ```

2. **端口是否被占用**:
   ```cmd
   # 检查端口41451是否被占用
   netstat -ano | findstr :41451
   
   # 如果被占用，结束相关进程或更改端口
   ```

3. **防火墙设置**:
   - 将AirSim环境添加到防火墙例外
   - 或临时关闭防火墙测试

4. **网络配置**:
   ```python
   # 测试连接
   import airsim
   
   client = airsim.MultirotorClient()
   try:
       client.confirmConnection()
       print("连接成功!")
   except Exception as e:
       print(f"连接失败: {e}")
   ```

#### 问题: AirSim环境启动失败

**常见原因及解决**:

1. **缺少Visual C++ Runtime**:
   - 下载安装 Microsoft Visual C++ Redistributable
   - 从微软官网获取最新版本

2. **显卡驱动问题**:
   ```cmd
   # 更新显卡驱动
   # NVIDIA: 从官网下载最新驱动
   # AMD: 使用AMD Software更新
   ```

3. **DirectX问题**:
   - 使用 DirectX End-User Runtime Web Installer 更新DirectX

### 配置文件问题

#### 问题: AirSim设置无效

**检查配置文件位置**:
```cmd
# 正确位置
%USERPROFILE%\Documents\AirSim\settings.json

# 检查文件是否存在
dir %USERPROFILE%\Documents\AirSim\
```

**验证JSON格式**:
```cmd
# 使用Python验证JSON格式
python -c "import json; json.load(open(r'%USERPROFILE%\Documents\AirSim\settings.json'))"
```

## 训练相关问题

### 训练不启动

#### 问题: 训练脚本执行失败

**检查步骤**:
1. **确认虚拟环境已激活**:
   ```cmd
   # 命令提示符应显示 (venv)
   where python
   # 应该指向 venv\Scripts\python.exe
   ```

2. **检查脚本路径**:
   ```cmd
   # 确保在项目根目录
   dir scripts\
   # 应该看到 train_sac.py, train_ppo.py 等文件
   ```

3. **查看详细错误**:
   ```cmd
   # 运行时添加详细输出
   python scripts\train_sac.py --log-level DEBUG
   ```

### 训练性能问题

#### 问题: 训练速度过慢

**优化策略**:

1. **降低环境复杂度**:
   ```yaml
   # config/environments/airsim.yaml
   sensors:
     camera:
       width: 64     # 从84降到64
       height: 64
   
   episode:
     max_steps: 500  # 减少步数
   ```

2. **调整训练参数**:
   ```yaml
   # config/algorithms/sac.yaml
   sac:
     train_freq: 4      # 降低训练频率
     batch_size: 128    # 减少批次大小
     learning_starts: 1000  # 提前开始学习
   ```

3. **使用更简单的环境**:
   - 从Blocks环境开始，而非复杂的LandscapeMountains

#### 问题: 内存使用过高

**解决方案**:
```yaml
# 减少缓存大小
sac:
  buffer_size: 100000  # 从1000000降到100000

# 使用更小的网络
network:
  policy_network:
    hidden_sizes: [128, 128]  # 从[256, 256]降低
```

### 训练结果问题

#### 问题: 训练不收敛

**排查步骤**:

1. **检查奖励函数**:
   ```yaml
   # config/environments/airsim.yaml
   reward:
     collision_penalty: -100.0    # 确保惩罚足够大
     goal_reward: 100.0          # 确保奖励明确
     time_penalty: -0.01         # 时间惩罚不要过大
   ```

2. **调整学习率**:
   ```yaml
   sac:
     learning_rate: 0.0001  # 降低学习率
   ```

3. **增加训练时间**:
   ```yaml
   training:
     total_timesteps: 1000000  # 增加训练步数
   ```

4. **检查环境设置**:
   - 确保目标位置合理
   - 检查初始位置不会立即碰撞
   - 验证动作空间范围合适

## 日志和调试

### 启用详细日志

```cmd
# 启用DEBUG级别日志
python experiments\scripts\train.py --algorithm sac --log-level DEBUG

# 查看特定组件日志
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# 你的代码
"
```

### 常用调试命令

```cmd
# 检查系统资源使用
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%')"

# 检查GPU状态 (如果有)
nvidia-smi

# 查看端口占用
netstat -ano | findstr :41451

# 检查进程
tasklist | findstr python
tasklist | findstr Blocks
```

### 日志文件位置

- **训练日志**: `experiments\logs\`
- **模型文件**: `models\`
- **实验结果**: `experiments\results\`
- **错误日志**: 通常在命令行输出中

## 重置和清理

### 完全重置环境

```cmd
# 停止所有相关进程
taskkill /f /im python.exe
taskkill /f /im Blocks.exe

# 清理虚拟环境
rmdir /s venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements/base.txt

# 清理AirSim缓存
rmdir /s %USERPROFILE%\Documents\AirSim\Logs
```

### 清理训练数据

```cmd
# 清理训练输出
rmdir /s experiments\logs
rmdir /s experiments\results
rmdir /s models

# 重新创建目录
mkdir experiments\logs
mkdir experiments\results
mkdir models
```

## 获取更多帮助

### 信息收集

在寻求帮助时，请提供：

1. **系统信息**:
   ```cmd
   # 运行诊断脚本
   python scripts\test_installation.py > diagnosis.txt
   
   # 系统信息
   systeminfo | findstr /B /C:"OS Name" /C:"OS Version"
   python --version
   pip list > installed_packages.txt
   ```

2. **错误信息**:
   - 完整的错误堆栈信息
   - 运行的具体命令
   - 配置文件内容

3. **运行环境**:
   - 硬件配置 (CPU, GPU, RAM)
   - AirSim环境版本
   - 是否使用GPU训练

### 支持渠道

1. **GitHub Issues**: 在项目仓库创建Issue
2. **项目文档**: 查看 `docs/` 目录下的其他文档
3. **社区讨论**: 参与GitHub Discussions

### 常用链接

- [AirSim官方文档](https://microsoft.github.io/AirSim/)
- [PyTorch官方文档](https://pytorch.org/docs/)
- [项目GitHub仓库](https://github.com/Nathanielneil/drone_rl_airsim)

记住：大多数问题都有解决方案，耐心排查通常能找到根本原因！