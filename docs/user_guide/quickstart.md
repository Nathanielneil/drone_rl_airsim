# 快速开始指南 - Drone RL AirSim

本指南帮助您快速在Windows系统上运行第一个强化学习训练任务。

## 5分钟快速部署

### 前置条件
- Windows 10/11 系统
- Python 3.8-3.11
- 8GB以上内存

### 步骤1: 获取项目

```cmd
# 克隆项目
git clone https://github.com/Nathanielneil/drone_rl_airsim.git
cd drone_rl_airsim
```

### 步骤2: 安装环境

```cmd
# 创建虚拟环境
python -m venv venv

# 激活环境
venv\Scripts\activate

# 安装依赖
pip install -r requirements/base.txt
```

### 步骤3: 下载AirSim环境

1. 访问 [AirSim Releases](https://github.com/microsoft/AirSim/releases)
2. 下载 "Blocks" 环境
3. 解压到 `C:\AirSim\Blocks\`

### 步骤4: 验证安装

```cmd
python scripts\test_installation.py
```

### 步骤5: 运行第一个训练

```cmd
# 启动AirSim (在新窗口)
C:\AirSim\Blocks\Blocks.exe

# 开始训练 (在项目目录)
python scripts\train_sac.py
```

恭喜！您已经成功启动了第一个训练任务。

## 详细使用指南

### 支持的算法

| 算法类型 | 算法名称 | 适用场景 | 训练脚本 |
|---------|---------|---------|----------|
| Actor-Critic | SAC | 连续控制，稳定性好 | `scripts\train_sac.py` |
| Policy-Based | PPO | 通用性强，易调参 | `scripts\train_ppo.py` |
| Actor-Critic | DDPG | 连续控制，确定性策略 | 需自定义配置 |
| Actor-Critic | TD3 | DDPG改进版 | 需自定义配置 |
| Value-Based | DQN | 离散控制 | 需自定义配置 |

### 自定义训练配置

#### 修改算法参数

编辑 `config/algorithms/sac.yaml`:
```yaml
sac:
  learning_rate: 0.0003    # 学习率
  batch_size: 256          # 批次大小
  buffer_size: 1000000     # 经验回放缓存大小
  learning_starts: 10000   # 开始学习的步数
```

#### 修改环境设置

编辑 `config/environments/airsim.yaml`:
```yaml
episode:
  max_steps: 1000          # 最大步数
  
reward:
  collision_penalty: -100.0  # 碰撞惩罚
  goal_reward: 100.0        # 到达目标奖励
  
sensors:
  camera:
    width: 84              # 图像宽度
    height: 84             # 图像高度
```

### 高级训练命令

```cmd
# 使用自定义配置训练
python experiments\scripts\train.py --algorithm sac --config config\algorithms\sac.yaml

# 指定实验名称
python experiments\scripts\train.py --algorithm ppo --experiment-name ppo_experiment_001

# 从检查点恢复训练
python experiments\scripts\train.py --algorithm sac --resume models\sac_checkpoint.zip

# 训练完成后立即评估
python experiments\scripts\train.py --algorithm ppo --evaluate
```

### 监控训练进度

#### 使用TensorBoard

```cmd
# 启动TensorBoard
tensorboard --logdir experiments\logs

# 在浏览器访问
# http://localhost:6006
```

#### 查看训练输出

训练期间会显示：
```
Episode: 100, Reward: 45.2, Steps: 856, Loss: 0.023
Episode: 101, Reward: 52.1, Steps: 934, Loss: 0.019
...
```

### 评估训练结果

```cmd
# 评估单个模型
python experiments\scripts\evaluate.py --model-path models\sac_final.zip --algorithm sac --num-episodes 10

# 比较多个模型
python experiments\scripts\compare.py --models models\sac.zip models\ppo.zip --algorithms sac ppo --num-episodes 20
```

## 可用环境

### 推荐的AirSim环境

1. **Blocks** - 基础测试环境
   - 简单几何体障碍物
   - 适合算法验证
   - 文件大小: ~200MB

2. **SimpleMaze** - 迷宫环境
   - 复杂迷宫结构
   - 适合导航训练
   - 文件大小: ~150MB

3. **LandscapeMountains** - 户外环境
   - 真实地形和纹理
   - 适合实际应用测试
   - 文件大小: ~2GB

### 环境切换

修改 `config/environments/airsim.yaml` 中的连接参数，或为不同环境创建专门的配置文件。

## 常用配置模板

### 快速原型验证配置

适用于快速测试算法的配置：

```yaml
# config/fast_prototype.yaml
training:
  total_timesteps: 50000    # 减少训练时间
  
sac:
  batch_size: 128           # 减少内存使用
  learning_starts: 1000     # 快速开始学习

episode:
  max_steps: 200            # 缩短episode长度
```

使用方法：
```cmd
python experiments\scripts\train.py --algorithm sac --config config\fast_prototype.yaml
```

### 高性能训练配置

适用于充分训练的配置：

```yaml
# config/high_performance.yaml
training:
  total_timesteps: 2000000  # 充分训练

sac:
  batch_size: 512           # 大批次训练
  buffer_size: 2000000      # 大经验缓存

episode:
  max_steps: 2000           # 长episode
```

### GPU优化配置

如果有NVIDIA GPU：

```cmd
# 安装GPU版PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证GPU可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

在配置文件中启用GPU：
```yaml
training:
  device: "cuda"            # 使用GPU
  batch_size: 512           # GPU可以处理更大批次
```

## 故障排除

### 常见错误及解决方案

**错误**: "ModuleNotFoundError: No module named 'airsim'"
**解决**: 
```cmd
pip install airsim
```

**错误**: "Connection refused to AirSim"
**解决**:
1. 确保AirSim环境已启动
2. 检查防火墙设置
3. 重启AirSim环境

**错误**: "CUDA out of memory"
**解决**:
```yaml
# 减少批次大小
sac:
  batch_size: 64
```

**错误**: 训练不收敛
**解决**:
1. 检查奖励函数设计
2. 调整学习率
3. 增加训练时间
4. 检查环境配置

### 性能优化建议

1. **降低图像分辨率**: 在 `config/environments/airsim.yaml` 中设置更小的图像尺寸
2. **减少episode长度**: 设置合适的 `max_steps`
3. **调整训练频率**: 修改 `train_freq` 参数
4. **使用较简单的环境**: 从Blocks环境开始测试

## 下一步

完成快速开始后，建议：

1. **深入学习**: 阅读 [详细文档](../README.md)
2. **算法对比**: 尝试不同算法并比较性能
3. **环境定制**: 学习如何创建自定义环境
4. **参数调优**: 深入理解各算法的超参数
5. **实际部署**: 将训练好的模型部署到真实无人机

## 获取帮助

- **文档**: 查看 `docs/` 目录下的详细文档
- **示例**: 参考 `examples/` 目录下的示例代码
- **问题反馈**: 在GitHub仓库创建Issue
- **讨论交流**: 参与GitHub Discussions

祝您使用愉快！