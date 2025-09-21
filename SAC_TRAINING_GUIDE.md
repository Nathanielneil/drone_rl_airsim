# SAC系列算法运行指南

## 🚀 SAC系列算法运行完整指南

### 🎯 SAC系列算法概览

**SAC (Soft Actor-Critic)** 是当前项目中最强大、最稳定的算法，提供三种训练模式：

1. **改进奖励SAC** - 新手推荐，平衡奖励设计
2. **目标导航SAC** - 高级功能，点到点导航
3. **标准SAC** - 基础版本，通用训练

## 📋 **运行前准备**

### 1. 环境检查
```bash
# 确保在正确的虚拟环境
conda activate rlsim

# 检查依赖
python scripts/test_goal_env.py
```

### 2. AirSim启动
```bash
# 启动AirSim仿真环境
# 双击运行你的UE4 AirSim项目
# 或使用命令行启动
```

### 3. 进入项目目录
```bash
cd D:\code\drone_rl_airsim
```

## 🎯 **方式一：改进奖励SAC训练** (推荐新手)

### 快速启动
```bash
python scripts/train_improved_rewards.py
```

### 详细配置运行
```bash
python scripts/train_improved_rewards.py \
    --experiment-name "my_improved_sac_training" \
    --total-timesteps 200000 \
    --log-level INFO
```

### 特性说明
- ✅ **平衡奖励设计**: 碰撞惩罚从-100降至-8
- ✅ **课程学习**: 自动难度调整
- ✅ **多组件奖励**: 进度+安全+效率+探索
- ✅ **快速收敛**: 相比标准版本提升50%收敛速度

### 配置文件
使用的配置: `configs/improved_training_config.yaml`

重要参数:
```yaml
algorithm:
  algorithm_name: "sac"
  buffer_size: 150000
  batch_size_gpu: 512
  
training:
  total_timesteps: 200000
  learning_rate: 0.0003
  
environment:
  collision_penalty: -8.0      # 改进：大幅降低碰撞惩罚
  progress_reward_scale: 8.0   # 增强前进奖励
  curriculum_enabled: true     # 启用课程学习
```

## 🎯 **方式二：目标导航SAC训练** (推荐高级用户)

### 快速启动
```bash
python scripts/train_goal_based_fixed.py
```

### 详细配置运行
```bash
python scripts/train_goal_based_fixed.py \
    --config configs/goal_based_training_config.yaml \
    --experiment-name "goal_navigation_training" \
    --total-timesteps 300000
```

### 特性说明
- ✅ **点到点导航**: 学习飞向指定目标点
- ✅ **智能目标验证**: 确保目标不与障碍物冲突
- ✅ **多目标任务**: 每episode完成多个目标
- ✅ **路径优化**: 学习最优飞行路径

### 目标空间配置
```yaml
environment:
  goal_range:
    x: [15, 45]    # 前方15-45米
    y: [-20, 20]   # 左右±20米
    z: [3, 8]      # 高度3-8米
  
  goal_tolerance: 4.0          # 到达阈值4米
  max_goals_per_episode: 2     # 每回合2个目标
  enable_goal_validation: true # 启用智能验证
```

## 🎯 **方式三：标准SAC训练** (通用版本)

### 运行方式
```bash
python scripts/modern_train.py --algorithm sac
```

### 自定义配置
```bash
python scripts/modern_train.py \
    --algorithm sac \
    --config configs/custom_config.yaml \
    --experiment-name "standard_sac_training" \
    --total-timesteps 150000
```

## 📊 **训练参数详解**

### 核心SAC参数
```yaml
algorithm:
  algorithm_name: "sac"
  
  # 缓冲区配置
  buffer_size: 150000          # CPU版本
  buffer_size_gpu: 300000      # GPU版本
  
  # 训练频率
  batch_size: 256              # CPU批次大小
  batch_size_gpu: 512          # GPU批次大小
  train_freq: 1                # 每步都训练
  gradient_steps: 1            # 梯度步数
  
  # SAC特有参数
  tau: 0.005                   # 软更新系数
  ent_coef: "auto"            # 自动熵系数调节
  target_entropy: "auto"       # 自动目标熵
  learning_starts: 2000        # 开始学习的步数
```

### 训练配置
```yaml
training:
  total_timesteps: 200000      # 总训练步数
  learning_rate: 0.0003        # 学习率
  gamma: 0.99                  # 折扣因子
  seed: 42                     # 随机种子
  device: "auto"               # 自动选择GPU/CPU
```

### GPU优化配置
```yaml
gpu:
  mixed_precision: true        # 混合精度训练
  enable_tf32: true           # TF32加速
  enable_cudnn_benchmark: true # CUDNN基准测试
  memory_fraction: 0.8        # GPU内存使用比例
```

## 🔧 **高级运行选项**

### 1. 自定义初始位置训练
```bash
# 先设置无人机位置
python scripts/set_drone_position.py set --x 20 --y 10 --z -8 --yaw 45

# 然后开始训练
python scripts/train_goal_based_fixed.py
```

### 2. 使用位置预设训练
```bash
# 设置到训练起点
python scripts/set_drone_position.py preset basic_training

# 开始训练
python scripts/train_improved_rewards.py
```

### 3. 随机初始位置训练
```bash
# 配置文件中启用随机位置
# configs/custom_config.yaml
environment:
  random_position:
    enabled: true
    x_range: [10, 30]
    y_range: [-15, 15]
    z_range: [-10, -3]
```

## 📈 **训练监控和分析**

### 1. 实时监控 - TensorBoard
```bash
# 新开终端窗口
tensorboard --logdir data/experiments

# 浏览器访问
http://localhost:6006
```

### 2. 控制台输出监控
训练过程中会显示：
```
Episode 150 | Step 23,450 | Reward: 45.67 | Length: 234 | FPS: 3.2 | Best: 78.90
Goals: 2 | Completion: 0.75 | Distance: 12.3m | Collision: 0.05
```

### 3. 训练数据分析
```bash
# 分析训练结果
python scripts/analyze_experiments.py list

# 详细分析特定实验
python scripts/analyze_experiments.py info --experiment-id goal_navigation_training

# 对比不同实验
python scripts/analyze_experiments.py compare --experiment-ids exp1 exp2
```

## 📁 **训练结果管理**

### 自动生成的文件结构
```
data/experiments/[实验名称]/
├── models/                  # 模型检查点
│   ├── sac_*.zip           # 定期保存的模型
│   └── final_model.zip     # 最终模型
├── logs/                   # 训练日志
│   └── training.log
├── tensorboard/            # TensorBoard数据
└── metadata.json          # 实验元数据
```

### 模型保存设置
```yaml
experiment:
  checkpoint_frequency: 5000   # 每5000步保存一次
  save_best_model: true       # 保存最佳模型
  
logging:
  log_interval: 1000          # 每1000步记录一次
  eval_freq: 15000           # 每15000步评估一次
  n_eval_episodes: 5         # 评估时运行5个回合
```

## ⚡ **性能优化建议**

### 1. GPU优化
```yaml
# 如果有高端GPU，可以增大批次大小
algorithm:
  batch_size_gpu: 1024        # RTX 3090/4090
  buffer_size_gpu: 500000     # 大内存GPU

gpu:
  memory_fraction: 0.9        # 使用更多GPU内存
```

### 2. CPU优化
```yaml
# 如果使用CPU训练
algorithm:
  batch_size: 128             # 较小批次
  buffer_size: 100000         # 较小缓冲区
```

### 3. 快速测试配置
```yaml
training:
  total_timesteps: 50000      # 减少训练步数
environment:
  max_episode_steps: 1000     # 减少episode长度
```

## 🔍 **故障排除**

### 常见问题及解决方案

#### 1. 导入错误
```bash
# 运行测试脚本诊断
python scripts/test_goal_env.py

# 如果仍有问题，使用修复版本
python scripts/train_goal_based_fixed.py
```

#### 2. AirSim连接失败
```bash
# 检查AirSim是否运行
# 检查端口设置 (默认41451)
# 检查防火墙设置
```

#### 3. GPU内存不足
```yaml
# 减少批次大小
algorithm:
  batch_size_gpu: 256         # 从512降到256
gpu:
  memory_fraction: 0.6        # 减少GPU内存使用
```

#### 4. 训练不收敛
```yaml
# 调整学习率
training:
  learning_rate: 0.0001       # 降低学习率

# 增加探索
algorithm:
  learning_starts: 5000       # 延迟学习开始
```

## 🎉 **成功训练的标志**

### 好的训练指标
- **奖励趋势**: 逐渐上升并趋于稳定
- **Episode长度**: 逐渐增加
- **碰撞率**: 逐渐降低
- **目标完成率**: 逐渐提高 (目标导航模式)
- **FPS**: 保持稳定 (>2.0)

### 预期训练时间
- **改进奖励SAC**: 2-4小时 (200k步)
- **目标导航SAC**: 3-6小时 (300k步)
- **标准SAC**: 2-3小时 (150k步)

*时间取决于硬件配置和环境复杂度*

## 🚀 **快速开始命令**

```bash
# 最简单的开始方式
python scripts/train_improved_rewards.py

# 高级功能
python scripts/train_goal_based_fixed.py

# 环境测试
python scripts/test_goal_env.py

# 位置设置
python scripts/set_drone_position.py preset basic_training
```

通过这些详细的指南，你可以充分利用SAC系列算法的强大功能，实现高效的无人机强化学习训练！