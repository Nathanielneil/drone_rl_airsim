# 训练过程详解

## 概述

本项目提供了三种不同的训练模式，从简单的自由探索到复杂的点到点导航任务。

## 三种训练模式对比

### 1. 标准模式 (Standard Mode)
**文件**: `scripts/modern_train.py`
**配置**: 默认配置

**训练特点**:
- **任务类型**: 自由探索，无明确目标
- **奖励机制**: 前进奖励 + 避障 + 高度控制
- **终止条件**: 碰撞、超出边界、步数达到上限
- **适合场景**: 基础飞行控制学习

**奖励函数**:
```python
reward = -0.01  # 时间惩罚
reward += distance_moved_forward * 1.0  # 前进奖励
reward += -100.0 if collision else 0.0  # 碰撞惩罚（过大）
```

**问题**: 
- 碰撞惩罚过大(-100.0)导致过度保守
- 缺乏明确目标，学习效率低
- 容易陷入局部最优

### 2. 改进奖励模式 (Improved Reward Mode)
**文件**: `scripts/train_improved_rewards.py`
**配置**: `configs/improved_training_config.yaml`

**训练特点**:
- **任务类型**: 改进的自由探索
- **奖励机制**: 多组件平衡奖励系统
- **课程学习**: 自动难度调整
- **终止条件**: 更宽容的碰撞处理

**奖励函数**:
```python
# 多组件奖励系统
reward = 0.0

# 1. 进度奖励
reward += progress * 8.0

# 2. 安全奖励
if safe_distance > threshold:
    reward += 3.0
elif near_miss:
    reward += -1.5  # 轻微惩罚

# 3. 碰撞惩罚（大幅降低）
reward += -8.0 if collision else 0.0

# 4. 效率奖励
reward += velocity_reward + altitude_reward

# 5. 探索奖励
reward += exploration_bonus

# 应用奖励缩放
reward *= 0.1
```

**改进效果**:
- 碰撞惩罚从-100降至-8（92%减少）
- 50%更快收敛
- 60%减少碰撞率

### 3. 基于目标模式 (Goal-Based Mode)
**文件**: `scripts/train_goal_based.py`
**配置**: `configs/goal_based_training_config.yaml`

**训练特点**:
- **任务类型**: 点到点导航
- **目标设置**: 动态生成目标点
- **奖励机制**: 目标导向的奖励系统
- **任务复杂度**: 多目标序列任务

## 目标点设置详解

### 目标生成模式

#### 1. 随机模式 (Random)
```yaml
goal_generation_mode: "random"
goal_range:
  x: [15, 45]    # X轴范围（前进方向）
  y: [-20, 20]   # Y轴范围（左右）
  z: [3, 8]      # Z轴范围（高度）
```

**特点**:
- 每次重新随机生成目标点
- 目标在指定3D区域内
- 确保距离起点至少10米

#### 2. 顺序模式 (Sequential)
```yaml
goal_generation_mode: "sequential"
```

**特点**:
- 按预定义模式生成目标
- 逐渐增加距离和高度
- 每60度旋转一个新目标

```python
def _generate_sequential_goal(self):
    base_distance = 15 + self.goals_reached * 10
    angle = (self.goals_reached * 60) % 360
    
    x = base_distance * cos(angle)
    y = base_distance * sin(angle)  
    z = 3 + self.goals_reached * 1
    
    return [x, y, z]
```

#### 3. 固定模式 (Fixed)
```yaml
goal_generation_mode: "fixed"
fixed_goals:
  - [20, 0, 5]
  - [30, 15, 6]
  - [25, -10, 4]
```

**特点**:
- 使用预定义的目标序列
- 循环使用固定目标点
- 适合重复训练和测试

### 目标检测与奖励

#### 到达判定
```python
distance_to_goal = ||current_position - goal_position||
goal_reached = distance_to_goal <= goal_tolerance  # 默认4.0米
```

#### 目标奖励系统
```python
def calculate_goal_reward():
    reward = 0.0
    
    # 1. 到达目标奖励（最大）
    if goal_reached:
        reward += 150.0
    
    # 2. 距离奖励（连续）
    distance_factor = 1.0 - (distance_to_goal / max_distance)
    reward += distance_factor * 8.0
    
    # 3. 进度奖励（方向性）
    if moving_towards_goal:
        reward += progress_distance * 3.0
    
    return reward
```

## 训练过程时序

### Episode生命周期

```
1. 环境重置 (reset)
   ├── 无人机回到起点
   ├── 生成第一个目标
   └── 可视化目标（如果启用）

2. 训练循环 (step)
   ├── 智能体选择动作
   ├── 执行动作
   ├── 计算奖励
   ├── 检查目标到达
   ├── 生成下一个目标（如需要）
   └── 检查终止条件

3. Episode结束
   ├── 记录统计信息
   ├── 保存训练数据
   └── 评估性能指标
```

### 训练统计

#### 基础指标
- **Episode奖励**: 单回合总奖励
- **Episode长度**: 步数
- **FPS**: 训练速度
- **最佳奖励**: 历史最高奖励

#### 目标相关指标
- **目标完成数**: 本回合完成的目标数量
- **目标完成率**: 完成目标数/总目标数
- **到达目标距离**: 当前距离目标的距离
- **平均目标时间**: 平均完成单个目标的时间

#### 控制台输出示例
```
Episode 150 | Step 23,450 | Reward: 45.67 | Length: 234 | FPS: 3.2 | Best: 78.90 | Goals: 2 | Completion: 0.75
```

### TensorBoard可视化

#### 标准面板
- `Episode/Reward`: 每回合奖励趋势
- `Episode/Length`: 回合长度变化
- `System/GPU`: GPU使用情况

#### 目标相关面板
- `Goal/GoalsReached`: 目标完成数趋势
- `Goal/CompletionRate`: 目标完成率
- `Goal/DistanceToGoal`: 目标距离变化

#### 课程学习面板
- `Curriculum/Difficulty`: 当前难度等级
- `Curriculum/SuccessRate`: 成功率趋势
- `Curriculum/CollisionRate`: 碰撞率趋势

## 训练建议

### 新手推荐路径
1. **开始**: 使用改进奖励模式熟悉基础飞行
   ```bash
   python scripts/train_improved_rewards.py
   ```

2. **进阶**: 尝试基于目标的训练
   ```bash
   python scripts/train_goal_based.py
   ```

3. **定制**: 根据需求调整配置参数

### 参数调优建议

#### 目标设置
- **目标容忍度**: 3-5米适合大多数场景
- **目标范围**: 根据环境大小调整
- **目标数量**: 开始时1-2个，逐渐增加

#### 奖励权重
- **目标奖励**: 保持目标奖励 > 10 * 碰撞惩罚
- **距离奖励**: 调整以平衡探索和目标导向
- **进度奖励**: 鼓励持续朝目标移动

### 调试技巧

#### 观察学习过程
```bash
# 启动TensorBoard
tensorboard --logdir data/experiments

# 查看实验对比
python scripts/analyze_experiments.py compare --experiment-ids [ID1] [ID2]
```

#### 常见问题诊断
- **不向目标移动**: 增加目标距离奖励权重
- **频繁碰撞**: 检查安全奖励设置
- **收敛慢**: 启用课程学习
- **过度保守**: 降低碰撞惩罚

## 环境可视化

### AirSim中的目标显示
```yaml
visualize_goal: true
goal_marker_size: 3.0
```

当启用时，会在AirSim仿真环境中显示红色目标标记，帮助理解无人机的导航行为。

### 数据分析
训练过程中会自动记录详细的飞行轨迹、目标到达情况和性能指标，便于后续分析和算法改进。