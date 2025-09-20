# 无人机初始位置设置指南

## 概述

本系统提供了多种方式来设置AirSim中四旋翼无人机的初始位置，支持固定位置、随机位置和预设位置。

## 方法1：独立脚本设置

### 基本用法

```bash
# 设置到指定位置
python scripts/set_drone_position.py set --x 10 --y 5 --z -8 --yaw 45

# 设置随机位置
python scripts/set_drone_position.py random --x-range -20 20 --y-range -20 20

# 获取当前位置
python scripts/set_drone_position.py get

# 使用预设位置
python scripts/set_drone_position.py preset training_start
```

### 高级功能

#### 创建位置预设
```bash
# 创建示例预设文件
python scripts/set_drone_position.py create-examples

# 查看预设文件
ls configs/position_presets/
```

#### 自定义预设
编辑 `configs/position_presets/training_positions.yaml`:
```yaml
my_custom_position:
  description: "我的自定义位置"
  position:
    vehicle: "Drone1"
    x: 25.0
    y: 10.0
    z: -6.0
    yaw: 90.0
```

## 方法2：训练环境集成

### 在配置文件中设置

#### 固定初始位置
```yaml
# configs/my_training_config.yaml
environment:
  # 固定起始位置
  initial_position:
    x: 15.0      # 前进15米
    y: -5.0      # 左移5米
    z: -8.0      # 高度8米
    yaw: 30.0    # 朝向30度
```

#### 随机初始位置
```yaml
environment:
  # 随机起始位置
  random_position:
    enabled: true
    x_range: [5, 25]       # X轴范围
    y_range: [-15, 15]     # Y轴范围
    z_range: [-12, -4]     # Z轴范围
    yaw_range: [0, 360]    # 朝向范围
```

### 运行时设置

```python
# 在reset时传递位置参数
import numpy as np

# 方法1: 通过options参数
position_options = {
    "initial_position": {
        "x": 20.0, "y": 0.0, "z": -5.0, "yaw": 0.0
    }
}
obs, info = env.reset(options=position_options)

# 方法2: 随机位置
random_options = {
    "random_position": {
        "enabled": True,
        "x_range": [-30, 30],
        "y_range": [-30, 30],
        "z_range": [-10, -3],
        "yaw_range": [0, 360]
    }
}
obs, info = env.reset(options=random_options)
```

## 方法3：程序化位置控制

### 直接API调用
```python
import airsim
import numpy as np

# 连接AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

def set_drone_position(x, y, z, yaw_degrees=0):
    """设置无人机位置"""
    # 创建位置和方向
    position = airsim.Vector3r(float(x), float(y), float(z))
    orientation = airsim.to_quaternion(0, 0, np.radians(yaw_degrees))
    pose = airsim.Pose(position, orientation)
    
    # 设置位置
    client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name="Drone1")
    
    # 重置物理状态
    client.reset()
    client.enableApiControl(True, vehicle_name="Drone1")
    client.armDisarm(True, vehicle_name="Drone1")

# 使用示例
set_drone_position(15, 10, -6, 45)
```

### 批量设置多架无人机
```python
def set_multiple_drones(positions):
    """设置多架无人机位置"""
    for pos in positions:
        vehicle = pos["vehicle"]
        x, y, z, yaw = pos["x"], pos["y"], pos["z"], pos["yaw"]
        
        position = airsim.Vector3r(float(x), float(y), float(z))
        orientation = airsim.to_quaternion(0, 0, np.radians(yaw))
        pose = airsim.Pose(position, orientation)
        
        client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=vehicle)

# 编队设置示例
formation = [
    {"vehicle": "Drone1", "x": 0, "y": 0, "z": -5, "yaw": 0},
    {"vehicle": "Drone2", "x": 10, "y": 0, "z": -5, "yaw": 0},
    {"vehicle": "Drone3", "x": 5, "y": 8.66, "z": -5, "yaw": 0}
]
set_multiple_drones(formation)
```

## 实际使用场景

### 1. 训练不同起点的策略
```bash
# 从原点开始训练
python scripts/train_goal_based.py

# 从自定义位置开始训练
python scripts/set_drone_position.py set --x 20 --y 10 --z -8
python scripts/train_goal_based.py
```

### 2. 测试环境鲁棒性
```bash
# 循环测试不同起点
for i in {1..10}; do
    python scripts/set_drone_position.py random
    python scripts/test_trained_model.py --episodes 5
done
```

### 3. 特定场景训练
```yaml
# 高空训练配置
environment:
  initial_position:
    x: 0.0
    y: 0.0
    z: -20.0    # 高空起点
    yaw: 0.0
  goal_range:
    x: [10, 50]
    y: [-30, 30]
    z: [15, 25]   # 高空目标
```

## 坐标系说明

### AirSim坐标系
- **X轴**: 前进方向（正值向前）
- **Y轴**: 右侧方向（正值向右）
- **Z轴**: 向下方向（负值表示空中高度）
- **Yaw**: 朝向角度（0度朝向X轴正方向，逆时针为正）

### 高度设置注意事项
```python
# 正确的高度设置
z = -5.0    # 表示距地面5米高
z = -10.0   # 表示距地面10米高

# 错误的设置
z = 5.0     # 表示地下5米（会发生碰撞）
```

## 预设位置库

### 内置预设
```bash
# 查看所有预设
python scripts/set_drone_position.py preset --list

# 常用预设
python scripts/set_drone_position.py preset basic_training      # 基础训练
python scripts/set_drone_position.py preset goal_navigation     # 目标导航
python scripts/set_drone_position.py preset obstacle_course     # 障碍训练
python scripts/set_drone_position.py preset high_altitude       # 高空训练
```

### 创建自定义预设
```yaml
# configs/position_presets/my_positions.yaml
my_scenarios:
  indoor_test:
    description: "室内测试环境"
    position:
      vehicle: "Drone1"
      x: 5.0
      y: 0.0
      z: -3.0
      yaw: 0.0
  
  outdoor_exploration:
    description: "户外探索环境"
    position:
      vehicle: "Drone1"
      x: 50.0
      y: 25.0
      z: -15.0
      yaw: 180.0
```

## 故障排除

### 常见问题
1. **位置设置无效**: 检查AirSim连接状态
2. **无人机卡在地面**: 确保Z值为负数
3. **朝向不正确**: 检查yaw角度单位（度vs弧度）

### 调试方法
```bash
# 检查当前位置
python scripts/set_drone_position.py get

# 验证连接
python -c "import airsim; client = airsim.MultirotorClient(); client.confirmConnection(); print('连接成功')"
```

通过这些方法，可以灵活地控制无人机的初始位置，满足各种训练和测试需求。