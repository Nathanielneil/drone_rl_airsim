# 训练数据管理规范

## 📁 数据组织结构

```
data/
├── experiments/                    # 实验数据总目录
│   ├── {experiment_id}/           # 实验唯一标识符
│   │   ├── metadata.json          # 实验元信息
│   │   ├── config/                # 配置文件
│   │   │   ├── experiment_config.yaml
│   │   │   ├── algorithm_config.yaml
│   │   │   └── environment_config.yaml
│   │   ├── logs/                  # 日志数据
│   │   │   ├── tensorboard/       # TensorBoard日志
│   │   │   │   ├── train/
│   │   │   │   ├── eval/
│   │   │   │   └── system/
│   │   │   ├── training.log       # 训练日志
│   │   │   ├── performance.log    # 性能日志
│   │   │   └── error.log          # 错误日志
│   │   ├── checkpoints/           # 检查点数据
│   │   │   ├── step_1000/
│   │   │   ├── step_5000/
│   │   │   ├── best_model/
│   │   │   └── final_model/
│   │   ├── metrics/               # 指标数据
│   │   │   ├── training_metrics.json
│   │   │   ├── evaluation_metrics.json
│   │   │   ├── episode_rewards.csv
│   │   │   ├── loss_curves.csv
│   │   │   └── performance_stats.json
│   │   ├── artifacts/             # 训练产物
│   │   │   ├── videos/           # 训练视频
│   │   │   ├── images/           # 截图记录
│   │   │   ├── trajectories/     # 飞行轨迹
│   │   │   └── analysis/         # 分析报告
│   │   └── reports/               # 生成报告
│   │       ├── training_summary.html
│   │       ├── performance_analysis.pdf
│   │       └── comparison_report.json
│   └── archive/                   # 历史实验存档
├── datasets/                      # 数据集管理
│   ├── airsim_trajectories/       # AirSim轨迹数据
│   ├── expert_demonstrations/     # 专家演示数据
│   └── evaluation_benchmarks/     # 评估基准数据
└── shared/                        # 共享数据
    ├── environment_maps/          # 环境地图
    ├── reference_models/          # 参考模型
    └── analysis_templates/        # 分析模板
```

## 🏷️ 实验标识符规范

### 标准格式
```
{algorithm}_{environment}_{date}_{time}_{tag}
```

### 示例
- `sac_airsim_20250920_143022_baseline`
- `ppo_airsim_20250920_150000_hyperopt`
- `sac_airsim_20250920_160000_production`

### 标签含义
- `baseline`: 基线实验
- `hyperopt`: 超参数优化
- `ablation`: 消融实验
- `comparison`: 对比实验
- `production`: 生产实验
- `debug`: 调试实验

## 📋 实验元数据格式

```json
{
  "experiment_id": "sac_airsim_20250920_143022_baseline",
  "name": "SAC Baseline Training",
  "description": "Initial SAC training with default hyperparameters",
  "created_date": "2025-09-20T14:30:22",
  "status": "completed",
  "duration": "02:15:30",
  "tags": ["baseline", "sac", "airsim"],
  "algorithm": {
    "name": "sac",
    "version": "modern_v1.0"
  },
  "environment": {
    "name": "airsim",
    "version": "1.8.1",
    "settings": "default"
  },
  "hardware": {
    "gpu": "RTX 3090",
    "cuda_version": "12.1",
    "cpu": "Intel i9-12900K",
    "memory": "64GB"
  },
  "hyperparameters": {
    "total_timesteps": 100000,
    "batch_size": 256,
    "learning_rate": 0.0003,
    "buffer_size": 1000000
  },
  "results": {
    "final_reward": -234.56,
    "best_reward": -156.78,
    "convergence_step": 75000,
    "training_stability": "stable"
  },
  "files": {
    "model": "checkpoints/best_model/model.pth",
    "config": "config/experiment_config.yaml",
    "tensorboard": "logs/tensorboard/",
    "metrics": "metrics/training_metrics.json"
  },
  "notes": "First successful training run with new architecture"
}
```

## 📊 指标数据格式

### training_metrics.json
```json
{
  "episode_rewards": [
    {"episode": 1, "reward": -1234.5, "length": 150, "time": "14:30:22"},
    {"episode": 2, "reward": -1100.2, "length": 200, "time": "14:32:15"}
  ],
  "loss_curves": {
    "policy_loss": [0.5, 0.45, 0.42, ...],
    "value_loss": [1.2, 1.1, 1.05, ...],
    "total_loss": [1.7, 1.55, 1.47, ...]
  },
  "performance_metrics": {
    "fps": [3.2, 3.5, 3.4, ...],
    "gpu_utilization": [85, 87, 84, ...],
    "memory_usage": [12.5, 13.1, 12.8, ...]
  }
}
```

## 🔍 TensorBoard日志组织

### 目录结构
```
logs/tensorboard/
├── train/                         # 训练日志
│   ├── scalars/                   # 标量指标
│   │   ├── episode_reward
│   │   ├── policy_loss
│   │   ├── value_loss
│   │   └── learning_rate
│   ├── histograms/                # 分布数据
│   │   ├── policy_weights
│   │   ├── value_weights
│   │   └── gradients
│   └── images/                    # 图像数据
│       ├── environment_state
│       ├── action_visualization
│       └── trajectory_plots
├── eval/                          # 评估日志
│   ├── episode_rewards
│   ├── success_rate
│   └── trajectory_analysis
└── system/                        # 系统监控
    ├── gpu_utilization
    ├── memory_usage
    ├── training_speed
    └── hardware_stats
```

## 🗂️ 数据生命周期管理

### 1. 实验创建阶段
- 生成唯一实验ID
- 创建目录结构
- 保存初始配置
- 记录系统信息

### 2. 训练运行阶段
- 实时记录指标数据
- 定期保存检查点
- 监控系统性能
- 生成可视化数据

### 3. 实验完成阶段
- 保存最终模型
- 生成总结报告
- 分析训练曲线
- 评估模型性能

### 4. 数据归档阶段
- 压缩历史数据
- 移动到存档目录
- 保留重要指标
- 清理临时文件

## 📈 数据分析工具

### TensorBoard启动
```bash
# 查看特定实验
tensorboard --logdir data/experiments/{experiment_id}/logs/tensorboard

# 比较多个实验
tensorboard --logdir data/experiments --reload_multifile=true

# 自定义端口
tensorboard --logdir data/experiments/{experiment_id}/logs/tensorboard --port 6007
```

### 数据导出工具
```python
# 导出训练指标
from src.utils.data_manager import DataManager
dm = DataManager()
metrics = dm.export_metrics("sac_airsim_20250920_143022_baseline")

# 生成对比报告
report = dm.compare_experiments([
    "sac_airsim_20250920_143022_baseline",
    "sac_airsim_20250920_150000_hyperopt"
])
```

## 🔄 自动化管理

### 自动清理策略
- 保留最近30天的开发实验
- 永久保留标记为production的实验
- 自动压缩30天以上的大文件
- 定期清理临时和缓存文件

### 备份策略
- 每日备份重要实验数据
- 实时同步TensorBoard日志
- 定期备份模型检查点
- 云端存储重要结果