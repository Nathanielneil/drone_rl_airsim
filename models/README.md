# 模型管理规范

## 📁 文件夹结构说明

```
models/
├── {algorithm}/              # 算法名称 (sac, ppo, dqn等)
│   ├── production/          # 生产环境模型
│   │   ├── v{X.Y.Z}_{date}_{tag}/  # 版本化模型目录
│   │   │   ├── model.pth           # 训练好的模型权重
│   │   │   ├── config.yaml         # 训练配置
│   │   │   ├── metadata.json       # 模型元信息
│   │   │   └── performance_metrics.json  # 性能指标
│   │   └── latest -> v{X.Y.Z}_{date}_{tag}/  # 指向最新版本
│   ├── development/         # 开发测试模型
│   │   └── exp_{name}_{timestamp}/  # 实验模型目录
│   └── archive/            # 历史模型存档
```

## 🏷️ 命名规范

### Production模型命名
- 格式: `v{major}.{minor}.{patch}_{YYYYMMDD}_{tag}`
- 示例: `v1.0.0_20250920_best`, `v1.1.0_20250925_stable`

### Development模型命名
- 格式: `exp_{experiment_name}_{YYYYMMDDHHmmss}`
- 示例: `exp_test_20250920143022`, `exp_long_training_20250920150000`

### 标签(tag)含义
- `best`: 当前最佳性能模型
- `stable`: 稳定版本模型
- `baseline`: 基线对比模型
- `milestone`: 里程碑版本

## 📋 metadata.json格式

```json
{
  "model_name": "modern_sac_v1.0.0",
  "algorithm": "sac",
  "created_date": "2025-09-20T14:30:22",
  "training_duration": "00:15:30",
  "total_timesteps": 100000,
  "final_reward": -245.67,
  "best_reward": -156.23,
  "episodes_trained": 150,
  "environment": "AirSim-v1.8.1",
  "gpu_used": "RTX 3090",
  "cuda_version": "12.1",
  "tags": ["best", "production"],
  "notes": "First production model with CUDA 12.1 optimization"
}
```

## 🔄 模型生命周期

1. **开发阶段**: 保存在 `development/` 目录
2. **验证通过**: 移动到 `production/` 并版本化
3. **过时淘汰**: 移动到 `archive/` 目录
4. **latest链接**: 始终指向最新的production模型

## 📊 使用示例

```python
# 加载最新生产模型
model_path = "models/sac/production/latest/model.pth"

# 加载特定版本
model_path = "models/sac/production/v1.0.0_20250920_best/model.pth"

# 开发模型
model_path = "models/sac/development/exp_test_20250920143022/model.pth"
```