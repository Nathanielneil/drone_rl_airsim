#!/usr/bin/env python3
"""
UE4厘米单位修正版：基于目标点的训练脚本
使用正确的UE4距离单位（厘米）进行训练
"""

import sys
import os
import logging
import time
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    print("🚁 目标导航训练 (UE4厘米单位修正版)")
    print("=" * 60)
    print("⚠️  重要：此版本使用正确的UE4距离单位（厘米）")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description='UE4单位修正版目标导航训练')
    parser.add_argument('--config', default='configs/goal_based_training_config_ue4_units.yaml', 
                       help='配置文件路径')
    parser.add_argument('--experiment-name', default='goal_navigation_ue4_units', 
                       help='实验名称')
    parser.add_argument('--total-timesteps', type=int, default=300000, 
                       help='训练步数')
    parser.add_argument('--log-level', default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    try:
        # 导入必要的模块
        import numpy as np
        import torch
        import yaml
        import gymnasium as gym
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from torch.utils.tensorboard import SummaryWriter
        
        # 动态导入项目模块
        from src.environments.airsim_env.goal_based_env import GoalBasedAirSimEnv
        from src.utils.data_manager import DataManager
        
        print(f"✅ 成功导入所有依赖")
        
        # 加载配置
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ 加载UE4单位修正配置: {config_path}")
        print(f"📝 实验名称: {args.experiment_name}")
        print(f"🎯 训练步数: {args.total_timesteps:,}")
        print()
        
        # 显示单位修正信息
        print("📏 UE4距离单位修正说明:")
        env_config = config.get('environment', {})
        goal_range = env_config.get('goal_range', {})
        
        print(f"   目标范围 (厘米):")
        print(f"     X: {goal_range.get('x', [])} cm → {[x/100 for x in goal_range.get('x', [])]} 米")
        print(f"     Y: {goal_range.get('y', [])} cm → {[y/100 for y in goal_range.get('y', [])]} 米") 
        print(f"     Z: {goal_range.get('z', [])} cm → {[z/100 for z in goal_range.get('z', [])]} 米")
        
        print(f"   关键参数 (厘米):")
        print(f"     目标容忍度: {env_config.get('goal_tolerance', 0)} cm → {env_config.get('goal_tolerance', 0)/100} 米")
        print(f"     安全距离: {env_config.get('min_clearance', 0)} cm → {env_config.get('min_clearance', 0)/100} 米")
        print(f"     最大速度: {env_config.get('max_velocity', 0)} cm/s → {env_config.get('max_velocity', 0)/100} 米/秒")
        print()
        
        # 创建环境
        print("🌍 创建UE4单位修正训练环境...")
        env = GoalBasedAirSimEnv(config=env_config)
        
        # 包装为向量化环境
        env = DummyVecEnv([lambda: env])
        print("✅ 环境创建成功")
        
        # 设置实验目录
        experiment_dir = Path(f"data/experiments/{args.experiment_name}")
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        model_dir = experiment_dir / "models"
        log_dir = experiment_dir / "logs"
        tensorboard_dir = experiment_dir / "tensorboard"
        
        for dir_path in [model_dir, log_dir, tensorboard_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"📁 实验目录: {experiment_dir}")
        
        # 创建数据管理器
        data_manager = DataManager(
            experiment_name=args.experiment_name,
            save_dir=str(experiment_dir)
        )
        
        # 创建SAC算法
        print("🧠 创建SAC算法 (UE4单位优化)...")
        algorithm_config = config.get('algorithm', {})
        training_config = config.get('training', {})
        
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=training_config.get('learning_rate', 3e-4),
            buffer_size=algorithm_config.get('buffer_size', 100000),
            batch_size=training_config.get('batch_size', 256),
            tau=algorithm_config.get('tau', 0.005),
            gamma=training_config.get('gamma', 0.99),
            train_freq=algorithm_config.get('train_freq', 1),
            gradient_steps=algorithm_config.get('gradient_steps', 1),
            learning_starts=algorithm_config.get('learning_starts', 1000),
            tensorboard_log=str(tensorboard_dir),
            verbose=1,
            seed=training_config.get('seed', 42)
        )
        
        print("✅ SAC算法创建成功")
        
        # 创建回调函数
        callbacks = []
        
        # 检查点保存
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=str(model_dir),
            name_prefix="sac_goal_navigation_ue4_units"
        )
        callbacks.append(checkpoint_callback)
        
        # 开始训练
        print("🚀 开始训练 (使用UE4厘米单位)...")
        print("-" * 60)
        print("💡 提示: 现在的目标距离是合理的米级范围")
        print("   - 目标范围: 15-45米 (而不是15-45厘米)")
        print("   - 安全距离: 3米 (而不是3厘米)")
        print("   - 飞行速度: 8米/秒 (而不是8厘米/秒)")
        print("-" * 60)
        
        start_time = time.time()
        
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=10
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print()
        print("🎉 训练完成!")
        print(f"⏱️  训练时间: {training_time:.2f}秒")
        
        # 保存最终模型
        final_model_path = model_dir / "final_model_ue4_units"
        model.save(str(final_model_path))
        print(f"💾 最终模型已保存: {final_model_path}")
        
        # 保存训练信息
        training_info = {
            "experiment_name": args.experiment_name,
            "config_file": str(config_path),
            "total_timesteps": args.total_timesteps,
            "training_time": training_time,
            "final_model_path": str(final_model_path),
            "tensorboard_log": str(tensorboard_dir),
            "units_correction": "UE4 centimeters used",
            "goal_range_cm": goal_range,
            "goal_range_meters": {
                "x": [x/100 for x in goal_range.get('x', [])],
                "y": [y/100 for y in goal_range.get('y', [])],
                "z": [z/100 for z in goal_range.get('z', [])]
            }
        }
        
        data_manager.save_experiment_metadata(training_info)
        
        print()
        print("📊 查看训练结果:")
        print(f"   TensorBoard: tensorboard --logdir {tensorboard_dir}")
        print(f"   模型文件: {model_dir}")
        print(f"   实验数据: {experiment_dir}")
        print()
        print("✅ UE4单位修正版训练完成!")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请检查以下项目:")
        print("1. 虚拟环境是否正确激活")
        print("2. 依赖包是否正确安装")
        print("3. AirSim是否正在运行")
        
    except FileNotFoundError as e:
        print(f"❌ 文件错误: {e}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        print(f"❌ 训练失败: {e}")
        raise

if __name__ == "__main__":
    main()