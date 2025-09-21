#!/usr/bin/env python3
"""
修复版本：基于目标点的训练脚本（无表情符号版本）
避免相对导入问题，直接实现训练逻辑
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
    print("无人机点到点导航训练 (修复版)")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description='目标导航训练')
    parser.add_argument('--config', default='configs/goal_based_training_config_real_env.yaml', 
                       help='配置文件路径')
    parser.add_argument('--experiment-name', default='goal_navigation_training', 
                       help='实验名称')
    parser.add_argument('--total-timesteps', type=int, default=250000, 
                       help='训练步数')
    parser.add_argument('--log-level', default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    try:
        # 导入必要的模块
        import numpy as np
        import torch
        import yaml
        import json
        import gymnasium as gym
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from torch.utils.tensorboard import SummaryWriter
        
        # 动态导入项目模块
        from src.environments.airsim_env.goal_based_env import GoalBasedAirSimEnv
        from src.utils.data_manager import DataManager
        
        print(f"成功导入所有依赖")
        
        # 加载配置
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"加载配置文件: {config_path}")
        print(f"实验名称: {args.experiment_name}")
        print(f"训练步数: {args.total_timesteps:,}")
        print()
        
        # 显示配置信息
        print("目标导航特性:")
        print("   明确的目标点设置")
        print("   距离奖励和到达奖励")
        print("   多目标序列任务")
        print("   渐进式难度调整")
        print("   目标可视化支持")
        print()
        
        # 创建环境
        print("创建训练环境...")
        env_config = config.get('environment', {})
        env = GoalBasedAirSimEnv(config=env_config)
        
        # 包装为向量化环境
        env = DummyVecEnv([lambda: env])
        print("环境创建成功")
        
        # 设置实验目录
        experiment_dir = Path(f"data/experiments/{args.experiment_name}")
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        model_dir = experiment_dir / "models"
        log_dir = experiment_dir / "logs"
        tensorboard_dir = experiment_dir / "tensorboard"
        
        for dir_path in [model_dir, log_dir, tensorboard_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"实验目录: {experiment_dir}")
        
        # 创建数据管理器
        data_manager = DataManager(base_dir="data")
        
        # 创建SAC算法
        print("创建SAC算法...")
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
        
        print("SAC算法创建成功")
        
        # 创建回调函数
        callbacks = []
        
        # 检查点保存
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=str(model_dir),
            name_prefix="sac_goal_navigation"
        )
        callbacks.append(checkpoint_callback)
        
        # 开始训练
        print("开始训练...")
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
        print("训练完成!")
        print(f"训练时间: {training_time:.2f}秒")
        
        # 保存最终模型
        final_model_path = model_dir / "final_model"
        model.save(str(final_model_path))
        print(f"最终模型已保存: {final_model_path}")
        
        # 保存训练信息
        training_info = {
            "experiment_name": args.experiment_name,
            "config_file": str(config_path),
            "total_timesteps": args.total_timesteps,
            "training_time": training_time,
            "final_model_path": str(final_model_path),
            "tensorboard_log": str(tensorboard_dir)
        }
        
        # 创建实验并保存元数据
        experiment_id = data_manager.create_experiment(
            algorithm="sac",
            environment="airsim_goal_based",
            name=args.experiment_name,
            description="目标导航训练 (修复版)",
            tags=["goal_navigation", "sac", "fixed_version"],
            hyperparameters=training_info
        )
        
        # 保存详细的训练信息
        metadata_file = experiment_dir / "training_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
        
        print()
        print("查看训练结果:")
        print(f"   TensorBoard: tensorboard --logdir {tensorboard_dir}")
        print(f"   模型文件: {model_dir}")
        print(f"   实验数据: {experiment_dir}")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请检查以下项目:")
        print("1. 虚拟环境是否正确激活")
        print("2. 依赖包是否正确安装")
        print("3. AirSim是否正在运行")
        
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        print(f"训练失败: {e}")
        raise

if __name__ == "__main__":
    main()