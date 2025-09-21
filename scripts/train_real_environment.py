#!/usr/bin/env python3
"""
真实仿真环境训练脚本
基于实际环境参数：X轴±10米，Y轴±10米，Z轴0.4-3米，含障碍物
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
    print("真实仿真环境训练")
    print("=" * 60)
    print("环境参数: X±10米, Y±10米, Z:0.4-3米, 含障碍物")
    print("空间规模: 20m×20m×2.6m = 1,040立方米")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description='真实仿真环境训练')
    parser.add_argument('--mode', choices=['improved', 'goal_based'], default='improved',
                       help='训练模式: improved(改进奖励) 或 goal_based(目标导航)')
    parser.add_argument('--experiment-name', default=None, 
                       help='实验名称 (默认自动生成)')
    parser.add_argument('--total-timesteps', type=int, default=None, 
                       help='训练步数 (默认根据模式选择)')
    parser.add_argument('--log-level', default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    # 根据模式设置默认参数
    if args.mode == 'improved':
        default_config = 'configs/improved_training_config_real_env.yaml'
        default_experiment = 'improved_rewards_real_env'
        default_timesteps = 150000
        mode_desc = "改进奖励系统"
    else:  # goal_based
        default_config = 'configs/goal_based_training_config_real_env.yaml'
        default_experiment = 'goal_based_real_env'
        default_timesteps = 250000
        mode_desc = "目标导航系统"
    
    config_file = default_config
    experiment_name = args.experiment_name or default_experiment
    total_timesteps = args.total_timesteps or default_timesteps
    
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
        if args.mode == 'improved':
            from src.environments.airsim_env.improved_reward_env import ImprovedRewardAirSimEnv as EnvClass
        else:
            from src.environments.airsim_env.goal_based_env import GoalBasedAirSimEnv as EnvClass
        
        from src.utils.data_manager import DataManager
        
        print(f"成功导入所有依赖")
        print(f"训练模式: {mode_desc}")
        
        # 加载配置
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"加载真实环境配置: {config_path}")
        print(f"实验名称: {experiment_name}")
        print(f"训练步数: {total_timesteps:,}")
        print()
        
        # 显示真实环境参数
        print("真实仿真环境参数:")
        env_config = config.get('environment', {})
        
        # 显示空间范围
        if args.mode == 'goal_based':
            goal_range = env_config.get('goal_range', {})
            print(f"   目标生成范围:")
            print(f"     X轴: {goal_range.get('x', [])} cm → {[x/100 for x in goal_range.get('x', [])]} 米")
            print(f"     Y轴: {goal_range.get('y', [])} cm → {[y/100 for y in goal_range.get('y', [])]} 米")
            print(f"     Z轴: {goal_range.get('z', [])} cm → {[z/100 for z in goal_range.get('z', [])]} 米")
            print(f"   总空间: 20m × 20m × 2.6m = 1,040立方米")
        
        # 显示关键参数
        print(f"   飞行参数:")
        print(f"     最大速度: {env_config.get('max_velocity', 0)} cm/s → {env_config.get('max_velocity', 0)/100} 米/秒")
        print(f"     起飞高度: {env_config.get('takeoff_height', 0)} cm → {env_config.get('takeoff_height', 0)/100} 米")
        print(f"     高度范围: {env_config.get('min_altitude', 0)}-{env_config.get('max_altitude', 0)} cm")
        print(f"     → {env_config.get('min_altitude', 0)/100}-{env_config.get('max_altitude', 0)/100} 米")
        
        if args.mode == 'goal_based':
            print(f"   目标参数:")
            print(f"     目标容忍度: {env_config.get('goal_tolerance', 0)} cm → {env_config.get('goal_tolerance', 0)/100} 米")
            print(f"     安全距离: {env_config.get('min_clearance', 0)} cm → {env_config.get('min_clearance', 0)/100} 米")
        
        print(f"   环境特点:")
        print(f"     紧凑空间 (20m×20m×2.6m)")
        print(f"     包含障碍物")
        print(f"     需要精确控制")
        print(f"     高度重视安全")
        print()
        
        # 创建环境
        print(f"创建真实环境训练环境 ({mode_desc})...")
        env = EnvClass(config=env_config)
        
        # 包装为向量化环境
        env = DummyVecEnv([lambda: env])
        print("环境创建成功")
        
        # 设置实验目录
        experiment_dir = Path(f"data/experiments/{experiment_name}")
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
        print("创建SAC算法 (真实环境优化)...")
        algorithm_config = config.get('algorithm', {})
        training_config = config.get('training', {})
        
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=training_config.get('learning_rate', 3e-4),
            buffer_size=algorithm_config.get('buffer_size', 80000),
            batch_size=training_config.get('batch_size', 128),
            tau=algorithm_config.get('tau', 0.005),
            gamma=training_config.get('gamma', 0.99),
            train_freq=algorithm_config.get('train_freq', 1),
            gradient_steps=algorithm_config.get('gradient_steps', 1),
            learning_starts=algorithm_config.get('learning_starts', 800),
            tensorboard_log=str(tensorboard_dir),
            verbose=1,
            seed=training_config.get('seed', 42)
        )
        
        print("SAC算法创建成功")
        
        # 创建回调函数
        callbacks = []
        
        # 检查点保存 (更频繁，因为训练步数较少)
        checkpoint_callback = CheckpointCallback(
            save_freq=3000,
            save_path=str(model_dir),
            name_prefix=f"sac_{args.mode}_real_env"
        )
        callbacks.append(checkpoint_callback)
        
        # 开始训练
        print("开始真实环境训练...")
        print("-" * 60)
        print("训练策略针对真实环境优化:")
        print("   较低飞行速度 (2.5-3米/秒)")
        print("   较小目标容忍度 (0.8米)")
        print("   增强安全奖励权重")
        print("   启用智能障碍物避让")
        print("   更精细的验证检查")
        print("-" * 60)
        
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=5  # 更频繁的日志输出
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print()
        print("真实环境训练完成!")
        print(f"训练时间: {training_time:.2f}秒")
        
        # 保存最终模型
        final_model_path = model_dir / f"final_model_{args.mode}_real_env"
        model.save(str(final_model_path))
        print(f"最终模型已保存: {final_model_path}")
        
        # 保存训练信息
        training_info = {
            "experiment_name": experiment_name,
            "training_mode": args.mode,
            "config_file": str(config_path),
            "total_timesteps": total_timesteps,
            "training_time": training_time,
            "final_model_path": str(final_model_path),
            "tensorboard_log": str(tensorboard_dir),
            "environment_type": "real_simulation_environment",
            "environment_specs": {
                "x_range_meters": [-10, 10],
                "y_range_meters": [-10, 10],
                "z_range_meters": [0.4, 3.0],
                "total_volume_m3": 1040,
                "has_obstacles": True,
                "space_type": "compact"
            }
        }
        
        if args.mode == 'goal_based':
            goal_range = env_config.get('goal_range', {})
            training_info["goal_range_cm"] = goal_range
            training_info["goal_range_meters"] = {
                "x": [x/100 for x in goal_range.get('x', [])],
                "y": [y/100 for y in goal_range.get('y', [])],
                "z": [z/100 for z in goal_range.get('z', [])]
            }
        
        # 创建实验并保存元数据
        experiment_id = data_manager.create_experiment(
            algorithm="sac",
            environment="airsim_goal_based" if args.mode == 'goal_based' else "airsim_improved",
            name=experiment_name,
            description=f"真实环境训练 - {mode_desc}",
            tags=["real_environment", args.mode, "obstacle_avoidance"],
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
        print()
        print("真实环境训练成功完成!")
        print("模型已针对20m×20m×2.6m的紧凑障碍物环境优化")
        
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