#!/usr/bin/env python3
"""
现代化训练脚本
针对 Windows 10 + AirSim 1.8.1 + UE4.7.2 + CUDA 12.1 优化
支持混合精度训练、性能监控、自动优化
"""

import os
import sys
import argparse
import logging
import time
import signal
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

# 项目导入
from core.config_manager import ConfigManager, create_config_from_args, ModernConfig
from environments.airsim_env.modern_airsim_env import ModernAirSimEnv
from environments.airsim_env.improved_reward_env import ImprovedRewardAirSimEnv
from algorithms.actor_critic.sac.modern_sac import ModernSAC
from utils.performance.gpu_manager import (
    PerformanceMonitor, 
    optimize_for_training,
    get_performance_monitor
)
from utils.model_manager import ModelManager
from utils.data_manager import DataManager
from utils.curriculum_manager import CurriculumManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class ModernTrainer:
    """现代化训练器"""
    
    def __init__(self, config: ModernConfig):
        self.config = config
        self.start_time = time.time()
        
        # 设置设备
        self.device = self._setup_device()
        
        # 设置随机种子
        self._setup_seed()
        
        # 创建环境
        self.env = self._create_environment()
        
        # 创建算法
        self.algorithm = self._create_algorithm()
        
        # 性能监控
        self.performance_monitor = get_performance_monitor()
        self.performance_monitor.start_monitoring()
        
        # 模型管理器
        self.model_manager = ModelManager()
        
        # 数据管理器 - 创建实验
        self.data_manager = DataManager()
        
        # 检查是否使用改进奖励和课程学习
        use_curriculum = getattr(self.config.environment, 'curriculum_enabled', False)
        experiment_tags = [self.config.algorithm.algorithm_name, "airsim", "cuda121", "modern"]
        if use_curriculum:
            experiment_tags.append("curriculum_learning")
            experiment_tags.append("improved_rewards")
        
        self.experiment_id = self.data_manager.create_experiment(
            algorithm=self.config.algorithm.algorithm_name,
            environment="airsim",
            name=f"{self.config.algorithm.algorithm_name.upper()} Training - {self.config.experiment.experiment_name}",
            description=f"Modern {self.config.algorithm.algorithm_name.upper()} training on AirSim 1.8.1 with CUDA 12.1",
            tags=experiment_tags,
            hyperparameters={
                "total_timesteps": self.config.training.total_timesteps,
                "batch_size": self.config.training.batch_size,
                "learning_rate": self.config.training.learning_rate,
                "gamma": self.config.training.gamma,
                "buffer_size": getattr(self.config.algorithm, 'buffer_size', 0),
                "tau": getattr(self.config.algorithm, 'tau', 0.0),
                "curriculum_enabled": use_curriculum
            }
        )
        logger.info(f"实验已创建: {self.experiment_id}")
        
        # 课程学习管理器
        self.curriculum_manager = None
        if use_curriculum and hasattr(self.env, 'get_reward_info'):
            self.curriculum_manager = CurriculumManager(self.env, self.config.environment.__dict__)
            logger.info("课程学习管理器已启用")
        
        # 设置日志记录 - 现在可以使用数据管理器路径
        self.writer = self._setup_tensorboard()
        
        # 训练统计
        self.total_timesteps = 0
        self.episode_count = 0
        self.best_reward = float('-inf')
        
        # 优雅关闭处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("现代化训练器初始化完成")
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        if self.config.training.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.training.device)
        
        logger.info(f"使用设备: {device}")
        
        if device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # GPU优化设置
            if self.config.gpu.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            if self.config.gpu.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
        
        return device
    
    def _setup_seed(self):
        """设置随机种子"""
        if self.config.training.seed is not None:
            seed = self.config.training.seed
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # 确保确定性（可能影响性能）
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            logger.info(f"设置随机种子: {seed}")
    
    def _create_environment(self) -> gym.Env:
        """创建环境"""
        # 基础环境配置
        env_config = {
            "host": self.config.environment.host,
            "port": self.config.environment.port,
            "vehicle_name": self.config.environment.vehicle_name,
            "camera_name": self.config.environment.camera_name,
            "image_type": self.config.environment.image_type,
            "image_width": self.config.environment.image_width,
            "image_height": self.config.environment.image_height,
            "max_episode_steps": self.config.environment.max_episode_steps,
            "action_space_type": self.config.environment.action_space_type,
            "max_velocity": self.config.environment.max_velocity,
            "max_altitude": self.config.environment.max_altitude,
            "min_altitude": self.config.environment.min_altitude,
            "takeoff_height": self.config.environment.takeoff_height,
            "use_gpu_processing": self.device.type == "cuda",
        }
        
        # 检查是否使用改进的奖励函数
        use_improved_rewards = getattr(self.config.environment, 'reward_scale', None) is not None
        
        if use_improved_rewards:
            # 添加改进奖励函数的配置
            env_config.update({
                "reward_scale": getattr(self.config.environment, 'reward_scale', 0.1),
                "progress_reward_scale": getattr(self.config.environment, 'progress_reward_scale', 10.0),
                "goal_reward": getattr(self.config.environment, 'goal_reward', 50.0),
                "safe_distance_threshold": getattr(self.config.environment, 'safe_distance_threshold', 3.0),
                "safety_reward_scale": getattr(self.config.environment, 'safety_reward_scale', 2.0),
                "near_miss_penalty": getattr(self.config.environment, 'near_miss_penalty', -2.0),
                "collision_penalty": getattr(self.config.environment, 'collision_penalty', -10.0),
                "time_penalty": getattr(self.config.environment, 'time_penalty', -0.02),
                "velocity_reward_scale": getattr(self.config.environment, 'velocity_reward_scale', 1.0),
                "altitude_reward_scale": getattr(self.config.environment, 'altitude_reward_scale', 0.5),
                "exploration_reward_scale": getattr(self.config.environment, 'exploration_reward_scale', 0.5),
                "hover_penalty": getattr(self.config.environment, 'hover_penalty', -0.5),
                "curriculum_enabled": getattr(self.config.environment, 'curriculum_enabled', True),
                "difficulty_level": getattr(self.config.environment, 'difficulty_level', 1),
                "collision_forgiveness": getattr(self.config.environment, 'collision_forgiveness', 3),
                "distance_shaping": getattr(self.config.environment, 'distance_shaping', True),
                "velocity_shaping": getattr(self.config.environment, 'velocity_shaping', True),
            })
            
            env = ImprovedRewardAirSimEnv(config=env_config)
            logger.info(f"创建改进奖励环境: {env_config['host']}:{env_config['port']}")
        else:
            # 使用原始环境配置
            env_config.update({
                "collision_penalty": getattr(self.config.environment, 'collision_penalty', -100.0),
                "goal_reward": getattr(self.config.environment, 'goal_reward', 100.0),
                "distance_reward_scale": getattr(self.config.environment, 'distance_reward_scale', 1.0),
                "velocity_penalty_scale": getattr(self.config.environment, 'velocity_penalty_scale', -0.1),
                "time_penalty": getattr(self.config.environment, 'time_penalty', -0.01),
            })
            
            env = ModernAirSimEnv(config=env_config)
            logger.info(f"创建标准环境: {env_config['host']}:{env_config['port']}")
        
        return env
    
    def _create_algorithm(self):
        """创建算法"""
        if self.config.algorithm.algorithm_name == "sac":
            # 根据设备调整配置
            if self.device.type == "cuda" and hasattr(self.config.algorithm, 'batch_size_gpu'):
                batch_size = self.config.algorithm.batch_size_gpu
                buffer_size = getattr(self.config.algorithm, 'buffer_size_gpu', self.config.algorithm.buffer_size)
            else:
                batch_size = self.config.training.batch_size
                buffer_size = self.config.algorithm.buffer_size
            
            algorithm = ModernSAC(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                learning_rate=self.config.training.learning_rate,
                buffer_size=buffer_size,
                batch_size=batch_size,
                tau=self.config.algorithm.tau,
                gamma=self.config.training.gamma,
                train_freq=self.config.algorithm.train_freq,
                gradient_steps=self.config.algorithm.gradient_steps,
                target_update_interval=self.config.algorithm.target_update_interval,
                ent_coef=self.config.algorithm.ent_coef,
                target_entropy=self.config.algorithm.target_entropy,
                device=self.device,
                seed=self.config.training.seed,
                use_mixed_precision=self.config.gpu.mixed_precision and self.device.type == "cuda",
                tensorboard_log=self.config.logging.tensorboard_log_dir,
            )
        else:
            raise ValueError(f"不支持的算法: {self.config.algorithm.algorithm_name}")
        
        logger.info(f"算法创建完成: {self.config.algorithm.algorithm_name}")
        return algorithm
    
    def _setup_tensorboard(self) -> SummaryWriter:
        """设置TensorBoard"""
        # 使用数据管理器的TensorBoard路径
        if hasattr(self, 'data_manager') and hasattr(self, 'experiment_id'):
            log_dir = self.data_manager.get_tensorboard_path(self.experiment_id, "train")
        else:
            # 回退到传统方式
            log_dir = Path(self.config.logging.tensorboard_log_dir) / self.config.experiment.experiment_name
            log_dir.mkdir(parents=True, exist_ok=True)
        
        writer = SummaryWriter(log_dir=str(log_dir))
        logger.info(f"TensorBoard日志目录: {log_dir}")
        
        return writer
    
    def _signal_handler(self, signum, frame):
        """信号处理器，用于优雅关闭"""
        logger.info(f"接收到信号 {signum}，开始优雅关闭...")
        self.save_checkpoint("emergency_checkpoint")
        self.cleanup()
        sys.exit(0)
    
    def train(self):
        """主训练循环"""
        logger.info("开始训练...")
        logger.info(f"总训练步数: {self.config.training.total_timesteps:,}")
        
        # 训练前优化
        optimize_for_training()
        
        observation, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_start_time = time.time()
        
        for step in range(self.config.training.total_timesteps):
            self.total_timesteps = step
            
            # 选择动作
            action, _ = self.algorithm.predict(observation, deterministic=False)
            
            # 执行动作
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            
            # 存储经验
            self.algorithm.replay_buffer.add(
                observation, next_observation, action, reward, terminated, [info]
            )
            
            # 更新统计
            episode_reward += reward
            episode_length += 1
            
            # 训练
            if step >= self.config.algorithm.learning_starts:
                if step % self.config.algorithm.train_freq == 0:
                    self.algorithm.train(
                        gradient_steps=self.config.algorithm.gradient_steps,
                        batch_size=None
                    )
            
            # 检查episode结束
            if terminated or truncated:
                # 记录episode信息
                episode_time = time.time() - episode_start_time
                self.last_episode_reward = episode_reward  # 记录最后一个episode的奖励
                self._log_episode(episode_reward, episode_length, episode_time, step)
                
                # 重置环境
                observation, info = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                episode_start_time = time.time()
                self.episode_count += 1
            else:
                observation = next_observation
            
            # 定期日志记录
            if step % self.config.logging.log_interval == 0:
                self._log_training_progress(step)
            
            # 定期保存
            if step % self.config.experiment.checkpoint_frequency == 0 and step > 0:
                self.save_checkpoint(f"checkpoint_{step}")
            
            # 性能监控
            if step % 100 == 0:
                self.performance_monitor.gpu_manager.auto_clear_cache()
        
        # 训练完成
        logger.info("训练完成")
        
        # 更新实验状态
        final_results = {
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.episode_count,
            "final_reward": getattr(self, 'last_episode_reward', 0.0),
            "best_reward": self.best_reward,
            "training_time_hours": (time.time() - self.start_time) / 3600
        }
        
        self.data_manager.update_experiment_status(
            experiment_id=self.experiment_id,
            status="completed",
            results=final_results,
            notes=f"Training completed successfully with {self.total_timesteps} timesteps"
        )
        
        self.save_final_model()
        self.generate_training_report()
    
    def _log_episode(self, reward: float, length: int, time_taken: float, step: int):
        """记录episode信息"""
        # 更新最佳奖励
        if reward > self.best_reward:
            self.best_reward = reward
            self.save_checkpoint("best_model", is_best=True)
        
        # 记录到数据管理器
        additional_data = {
            "step": step,
            "fps": length / time_taken if time_taken > 0 else 0,
            "best_reward": self.best_reward
        }
        
        # 如果使用改进环境，添加奖励组件信息
        collision_occurred = False
        if hasattr(self.env, 'get_reward_info'):
            reward_info = self.env.get_reward_info()
            additional_data.update(reward_info)
            collision_occurred = reward_info.get('collision_count', 0) > 0
        
        # 更新课程学习管理器
        if self.curriculum_manager:
            self.curriculum_manager.update_performance(reward, length, collision_occurred)
            
            # 添加课程学习信息
            curriculum_stats = self.curriculum_manager.get_current_stats()
            additional_data.update({
                "curriculum_difficulty": curriculum_stats.get("current_difficulty", 1),
                "curriculum_success_rate": curriculum_stats.get("success_rate", 0.0),
                "curriculum_collision_rate": curriculum_stats.get("collision_rate", 0.0)
            })
        
        self.data_manager.save_episode_data(
            experiment_id=self.experiment_id,
            episode=self.episode_count,
            reward=reward,
            length=length,
            time_taken=time_taken,
            additional_data=additional_data
        )
        
        # TensorBoard日志
        self.writer.add_scalar("Episode/Reward", reward, self.episode_count)
        self.writer.add_scalar("Episode/Length", length, self.episode_count)
        self.writer.add_scalar("Episode/Time", time_taken, self.episode_count)
        
        # 课程学习相关日志
        if self.curriculum_manager:
            curriculum_stats = self.curriculum_manager.get_current_stats()
            self.writer.add_scalar("Curriculum/Difficulty", curriculum_stats.get("current_difficulty", 1), self.episode_count)
            self.writer.add_scalar("Curriculum/SuccessRate", curriculum_stats.get("success_rate", 0.0), self.episode_count)
            self.writer.add_scalar("Curriculum/CollisionRate", curriculum_stats.get("collision_rate", 0.0), self.episode_count)
        
        # 性能监控
        fps = length / time_taken if time_taken > 0 else 0
        self.performance_monitor.record_fps(fps)
        
        # 控制台输出
        if self.episode_count % 10 == 0:
            base_info = (f"Episode {self.episode_count:,} | "
                        f"Step {step:,} | "
                        f"Reward: {reward:.2f} | "
                        f"Length: {length} | "
                        f"FPS: {fps:.1f} | "
                        f"Best: {self.best_reward:.2f}")
            
            # 添加课程学习信息
            if self.curriculum_manager:
                curriculum_stats = self.curriculum_manager.get_current_stats()
                base_info += (f" | Difficulty: {curriculum_stats.get('current_difficulty', 1)} | "
                             f"Success: {curriculum_stats.get('success_rate', 0.0):.2f}")
            
            logger.info(base_info)
    
    def _log_training_progress(self, step: int):
        """记录训练进度"""
        # 算法统计
        if hasattr(self.algorithm, 'get_performance_stats'):
            stats = self.algorithm.get_performance_stats()
            
            # 提取损失数据
            loss_data = {}
            for key, value in stats.items():
                self.writer.add_scalar(f"Algorithm/{key}", value, step)
                if 'loss' in key.lower() or 'error' in key.lower():
                    loss_data[key] = value
            
            # 保存损失数据到数据管理器
            if loss_data:
                self.data_manager.save_loss_data(
                    experiment_id=self.experiment_id,
                    step=step,
                    losses=loss_data
                )
        
        # 性能统计
        perf_stats = self.performance_monitor.get_detailed_stats()
        
        # GPU统计
        if 'gpu_stats' in perf_stats:
            for key, value in perf_stats['gpu_stats'].items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"GPU/{key}", value, step)
        
        # 系统统计
        if 'performance_metrics' in perf_stats:
            for key, value in perf_stats['performance_metrics'].items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"System/{key}", value, step)
        
        # 总训练时间
        elapsed_time = time.time() - self.start_time
        self.writer.add_scalar("Training/ElapsedTime", elapsed_time, step)
        self.writer.add_scalar("Training/StepsPerSecond", step / elapsed_time, step)
    
    def save_checkpoint(self, name: str, is_best: bool = False):
        """保存检查点"""
        # 临时保存到传统位置
        checkpoint_dir = Path(self.config.experiment.model_save_dir) / self.config.experiment.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        temp_checkpoint_path = checkpoint_dir / f"{name}.pth"
        
        # 保存算法到临时位置
        self.algorithm.save(temp_checkpoint_path)
        
        # 收集性能指标
        elapsed_time = time.time() - self.start_time
        performance_metrics = {
            "total_timesteps": self.total_timesteps,
            "episodes_trained": self.episode_count,
            "final_reward": getattr(self, 'last_reward', 0.0),
            "best_reward": self.best_reward,
            "training_duration": str(datetime.timedelta(seconds=int(elapsed_time))),
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
        }
        
        # 确定模型类型和标签
        if is_best and self.total_timesteps >= 10000:  # 只有训练足够长时间的best模型才进production
            model_type = "production"
            version = "1.0.0"  # 可以从配置或参数获取
            tag = "best"
            notes = f"Best model from experiment {self.config.experiment.experiment_name}"
        else:
            model_type = "development"
            version = None
            tag = name  # best, final等
            notes = f"Development checkpoint: {name}"
        
        # 使用模型管理器保存
        try:
            saved_dir = self.model_manager.save_model(
                model_path=str(temp_checkpoint_path),
                config=self.config.__dict__ if hasattr(self.config, '__dict__') else {},
                algorithm=self.config.algorithm.algorithm_name,
                experiment_name=self.config.experiment.experiment_name,
                performance_metrics=performance_metrics,
                model_type=model_type,
                version=version,
                tag=tag,
                notes=notes
            )
            logger.info(f"模型已保存到科学目录结构: {saved_dir}")
        except Exception as e:
            logger.warning(f"科学保存失败，使用传统方式: {e}")
            logger.info(f"检查点已保存: {temp_checkpoint_path}")
        
        # 保存训练状态（保持兼容性）
        state_path = checkpoint_dir / f"{name}_state.json"
        training_state = {
            "total_timesteps": self.total_timesteps,
            "episode_count": self.episode_count,
            "best_reward": self.best_reward,
            "training_time": elapsed_time,
        }
        
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
    
    def save_final_model(self):
        """保存最终模型"""
        self.save_checkpoint("final_model")
        
        # 保存配置
        config_path = Path(self.config.experiment.model_save_dir) / self.config.experiment.experiment_name / "config.yaml"
        config_manager = ConfigManager()
        config_manager.save_config(self.config, config_path)
    
    def generate_training_report(self):
        """生成训练报告"""
        report_dir = Path(self.config.experiment.save_dir) / self.config.experiment.experiment_name
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 性能报告
        performance_report_path = report_dir / "performance_report.json"
        self.performance_monitor.save_performance_report(performance_report_path)
        
        # 训练报告
        training_report = {
            "experiment_name": self.config.experiment.experiment_name,
            "algorithm": self.config.algorithm.algorithm_name,
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.episode_count,
            "best_reward": self.best_reward,
            "training_time_hours": (time.time() - self.start_time) / 3600,
            "final_performance": self.performance_monitor.get_performance_metrics().__dict__,
            "config": self.config.__dict__,
        }
        
        report_path = report_dir / "training_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(training_report, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"训练报告已保存: {report_path}")
    
    def cleanup(self):
        """清理资源"""
        try:
            # 如果实验被中断，更新状态
            if hasattr(self, 'data_manager') and hasattr(self, 'experiment_id'):
                self.data_manager.update_experiment_status(
                    experiment_id=self.experiment_id,
                    status="stopped",
                    results={
                        "total_timesteps": getattr(self, 'total_timesteps', 0),
                        "total_episodes": getattr(self, 'episode_count', 0),
                        "best_reward": getattr(self, 'best_reward', float('-inf'))
                    },
                    notes="Training was interrupted or stopped early"
                )
        except Exception as e:
            logger.warning(f"无法更新实验状态: {e}")
        
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.stop_monitoring()
        
        if hasattr(self, 'env'):
            self.env.close()
        
        if hasattr(self, 'writer'):
            self.writer.close()
        
        logger.info("资源清理完成")


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="现代化Drone RL训练脚本")
    
    # 基础参数
    parser.add_argument("--algorithm", type=str, default="sac",
                       choices=["sac", "ppo"], help="RL算法")
    parser.add_argument("--config", type=str, default=None,
                       help="配置文件路径")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="实验名称")
    
    # 训练参数
    parser.add_argument("--total-timesteps", type=int, default=None,
                       help="总训练步数")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="学习率")
    parser.add_argument("--device", type=str, default=None,
                       choices=["auto", "cpu", "cuda"], help="计算设备")
    parser.add_argument("--seed", type=int, default=None,
                       help="随机种子")
    
    # 环境参数
    parser.add_argument("--env-host", type=str, default="127.0.0.1",
                       help="AirSim主机地址")
    parser.add_argument("--env-port", type=int, default=41451,
                       help="AirSim端口")
    
    # 日志参数
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    parser.add_argument("--tensorboard-log", type=str, default=None,
                       help="TensorBoard日志目录")
    
    # 性能参数
    parser.add_argument("--disable-performance-monitoring", action="store_true",
                       help="禁用性能监控")
    parser.add_argument("--disable-mixed-precision", action="store_true",
                       help="禁用混合精度训练")
    
    # 检查点
    parser.add_argument("--resume", type=str, default=None,
                       help="从检查点恢复训练")
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # 创建配置
        logger.info("创建训练配置...")
        config = create_config_from_args(args)
        
        # 应用命令行覆盖
        if args.total_timesteps:
            config.training.total_timesteps = args.total_timesteps
        if args.batch_size:
            config.training.batch_size = args.batch_size
        if args.learning_rate:
            config.training.learning_rate = args.learning_rate
        if args.device:
            config.training.device = args.device
        if args.seed is not None:
            config.training.seed = args.seed
        if args.tensorboard_log:
            config.logging.tensorboard_log_dir = args.tensorboard_log
        if args.disable_mixed_precision:
            config.gpu.mixed_precision = False
        
        # 环境覆盖
        if args.env_host:
            config.environment.host = args.env_host
        if args.env_port:
            config.environment.port = args.env_port
        
        # 创建训练器
        logger.info("初始化训练器...")
        trainer = ModernTrainer(config)
        
        # 恢复训练（如果指定）
        if args.resume:
            logger.info(f"从检查点恢复训练: {args.resume}")
            trainer.algorithm.load(args.resume)
        
        # 开始训练
        trainer.train()
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        
        # 更新实验状态为失败
        if 'trainer' in locals() and hasattr(trainer, 'data_manager'):
            try:
                trainer.data_manager.update_experiment_status(
                    experiment_id=trainer.experiment_id,
                    status="failed",
                    results={
                        "total_timesteps": getattr(trainer, 'total_timesteps', 0),
                        "total_episodes": getattr(trainer, 'episode_count', 0),
                        "best_reward": getattr(trainer, 'best_reward', float('-inf'))
                    },
                    notes=f"Training failed with error: {str(e)}"
                )
            except Exception as cleanup_error:
                logger.warning(f"无法更新失败状态: {cleanup_error}")
        
        raise
    finally:
        # 清理资源
        if 'trainer' in locals():
            trainer.cleanup()


if __name__ == "__main__":
    main()