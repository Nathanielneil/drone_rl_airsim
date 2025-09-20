"""
现代化配置管理系统
支持YAML配置、环境变量、命令行参数覆盖
针对Windows 10 + AirSim 1.8.1 + CUDA 12.1优化
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import yaml
import json
from dataclasses import dataclass, field, asdict
from omegaconf import OmegaConf, DictConfig
import argparse

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    total_timesteps: int = 1_000_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    device: str = "auto"  # auto, cpu, cuda
    seed: Optional[int] = None
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 0.5


@dataclass
class GPUConfig:
    """GPU配置"""
    memory_fraction: float = 0.9
    allow_growth: bool = True
    enable_tf32: bool = True
    enable_cudnn_benchmark: bool = True
    mixed_precision: bool = True
    gradient_checkpointing: bool = False


@dataclass
class EnvironmentConfig:
    """环境配置"""
    name: str = "ModernAirSim-v1"
    max_episode_steps: int = 1000
    action_space_type: str = "continuous"  # continuous, discrete
    observation_space_type: str = "mixed"  # image, state, mixed
    
    # AirSim设置
    host: str = "127.0.0.1"
    port: int = 41451
    vehicle_name: str = "Drone1"
    camera_name: str = "front_center"
    image_type: str = "DepthVis"
    image_width: int = 84
    image_height: int = 84
    
    # 飞行参数
    max_velocity: float = 10.0
    max_altitude: float = 50.0
    min_altitude: float = -10.0
    takeoff_height: float = 2.0
    
    # 奖励设置
    collision_penalty: float = -100.0
    goal_reward: float = 100.0
    distance_reward_scale: float = 1.0
    velocity_penalty_scale: float = -0.1
    time_penalty: float = -0.01


@dataclass
class AlgorithmConfig:
    """算法配置基类"""
    algorithm_name: str = "sac"
    policy_type: str = "MlpPolicy"
    batch_size: int = 256
    learning_starts: int = 10000
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 1


@dataclass
class SACConfig(AlgorithmConfig):
    """SAC算法配置"""
    algorithm_name: str = "sac"
    buffer_size: int = 1_000_000
    tau: float = 0.005
    ent_coef: Union[str, float] = "auto"
    target_entropy: Union[str, float] = "auto"
    use_sde: bool = False
    sde_sample_freq: int = -1
    
    # 网络架构
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    activation_fn: str = "relu"
    
    # GPU特定设置
    batch_size_gpu: int = 512
    buffer_size_gpu: int = 2_000_000


@dataclass
class PPOConfig(AlgorithmConfig):
    """PPO算法配置"""
    algorithm_name: str = "ppo"
    n_steps: int = 2048
    n_epochs: int = 10
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    gae_lambda: float = 0.95
    use_sde: bool = False
    normalize_advantage: bool = True
    
    # 网络架构
    net_arch: List[int] = field(default_factory=lambda: [64, 64])
    activation_fn: str = "tanh"


@dataclass
class LoggingConfig:
    """日志配置"""
    log_level: str = "INFO"
    log_interval: int = 1000
    eval_interval: int = 10000
    save_interval: int = 50000
    tensorboard_log_dir: str = "experiments/logs"
    save_replay_buffer: bool = False
    
    # 性能监控
    monitor_performance: bool = True
    monitor_interval: float = 1.0
    save_performance_report: bool = True


@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str = "drone_rl_experiment"
    save_dir: str = "experiments/results"
    model_save_dir: str = "models"
    checkpoint_frequency: int = 10000
    max_checkpoints: int = 5
    
    # 评估设置
    eval_episodes: int = 10
    eval_deterministic: bool = True
    eval_render: bool = False


@dataclass
class ModernConfig:
    """现代化配置类"""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    algorithm: Union[SACConfig, PPOConfig] = field(default_factory=SACConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.config_cache = {}
        
        # 注册算法配置类
        self.algorithm_configs = {
            "sac": SACConfig,
            "ppo": PPOConfig,
        }
        
        logger.info(f"配置管理器初始化，配置目录: {self.config_dir}")
    
    def load_config(self, config_path: Union[str, Path]) -> DictConfig:
        """加载配置文件"""
        config_path = Path(config_path)
        
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 检查缓存
        cache_key = str(config_path)
        if cache_key in self.config_cache:
            logger.debug(f"从缓存加载配置: {config_path}")
            return self.config_cache[cache_key]
        
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            # 转换为OmegaConf
            config = OmegaConf.create(config_dict)
            
            # 缓存配置
            self.config_cache[cache_key] = config
            
            logger.info(f"配置加载成功: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"配置加载失败 {config_path}: {e}")
            raise
    
    def merge_configs(self, *configs: DictConfig) -> DictConfig:
        """合并多个配置"""
        merged = OmegaConf.create({})
        
        for config in configs:
            merged = OmegaConf.merge(merged, config)
        
        return merged
    
    def create_modern_config(
        self, 
        algorithm: str = "sac",
        base_config_path: Optional[str] = None,
        overrides: Optional[Dict] = None
    ) -> ModernConfig:
        """创建现代化配置对象"""
        
        # 加载基础配置
        if base_config_path:
            base_config = self.load_config(base_config_path)
        else:
            base_config = self.load_config("default.yaml")
        
        # 加载算法配置
        algorithm_config = self.load_config(f"algorithms/{algorithm}.yaml")
        
        # 合并配置
        merged_config = self.merge_configs(base_config, algorithm_config)
        
        # 应用覆盖
        if overrides:
            override_config = OmegaConf.create(overrides)
            merged_config = OmegaConf.merge(merged_config, override_config)
        
        # 转换为现代化配置对象
        config = self._dict_to_modern_config(OmegaConf.to_container(merged_config, resolve=True), algorithm)
        
        # 验证配置
        self.validate_config(config)
        
        return config
    
    def _dict_to_modern_config(self, config_dict: Dict, algorithm: str) -> ModernConfig:
        """将字典转换为现代化配置对象"""
        
        # 创建各个配置组件
        training_config = TrainingConfig(**config_dict.get("training", {}))
        gpu_config = GPUConfig(**config_dict.get("gpu", {}))
        environment_config = EnvironmentConfig(**config_dict.get("environment", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))
        experiment_config = ExperimentConfig(**config_dict.get("experiment", {}))
        
        # 创建算法配置
        algorithm_config_class = self.algorithm_configs.get(algorithm, SACConfig)
        algorithm_config = algorithm_config_class(**config_dict.get(algorithm, {}))
        
        return ModernConfig(
            training=training_config,
            gpu=gpu_config,
            environment=environment_config,
            algorithm=algorithm_config,
            logging=logging_config,
            experiment=experiment_config
        )
    
    def validate_config(self, config: ModernConfig):
        """验证配置"""
        errors = []
        
        # 验证训练配置
        if config.training.total_timesteps <= 0:
            errors.append("total_timesteps必须大于0")
        
        if config.training.batch_size <= 0:
            errors.append("batch_size必须大于0")
        
        if not 0 < config.training.learning_rate < 1:
            errors.append("learning_rate必须在(0, 1)范围内")
        
        # 验证GPU配置
        if not 0 < config.gpu.memory_fraction <= 1:
            errors.append("memory_fraction必须在(0, 1]范围内")
        
        # 验证环境配置
        if config.environment.max_episode_steps <= 0:
            errors.append("max_episode_steps必须大于0")
        
        if config.environment.action_space_type not in ["continuous", "discrete"]:
            errors.append("action_space_type必须是'continuous'或'discrete'")
        
        # 验证算法配置
        if isinstance(config.algorithm, SACConfig):
            if config.algorithm.buffer_size <= config.training.batch_size:
                errors.append("buffer_size必须大于batch_size")
        
        if errors:
            raise ValueError(f"配置验证失败: {'; '.join(errors)}")
        
        logger.info("配置验证通过")
    
    def save_config(self, config: ModernConfig, filepath: Union[str, Path]):
        """保存配置"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典
        config_dict = asdict(config)
        
        # 保存为YAML
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"配置已保存到: {filepath}")
    
    def load_from_cli(self, args: argparse.Namespace) -> Dict:
        """从命令行参数加载配置覆盖"""
        overrides = {}
        
        # 训练参数
        if hasattr(args, 'batch_size') and args.batch_size:
            overrides['training'] = overrides.get('training', {})
            overrides['training']['batch_size'] = args.batch_size
        
        if hasattr(args, 'learning_rate') and args.learning_rate:
            overrides['training'] = overrides.get('training', {})
            overrides['training']['learning_rate'] = args.learning_rate
        
        if hasattr(args, 'device') and args.device:
            overrides['training'] = overrides.get('training', {})
            overrides['training']['device'] = args.device
        
        # 实验参数
        if hasattr(args, 'experiment_name') and args.experiment_name:
            overrides['experiment'] = overrides.get('experiment', {})
            overrides['experiment']['experiment_name'] = args.experiment_name
        
        return overrides
    
    def get_optimized_config_for_hardware(self, base_config: ModernConfig) -> ModernConfig:
        """根据硬件优化配置"""
        import torch
        import psutil
        
        # 复制配置
        config = asdict(base_config)
        
        # GPU优化
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = gpu_props.total_memory / 1024**3
            
            logger.info(f"检测到GPU: {gpu_props.name}, 显存: {gpu_memory_gb:.1f}GB")
            
            # 根据显存调整batch size
            if gpu_memory_gb >= 24:  # 高端GPU
                if isinstance(base_config.algorithm, SACConfig):
                    config['algorithm']['batch_size_gpu'] = 1024
                    config['algorithm']['buffer_size_gpu'] = 5_000_000
                config['training']['batch_size'] = 512
            elif gpu_memory_gb >= 12:  # 中端GPU
                if isinstance(base_config.algorithm, SACConfig):
                    config['algorithm']['batch_size_gpu'] = 512
                    config['algorithm']['buffer_size_gpu'] = 2_000_000
                config['training']['batch_size'] = 256
            elif gpu_memory_gb >= 6:  # 入门级GPU
                if isinstance(base_config.algorithm, SACConfig):
                    config['algorithm']['batch_size_gpu'] = 256
                    config['algorithm']['buffer_size_gpu'] = 1_000_000
                config['training']['batch_size'] = 128
            else:  # 低端GPU
                config['training']['device'] = "cpu"
                config['gpu']['mixed_precision'] = False
                config['training']['batch_size'] = 64
            
            # 启用GPU优化
            config['gpu']['enable_tf32'] = True
            config['gpu']['enable_cudnn_benchmark'] = True
            config['gpu']['mixed_precision'] = True
            
        else:
            # CPU优化
            logger.info("未检测到GPU，使用CPU训练")
            config['training']['device'] = "cpu"
            config['gpu']['mixed_precision'] = False
            config['training']['batch_size'] = 64
            
            # 根据CPU核心数调整参数
            cpu_count = psutil.cpu_count(logical=False)
            if cpu_count >= 8:
                config['training']['batch_size'] = 128
        
        # 内存优化
        total_ram_gb = psutil.virtual_memory().total / 1024**3
        if total_ram_gb < 16:
            # 低内存系统
            if isinstance(base_config.algorithm, SACConfig):
                config['algorithm']['buffer_size'] = 500_000
            config['training']['batch_size'] = min(config['training']['batch_size'], 64)
        
        return self._dict_to_modern_config(config, base_config.algorithm.algorithm_name)
    
    def create_experiment_config(
        self,
        algorithm: str,
        experiment_name: str,
        custom_config: Optional[Dict] = None
    ) -> ModernConfig:
        """创建实验配置"""
        
        # 基础配置
        config = self.create_modern_config(algorithm=algorithm)
        
        # 设置实验名称
        config.experiment.experiment_name = experiment_name
        
        # 应用自定义配置
        if custom_config:
            config_dict = asdict(config)
            for key, value in custom_config.items():
                if '.' in key:
                    # 支持嵌套键，如 "training.batch_size"
                    keys = key.split('.')
                    target = config_dict
                    for k in keys[:-1]:
                        target = target[k]
                    target[keys[-1]] = value
                else:
                    config_dict[key] = value
            
            config = self._dict_to_modern_config(config_dict, algorithm)
        
        # 硬件优化
        config = self.get_optimized_config_for_hardware(config)
        
        return config


def create_config_from_args(args: argparse.Namespace) -> ModernConfig:
    """从命令行参数创建配置"""
    config_manager = ConfigManager()
    
    # 基础参数
    algorithm = getattr(args, 'algorithm', 'sac')
    experiment_name = getattr(args, 'experiment_name', f'{algorithm}_experiment')
    
    # CLI覆盖
    cli_overrides = config_manager.load_from_cli(args)
    
    # 创建配置
    config = config_manager.create_experiment_config(
        algorithm=algorithm,
        experiment_name=experiment_name,
        custom_config=cli_overrides
    )
    
    return config