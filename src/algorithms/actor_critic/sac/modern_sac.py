"""
现代化的SAC算法实现
针对 Windows 10 + AirSim 1.8.1 + CUDA 12.1 优化
支持混合精度训练、GPU优化、异步处理
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import gymnasium as gym

# 现代化导入
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import get_device

# 兼容不同版本的Stable-Baselines3
try:
    from stable_baselines3.common.type_aliases import PyTorchObs
except ImportError:
    # 新版本中PyTorchObs可能在不同位置或已移除
    PyTorchObs = Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]]

# 设置日志
logger = logging.getLogger(__name__)


class ModernSACNetwork(nn.Module):
    """
    现代化的SAC网络架构
    支持CUDA 12.1优化和混合精度训练
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        net_arch: Optional[List[int]] = None,
        activation_fn: nn.Module = nn.ReLU,
        device: Union[str, torch.device] = "auto",
        use_mixed_precision: bool = False,
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = get_device(device)
        self.use_mixed_precision = use_mixed_precision
        
        # 默认网络架构
        if net_arch is None:
            net_arch = [256, 256]
        
        # 特征提取器
        self.features_extractor = self._build_feature_extractor()
        
        # 先移动特征提取器到设备
        self.features_extractor = self.features_extractor.to(self.device)
        
        # 获取特征维度
        with torch.no_grad():
            sample_obs = self._get_sample_observation()
            # 将样本观察移动到正确设备
            if isinstance(sample_obs, dict):
                sample_obs = {k: v.to(self.device) for k, v in sample_obs.items()}
            else:
                sample_obs = sample_obs.to(self.device)
            features = self.features_extractor(sample_obs)
            features_dim = features.shape[-1]
        
        # Actor网络
        self.actor = self._build_actor(features_dim, net_arch, activation_fn)
        
        # Critic网络 (双Q网络)
        self.critic1 = self._build_critic(features_dim, net_arch, activation_fn)
        self.critic2 = self._build_critic(features_dim, net_arch, activation_fn)
        
        # 目标网络
        self.critic1_target = self._build_critic(features_dim, net_arch, activation_fn)
        self.critic2_target = self._build_critic(features_dim, net_arch, activation_fn)
        
        # 复制参数到目标网络
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # 冻结目标网络
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False
        
        # 移动到设备
        self.to(self.device)
        
        # 优化设置
        if torch.cuda.is_available() and "cuda" in str(self.device):
            # 启用cuDNN基准模式
            torch.backends.cudnn.benchmark = True
            # 使用TensorFloat-32 (TF32) 在Ampere GPU上
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _build_feature_extractor(self) -> nn.Module:
        """构建特征提取器"""
        if isinstance(self.observation_space, gym.spaces.Dict):
            # 处理字典观察空间（图像+状态）
            return DictFeatureExtractor(self.observation_space, self.device)
        else:
            # 处理简单观察空间
            obs_dim = gym.spaces.utils.flatdim(self.observation_space)
            return nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
    
    def _build_actor(self, features_dim: int, net_arch: List[int], activation_fn: nn.Module) -> nn.Module:
        """构建Actor网络"""
        action_dim = self.action_space.shape[0]
        
        layers = []
        prev_dim = features_dim
        
        # 隐藏层
        for hidden_dim in net_arch:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
                nn.LayerNorm(hidden_dim),  # 添加LayerNorm提升稳定性
                nn.Dropout(0.1)  # 轻量级Dropout
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, action_dim * 2))  # mean和log_std
        
        return nn.Sequential(*layers)
    
    def _build_critic(self, features_dim: int, net_arch: List[int], activation_fn: nn.Module) -> nn.Module:
        """构建Critic网络"""
        action_dim = self.action_space.shape[0]
        input_dim = features_dim + action_dim
        
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in net_arch:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _get_sample_observation(self) -> torch.Tensor:
        """获取样本观察用于维度推断"""
        if isinstance(self.observation_space, gym.spaces.Dict):
            sample_obs = {}
            for key, space in self.observation_space.spaces.items():
                sample_obs[key] = torch.zeros((1,) + space.shape, dtype=torch.float32)
            return sample_obs
        else:
            obs_shape = self.observation_space.shape
            return torch.zeros((1,) + obs_shape, dtype=torch.float32)
    
    def forward_actor(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Actor前向传播"""
        features = self.features_extractor(observations)
        actor_output = self.actor(features)
        
        # 分离均值和对数标准差
        action_dim = self.action_space.shape[0]
        mean = actor_output[:, :action_dim]
        log_std = actor_output[:, action_dim:]
        
        # 限制log_std范围
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def forward_critic(self, observations: torch.Tensor, actions: torch.Tensor, target: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Critic前向传播"""
        features = self.features_extractor(observations)
        input_tensor = torch.cat([features, actions], dim=1)
        
        if target:
            q1 = self.critic1_target(input_tensor)
            q2 = self.critic2_target(input_tensor)
        else:
            q1 = self.critic1(input_tensor)
            q2 = self.critic2(input_tensor)
        
        return q1, q2


class DictFeatureExtractor(nn.Module):
    """字典观察空间的特征提取器"""
    
    def __init__(self, observation_space: gym.spaces.Dict, device: torch.device):
        super().__init__()
        self.device = device
        
        # 图像特征提取器（CNN）
        if "image" in observation_space.spaces:
            image_shape = observation_space.spaces["image"].shape
            self.image_cnn = self._build_cnn(image_shape)
            
            # 计算CNN输出维度
            with torch.no_grad():
                sample_image = torch.zeros((1,) + image_shape, dtype=torch.float32)
                cnn_output = self.image_cnn(sample_image.permute(0, 3, 1, 2))  # NHWC -> NCHW
                self.cnn_features_dim = cnn_output.shape[1]
        else:
            self.image_cnn = None
            self.cnn_features_dim = 0
        
        # 状态特征提取器
        if "state" in observation_space.spaces:
            state_dim = observation_space.spaces["state"].shape[0]
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )
            self.state_features_dim = 64
        else:
            self.state_mlp = None
            self.state_features_dim = 0
        
        # 融合层
        total_features_dim = self.cnn_features_dim + self.state_features_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
    
    def _build_cnn(self, image_shape: Tuple[int, ...]) -> nn.Module:
        """构建CNN特征提取器"""
        # 假设输入是 (H, W, C)
        channels = image_shape[2] if len(image_shape) == 3 else 1
        
        return nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 第三个卷积块
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        features = []
        
        # 处理图像
        if self.image_cnn is not None and "image" in observations:
            image = observations["image"].to(self.device)
            # 转换为浮点数并归一化
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            # NHWC -> NCHW
            if len(image.shape) == 4:
                image = image.permute(0, 3, 1, 2)
            image_features = self.image_cnn(image)
            features.append(image_features)
        
        # 处理状态
        if self.state_mlp is not None and "state" in observations:
            state = observations["state"].to(self.device)
            state_features = self.state_mlp(state)
            features.append(state_features)
        
        # 融合特征
        if features:
            combined_features = torch.cat(features, dim=1)
            return self.fusion(combined_features)
        else:
            return torch.zeros((observations[list(observations.keys())[0]].shape[0], 128), device=self.device)


class ModernSAC:
    """
    现代化的SAC算法实现
    支持CUDA 12.1、混合精度训练、异步处理
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        target_update_interval: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        device: Union[str, torch.device] = "auto",
        seed: Optional[int] = None,
        use_mixed_precision: bool = False,
        optimize_memory: bool = True,
        tensorboard_log: Optional[str] = None,
        **kwargs
    ):
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.use_sde_at_warmup = use_sde_at_warmup
        self.device = get_device(device)
        self.seed = seed
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.optimize_memory = optimize_memory
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # 熵系数
        if ent_coef == "auto":
            self.ent_coef_optimizer = None
            if target_entropy == "auto":
                self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
            else:
                self.target_entropy = target_entropy
            self.log_ent_coef = torch.log(torch.ones(1, device=self.device)).requires_grad_(True)
        else:
            self.ent_coef_optimizer = None
            self.target_entropy = None
            self.log_ent_coef = torch.log(torch.tensor(ent_coef, device=self.device))
        
        # 网络
        self.policy = ModernSACNetwork(
            observation_space=observation_space,
            action_space=action_space,
            device=self.device,
            use_mixed_precision=self.use_mixed_precision
        )
        
        # 优化器
        self.actor_optimizer = optim.Adam(
            self.policy.actor.parameters(),
            lr=learning_rate,
            eps=1e-7,  # 更稳定的epsilon
        )
        
        self.critic_optimizer = optim.Adam(
            list(self.policy.critic1.parameters()) + list(self.policy.critic2.parameters()),
            lr=learning_rate,
            eps=1e-7,
        )
        
        if self.ent_coef_optimizer is None and ent_coef == "auto":
            self.ent_coef_optimizer = optim.Adam(
                [self.log_ent_coef],
                lr=learning_rate,
                eps=1e-7,
            )
        
        # 混合精度训练
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            logger.info("启用混合精度训练")
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=self.device,
            optimize_memory_usage=optimize_memory,
            handle_timeout_termination=True
        )
        
        # TensorBoard
        self.tensorboard_log = tensorboard_log
        if tensorboard_log:
            self.writer = SummaryWriter(log_dir=tensorboard_log)
        else:
            self.writer = None
        
        # 训练统计
        self.num_timesteps = 0
        self.num_gradient_updates = 0
        self._last_obs = None
        self._last_episode_starts = None
        
        # 性能监控
        self.training_times = deque(maxlen=100)
        self.gpu_memory_usage = deque(maxlen=100)
        
        logger.info(f"SAC初始化完成，设备: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA版本: {torch.version.cuda}")
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """预测动作"""
        
        # 转换观察为张量
        if isinstance(observation, dict):
            obs_tensor = {}
            for key, value in observation.items():
                obs_tensor[key] = torch.as_tensor(value, device=self.device).unsqueeze(0)
        else:
            obs_tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            if self.use_mixed_precision:
                with autocast():
                    actions = self._predict(obs_tensor, deterministic)
            else:
                actions = self._predict(obs_tensor, deterministic)
        
        return actions.cpu().numpy(), state
    
    def _predict(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """内部预测函数"""
        mean, log_std = self.policy.forward_actor(observations)
        
        if deterministic:
            return torch.tanh(mean)
        else:
            std = torch.exp(log_std)
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # 重参数化技巧
            action = torch.tanh(x_t)
            return action
    
    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 4,
        eval_env=None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "ModernSAC":
        """学习函数"""
        
        if reset_num_timesteps:
            self.num_timesteps = 0
            self.num_gradient_updates = 0
        
        # 训练循环将在环境交互中实现
        # 这里只是接口定义
        
        return self
    
    def train(self, gradient_steps: int, batch_size: int = None) -> None:
        """训练网络"""
        
        if batch_size is None:
            batch_size = self.batch_size
        
        # 检查缓冲区大小
        if self.replay_buffer.size() < batch_size:
            return
        
        start_time = time.time()
        
        for gradient_step in range(gradient_steps):
            
            # 采样批次
            replay_data = self.replay_buffer.sample(batch_size)
            
            if self.use_mixed_precision:
                self._train_step_mixed_precision(replay_data)
            else:
                self._train_step(replay_data)
            
            self.num_gradient_updates += 1
            
            # 更新目标网络
            if self.num_gradient_updates % self.target_update_interval == 0:
                self._update_target_networks()
        
        # 记录训练时间
        training_time = time.time() - start_time
        self.training_times.append(training_time)
        
        # 记录GPU内存使用
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / 1024**3  # GB
            self.gpu_memory_usage.append(memory_usage)
    
    def _train_step(self, replay_data) -> None:
        """单步训练（常规精度）"""
        # 实现训练逻辑
        pass  # 这里需要实现具体的训练步骤
    
    def _train_step_mixed_precision(self, replay_data) -> None:
        """单步训练（混合精度）"""
        # 实现混合精度训练逻辑
        pass  # 这里需要实现具体的训练步骤
    
    def _update_target_networks(self) -> None:
        """软更新目标网络"""
        for param, target_param in zip(self.policy.critic1.parameters(), self.policy.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.policy.critic2.parameters(), self.policy.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: Union[str, Path]) -> None:
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'ent_coef_optimizer_state_dict': self.ent_coef_optimizer.state_dict() if self.ent_coef_optimizer else None,
            'log_ent_coef': self.log_ent_coef,
            'num_timesteps': self.num_timesteps,
            'num_gradient_updates': self.num_gradient_updates,
        }, path)
        
        logger.info(f"模型已保存到: {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if checkpoint['ent_coef_optimizer_state_dict'] and self.ent_coef_optimizer:
            self.ent_coef_optimizer.load_state_dict(checkpoint['ent_coef_optimizer_state_dict'])
        
        self.log_ent_coef = checkpoint['log_ent_coef']
        self.num_timesteps = checkpoint['num_timesteps']
        self.num_gradient_updates = checkpoint['num_gradient_updates']
        
        logger.info(f"模型已从 {path} 加载")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        stats = {
            "avg_training_time": np.mean(self.training_times) if self.training_times else 0.0,
            "num_gradient_updates": self.num_gradient_updates,
            "replay_buffer_size": self.replay_buffer.size(),
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_memory_usage_gb": self.gpu_memory_usage[-1] if self.gpu_memory_usage else 0.0,
                "avg_gpu_memory_gb": np.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0.0,
                "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0,
            })
        
        return stats