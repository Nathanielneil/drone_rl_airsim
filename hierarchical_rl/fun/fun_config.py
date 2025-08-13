#!/usr/bin/env python
"""
Configuration settings for FuN (FeUdal Networks) algorithm.
"""

import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class FuNConfig:
    """Configuration for FuN (FeUdal Networks) algorithm."""
    
    # Environment settings
    env_name: str = "AirSimEnv-v42"
    max_episode_steps: int = 512
    
    # Network architecture
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    layer_norm: bool = True
    
    # FuN-specific parameters
    manager_horizon: int = 10  # Manager decision frequency (c steps)
    embedding_dim: int = 256  # State embedding dimension
    goal_dim: int = 16  # Manager goal dimension
    
    # Manager network
    manager_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [512, 256])
    manager_lr: float = 3e-4
    
    # Worker network  
    worker_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [256, 256])
    worker_lr: float = 3e-4
    
    # Intrinsic motivation
    alpha: float = 0.5  # Intrinsic reward coefficient
    dilation: int = 10  # Transition policy gradient dilation
    
    # Training parameters
    batch_size: int = 256
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    
    # PPO parameters (for policy updates)
    ppo_epochs: int = 4
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Experience buffer
    buffer_size: int = 2048  # Rollout buffer size
    
    # Training schedule
    train_freq: int = 1
    save_freq: int = 1000
    
    # Evaluation
    eval_freq: int = 5000
    eval_episodes: int = 10
    
    # Logging
    log_freq: int = 100
    tensorboard_log: bool = True
    
    # Device
    device: str = "cuda"
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.manager_horizon > 0, "Manager horizon must be positive"
        assert self.embedding_dim > 0, "Embedding dimension must be positive"
        assert self.goal_dim > 0, "Goal dimension must be positive"
        assert 0 < self.alpha <= 1, "Intrinsic reward coefficient must be in (0, 1]"
        assert self.dilation > 0, "Dilation parameter must be positive"