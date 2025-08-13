#!/usr/bin/env python
"""
Configuration settings for HIRO algorithm.
"""

import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class HIROConfig:
    """Configuration for HIRO (HIerarchical RL with Off-policy correction) algorithm."""
    
    # Environment settings
    env_name: str = "AirSimEnv-v42"
    max_episode_steps: int = 512
    
    # Network architecture
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    layer_norm: bool = True
    
    # HIRO-specific parameters
    subgoal_dim: int = 3  # Subgoal dimension (position)
    subgoal_freq: int = 10  # Subgoal frequency (every k steps)
    subgoal_scale: float = 10.0  # Subgoal scaling factor
    
    # Goal relabeling parameters
    her_ratio: float = 0.8  # Hindsight experience replay ratio
    off_policy_correction: bool = True  # Enable off-policy correction
    correction_radius: float = 2.0  # Off-policy correction radius
    
    # High-level policy
    high_level_lr: float = 3e-4
    high_level_noise: float = 0.2
    high_level_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [256, 256])
    
    # Low-level policy  
    low_level_lr: float = 3e-4
    low_level_noise: float = 0.1
    low_level_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [256, 256])
    
    # Training parameters
    batch_size: int = 256
    gamma: float = 0.98  # Discount factor
    tau: float = 0.005  # Soft update rate
    
    # Replay buffer
    buffer_size: int = 1000000
    min_buffer_size: int = 10000
    
    # Training schedule
    train_freq: int = 1
    update_freq: int = 2  # Update frequency for target networks
    high_level_train_freq: int = 2  # High-level training frequency
    
    # Exploration
    noise_decay: float = 0.9995
    min_noise: float = 0.05
    random_eps: float = 0.1
    
    # Evaluation
    eval_freq: int = 5000
    eval_episodes: int = 10
    
    # Logging
    log_freq: int = 100
    save_freq: int = 1000
    tensorboard_log: bool = True
    
    # Device
    device: str = "cuda"
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.subgoal_dim > 0, "Subgoal dimension must be positive"
        assert self.subgoal_freq > 0, "Subgoal frequency must be positive"
        assert self.subgoal_scale > 0, "Subgoal scale must be positive"
        assert 0 < self.her_ratio <= 1, "HER ratio must be in (0, 1]"
        assert self.correction_radius > 0, "Correction radius must be positive"