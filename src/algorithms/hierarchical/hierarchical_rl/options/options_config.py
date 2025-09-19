#!/usr/bin/env python
"""
Configuration settings for Options Framework algorithm.
"""

import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class OptionsConfig:
    """Configuration for Options Framework algorithm."""
    
    # Environment settings
    env_name: str = "AirSimEnv-v42"
    max_episode_steps: int = 512
    
    # Network architecture
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    layer_norm: bool = True
    
    # Options-specific parameters
    num_options: int = 8  # Number of options/skills
    option_min_length: int = 4  # Minimum option duration
    option_max_length: int = 20  # Maximum option duration
    
    # Option discovery
    use_diversity_bonus: bool = True  # Encourage option diversity
    diversity_coef: float = 0.1  # Diversity bonus coefficient
    use_mutual_info: bool = True  # Use mutual information for option discovery
    mi_coef: float = 0.1  # Mutual information coefficient
    
    # Network architectures
    policy_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [256, 256])
    option_policy_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [256, 256])
    termination_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [128, 64])
    
    # Training parameters
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    
    # PPO parameters
    ppo_epochs: int = 4
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Option-critic parameters
    termination_reg: float = 0.01  # Termination regularization
    deliberation_cost: float = 0.0  # Cost for switching options
    
    # Experience buffer
    buffer_size: int = 2048
    
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
        assert self.num_options > 1, "Number of options must be greater than 1"
        assert self.option_min_length > 0, "Option minimum length must be positive"
        assert self.option_max_length > self.option_min_length, "Option max length must be greater than min length"
        assert 0 <= self.diversity_coef <= 1, "Diversity coefficient must be in [0, 1]"
        assert 0 <= self.mi_coef <= 1, "Mutual information coefficient must be in [0, 1]"