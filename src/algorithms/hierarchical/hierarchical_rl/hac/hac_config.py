#!/usr/bin/env python
"""
Configuration settings for HAC algorithm.
"""

import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class HACConfig:
    """Configuration for HAC (Hindsight Action Control) algorithm."""
    
    # Environment settings
    env_name: str = "AirSimEnv-v42"
    max_episode_steps: int = 512
    
    # Network architecture
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    layer_norm: bool = True
    
    # HAC-specific parameters
    num_levels: int = 2  # Number of hierarchy levels
    subgoal_test_perc: float = 0.3  # Percentage of subgoal testing
    atomic_noise: float = 0.5  # Increased noise for atomic actions exploration
    subgoal_noise: float = 0.1  # Increased noise for subgoals exploration
    
    # Goal settings
    goal_dim: int = 3  # Position goal (x, y, z)
    subgoal_bounds: List[float] = dataclasses.field(default_factory=lambda: [-10.0, 10.0])
    max_actions: int = 20  # Max actions per subgoal
    
    # Training parameters
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.98  # Discount factor
    tau: float = 0.05  # Soft update rate
    
    # Replay buffer
    buffer_size: int = 1000000
    her_ratio: float = 0.8  # Hindsight experience replay ratio
    future_k: int = 4  # Number of future states for HER
    
    # Exploration
    noise_eps: float = 0.2
    random_eps: float = 0.3
    noise_decay: float = 0.9999
    
    # Training schedule
    train_freq: int = 1  # Training frequency
    target_update_freq: int = 2  # Target network update frequency
    save_freq: int = 1000  # Model save frequency
    
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
        assert self.num_levels >= 2, "HAC requires at least 2 hierarchy levels"
        assert 0 < self.subgoal_test_perc < 1, "Subgoal test percentage must be between 0 and 1"
        assert self.goal_dim > 0, "Goal dimension must be positive"
        assert len(self.subgoal_bounds) == 2, "Subgoal bounds must be [min, max]"
        assert self.subgoal_bounds[0] < self.subgoal_bounds[1], "Invalid subgoal bounds"