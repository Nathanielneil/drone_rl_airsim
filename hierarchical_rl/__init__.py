"""
Hierarchical Reinforcement Learning Suite for Drone Navigation
============================================================

This package implements various hierarchical reinforcement learning algorithms
specifically designed for UAV autonomous navigation in AirSim environments.

Available Algorithms:
- HAC (Hindsight Action Control): Goal-conditioned hierarchical learning
- FuN (FeUdal Networks): Manager-worker feudal architecture
- HIRO (HIerarchical RL with Off-policy correction): Off-policy hierarchical learning
- Options Framework: Semi-Markov decision process with temporal abstractions

Author: Drone RL Team
Email: guowei_ni@bit.edu.cn
"""

__version__ = "1.0.0"
__author__ = "Drone RL Team"

from .common.base_hierarchical_agent import BaseHierarchicalAgent
from .common.hierarchical_replay_buffer import HierarchicalReplayBuffer
from .envs.hierarchical_airsim_env import HierarchicalAirSimEnv

__all__ = [
    "BaseHierarchicalAgent",
    "HierarchicalReplayBuffer", 
    "HierarchicalAirSimEnv"
]