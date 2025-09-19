"""
FuN (FeUdal Networks) Implementation
===================================

Manager-worker hierarchical architecture with intrinsic motivation.
Based on "FeUdal Networks for Hierarchical Reinforcement Learning" (Vezhnevets et al., 2017).
"""

from .fun_agent import FuNAgent
from .fun_config import FuNConfig

__all__ = ["FuNAgent", "FuNConfig"]