"""
HIRO (HIerarchical RL with Off-policy correction) Implementation
==============================================================

Data-efficient hierarchical reinforcement learning with goal relabeling.
Based on "Data-Efficient Hierarchical Reinforcement Learning" (Nachum et al., 2018).
"""

from .hiro_agent import HIROAgent
from .hiro_config import HIROConfig

__all__ = ["HIROAgent", "HIROConfig"]