"""
HAC (Hindsight Action Control) Implementation
============================================

Goal-conditioned hierarchical reinforcement learning with hindsight experience replay.
Based on "Learning Multi-Level Hierarchies with Hindsight" (Levy et al., 2017).
"""

from .hac_agent import HACAgent
from .hac_config import HACConfig

__all__ = ["HACAgent", "HACConfig"]