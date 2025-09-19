#!/usr/bin/env python
"""
Goal generation strategies for hierarchical reinforcement learning.
"""

import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


class GoalGenerator(ABC):
    """Abstract base class for goal generation strategies."""
    
    @abstractmethod
    def generate_goal(self, state: np.ndarray) -> np.ndarray:
        """Generate a goal given the current state."""
        pass


class RandomGoalGenerator(GoalGenerator):
    """Random goal generator within specified bounds."""
    
    def __init__(self, goal_bounds: Tuple[float, float], goal_dim: int = 3):
        self.goal_bounds = goal_bounds
        self.goal_dim = goal_dim
    
    def generate_goal(self, state: np.ndarray) -> np.ndarray:
        """Generate random goal within bounds."""
        return np.random.uniform(
            self.goal_bounds[0], 
            self.goal_bounds[1], 
            size=self.goal_dim
        ).astype(np.float32)


class CurriculumGoalGenerator(GoalGenerator):
    """Curriculum-based goal generator with increasing difficulty."""
    
    def __init__(self, initial_radius: float = 5.0, max_radius: float = 20.0, goal_dim: int = 3):
        self.initial_radius = initial_radius
        self.max_radius = max_radius
        self.goal_dim = goal_dim
        self.current_level = 1
        self.success_rate = 0.0
    
    def generate_goal(self, state: np.ndarray) -> np.ndarray:
        """Generate goal based on curriculum level."""
        current_radius = min(
            self.max_radius,
            self.initial_radius * self.current_level
        )
        
        # Random direction
        direction = np.random.randn(self.goal_dim)
        direction /= np.linalg.norm(direction)
        
        # Random distance within curriculum radius
        distance = np.random.uniform(self.initial_radius, current_radius)
        
        goal = state[:self.goal_dim] + direction * distance
        return goal.astype(np.float32)
    
    def update_curriculum(self, success_rate: float):
        """Update curriculum based on success rate."""
        self.success_rate = success_rate
        if success_rate > 0.8 and self.current_level < 5:
            self.current_level += 1
        elif success_rate < 0.3 and self.current_level > 1:
            self.current_level -= 1