#!/usr/bin/env python
"""
Intrinsic motivation mechanisms for hierarchical reinforcement learning.
"""

import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class IntrinsicMotivation(ABC):
    """Abstract base class for intrinsic motivation mechanisms."""
    
    @abstractmethod
    def compute_reward(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        next_state: np.ndarray,
        **kwargs
    ) -> float:
        """Compute intrinsic reward."""
        pass


class CuriosityDrivenMotivation(IntrinsicMotivation):
    """Curiosity-driven intrinsic motivation based on prediction error."""
    
    def __init__(self, state_history_size: int = 1000):
        self.state_history = []
        self.max_history_size = state_history_size
    
    def compute_reward(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        next_state: np.ndarray,
        **kwargs
    ) -> float:
        """Compute curiosity reward based on state novelty."""
        if len(self.state_history) == 0:
            novelty = 1.0
        else:
            # Find minimum distance to previous states
            distances = [np.linalg.norm(next_state - prev_state) 
                        for prev_state in self.state_history[-100:]]  # Last 100 states
            novelty = min(1.0, min(distances))
        
        # Add to history
        self.state_history.append(next_state.copy())
        if len(self.state_history) > self.max_history_size:
            self.state_history.pop(0)
        
        return novelty


class InformationGainMotivation(IntrinsicMotivation):
    """Information gain based intrinsic motivation."""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.state_action_counts = {}
    
    def compute_reward(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        next_state: np.ndarray,
        **kwargs
    ) -> float:
        """Compute information gain reward."""
        # Discretize state-action for counting
        state_key = tuple(np.round(state[:5], 1))  # Use first 5 dimensions
        action_key = tuple(np.round(action, 1))
        sa_key = (state_key, action_key)
        
        # Count visits
        count = self.state_action_counts.get(sa_key, 0)
        self.state_action_counts[sa_key] = count + 1
        
        # Information gain reward (higher for less visited state-actions)
        info_gain = 1.0 / np.sqrt(count + 1)
        return self.alpha * info_gain


class SkillDiversityMotivation(IntrinsicMotivation):
    """Promote diversity in learned skills/options."""
    
    def __init__(self, num_skills: int, diversity_coef: float = 0.1):
        self.num_skills = num_skills
        self.diversity_coef = diversity_coef
        self.skill_trajectories = {i: [] for i in range(num_skills)}
    
    def compute_reward(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        next_state: np.ndarray,
        skill_id: int = 0,
        **kwargs
    ) -> float:
        """Compute diversity reward for skill learning."""
        if skill_id not in self.skill_trajectories:
            return 0.0
        
        # Add state to skill trajectory
        self.skill_trajectories[skill_id].append(next_state.copy())
        
        # Keep only recent states
        max_trajectory_length = 1000
        if len(self.skill_trajectories[skill_id]) > max_trajectory_length:
            self.skill_trajectories[skill_id].pop(0)
        
        # Compute diversity reward
        diversity_reward = 0.0
        
        # Compare with other skills
        for other_skill in range(self.num_skills):
            if other_skill == skill_id or not self.skill_trajectories[other_skill]:
                continue
            
            # Find minimum distance to other skill's trajectory
            other_states = self.skill_trajectories[other_skill][-100:]  # Recent states
            distances = [np.linalg.norm(next_state - other_state) 
                        for other_state in other_states]
            
            if distances:
                min_distance = min(distances)
                diversity_reward += min_distance
        
        return self.diversity_coef * diversity_reward / max(1, self.num_skills - 1)