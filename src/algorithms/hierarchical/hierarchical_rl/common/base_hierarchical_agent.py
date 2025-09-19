#!/usr/bin/env python
"""
Base class for hierarchical reinforcement learning agents.
Provides common interface and utilities for all HRL algorithms.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseHierarchicalAgent(ABC):
    """
    Abstract base class for hierarchical RL agents.
    
    Defines the common interface and shared functionality for all HRL algorithms
    including HAC, FuN, HIRO, and Options Framework.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        max_action: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Initialize the hierarchical agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space  
            goal_dim: Dimension of goal space
            max_action: Maximum action value
            device: Computing device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.max_action = max_action
        self.device = torch.device(device)
        
        # Initialize networks (to be defined by subclasses)
        self.high_level_policy = None
        self.low_level_policy = None
        
        # Training statistics
        self.stats = {
            'high_level_losses': [],
            'low_level_losses': [],
            'success_rates': [],
            'episode_rewards': [],
            'intrinsic_rewards': [],
            'goal_achievement_rates': []
        }
        
        logger.info(f"Initialized {self.__class__.__name__} on {self.device}")
    
    @abstractmethod
    def select_action(
        self, 
        state: np.ndarray, 
        goal: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select action using hierarchical policy.
        
        Args:
            state: Current state
            goal: Current goal
            deterministic: Whether to use deterministic action selection
            
        Returns:
            action: Selected action
            info: Additional information (subgoals, values, etc.)
        """
        pass
    
    @abstractmethod
    def update(
        self, 
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, float]:
        """
        Update agent parameters using a batch of experiences.
        
        Args:
            batch: Batch of experiences
            
        Returns:
            Dictionary of loss values and metrics
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save agent parameters to file."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load agent parameters from file."""
        pass
    
    def generate_subgoal(
        self, 
        state: np.ndarray, 
        final_goal: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Generate intermediate subgoal.
        
        Args:
            state: Current state
            final_goal: Final goal
            
        Returns:
            Generated subgoal
        """
        if self.high_level_policy is None:
            # Default: linear interpolation towards final goal
            alpha = kwargs.get('alpha', 0.5)
            return state + alpha * (final_goal - state)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            goal_tensor = torch.FloatTensor(final_goal).unsqueeze(0).to(self.device)
            subgoal = self.high_level_policy(state_tensor, goal_tensor)
            return subgoal.cpu().numpy().squeeze()
    
    def compute_intrinsic_reward(
        self,
        state: np.ndarray,
        action: np.ndarray, 
        next_state: np.ndarray,
        subgoal: np.ndarray,
        **kwargs
    ) -> float:
        """
        Compute intrinsic reward for low-level policy.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            subgoal: Current subgoal
            
        Returns:
            Intrinsic reward value
        """
        # Default: negative distance to subgoal
        goal_distance = np.linalg.norm(next_state[:self.goal_dim] - subgoal)
        return -goal_distance
    
    def is_subgoal_achieved(
        self,
        state: np.ndarray,
        subgoal: np.ndarray,
        threshold: float = 1.0
    ) -> bool:
        """
        Check if subgoal is achieved.
        
        Args:
            state: Current state
            subgoal: Target subgoal
            threshold: Achievement threshold
            
        Returns:
            True if subgoal is achieved
        """
        distance = np.linalg.norm(state[:self.goal_dim] - subgoal)
        return distance <= threshold
    
    def update_stats(self, **kwargs) -> None:
        """Update training statistics."""
        for key, value in kwargs.items():
            if key in self.stats:
                self.stats[key].append(value)
    
    def get_stats(self) -> Dict[str, List]:
        """Get training statistics."""
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset training statistics."""
        for key in self.stats:
            self.stats[key].clear()


class HierarchicalNetwork(nn.Module):
    """
    Base network architecture for hierarchical policies.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer with tanh activation for bounded actions
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Tanh()  # Bounded output for stable training
        ])
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Linear):
            # Use proper gain for better initialization
            if hasattr(m, 'out_features') and m.out_features <= 10:  # Output layer
                nn.init.orthogonal_(m.weight, gain=0.1)  # Small gain for output
            else:  # Hidden layers
                nn.init.orthogonal_(m.weight, gain=1.0)  # Standard gain for hidden layers
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)