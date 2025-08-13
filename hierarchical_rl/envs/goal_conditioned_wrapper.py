#!/usr/bin/env python
"""
Goal-conditioned wrapper for environments.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any


class GoalConditionedWrapper(gym.Wrapper):
    """
    Wrapper that adds goal-conditioning to any environment.
    
    Transforms observations to include achieved and desired goals,
    following the HER (Hindsight Experience Replay) convention.
    """
    
    def __init__(
        self,
        env: gym.Env,
        goal_dim: int = 3,
        goal_bounds: Tuple[float, float] = (-10.0, 10.0),
        goal_threshold: float = 1.0,
        sparse_reward: bool = False
    ):
        """
        Initialize goal-conditioned wrapper.
        
        Args:
            env: Base environment
            goal_dim: Dimension of goal space
            goal_bounds: Bounds for random goal generation
            goal_threshold: Distance threshold for goal achievement
            sparse_reward: Whether to use sparse (0/1) or dense rewards
        """
        super().__init__(env)
        
        self.goal_dim = goal_dim
        self.goal_bounds = np.array(goal_bounds)
        self.goal_threshold = goal_threshold
        self.sparse_reward = sparse_reward
        
        # Store original spaces
        self.original_observation_space = env.observation_space
        self.original_action_space = env.action_space
        
        # Create goal-conditioned observation space
        self.observation_space = spaces.Dict({
            'observation': self.original_observation_space,
            'achieved_goal': spaces.Box(-np.inf, np.inf, (goal_dim,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, (goal_dim,), dtype=np.float32)
        })
        
        # Current goal
        self.current_goal = None
        
    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and sample new goal."""
        obs, info = self.env.reset(**kwargs)
        
        # Sample new goal
        self.current_goal = self._sample_goal(obs)
        
        # Create goal-conditioned observation
        achieved_goal = self._extract_achieved_goal(obs)
        
        goal_obs = {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': self.current_goal.copy()
        }
        
        # Add goal information to info
        info['goal'] = self.current_goal.copy()
        
        return goal_obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute action and compute goal-conditioned reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract achieved goal
        achieved_goal = self._extract_achieved_goal(obs)
        
        # Compute goal-conditioned reward
        goal_reward = self._compute_goal_reward(achieved_goal, self.current_goal)
        
        # Check goal achievement
        goal_achieved = self._is_goal_achieved(achieved_goal, self.current_goal)
        
        # Create goal-conditioned observation
        goal_obs = {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': self.current_goal.copy()
        }
        
        # Update info
        info.update({
            'goal': self.current_goal.copy(),
            'achieved_goal': achieved_goal.copy(),
            'goal_achieved': goal_achieved,
            'goal_distance': np.linalg.norm(achieved_goal - self.current_goal),
            'original_reward': reward,
            'goal_reward': goal_reward
        })
        
        # Use goal reward or combine with original reward
        total_reward = goal_reward if self.sparse_reward else (reward + goal_reward)
        
        return goal_obs, total_reward, terminated, truncated, info
    
    def _sample_goal(self, obs: np.ndarray) -> np.ndarray:
        """Sample a random goal."""
        return np.random.uniform(
            self.goal_bounds[0], 
            self.goal_bounds[1], 
            size=self.goal_dim
        ).astype(np.float32)
    
    def _extract_achieved_goal(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract achieved goal from observation.
        
        This is environment-specific and should be overridden.
        Default: use first goal_dim elements of observation.
        """
        if isinstance(obs, dict):
            # If observation is dict, try to find position information
            if 'position' in obs:
                return obs['position'][:self.goal_dim].astype(np.float32)
            elif 'inform_vector' in obs:
                return obs['inform_vector'][:self.goal_dim].astype(np.float32)
            else:
                # Flatten and use first elements
                flat_obs = np.concatenate([v.flatten() for v in obs.values()])
                return flat_obs[:self.goal_dim].astype(np.float32)
        else:
            # Simple array observation
            flat_obs = obs.flatten()
            return flat_obs[:self.goal_dim].astype(np.float32)
    
    def _compute_goal_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """Compute reward based on goal achievement."""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        
        if self.sparse_reward:
            return 1.0 if distance <= self.goal_threshold else 0.0
        else:
            # Dense reward: negative distance with achievement bonus
            reward = -distance
            if distance <= self.goal_threshold:
                reward += 10.0
            return reward
    
    def _is_goal_achieved(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """Check if goal is achieved."""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance <= self.goal_threshold
    
    def set_goal(self, goal: np.ndarray) -> None:
        """Set a specific goal."""
        self.current_goal = goal.astype(np.float32)
    
    def get_goal(self) -> np.ndarray:
        """Get current goal."""
        return self.current_goal.copy() if self.current_goal is not None else np.zeros(self.goal_dim)
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict) -> float:
        """
        Compute reward for HER-style goal relabeling.
        
        This method is used by HER to compute rewards for relabeled goals.
        """
        return self._compute_goal_reward(achieved_goal, desired_goal)
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render environment."""
        return self.env.render(mode)