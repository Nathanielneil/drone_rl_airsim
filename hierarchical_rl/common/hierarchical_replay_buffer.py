#!/usr/bin/env python
"""
Hierarchical replay buffer for storing multi-level experiences.
Supports goal relabeling and hindsight experience replay.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import collections
import random


class HierarchicalReplayBuffer:
    """
    Replay buffer designed for hierarchical reinforcement learning.
    
    Features:
    - Multi-level experience storage (high-level and low-level)
    - Goal relabeling for hindsight experience replay
    - Temporal abstraction support
    - Intrinsic reward computation
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        max_episode_length: int = 1000,
        her_ratio: float = 0.8,
        future_k: int = 4
    ):
        """
        Initialize hierarchical replay buffer.
        
        Args:
            capacity: Maximum buffer size
            state_dim: State dimension
            action_dim: Action dimension
            goal_dim: Goal dimension
            max_episode_length: Maximum episode length
            her_ratio: Ratio of HER samples
            future_k: Number of future states for goal relabeling
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.max_episode_length = max_episode_length
        self.her_ratio = her_ratio
        self.future_k = future_k
        
        # Low-level experience buffer
        self.low_level_buffer = {
            'states': np.zeros((capacity, state_dim), dtype=np.float32),
            'actions': np.zeros((capacity, action_dim), dtype=np.float32),
            'rewards': np.zeros(capacity, dtype=np.float32),
            'next_states': np.zeros((capacity, state_dim), dtype=np.float32),
            'goals': np.zeros((capacity, goal_dim), dtype=np.float32),
            'dones': np.zeros(capacity, dtype=bool),
            'intrinsic_rewards': np.zeros(capacity, dtype=np.float32)
        }
        
        # High-level experience buffer
        self.high_level_buffer = {
            'states': np.zeros((capacity, state_dim), dtype=np.float32),
            'subgoals': np.zeros((capacity, goal_dim), dtype=np.float32),
            'rewards': np.zeros(capacity, dtype=np.float32),
            'next_states': np.zeros((capacity, state_dim), dtype=np.float32),
            'final_goals': np.zeros((capacity, goal_dim), dtype=np.float32),
            'dones': np.zeros(capacity, dtype=bool),
            'episode_lengths': np.zeros(capacity, dtype=int)
        }
        
        # Episode storage for HER
        self.episode_buffer = []
        self.current_episode = {
            'states': [],
            'actions': [], 
            'rewards': [],
            'goals': [],
            'achieved_goals': [],
            'infos': []
        }
        
        self.low_ptr = 0
        self.high_ptr = 0
        self.low_size = 0
        self.high_size = 0
        
    def store_low_level(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        goal: np.ndarray,
        done: bool,
        intrinsic_reward: float = 0.0
    ) -> None:
        """Store low-level experience."""
        self.low_level_buffer['states'][self.low_ptr] = state
        self.low_level_buffer['actions'][self.low_ptr] = action
        self.low_level_buffer['rewards'][self.low_ptr] = reward
        self.low_level_buffer['next_states'][self.low_ptr] = next_state
        self.low_level_buffer['goals'][self.low_ptr] = goal
        self.low_level_buffer['dones'][self.low_ptr] = done
        self.low_level_buffer['intrinsic_rewards'][self.low_ptr] = intrinsic_reward
        
        self.low_ptr = (self.low_ptr + 1) % self.capacity
        self.low_size = min(self.low_size + 1, self.capacity)
    
    def store_high_level(
        self,
        state: np.ndarray,
        subgoal: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        final_goal: np.ndarray,
        done: bool,
        episode_length: int = 1
    ) -> None:
        """Store high-level experience."""
        self.high_level_buffer['states'][self.high_ptr] = state
        self.high_level_buffer['subgoals'][self.high_ptr] = subgoal
        self.high_level_buffer['rewards'][self.high_ptr] = reward
        self.high_level_buffer['next_states'][self.high_ptr] = next_state
        self.high_level_buffer['final_goals'][self.high_ptr] = final_goal
        self.high_level_buffer['dones'][self.high_ptr] = done
        self.high_level_buffer['episode_lengths'][self.high_ptr] = episode_length
        
        self.high_ptr = (self.high_ptr + 1) % self.capacity
        self.high_size = min(self.high_size + 1, self.capacity)
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        goal: np.ndarray,
        done: bool,
        info: Dict[str, Any] = None
    ) -> None:
        """Store transition for episode-based HER."""
        self.current_episode['states'].append(state.copy())
        self.current_episode['actions'].append(action.copy())
        self.current_episode['rewards'].append(reward)
        self.current_episode['goals'].append(goal.copy())
        self.current_episode['achieved_goals'].append(next_state[:self.goal_dim].copy())
        self.current_episode['infos'].append(info or {})
    
    def store_episode(self) -> None:
        """Store completed episode and generate HER samples."""
        if len(self.current_episode['states']) == 0:
            return
        
        episode = dict(self.current_episode)
        self.episode_buffer.append(episode)
        
        # Apply HER to the episode
        self._apply_hindsight_experience_replay(episode)
        
        # Clear current episode
        for key in self.current_episode:
            self.current_episode[key].clear()
        
        # Maintain episode buffer size
        if len(self.episode_buffer) > 1000:  # Keep last 1000 episodes
            self.episode_buffer.pop(0)
    
    def _apply_hindsight_experience_replay(self, episode: Dict[str, List]) -> None:
        """Apply HER to generate additional training samples."""
        episode_length = len(episode['states'])
        if episode_length == 0:
            return
        
        # Store original episode
        for t in range(episode_length - 1):
            self.store_low_level(
                state=episode['states'][t],
                action=episode['actions'][t],
                reward=episode['rewards'][t],
                next_state=episode['states'][t + 1],
                goal=episode['goals'][t],
                done=(t == episode_length - 2),
                intrinsic_reward=0.0
            )
        
        # Generate HER samples
        her_samples = int(episode_length * self.her_ratio)
        for _ in range(her_samples):
            t = np.random.randint(0, episode_length - 1)
            
            # Sample future state as goal
            future_t = np.random.randint(t, episode_length)
            relabeled_goal = episode['achieved_goals'][future_t]
            
            # Compute relabeled reward
            achieved_goal = episode['achieved_goals'][t + 1]
            relabeled_reward = self._compute_reward(achieved_goal, relabeled_goal)
            
            # Store HER sample
            self.store_low_level(
                state=episode['states'][t],
                action=episode['actions'][t],
                reward=relabeled_reward,
                next_state=episode['states'][t + 1],
                goal=relabeled_goal,
                done=self._is_goal_achieved(achieved_goal, relabeled_goal),
                intrinsic_reward=relabeled_reward
            )
    
    def _compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """Compute reward based on goal achievement."""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return -distance  # Dense reward
    
    def _is_goal_achieved(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """Check if goal is achieved."""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance < 0.5  # Achievement threshold
    
    def sample_low_level(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch for low-level policy training."""
        if self.low_size == 0:
            return {}
        
        indices = np.random.randint(0, self.low_size, size=batch_size)
        
        batch = {}
        for key, values in self.low_level_buffer.items():
            if key in ['dones']:
                batch[key] = torch.BoolTensor(values[indices])
            else:
                batch[key] = torch.FloatTensor(values[indices])
        
        return batch
    
    def sample_high_level(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch for high-level policy training."""
        if self.high_size == 0:
            return {}
        
        indices = np.random.randint(0, self.high_size, size=batch_size)
        
        batch = {}
        for key, values in self.high_level_buffer.items():
            if key in ['dones']:
                batch[key] = torch.BoolTensor(values[indices])
            elif key in ['episode_lengths']:
                batch[key] = torch.LongTensor(values[indices])
            else:
                batch[key] = torch.FloatTensor(values[indices])
        
        return batch
    
    def sample_mixed(
        self, 
        low_batch_size: int, 
        high_batch_size: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Sample both low-level and high-level batches."""
        low_batch = self.sample_low_level(low_batch_size)
        high_batch = self.sample_high_level(high_batch_size)
        return low_batch, high_batch
    
    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.low_size >= batch_size
    
    def __len__(self) -> int:
        """Return buffer size."""
        return self.low_size
    
    def clear(self) -> None:
        """Clear all buffers."""
        self.low_ptr = 0
        self.high_ptr = 0
        self.low_size = 0
        self.high_size = 0
        self.episode_buffer.clear()
        for key in self.current_episode:
            self.current_episode[key].clear()


class PrioritizedHierarchicalReplayBuffer(HierarchicalReplayBuffer):
    """
    Prioritized version of hierarchical replay buffer.
    Uses TD error for sample prioritization.
    """
    
    def __init__(self, *args, alpha: float = 0.6, beta: float = 0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        
        # Priority trees for efficient sampling
        self.low_priorities = np.ones(self.capacity, dtype=np.float32)
        self.high_priorities = np.ones(self.capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def store_low_level(self, *args, priority: float = None, **kwargs) -> None:
        """Store low-level experience with priority."""
        super().store_low_level(*args, **kwargs)
        
        if priority is None:
            priority = self.max_priority
        
        self.low_priorities[self.low_ptr - 1] = priority
        self.max_priority = max(self.max_priority, priority)
    
    def sample_low_level(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch using prioritized sampling."""
        if self.low_size == 0:
            return {}
        
        # Compute sampling probabilities
        priorities = self.low_priorities[:self.low_size] ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.low_size, batch_size, p=probabilities)
        
        # Compute importance sampling weights
        weights = (self.low_size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Create batch
        batch = {}
        for key, values in self.low_level_buffer.items():
            if key in ['dones']:
                batch[key] = torch.BoolTensor(values[indices])
            else:
                batch[key] = torch.FloatTensor(values[indices])
        
        batch['indices'] = torch.LongTensor(indices)
        batch['weights'] = torch.FloatTensor(weights)
        
        return batch
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update sample priorities."""
        for idx, priority in zip(indices, priorities):
            self.low_priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)