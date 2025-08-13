#!/usr/bin/env python
"""
HAC (Hindsight Action Control) Agent Implementation
==================================================

Implements goal-conditioned hierarchical reinforcement learning with hindsight experience replay.
Based on "Learning Multi-Level Hierarchies with Hindsight" (Levy et al., 2017).

Key Features:
- Multi-level hierarchical policy learning
- Hindsight experience replay for sparse rewards
- Subgoal testing for improved learning
- Goal-conditioned value functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import copy
import logging

from ..common.base_hierarchical_agent import BaseHierarchicalAgent, HierarchicalNetwork
from ..common.hierarchical_replay_buffer import HierarchicalReplayBuffer
from .hac_config import HACConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HACLevel:
    """Individual level in HAC hierarchy."""
    
    def __init__(
        self,
        level_id: int,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        config: HACConfig,
        is_lowest_level: bool = False
    ):
        self.level_id = level_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.config = config
        self.is_lowest_level = is_lowest_level
        self.device = torch.device(config.device)
        
        # Input dimension: state + goal
        input_dim = state_dim + goal_dim
        
        # Output dimension: action or subgoal
        output_dim = action_dim if is_lowest_level else goal_dim
        
        
        # Actor network
        self.actor = HierarchicalNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation
        ).to(self.device)
        
        # Critic network (Q-function)
        self.critic = HierarchicalNetwork(
            input_dim=input_dim + output_dim,
            output_dim=1,
            hidden_dims=config.hidden_dims,
            activation=config.activation
        ).to(self.device)
        
        # Target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # Replay buffer
        self.replay_buffer = HierarchicalReplayBuffer(
            capacity=config.buffer_size // config.num_levels,
            state_dim=state_dim,
            action_dim=output_dim,
            goal_dim=goal_dim,
            her_ratio=config.her_ratio,
            future_k=config.future_k
        )
        
        # Noise for exploration
        self.noise_eps = config.noise_eps
        self.action_bounds = None
        if is_lowest_level:
            self.action_bounds = [-1.0, 1.0]  # Normalized action bounds
            # Action scaling for environment compatibility
            self.action_scale = 2.0  # Scale to [-2.0, 2.0] for AirSim environment
        else:
            self.action_bounds = config.subgoal_bounds
    
    def select_action(
        self, 
        state: np.ndarray, 
        goal: np.ndarray,
        deterministic: bool = False,
        test_subgoal: bool = False
    ) -> np.ndarray:
        """Select action or subgoal."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
            
            input_tensor = torch.cat([state_tensor, goal_tensor], dim=-1)
            action = self.actor(input_tensor).cpu().numpy().squeeze()
            
            # Add exploration noise
            if not deterministic and not test_subgoal:
                if self.is_lowest_level:
                    # Increase exploration noise for atomic actions
                    noise = np.random.normal(0, self.config.atomic_noise * 2.0, size=action.shape)
                else:
                    # Increase exploration noise for subgoals
                    noise = np.random.normal(0, self.config.subgoal_noise * 1.5, size=action.shape)
                action += noise
            
            # Clip to bounds
            if self.action_bounds is not None:
                action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
            
            # Apply action scaling for lowest level (atomic actions)
            if self.is_lowest_level and hasattr(self, 'action_scale'):
                action = action * self.action_scale
            
            return action
    
    def get_value(
        self, 
        state: np.ndarray, 
        goal: np.ndarray, 
        action: np.ndarray
    ) -> float:
        """Get Q-value for state-goal-action."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            input_tensor = torch.cat([state_tensor, goal_tensor, action_tensor], dim=-1)
            value = self.critic(input_tensor).cpu().numpy().squeeze()
            
            return value
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update actor and critic networks."""
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        goals = batch['goals'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        batch_size = states.size(0)
        
        # Current Q-values
        current_input = torch.cat([states, goals, actions], dim=-1)
        current_q = self.critic(current_input).squeeze()
        
        # Target Q-values
        with torch.no_grad():
            next_input = torch.cat([next_states, goals], dim=-1)
            next_actions = self.actor_target(next_input)
            
            # Add target policy smoothing noise
            noise = torch.randn_like(next_actions) * 0.2
            noise = torch.clamp(noise, -0.5, 0.5)
            next_actions += noise
            
            if self.action_bounds is not None:
                next_actions = torch.clamp(
                    next_actions, 
                    self.action_bounds[0], 
                    self.action_bounds[1]
                )
            
            next_q_input = torch.cat([next_states, goals, next_actions], dim=-1)
            next_q = self.critic_target(next_q_input).squeeze()
            
            target_q = rewards + (1 - dones.float()) * self.config.gamma * next_q
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor loss (delayed policy update)
        actor_input = torch.cat([states, goals], dim=-1)
        predicted_actions = self.actor(actor_input)
        actor_q_input = torch.cat([states, goals, predicted_actions], dim=-1)
        actor_loss = -self.critic(actor_q_input).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_q.mean().item()
        }
    
    def soft_update(self) -> None:
        """Soft update target networks."""
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data
            )
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data
            )


class HACAgent(BaseHierarchicalAgent):
    """
    HAC (Hindsight Action Control) Agent.
    
    Implements multi-level hierarchical learning with hindsight experience replay.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        config: HACConfig,
        **kwargs
    ):
        super().__init__(state_dim, action_dim, goal_dim, device=config.device, **kwargs)
        
        self.config = config
        self.num_levels = config.num_levels
        
        # Create hierarchy levels
        self.levels = []
        for i in range(self.num_levels):
            is_lowest = (i == self.num_levels - 1)
            level = HACLevel(
                level_id=i,
                state_dim=state_dim,
                action_dim=action_dim if is_lowest else goal_dim,
                goal_dim=goal_dim,
                config=config,
                is_lowest_level=is_lowest
            )
            self.levels.append(level)
        
        # Training state
        self.training_step = 0
        self.episode_count = 0
        
        # Subgoal testing
        self.subgoal_test_episodes = []
        
        logger.info(f"Initialized HAC agent with {self.num_levels} levels")
    
    def select_action(
        self, 
        state: np.ndarray, 
        goal: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select action using hierarchical policy.
        
        Returns action and additional info including subgoals.
        """
        info = {'subgoals': []}
        
        current_state = state.copy()
        current_goal = goal.copy()
        
        # Forward pass through hierarchy
        for level_id in range(self.num_levels):
            level = self.levels[level_id]
            
            # Check if this is subgoal testing episode
            test_subgoal = (
                level_id < self.num_levels - 1 and
                self.episode_count in self.subgoal_test_episodes
            )
            
            action_or_subgoal = level.select_action(
                current_state, current_goal, deterministic, test_subgoal
            )
            
            if level_id < self.num_levels - 1:
                # This is a subgoal
                info['subgoals'].append(action_or_subgoal.copy())
                current_goal = action_or_subgoal
            else:
                # This is the final action
                final_action = action_or_subgoal
        
        return final_action, info
    
    def update(
        self, 
        batch: Dict[str, torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Update all levels in the hierarchy."""
        total_losses = {}
        
        for level_id, level in enumerate(self.levels):
            if level.replay_buffer.can_sample(self.config.batch_size):
                level_batch = level.replay_buffer.sample_low_level(self.config.batch_size)
                level_losses = level.update(level_batch)
                
                # Add level prefix to loss names
                for key, value in level_losses.items():
                    total_losses[f'level_{level_id}_{key}'] = value
        
        self.training_step += 1
        return total_losses
    
    def store_transition(
        self,
        level_id: int,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        goal: np.ndarray,
        done: bool,
        **kwargs
    ) -> None:
        """Store transition for specific level."""
        if 0 <= level_id < self.num_levels:
            intrinsic_reward = kwargs.get('intrinsic_reward', reward)
            self.levels[level_id].replay_buffer.store_low_level(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                goal=goal,
                done=done,
                intrinsic_reward=intrinsic_reward
            )
    
    def end_episode(self) -> None:
        """Called at the end of each episode."""
        # Store episodes in replay buffers with HER
        for level in self.levels:
            level.replay_buffer.store_episode()
        
        self.episode_count += 1
        
        # Schedule subgoal testing episodes
        if np.random.random() < self.config.subgoal_test_perc:
            self.subgoal_test_episodes.append(self.episode_count)
    
    def compute_intrinsic_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        subgoal: np.ndarray,
        level_id: int = 0,
        **kwargs
    ) -> float:
        """Compute intrinsic reward for hierarchical learning."""
        if level_id >= self.num_levels - 1:
            # Lowest level uses external reward
            return 0.0
        
        # Goal-conditioned intrinsic reward
        achieved_goal = next_state[:self.goal_dim]
        goal_distance = np.linalg.norm(achieved_goal - subgoal)
        
        # Dense reward based on distance to subgoal
        intrinsic_reward = -goal_distance
        
        # Bonus for achieving subgoal
        if goal_distance < 1.0:  # Achievement threshold
            intrinsic_reward += 10.0
        
        return intrinsic_reward
    
    def is_subgoal_achieved(
        self,
        state: np.ndarray,
        subgoal: np.ndarray,
        threshold: float = 1.0
    ) -> bool:
        """Check if subgoal is achieved."""
        achieved_goal = state[:self.goal_dim]
        distance = np.linalg.norm(achieved_goal - subgoal)
        return distance <= threshold
    
    def save(self, filepath: str) -> None:
        """Save all levels to file."""
        save_dict = {
            'config': self.config,
            'training_step': self.training_step,
            'episode_count': self.episode_count
        }
        
        # Save each level
        for level_id, level in enumerate(self.levels):
            level_dict = {
                'actor': level.actor.state_dict(),
                'critic': level.critic.state_dict(),
                'actor_target': level.actor_target.state_dict(),
                'critic_target': level.critic_target.state_dict(),
                'actor_optimizer': level.actor_optimizer.state_dict(),
                'critic_optimizer': level.critic_optimizer.state_dict()
            }
            save_dict[f'level_{level_id}'] = level_dict
        
        torch.save(save_dict, filepath)
        logger.info(f"HAC agent saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load all levels from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        
        # Load each level
        for level_id, level in enumerate(self.levels):
            level_key = f'level_{level_id}'
            if level_key in checkpoint:
                level_dict = checkpoint[level_key]
                level.actor.load_state_dict(level_dict['actor'])
                level.critic.load_state_dict(level_dict['critic'])
                level.actor_target.load_state_dict(level_dict['actor_target'])
                level.critic_target.load_state_dict(level_dict['critic_target'])
                level.actor_optimizer.load_state_dict(level_dict['actor_optimizer'])
                level.critic_optimizer.load_state_dict(level_dict['critic_optimizer'])
        
        logger.info(f"HAC agent loaded from {filepath}")
    
    def get_stats(self) -> Dict[str, List]:
        """Get training statistics from all levels."""
        combined_stats = dict(self.stats)
        
        for level_id, level in enumerate(self.levels):
            buffer_size = len(level.replay_buffer)
            combined_stats[f'level_{level_id}_buffer_size'] = [buffer_size]
        
        return combined_stats