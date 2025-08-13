#!/usr/bin/env python
"""
HIRO (HIerarchical RL with Off-policy correction) Agent Implementation
=====================================================================

Data-efficient hierarchical reinforcement learning with goal relabeling.
Based on "Data-Efficient Hierarchical Reinforcement Learning" (Nachum et al., 2018).

Key Features:
- Two-level hierarchy with high-level and low-level policies
- Off-policy correction for improved sample efficiency
- Goal relabeling with hindsight experience replay
- Continuous action spaces for both levels
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
from .hiro_config import HIROConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DDPGActor(nn.Module):
    """DDPG Actor network for continuous control."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], max_action: float = 1.0):
        super().__init__()
        
        self.max_action = max_action
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.network(x)


class DDPGCritic(nn.Module):
    """DDPG Critic network for Q-value estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0], 1)
        )
        
        # Q2 network (for TD3-style double Q-learning)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0], 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


class HIROReplayBuffer:
    """Specialized replay buffer for HIRO with off-policy correction."""
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        subgoal_dim: int,
        her_ratio: float = 0.8
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.subgoal_dim = subgoal_dim
        self.her_ratio = her_ratio
        
        # High-level buffer
        self.high_level_buffer = {
            'states': np.zeros((capacity, state_dim), dtype=np.float32),
            'subgoals': np.zeros((capacity, subgoal_dim), dtype=np.float32),
            'rewards': np.zeros(capacity, dtype=np.float32),
            'next_states': np.zeros((capacity, state_dim), dtype=np.float32),
            'dones': np.zeros(capacity, dtype=bool)
        }
        
        # Low-level buffer
        self.low_level_buffer = {
            'states': np.zeros((capacity, state_dim), dtype=np.float32),
            'actions': np.zeros((capacity, action_dim), dtype=np.float32),
            'rewards': np.zeros(capacity, dtype=np.float32),
            'next_states': np.zeros((capacity, state_dim), dtype=np.float32),
            'subgoals': np.zeros((capacity, subgoal_dim), dtype=np.float32),
            'dones': np.zeros(capacity, dtype=bool)
        }
        
        self.high_ptr = 0
        self.low_ptr = 0
        self.high_size = 0
        self.low_size = 0
        
        # Episode storage for HER
        self.episodes = []
        self.current_episode = {'states': [], 'actions': [], 'rewards': [], 'subgoals': []}
    
    def store_high_level(
        self,
        state: np.ndarray,
        subgoal: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store high-level transition."""
        self.high_level_buffer['states'][self.high_ptr] = state
        self.high_level_buffer['subgoals'][self.high_ptr] = subgoal
        self.high_level_buffer['rewards'][self.high_ptr] = reward
        self.high_level_buffer['next_states'][self.high_ptr] = next_state
        self.high_level_buffer['dones'][self.high_ptr] = done
        
        self.high_ptr = (self.high_ptr + 1) % self.capacity
        self.high_size = min(self.high_size + 1, self.capacity)
    
    def store_low_level(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        subgoal: np.ndarray,
        done: bool
    ) -> None:
        """Store low-level transition."""
        self.low_level_buffer['states'][self.low_ptr] = state
        self.low_level_buffer['actions'][self.low_ptr] = action
        self.low_level_buffer['rewards'][self.low_ptr] = reward
        self.low_level_buffer['next_states'][self.low_ptr] = next_state
        self.low_level_buffer['subgoals'][self.low_ptr] = subgoal
        self.low_level_buffer['dones'][self.low_ptr] = done
        
        self.low_ptr = (self.low_ptr + 1) % self.capacity
        self.low_size = min(self.low_size + 1, self.capacity)
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        subgoal: np.ndarray
    ) -> None:
        """Store transition for HER processing."""
        self.current_episode['states'].append(state.copy())
        self.current_episode['actions'].append(action.copy())
        self.current_episode['rewards'].append(reward)
        self.current_episode['subgoals'].append(subgoal.copy())
    
    def end_episode(self) -> None:
        """Process episode with HER."""
        if len(self.current_episode['states']) > 0:
            self.episodes.append(dict(self.current_episode))
            self._apply_her(self.current_episode)
            
            # Clear current episode
            for key in self.current_episode:
                self.current_episode[key].clear()
            
            # Maintain episode buffer size
            if len(self.episodes) > 1000:
                self.episodes.pop(0)
    
    def _apply_her(self, episode: Dict[str, List]) -> None:
        """Apply hindsight experience replay."""
        episode_length = len(episode['states'])
        if episode_length <= 1:
            return
        
        # Generate HER samples
        num_her_samples = int(episode_length * self.her_ratio)
        for _ in range(num_her_samples):
            t = np.random.randint(0, episode_length - 1)
            
            # Sample future achieved goal as relabeled subgoal
            future_t = np.random.randint(t + 1, episode_length)
            achieved_goal = episode['states'][future_t][:self.subgoal_dim]
            
            # Compute relabeled reward (dense reward based on distance)
            current_pos = episode['states'][t + 1][:self.subgoal_dim]
            relabeled_reward = -np.linalg.norm(current_pos - achieved_goal)
            
            # Store HER sample in low-level buffer
            self.store_low_level(
                state=episode['states'][t],
                action=episode['actions'][t],
                reward=relabeled_reward,
                next_state=episode['states'][t + 1],
                subgoal=achieved_goal,
                done=(t == episode_length - 2)
            )
    
    def sample_high_level(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample high-level batch."""
        if self.high_size < batch_size:
            return None
        
        indices = np.random.randint(0, self.high_size, size=batch_size)
        
        batch = {}
        for key, values in self.high_level_buffer.items():
            if key == 'dones':
                batch[key] = torch.BoolTensor(values[indices])
            else:
                batch[key] = torch.FloatTensor(values[indices])
        
        return batch
    
    def sample_low_level(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample low-level batch."""
        if self.low_size < batch_size:
            return None
        
        indices = np.random.randint(0, self.low_size, size=batch_size)
        
        batch = {}
        for key, values in self.low_level_buffer.items():
            if key == 'dones':
                batch[key] = torch.BoolTensor(values[indices])
            else:
                batch[key] = torch.FloatTensor(values[indices])
        
        return batch


class HIROAgent(BaseHierarchicalAgent):
    """
    HIRO (HIerarchical RL with Off-policy correction) Agent.
    
    Implements two-level hierarchy with off-policy correction and goal relabeling.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: HIROConfig,
        **kwargs
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=config.subgoal_dim,
            max_action=1.0,
            device=config.device,
            **kwargs
        )
        
        self.config = config
        self.subgoal_dim = config.subgoal_dim
        
        # High-level policy (generates subgoals)
        self.high_level_actor = DDPGActor(
            input_dim=state_dim,
            output_dim=config.subgoal_dim,
            hidden_dims=config.high_level_hidden_dims,
            max_action=config.subgoal_scale
        ).to(self.device)
        
        self.high_level_critic = DDPGCritic(
            state_dim=state_dim,
            action_dim=config.subgoal_dim,
            hidden_dims=config.high_level_hidden_dims
        ).to(self.device)
        
        # Low-level policy (executes actions)
        self.low_level_actor = DDPGActor(
            input_dim=state_dim + config.subgoal_dim,
            output_dim=action_dim,
            hidden_dims=config.low_level_hidden_dims,
            max_action=1.0
        ).to(self.device)
        
        self.low_level_critic = DDPGCritic(
            state_dim=state_dim + config.subgoal_dim,
            action_dim=action_dim,
            hidden_dims=config.low_level_hidden_dims
        ).to(self.device)
        
        # Target networks
        self.high_level_actor_target = copy.deepcopy(self.high_level_actor)
        self.high_level_critic_target = copy.deepcopy(self.high_level_critic)
        self.low_level_actor_target = copy.deepcopy(self.low_level_actor)
        self.low_level_critic_target = copy.deepcopy(self.low_level_critic)
        
        # Optimizers
        self.high_level_actor_optimizer = optim.Adam(self.high_level_actor.parameters(), lr=config.high_level_lr)
        self.high_level_critic_optimizer = optim.Adam(self.high_level_critic.parameters(), lr=config.high_level_lr)
        self.low_level_actor_optimizer = optim.Adam(self.low_level_actor.parameters(), lr=config.low_level_lr)
        self.low_level_critic_optimizer = optim.Adam(self.low_level_critic.parameters(), lr=config.low_level_lr)
        
        # Replay buffer
        self.replay_buffer = HIROReplayBuffer(
            capacity=config.buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            subgoal_dim=config.subgoal_dim,
            her_ratio=config.her_ratio
        )
        
        # Training state
        self.training_step = 0
        self.current_subgoal = None
        self.subgoal_step = 0
        self.episode_step = 0
        
        # Noise levels
        self.high_level_noise = config.high_level_noise
        self.low_level_noise = config.low_level_noise
        
        logger.info("Initialized HIRO agent")
    
    def select_action(
        self,
        state: np.ndarray,
        goal: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using hierarchical policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # High-level policy (generate subgoal every k steps)
            if self.subgoal_step <= 0 or self.current_subgoal is None:
                subgoal = self.high_level_actor(state_tensor).cpu().numpy().squeeze()
                
                # Add exploration noise
                if not deterministic:
                    noise = np.random.normal(0, self.high_level_noise, size=subgoal.shape)
                    subgoal += noise
                
                # Convert relative subgoal to absolute
                current_pos = state[:self.subgoal_dim]
                self.current_subgoal = current_pos + subgoal
                self.subgoal_step = self.config.subgoal_freq
            
            # Low-level policy (select action to reach subgoal)
            low_level_input = torch.cat([
                state_tensor,
                torch.FloatTensor(self.current_subgoal).unsqueeze(0).to(self.device)
            ], dim=-1)
            
            action = self.low_level_actor(low_level_input).cpu().numpy().squeeze()
            
            # Add exploration noise
            if not deterministic:
                noise = np.random.normal(0, self.low_level_noise, size=action.shape)
                action += noise
                action = np.clip(action, -1.0, 1.0)
            
            self.subgoal_step -= 1
            self.episode_step += 1
            
            info = {
                'subgoal': self.current_subgoal.copy(),
                'relative_subgoal': self.current_subgoal - state[:self.subgoal_dim],
                'subgoal_step': self.subgoal_step
            }
            
            return action, info
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Dict[str, Any]
    ) -> None:
        """Store transition in replay buffer."""
        subgoal = info.get('subgoal', np.zeros(self.subgoal_dim))
        
        # Store in episode buffer for HER
        self.replay_buffer.store_transition(state, action, reward, subgoal)
        
        # Store low-level transition
        intrinsic_reward = self._compute_intrinsic_reward(next_state, subgoal)
        self.replay_buffer.store_low_level(
            state=state,
            action=action,
            reward=intrinsic_reward,
            next_state=next_state,
            subgoal=subgoal,
            done=done
        )
        
        # Store high-level transition (every k steps or at episode end)
        if self.subgoal_step <= 0 or done:
            self.replay_buffer.store_high_level(
                state=state,
                subgoal=info.get('relative_subgoal', np.zeros(self.subgoal_dim)),
                reward=reward,
                next_state=next_state,
                done=done
            )
    
    def _compute_intrinsic_reward(self, state: np.ndarray, subgoal: np.ndarray) -> float:
        """Compute intrinsic reward for low-level policy."""
        current_pos = state[:self.subgoal_dim]
        distance = np.linalg.norm(current_pos - subgoal)
        
        # Dense reward based on distance to subgoal
        intrinsic_reward = -distance
        
        # Bonus for achieving subgoal
        if distance < 1.0:
            intrinsic_reward += 5.0
        
        return intrinsic_reward
    
    def update(self, **kwargs) -> Dict[str, float]:
        """Update both high-level and low-level policies."""
        if (self.replay_buffer.low_size < self.config.min_buffer_size or
            self.replay_buffer.high_size < self.config.min_buffer_size):
            return {}
        
        losses = {}
        
        # Update low-level policy
        low_level_losses = self._update_low_level()
        losses.update(low_level_losses)
        
        # Update high-level policy (less frequently)
        if self.training_step % self.config.high_level_train_freq == 0:
            high_level_losses = self._update_high_level()
            losses.update(high_level_losses)
        
        # Update target networks
        if self.training_step % self.config.update_freq == 0:
            self._soft_update_targets()
        
        # Decay noise
        self.high_level_noise = max(
            self.config.min_noise,
            self.high_level_noise * self.config.noise_decay
        )
        self.low_level_noise = max(
            self.config.min_noise,
            self.low_level_noise * self.config.noise_decay
        )
        
        self.training_step += 1
        return losses
    
    def _update_low_level(self) -> Dict[str, float]:
        """Update low-level policy using DDPG."""
        batch = self.replay_buffer.sample_low_level(self.config.batch_size)
        if batch is None:
            return {}
        
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        subgoals = batch['subgoals'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Prepare inputs
        current_input = torch.cat([states, subgoals], dim=-1)
        next_input = torch.cat([next_states, subgoals], dim=-1)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.low_level_actor_target(next_input)
            # Target policy smoothing
            noise = torch.clamp(torch.randn_like(next_actions) * 0.2, -0.5, 0.5)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)
            
            target_q1, target_q2 = self.low_level_critic_target(next_input, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones.float()) * self.config.gamma * target_q
        
        current_q1, current_q2 = self.low_level_critic(current_input, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.low_level_critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.low_level_critic.parameters(), 1.0)
        self.low_level_critic_optimizer.step()
        
        # Actor update (delayed)
        if self.training_step % 2 == 0:
            predicted_actions = self.low_level_actor(current_input)
            actor_loss = -self.low_level_critic.q1_forward(current_input, predicted_actions).mean()
            
            self.low_level_actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.low_level_actor.parameters(), 1.0)
            self.low_level_actor_optimizer.step()
        else:
            actor_loss = torch.tensor(0.0)
        
        return {
            'low_level_critic_loss': critic_loss.item(),
            'low_level_actor_loss': actor_loss.item(),
            'low_level_q_value': current_q1.mean().item()
        }
    
    def _update_high_level(self) -> Dict[str, float]:
        """Update high-level policy using DDPG."""
        batch = self.replay_buffer.sample_high_level(self.config.batch_size)
        if batch is None:
            return {}
        
        states = batch['states'].to(self.device)
        subgoals = batch['subgoals'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Critic update
        with torch.no_grad():
            next_subgoals = self.high_level_actor_target(next_states)
            # Target policy smoothing
            noise = torch.clamp(torch.randn_like(next_subgoals) * 0.2, -0.5, 0.5)
            next_subgoals = torch.clamp(next_subgoals + noise, -self.config.subgoal_scale, self.config.subgoal_scale)
            
            target_q1, target_q2 = self.high_level_critic_target(next_states, next_subgoals)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones.float()) * self.config.gamma * target_q
        
        current_q1, current_q2 = self.high_level_critic(states, subgoals)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.high_level_critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.high_level_critic.parameters(), 1.0)
        self.high_level_critic_optimizer.step()
        
        # Actor update (delayed)
        if self.training_step % 2 == 0:
            predicted_subgoals = self.high_level_actor(states)
            actor_loss = -self.high_level_critic.q1_forward(states, predicted_subgoals).mean()
            
            self.high_level_actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.high_level_actor.parameters(), 1.0)
            self.high_level_actor_optimizer.step()
        else:
            actor_loss = torch.tensor(0.0)
        
        return {
            'high_level_critic_loss': critic_loss.item(),
            'high_level_actor_loss': actor_loss.item(),
            'high_level_q_value': current_q1.mean().item()
        }
    
    def _soft_update_targets(self) -> None:
        """Soft update target networks."""
        tau = self.config.tau
        
        for param, target_param in zip(self.high_level_actor.parameters(), self.high_level_actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        for param, target_param in zip(self.high_level_critic.parameters(), self.high_level_critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        for param, target_param in zip(self.low_level_actor.parameters(), self.low_level_actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        for param, target_param in zip(self.low_level_critic.parameters(), self.low_level_critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def end_episode(self) -> None:
        """Called at the end of each episode."""
        self.replay_buffer.end_episode()
        self.current_subgoal = None
        self.subgoal_step = 0
        self.episode_step = 0
    
    def save(self, filepath: str) -> None:
        """Save agent parameters."""
        save_dict = {
            'config': self.config,
            'high_level_actor': self.high_level_actor.state_dict(),
            'high_level_critic': self.high_level_critic.state_dict(),
            'low_level_actor': self.low_level_actor.state_dict(),
            'low_level_critic': self.low_level_critic.state_dict(),
            'high_level_actor_target': self.high_level_actor_target.state_dict(),
            'high_level_critic_target': self.high_level_critic_target.state_dict(),
            'low_level_actor_target': self.low_level_actor_target.state_dict(),
            'low_level_critic_target': self.low_level_critic_target.state_dict(),
            'optimizers': {
                'high_level_actor': self.high_level_actor_optimizer.state_dict(),
                'high_level_critic': self.high_level_critic_optimizer.state_dict(),
                'low_level_actor': self.low_level_actor_optimizer.state_dict(),
                'low_level_critic': self.low_level_critic_optimizer.state_dict()
            },
            'training_step': self.training_step,
            'high_level_noise': self.high_level_noise,
            'low_level_noise': self.low_level_noise
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"HIRO agent saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load agent parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.high_level_actor.load_state_dict(checkpoint['high_level_actor'])
        self.high_level_critic.load_state_dict(checkpoint['high_level_critic'])
        self.low_level_actor.load_state_dict(checkpoint['low_level_actor'])
        self.low_level_critic.load_state_dict(checkpoint['low_level_critic'])
        
        self.high_level_actor_target.load_state_dict(checkpoint['high_level_actor_target'])
        self.high_level_critic_target.load_state_dict(checkpoint['high_level_critic_target'])
        self.low_level_actor_target.load_state_dict(checkpoint['low_level_actor_target'])
        self.low_level_critic_target.load_state_dict(checkpoint['low_level_critic_target'])
        
        optimizers = checkpoint['optimizers']
        self.high_level_actor_optimizer.load_state_dict(optimizers['high_level_actor'])
        self.high_level_critic_optimizer.load_state_dict(optimizers['high_level_critic'])
        self.low_level_actor_optimizer.load_state_dict(optimizers['low_level_actor'])
        self.low_level_critic_optimizer.load_state_dict(optimizers['low_level_critic'])
        
        self.training_step = checkpoint.get('training_step', 0)
        self.high_level_noise = checkpoint.get('high_level_noise', self.config.high_level_noise)
        self.low_level_noise = checkpoint.get('low_level_noise', self.config.low_level_noise)
        
        logger.info(f"HIRO agent loaded from {filepath}")