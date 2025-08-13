#!/usr/bin/env python
"""
FuN (FeUdal Networks) Agent Implementation
=========================================

Manager-worker hierarchical architecture with intrinsic motivation.
Based on "FeUdal Networks for Hierarchical Reinforcement Learning" (Vezhnevets et al., 2017).

Key Features:
- Manager-worker feudal architecture
- Goal embedding and transition policies
- Intrinsic motivation through cosine similarity
- Dilated LSTM for temporal abstraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
import logging

from ..common.base_hierarchical_agent import BaseHierarchicalAgent
from .fun_config import FuNConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ManagerNetwork(nn.Module):
    """Manager network that sets goals for the worker."""
    
    def __init__(self, embedding_dim: int, goal_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.goal_dim = goal_dim
        
        # Dilated LSTM for temporal abstraction
        self.lstm = nn.LSTM(embedding_dim, hidden_dims[0], batch_first=True)
        
        # Value function
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0], 1)
        )
        
        # Goal generation
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0], goal_dim)
        )
        
        self.hidden_state = None
    
    def forward(self, state_embedding: torch.Tensor, hidden: Optional[Tuple] = None):
        """
        Forward pass through manager network.
        
        Args:
            state_embedding: State embedding [batch, seq_len, embedding_dim]
            hidden: LSTM hidden state
            
        Returns:
            goal: Generated goal [batch, seq_len, goal_dim]  
            value: State value [batch, seq_len, 1]
            hidden: Updated LSTM hidden state
        """
        lstm_out, hidden = self.lstm(state_embedding, hidden)
        
        goal = self.goal_head(lstm_out)
        # Normalize goal to unit sphere
        goal = F.normalize(goal, dim=-1)
        
        value = self.value_head(lstm_out)
        
        return goal, value, hidden
    
    def reset_hidden(self, batch_size: int = 1):
        """Reset LSTM hidden state."""
        self.hidden_state = None


class WorkerNetwork(nn.Module):
    """Worker network that takes actions conditioned on manager goals."""
    
    def __init__(
        self, 
        embedding_dim: int, 
        goal_dim: int, 
        action_dim: int,
        hidden_dims: List[int],
        continuous_actions: bool = True
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.continuous_actions = continuous_actions
        
        # Combine state embedding and goal
        input_dim = embedding_dim + goal_dim
        
        # Policy network
        self.policy_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.policy_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Action head
        if continuous_actions:
            self.action_mean = nn.Linear(prev_dim, action_dim)
            self.action_logstd = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_logits = nn.Linear(prev_dim, action_dim)
        
        # Value function
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1] if hidden_dims else input_dim),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] if hidden_dims else input_dim, 1)
        )
    
    def forward(self, state_embedding: torch.Tensor, goal: torch.Tensor):
        """
        Forward pass through worker network.
        
        Args:
            state_embedding: State embedding [batch, embedding_dim]
            goal: Manager goal [batch, goal_dim]
            
        Returns:
            action_dist: Action distribution
            value: State value [batch, 1]
        """
        # Concatenate state and goal
        x = torch.cat([state_embedding, goal], dim=-1)
        
        # Forward through policy layers
        for layer in self.policy_layers:
            x = layer(x)
        
        # Action distribution
        if self.continuous_actions:
            action_mean = torch.tanh(self.action_mean(x))
            action_std = torch.exp(self.action_logstd.expand_as(action_mean))
            action_dist = Normal(action_mean, action_std)
        else:
            action_logits = self.action_logits(x)
            action_dist = Categorical(logits=action_logits)
        
        # Value
        value = self.value_head(x)
        
        return action_dist, value


class StateEncoder(nn.Module):
    """Encode raw observations to embeddings."""
    
    def __init__(self, obs_dim: int, embedding_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to embedding."""
        return self.encoder(obs)


class FuNRolloutBuffer:
    """Rollout buffer for FuN algorithm."""
    
    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int, goal_dim: int):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        
        # Initialize buffers
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.goals = np.zeros((buffer_size, goal_dim), dtype=np.float32)
        self.intrinsic_rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)
        
        # Manager specific
        self.manager_values = np.zeros(buffer_size, dtype=np.float32)
        self.manager_goals = np.zeros((buffer_size, goal_dim), dtype=np.float32)
        
        self.ptr = 0
        self.full = False
    
    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        goal: np.ndarray,
        intrinsic_reward: float,
        done: bool,
        manager_value: float = 0.0
    ):
        """Store transition."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.goals[self.ptr] = goal
        self.intrinsic_rewards[self.ptr] = intrinsic_reward
        self.dones[self.ptr] = done
        self.manager_values[self.ptr] = manager_value
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True
    
    def get(self) -> Dict[str, np.ndarray]:
        """Get all stored data."""
        size = self.buffer_size if self.full else self.ptr
        
        return {
            'observations': self.observations[:size],
            'actions': self.actions[:size], 
            'rewards': self.rewards[:size],
            'values': self.values[:size],
            'log_probs': self.log_probs[:size],
            'goals': self.goals[:size],
            'intrinsic_rewards': self.intrinsic_rewards[:size],
            'dones': self.dones[:size],
            'manager_values': self.manager_values[:size]
        }
    
    def clear(self):
        """Clear buffer."""
        self.ptr = 0
        self.full = False


class FuNAgent(BaseHierarchicalAgent):
    """
    FuN (FeUdal Networks) Agent.
    
    Implements manager-worker hierarchical architecture with intrinsic motivation.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: FuNConfig,
        continuous_actions: bool = True,
        **kwargs
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=config.goal_dim,
            device=config.device,
            **kwargs
        )
        
        self.config = config
        self.continuous_actions = continuous_actions
        
        # Networks
        self.state_encoder = StateEncoder(
            obs_dim=state_dim,
            embedding_dim=config.embedding_dim,
            hidden_dims=[256, 256]
        ).to(self.device)
        
        self.manager = ManagerNetwork(
            embedding_dim=config.embedding_dim,
            goal_dim=config.goal_dim,
            hidden_dims=config.manager_hidden_dims
        ).to(self.device)
        
        self.worker = WorkerNetwork(
            embedding_dim=config.embedding_dim,
            goal_dim=config.goal_dim,
            action_dim=action_dim,
            hidden_dims=config.worker_hidden_dims,
            continuous_actions=continuous_actions
        ).to(self.device)
        
        # Optimizers
        self.manager_optimizer = optim.Adam(
            list(self.state_encoder.parameters()) + list(self.manager.parameters()),
            lr=config.manager_lr
        )
        self.worker_optimizer = optim.Adam(self.worker.parameters(), lr=config.worker_lr)
        
        # Rollout buffer
        self.buffer = FuNRolloutBuffer(
            buffer_size=config.buffer_size,
            obs_dim=state_dim,
            action_dim=action_dim,
            goal_dim=config.goal_dim
        )
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.current_goal = None
        self.manager_hidden = None
        self.goal_horizon = 0
        
        logger.info("Initialized FuN agent")
    
    def select_action(
        self, 
        state: np.ndarray,
        goal: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using manager-worker hierarchy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Encode state
            state_embedding = self.state_encoder(state_tensor)
            
            # Manager decision (every c steps)
            if self.goal_horizon <= 0 or self.current_goal is None:
                manager_goal, manager_value, self.manager_hidden = self.manager(
                    state_embedding.unsqueeze(1), self.manager_hidden
                )
                self.current_goal = manager_goal.squeeze(1).cpu().numpy()
                self.goal_horizon = self.config.manager_horizon
                manager_value_scalar = manager_value.squeeze().cpu().numpy()
            else:
                manager_value_scalar = 0.0
            
            # Worker action selection
            goal_tensor = torch.FloatTensor(self.current_goal).to(self.device)
            action_dist, worker_value = self.worker(state_embedding, goal_tensor)
            
            if deterministic:
                if self.continuous_actions:
                    action = action_dist.mean
                else:
                    action = action_dist.probs.argmax(dim=-1, keepdim=True).float()
            else:
                action = action_dist.sample()
            
            log_prob = action_dist.log_prob(action)
            if self.continuous_actions:
                log_prob = log_prob.sum(dim=-1)
            
            # Convert to numpy
            action_np = action.squeeze().cpu().numpy()
            log_prob_np = log_prob.squeeze().cpu().numpy()
            worker_value_np = worker_value.squeeze().cpu().numpy()
            
            # Compute intrinsic reward
            intrinsic_reward = self._compute_intrinsic_reward(
                state_embedding.squeeze().cpu().numpy(),
                self.current_goal.squeeze()
            )
            
            self.goal_horizon -= 1
            
            info = {
                'goal': self.current_goal.copy(),
                'intrinsic_reward': intrinsic_reward,
                'worker_value': worker_value_np,
                'manager_value': manager_value_scalar,
                'log_prob': log_prob_np
            }
            
            return action_np, info
    
    def _compute_intrinsic_reward(
        self, 
        state_embedding: np.ndarray,
        goal: np.ndarray
    ) -> float:
        """Compute intrinsic reward based on goal alignment."""
        # Cosine similarity between state embedding and goal
        state_norm = np.linalg.norm(state_embedding)
        goal_norm = np.linalg.norm(goal)
        
        if state_norm == 0 or goal_norm == 0:
            return 0.0
        
        cosine_sim = np.dot(state_embedding[:len(goal)], goal) / (state_norm * goal_norm)
        return cosine_sim
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Dict[str, Any]
    ) -> None:
        """Store transition in rollout buffer."""
        self.buffer.store(
            obs=state,
            action=action,
            reward=reward,
            value=info.get('worker_value', 0.0),
            log_prob=info.get('log_prob', 0.0),
            goal=info.get('goal', np.zeros(self.config.goal_dim)),
            intrinsic_reward=info.get('intrinsic_reward', 0.0),
            done=done,
            manager_value=info.get('manager_value', 0.0)
        )
        
        self.step_count += 1
    
    def update(self, **kwargs) -> Dict[str, float]:
        """Update manager and worker networks."""
        buffer_data = self.buffer.get()
        
        if len(buffer_data['observations']) < self.config.batch_size:
            return {}
        
        # Convert to tensors
        obs = torch.FloatTensor(buffer_data['observations']).to(self.device)
        actions = torch.FloatTensor(buffer_data['actions']).to(self.device)
        rewards = torch.FloatTensor(buffer_data['rewards']).to(self.device)
        old_values = torch.FloatTensor(buffer_data['values']).to(self.device)
        old_log_probs = torch.FloatTensor(buffer_data['log_probs']).to(self.device)
        goals = torch.FloatTensor(buffer_data['goals']).to(self.device)
        intrinsic_rewards = torch.FloatTensor(buffer_data['intrinsic_rewards']).to(self.device)
        dones = torch.BoolTensor(buffer_data['dones']).to(self.device)
        
        # Compute advantages and returns
        with torch.no_grad():
            state_embeddings = self.state_encoder(obs)
            _, new_values = self.worker(state_embeddings, goals)
            new_values = new_values.squeeze()
            
            # Combined rewards (extrinsic + intrinsic)
            combined_rewards = rewards + self.config.alpha * intrinsic_rewards
            
            # GAE computation
            advantages, returns = self._compute_gae(
                combined_rewards, old_values, dones, new_values[-1]
            )
        
        # Update worker
        worker_losses = self._update_worker(
            obs, actions, goals, old_log_probs, advantages, returns, old_values
        )
        
        # Update manager (less frequently)
        manager_losses = {}
        if self.step_count % self.config.manager_horizon == 0:
            manager_losses = self._update_manager(obs, rewards, dones)
        
        # Clear buffer
        self.buffer.clear()
        
        # Combine losses
        total_losses = {**worker_losses, **manager_losses}
        
        return total_losses
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_val = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def _update_worker(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        goals: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor
    ) -> Dict[str, float]:
        """Update worker network using PPO."""
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.config.ppo_epochs):
            # Forward pass
            state_embeddings = self.state_encoder(obs)
            action_dist, values = self.worker(state_embeddings, goals)
            values = values.squeeze()
            
            # Policy loss
            if self.continuous_actions:
                log_probs = action_dist.log_prob(actions).sum(dim=-1)
            else:
                log_probs = action_dist.log_prob(actions.long())
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_pred_clipped = old_values + torch.clamp(
                values - old_values, -self.config.clip_param, self.config.clip_param
            )
            value_loss1 = (values - returns).pow(2)
            value_loss2 = (value_pred_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
            
            # Entropy
            entropy = action_dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy
            
            # Update
            self.worker_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.worker.parameters(), self.config.max_grad_norm)
            self.worker_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        return {
            'worker_policy_loss': total_policy_loss / self.config.ppo_epochs,
            'worker_value_loss': total_value_loss / self.config.ppo_epochs,
            'worker_entropy': total_entropy / self.config.ppo_epochs
        }
    
    def _update_manager(
        self,
        obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Update manager network."""
        # Simple manager update based on external rewards
        state_embeddings = self.state_encoder(obs)
        
        # Create sequences for LSTM
        seq_len = min(self.config.manager_horizon, len(obs))
        sequences = []
        for i in range(0, len(obs) - seq_len + 1, seq_len):
            sequences.append(state_embeddings[i:i+seq_len].unsqueeze(0))
        
        if not sequences:
            return {}
        
        sequence_batch = torch.cat(sequences, dim=0)
        
        # Forward pass
        goals, values, _ = self.manager(sequence_batch)
        
        # Simple value loss based on episode rewards
        episode_returns = []
        current_return = 0
        for i in reversed(range(len(rewards))):
            current_return = rewards[i] + self.config.gamma * current_return * (1 - dones[i].float())
            episode_returns.insert(0, current_return)
        
        target_values = torch.FloatTensor(episode_returns[:len(values.flatten())]).to(self.device)
        manager_value_loss = F.mse_loss(values.flatten(), target_values)
        
        # Update manager
        self.manager_optimizer.zero_grad()
        manager_value_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.manager.parameters()) + list(self.state_encoder.parameters()),
            self.config.max_grad_norm
        )
        self.manager_optimizer.step()
        
        return {
            'manager_value_loss': manager_value_loss.item()
        }
    
    def reset_episode(self) -> None:
        """Reset for new episode."""
        self.current_goal = None
        self.manager_hidden = None
        self.goal_horizon = 0
        self.episode_count += 1
    
    def save(self, filepath: str) -> None:
        """Save agent parameters."""
        save_dict = {
            'config': self.config,
            'state_encoder': self.state_encoder.state_dict(),
            'manager': self.manager.state_dict(),
            'worker': self.worker.state_dict(),
            'manager_optimizer': self.manager_optimizer.state_dict(),
            'worker_optimizer': self.worker_optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"FuN agent saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load agent parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.state_encoder.load_state_dict(checkpoint['state_encoder'])
        self.manager.load_state_dict(checkpoint['manager'])
        self.worker.load_state_dict(checkpoint['worker'])
        self.manager_optimizer.load_state_dict(checkpoint['manager_optimizer'])
        self.worker_optimizer.load_state_dict(checkpoint['worker_optimizer'])
        self.step_count = checkpoint.get('step_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        
        logger.info(f"FuN agent loaded from {filepath}")