#!/usr/bin/env python
"""
Options Framework Agent Implementation
=====================================

Temporal abstractions through options and semi-MDPs.
Based on "Between MDPs and Semi-MDPs" (Sutton et al., 1999).
Extended with Option-Critic (Bacon et al., 2017) and diversity mechanisms.

Key Features:
- Multiple learned options/skills  
- Option termination conditions
- Intra-option and option selection policies
- Diversity bonuses for option discovery
- Semi-Markov decision process formulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

from ..common.base_hierarchical_agent import BaseHierarchicalAgent
from .options_config import OptionsConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptionPolicy(nn.Module):
    """Individual option policy network."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        continuous_actions: bool = True
    ):
        super().__init__()
        
        self.continuous_actions = continuous_actions
        self.action_dim = action_dim
        
        # Policy network
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Action head
        if continuous_actions:
            self.action_mean = nn.Linear(prev_dim, action_dim)
            self.action_logstd = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_logits = nn.Linear(prev_dim, action_dim)
        
        # Value function
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """Forward pass through option policy."""
        x = self.shared_layers(state)
        
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


class TerminationNetwork(nn.Module):
    """Option termination network."""
    
    def __init__(self, state_dim: int, num_options: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_options))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Get termination probabilities for all options."""
        return self.network(state)


class OptionSelector(nn.Module):
    """High-level option selection policy."""
    
    def __init__(self, state_dim: int, num_options: int, hidden_dims: List[int]):
        super().__init__()
        
        self.num_options = num_options
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_options))
        
        self.network = nn.Sequential(*layers)
        
        # Value function for option selection
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """Select option and compute state value."""
        x = self.network[:-1](state)  # All layers except the last one
        
        option_logits = self.network[-1](x)
        option_dist = Categorical(logits=option_logits)
        
        value = self.value_head(x)
        
        return option_dist, value


class OptionsRolloutBuffer:
    """Rollout buffer for options framework."""
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, num_options: int):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options
        
        # Initialize buffers
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.options = np.zeros(buffer_size, dtype=np.int64)
        self.option_log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.terminations = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)
        
        # Option-specific rewards
        self.intrinsic_rewards = np.zeros(buffer_size, dtype=np.float32)
        self.diversity_bonuses = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.full = False
    
    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        option: int,
        option_log_prob: float,
        termination: float,
        done: bool,
        intrinsic_reward: float = 0.0,
        diversity_bonus: float = 0.0
    ):
        """Store transition."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.options[self.ptr] = option
        self.option_log_probs[self.ptr] = option_log_prob
        self.terminations[self.ptr] = termination
        self.dones[self.ptr] = done
        self.intrinsic_rewards[self.ptr] = intrinsic_reward
        self.diversity_bonuses[self.ptr] = diversity_bonus
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True
    
    def get(self) -> Dict[str, np.ndarray]:
        """Get all stored data."""
        size = self.buffer_size if self.full else self.ptr
        
        return {
            'states': self.states[:size],
            'actions': self.actions[:size],
            'rewards': self.rewards[:size],
            'values': self.values[:size],
            'log_probs': self.log_probs[:size],
            'options': self.options[:size],
            'option_log_probs': self.option_log_probs[:size],
            'terminations': self.terminations[:size],
            'dones': self.dones[:size],
            'intrinsic_rewards': self.intrinsic_rewards[:size],
            'diversity_bonuses': self.diversity_bonuses[:size]
        }
    
    def clear(self):
        """Clear buffer."""
        self.ptr = 0
        self.full = False


class OptionsAgent(BaseHierarchicalAgent):
    """
    Options Framework Agent.
    
    Implements temporal abstractions through learned options/skills.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: OptionsConfig,
        continuous_actions: bool = True,
        **kwargs
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=config.num_options,  # Using number of options as "goal" dim
            device=config.device,
            **kwargs
        )
        
        self.config = config
        self.continuous_actions = continuous_actions
        self.num_options = config.num_options
        
        # Option policies
        self.option_policies = nn.ModuleList([
            OptionPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=config.option_policy_hidden_dims,
                continuous_actions=continuous_actions
            ) for _ in range(config.num_options)
        ]).to(self.device)
        
        # Option selector (high-level policy)
        self.option_selector = OptionSelector(
            state_dim=state_dim,
            num_options=config.num_options,
            hidden_dims=config.policy_hidden_dims
        ).to(self.device)
        
        # Termination network
        self.termination_network = TerminationNetwork(
            state_dim=state_dim,
            num_options=config.num_options,
            hidden_dims=config.termination_hidden_dims
        ).to(self.device)
        
        # Optimizers
        self.option_policies_optimizer = optim.Adam(
            self.option_policies.parameters(),
            lr=config.learning_rate
        )
        self.option_selector_optimizer = optim.Adam(
            self.option_selector.parameters(),
            lr=config.learning_rate
        )
        self.termination_optimizer = optim.Adam(
            self.termination_network.parameters(),
            lr=config.learning_rate
        )
        
        # Rollout buffer
        self.buffer = OptionsRolloutBuffer(
            buffer_size=config.buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            num_options=config.num_options
        )
        
        # Training state
        self.current_option = None
        self.option_length = 0
        self.episode_step = 0
        self.total_steps = 0
        
        # Option statistics for diversity
        self.option_visit_counts = np.zeros(config.num_options)
        self.option_state_coverage = [[] for _ in range(config.num_options)]
        
        logger.info(f"Initialized Options agent with {config.num_options} options")
    
    def select_action(
        self,
        state: np.ndarray,
        goal: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using options framework."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Option selection (if no current option or option terminated)
            if self.current_option is None or self._should_terminate_option(state_tensor):
                option_dist, option_value = self.option_selector(state_tensor)
                
                if deterministic:
                    self.current_option = option_dist.probs.argmax().item()
                    option_log_prob = option_dist.log_prob(torch.tensor([self.current_option]).to(self.device))
                else:
                    option_sample = option_dist.sample()
                    self.current_option = option_sample.item()
                    option_log_prob = option_dist.log_prob(option_sample)
                
                self.option_length = 0
                option_log_prob_scalar = option_log_prob.cpu().numpy().item()
                option_value_scalar = option_value.cpu().numpy().item()
            else:
                option_log_prob_scalar = 0.0
                option_value_scalar = 0.0
            
            # Action selection using current option policy
            option_policy = self.option_policies[self.current_option]
            action_dist, value = option_policy(state_tensor)
            
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
            
            # Get termination probability
            termination_probs = self.termination_network(state_tensor)
            current_termination = termination_probs[0, self.current_option]
            
            # Convert to numpy
            action_np = action.squeeze().cpu().numpy()
            log_prob_np = log_prob.squeeze().cpu().numpy()
            value_np = value.squeeze().cpu().numpy()
            termination_np = current_termination.cpu().numpy()
            
            # Compute intrinsic rewards and bonuses
            intrinsic_reward = self._compute_intrinsic_reward(state, self.current_option)
            diversity_bonus = self._compute_diversity_bonus(state, self.current_option)
            
            # Update statistics
            self.option_length += 1
            self.episode_step += 1
            self.total_steps += 1
            self.option_visit_counts[self.current_option] += 1
            if self.config.use_diversity_bonus:
                self.option_state_coverage[self.current_option].append(state.copy())
            
            info = {
                'option': self.current_option,
                'option_log_prob': option_log_prob_scalar,
                'option_value': option_value_scalar,
                'termination_prob': termination_np,
                'option_length': self.option_length,
                'intrinsic_reward': intrinsic_reward,
                'diversity_bonus': diversity_bonus
            }
            
            return action_np, info
    
    def _should_terminate_option(self, state: torch.Tensor) -> bool:
        """Determine if current option should terminate."""
        if self.current_option is None:
            return True
        
        # Minimum option length constraint
        if self.option_length < self.config.option_min_length:
            return False
        
        # Maximum option length constraint
        if self.option_length >= self.config.option_max_length:
            return True
        
        # Termination network decision
        termination_probs = self.termination_network(state)
        termination_prob = termination_probs[0, self.current_option].item()
        
        # Stochastic termination
        return np.random.random() < termination_prob
    
    def _compute_intrinsic_reward(self, state: np.ndarray, option: int) -> float:
        """Compute intrinsic reward for option execution."""
        # Simple intrinsic reward based on option consistency
        if self.option_length == 1:
            return 0.1  # Small bonus for starting new option
        return 0.0
    
    def _compute_diversity_bonus(self, state: np.ndarray, option: int) -> float:
        """Compute diversity bonus to encourage option exploration."""
        if not self.config.use_diversity_bonus:
            return 0.0
        
        # State novelty bonus for the current option
        if len(self.option_state_coverage[option]) == 0:
            return self.config.diversity_coef
        
        # Find minimum distance to previous states visited by this option
        min_distance = float('inf')
        for prev_state in self.option_state_coverage[option][-100:]:  # Last 100 states
            distance = np.linalg.norm(state - prev_state)
            min_distance = min(min_distance, distance)
        
        # Bonus inversely proportional to minimum distance
        diversity_bonus = self.config.diversity_coef * min(1.0, min_distance)
        
        return diversity_bonus
    
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
        # Combined reward (extrinsic + intrinsic + diversity)
        combined_reward = (
            reward + 
            info.get('intrinsic_reward', 0.0) + 
            info.get('diversity_bonus', 0.0)
        )
        
        self.buffer.store(
            state=state,
            action=action,
            reward=combined_reward,
            value=info.get('option_value', 0.0),
            log_prob=info.get('log_prob', 0.0),
            option=info.get('option', 0),
            option_log_prob=info.get('option_log_prob', 0.0),
            termination=info.get('termination_prob', 0.0),
            done=done,
            intrinsic_reward=info.get('intrinsic_reward', 0.0),
            diversity_bonus=info.get('diversity_bonus', 0.0)
        )
    
    def update(self, **kwargs) -> Dict[str, float]:
        """Update option policies, selector, and termination network."""
        buffer_data = self.buffer.get()
        
        if len(buffer_data['states']) < self.config.batch_size:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(buffer_data['states']).to(self.device)
        actions = torch.FloatTensor(buffer_data['actions']).to(self.device)
        rewards = torch.FloatTensor(buffer_data['rewards']).to(self.device)
        old_values = torch.FloatTensor(buffer_data['values']).to(self.device)
        old_log_probs = torch.FloatTensor(buffer_data['log_probs']).to(self.device)
        options = torch.LongTensor(buffer_data['options']).to(self.device)
        old_option_log_probs = torch.FloatTensor(buffer_data['option_log_probs']).to(self.device)
        terminations = torch.FloatTensor(buffer_data['terminations']).to(self.device)
        dones = torch.BoolTensor(buffer_data['dones']).to(self.device)
        
        # Compute advantages and returns
        with torch.no_grad():
            # Use option-specific value functions
            new_values = torch.zeros_like(old_values)
            for i, option in enumerate(options):
                _, value = self.option_policies[option.item()](states[i:i+1])
                new_values[i] = value.squeeze()
            
            # GAE computation
            advantages, returns = self._compute_gae(rewards, old_values, dones, new_values[-1])
        
        # Update option policies
        option_policy_losses = self._update_option_policies(
            states, actions, options, old_log_probs, advantages, returns, old_values
        )
        
        # Update option selector
        selector_losses = self._update_option_selector(
            states, options, old_option_log_probs, advantages
        )
        
        # Update termination network
        termination_losses = self._update_termination_network(
            states, options, terminations, advantages
        )
        
        # Clear buffer
        self.buffer.clear()
        
        # Combine all losses
        total_losses = {**option_policy_losses, **selector_losses, **termination_losses}
        
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
    
    def _update_option_policies(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        options: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor
    ) -> Dict[str, float]:
        """Update all option policies using PPO."""
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(self.config.ppo_epochs):
            # Group by options for efficient processing
            unique_options = torch.unique(options)
            
            for option in unique_options:
                option_mask = (options == option)
                if not option_mask.any():
                    continue
                
                option_states = states[option_mask]
                option_actions = actions[option_mask]
                option_advantages = advantages[option_mask]
                option_returns = returns[option_mask]
                option_old_log_probs = old_log_probs[option_mask]
                option_old_values = old_values[option_mask]
                
                # Forward pass through option policy
                action_dist, values = self.option_policies[option.item()](option_states)
                values = values.squeeze()
                
                # Policy loss
                if self.continuous_actions:
                    log_probs = action_dist.log_prob(option_actions).sum(dim=-1)
                else:
                    log_probs = action_dist.log_prob(option_actions.long())
                
                ratio = torch.exp(log_probs - option_old_log_probs)
                surr1 = ratio * option_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
                ) * option_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_pred_clipped = option_old_values + torch.clamp(
                    values - option_old_values, -self.config.clip_param, self.config.clip_param
                )
                value_loss1 = (values - option_returns).pow(2)
                value_loss2 = (value_pred_clipped - option_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # Entropy
                entropy = action_dist.entropy().mean()
                
                # Total loss for this option
                option_loss = (
                    policy_loss + 
                    self.config.value_loss_coef * value_loss - 
                    self.config.entropy_coef * entropy
                )
                
                # Backward pass
                self.option_policies_optimizer.zero_grad()
                option_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.option_policies[option.item()].parameters(),
                    self.config.max_grad_norm
                )
                self.option_policies_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        num_options_updated = len(torch.unique(options))
        return {
            'option_policy_loss': total_policy_loss / (self.config.ppo_epochs * num_options_updated),
            'option_value_loss': total_value_loss / (self.config.ppo_epochs * num_options_updated),
            'option_entropy': total_entropy / (self.config.ppo_epochs * num_options_updated)
        }
    
    def _update_option_selector(
        self,
        states: torch.Tensor,
        options: torch.Tensor,
        old_option_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> Dict[str, float]:
        """Update option selector using policy gradient."""
        # Only update when options were actually selected (non-zero log probs)
        selection_mask = (old_option_log_probs != 0)
        if not selection_mask.any():
            return {'selector_loss': 0.0}
        
        selector_states = states[selection_mask]
        selector_options = options[selection_mask]
        selector_advantages = advantages[selection_mask]
        selector_old_log_probs = old_option_log_probs[selection_mask]
        
        option_dist, _ = self.option_selector(selector_states)
        log_probs = option_dist.log_prob(selector_options)
        
        # Policy gradient loss
        ratio = torch.exp(log_probs - selector_old_log_probs)
        selector_loss = -(ratio * selector_advantages).mean()
        
        # Update
        self.option_selector_optimizer.zero_grad()
        selector_loss.backward()
        nn.utils.clip_grad_norm_(self.option_selector.parameters(), self.config.max_grad_norm)
        self.option_selector_optimizer.step()
        
        return {'selector_loss': selector_loss.item()}
    
    def _update_termination_network(
        self,
        states: torch.Tensor,
        options: torch.Tensor,
        terminations: torch.Tensor,
        advantages: torch.Tensor
    ) -> Dict[str, float]:
        """Update termination network."""
        termination_probs = self.termination_network(states)
        
        # Gather termination probabilities for current options
        current_terminations = termination_probs.gather(1, options.unsqueeze(1)).squeeze()
        
        # Termination loss (encourage termination when advantage is negative)
        termination_targets = (advantages < 0).float()
        termination_loss = F.binary_cross_entropy(current_terminations, termination_targets)
        
        # Regularization to prevent too frequent terminations
        termination_reg = self.config.termination_reg * termination_probs.mean()
        
        total_termination_loss = termination_loss + termination_reg
        
        # Update
        self.termination_optimizer.zero_grad()
        total_termination_loss.backward()
        nn.utils.clip_grad_norm_(self.termination_network.parameters(), self.config.max_grad_norm)
        self.termination_optimizer.step()
        
        return {
            'termination_loss': termination_loss.item(),
            'termination_reg': termination_reg.item()
        }
    
    def reset_episode(self) -> None:
        """Reset for new episode."""
        self.current_option = None
        self.option_length = 0
        self.episode_step = 0
    
    def get_option_statistics(self) -> Dict[str, Any]:
        """Get option usage statistics."""
        total_visits = self.option_visit_counts.sum()
        if total_visits == 0:
            option_frequencies = np.zeros(self.num_options)
        else:
            option_frequencies = self.option_visit_counts / total_visits
        
        return {
            'option_visit_counts': self.option_visit_counts.copy(),
            'option_frequencies': option_frequencies,
            'option_diversity_score': 1.0 - np.var(option_frequencies),
            'total_steps': self.total_steps
        }
    
    def save(self, filepath: str) -> None:
        """Save agent parameters."""
        save_dict = {
            'config': self.config,
            'option_policies': [policy.state_dict() for policy in self.option_policies],
            'option_selector': self.option_selector.state_dict(),
            'termination_network': self.termination_network.state_dict(),
            'optimizers': {
                'option_policies': self.option_policies_optimizer.state_dict(),
                'option_selector': self.option_selector_optimizer.state_dict(),
                'termination': self.termination_optimizer.state_dict()
            },
            'option_visit_counts': self.option_visit_counts,
            'total_steps': self.total_steps
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"Options agent saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load agent parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load option policies
        for i, policy_state in enumerate(checkpoint['option_policies']):
            self.option_policies[i].load_state_dict(policy_state)
        
        self.option_selector.load_state_dict(checkpoint['option_selector'])
        self.termination_network.load_state_dict(checkpoint['termination_network'])
        
        # Load optimizers
        optimizers = checkpoint['optimizers']
        self.option_policies_optimizer.load_state_dict(optimizers['option_policies'])
        self.option_selector_optimizer.load_state_dict(optimizers['option_selector'])
        self.termination_optimizer.load_state_dict(optimizers['termination'])
        
        # Load statistics
        self.option_visit_counts = checkpoint.get('option_visit_counts', np.zeros(self.num_options))
        self.total_steps = checkpoint.get('total_steps', 0)
        
        logger.info(f"Options agent loaded from {filepath}")