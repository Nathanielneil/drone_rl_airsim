"""
A3C (Asynchronous Advantage Actor-Critic) Implementation for UAV Navigation
Based on the original A3C paper and adapted for AirSim environment
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import threading
import time
import gym
import os
import sys
import random
from collections import deque
from pathlib import Path
from tensorboardX import SummaryWriter
from tqdm import trange

# Add path to import AirSim environment
from gym_airsim.envs.AirGym import AirSimEnv


class A3CNetwork(nn.Module):
    """Simplified A3C network for AirSim inform_vector (9-dimensional input)"""
    
    def __init__(self, input_dim, num_actions, hidden_size=256):
        super(A3CNetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, num_actions)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor and critic outputs
        policy_logits = self.actor(x)
        value = self.critic(x)
        
        return policy_logits, value
    
    def act(self, state):
        """Select action using current policy"""
        policy_logits, value = self.forward(state)
        policy = F.softmax(policy_logits, dim=-1)
        action_dist = Categorical(policy)
        action = action_dist.sample()
        
        return action.item(), action_dist.log_prob(action).unsqueeze(0), value.squeeze(0)


class A3CConfig:
    """Configuration for A3C"""
    def __init__(self):
        # Network architecture
        self.input_dim = 9  # AirSim inform_vector dimension
        self.num_actions = 7  # AirSim discrete actions
        self.hidden_size = 256
        
        # Training parameters
        self.lr = 3e-4  # Increased learning rate
        self.gamma = 0.99
        self.n_steps = 5  # Reduced for faster updates
        self.max_timesteps = 10000  # Reduced for testing
        self.max_episode_steps = 100  # Reduced episode length
        self.num_workers = 1  # Simplified to single worker
        
        # Loss coefficients
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.1  # Increased entropy for more exploration
        self.max_grad_norm = 1.0
        
        # Environment
        self.env_name = 'AirGym'
        
        # Model saving
        self.model_save_path = 'models/a3c/'
        self.log_dir = 'logs/a3c/'


class A3CAgent:
    """A3C Agent for training"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create environment
        self.env = AirSimEnv(need_render=False)
        
        # Create network
        self.network = A3CNetwork(config.input_dim, config.num_actions, config.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)
        
        # Create directories
        os.makedirs(config.model_save_path, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Logger
        self.logger = SummaryWriter(config.log_dir)
        
        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
    def compute_loss(self, states, actions, rewards, values, log_probs, next_value):
        """Compute A3C loss"""
        returns = []
        R = next_value
        
        # Compute returns backwards
        for r in reversed(rewards):
            R = r + self.config.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        
        # Compute advantages
        advantages = returns - values
        
        # Actor loss (policy gradient)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy loss for exploration
        entropy_loss = -(log_probs * torch.exp(log_probs)).mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     self.config.value_loss_coef * value_loss - 
                     self.config.entropy_coef * entropy_loss)
        
        return total_loss, policy_loss, value_loss, entropy_loss
    
    def train_step(self, states, actions, rewards, values, log_probs, next_value):
        """Perform one training step"""
        loss, policy_loss, value_loss, entropy_loss = self.compute_loss(
            states, actions, rewards, values, log_probs, next_value
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item(), entropy_loss.item()
    
    def train(self):
        """Main training loop"""
        episode_count = 0
        step_count = 0
        
        print(f"Starting A3C training on {self.device}")
        
        # Initialize environment once
        obs = self.env.reset()
        state = torch.FloatTensor(obs[1]).to(self.device)
        episode_reward = 0
        episode_length = 0
        
        while step_count < self.config.max_timesteps:
            # Collect trajectory
            states, actions, rewards, values, log_probs = [], [], [], [], []
            
            for _ in range(self.config.n_steps):
                if step_count >= self.config.max_timesteps:
                    break
                    
                # Select action
                action, log_prob, value = self.network.act(state)
                
                # Take action in environment
                next_obs, reward, done, _ = self.env.step([action])
                next_state = torch.FloatTensor(next_obs[1]).to(self.device)
                
                # Store experience
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                
                # Update state and counters
                state = next_state
                episode_reward += reward
                episode_length += 1
                step_count += 1
                
                if done or episode_length >= self.config.max_episode_steps:
                    # Episode finished
                    episode_count += 1
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    
                    # Print progress
                    if episode_count % 5 == 0:
                        avg_reward = np.mean(list(self.episode_rewards)[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                        print(f"Episode {episode_count}: Steps={step_count}, Reward={episode_reward:.2f}, Avg Reward={avg_reward:.2f}")
                    
                    # Reset environment
                    obs = self.env.reset()
                    state = torch.FloatTensor(obs[1]).to(self.device)
                    episode_reward = 0
                    episode_length = 0
                    break
            
            # Train the network if we have experience
            if len(states) > 0:
                # Compute next value for bootstrapping
                if done:
                    next_value = 0
                else:
                    with torch.no_grad():
                        _, next_value = self.network.forward(state)
                        next_value = next_value.item()
                
                loss, policy_loss, value_loss, entropy_loss = self.train_step(
                    states, actions, rewards, values, log_probs, next_value
                )
                
                # Log training metrics
                if step_count % 500 == 0:
                    print(f"Step {step_count}: Loss={loss:.4f}, Policy={policy_loss:.4f}, Value={value_loss:.4f}, Entropy={entropy_loss:.4f}")
            
            # Save model
            if step_count % 2500 == 0 and step_count > 0:
                torch.save({
                    'model': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'step': step_count
                }, os.path.join(self.config.model_save_path, f'a3c_{step_count}.pth'))
        
        # Save final model
        torch.save({
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'timestep': self.config.max_timesteps
        }, os.path.join(self.config.model_save_path, 'a3c_final.pth'))
        
        print('A3C training completed!')
        self.logger.close()


def train_a3c():
    """Main function to train A3C"""
    config = A3CConfig()
    agent = A3CAgent(config)
    agent.train()


if __name__ == '__main__':
    train_a3c()