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

# Add path to import AirSim environment
sys.path.append(os.path.join(os.path.dirname(__file__), 'gym_airsim'))
from envs.AirGym import AirGym

from utils.storage import RolloutStorage
from utils.logger import Logger
from algorithm.model import CNNPolicy


class A3CNetwork(nn.Module):
    """Shared network for A3C with separate actor and critic heads"""
    
    def __init__(self, input_shape, num_actions, hidden_size=512):
        super(A3CNetwork, self).__init__()
        
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of flattened features
        conv_out_size = self._get_conv_out_size(input_shape)
        
        # LSTM layer
        self.lstm = nn.LSTMCell(conv_out_size, hidden_size)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, num_actions)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_conv_out_size(self, shape):
        """Calculate the output size of convolutional layers"""
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(np.prod(o.size()))
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, hidden_state):
        """Forward pass through the network"""
        batch_size = x.size(0)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        
        # LSTM
        hx, cx = hidden_state
        hx, cx = self.lstm(x, (hx, cx))
        
        # Actor and critic outputs
        policy_logits = self.actor(hx)
        value = self.critic(hx)
        
        return policy_logits, value, (hx, cx)
    
    def init_hidden(self, batch_size):
        """Initialize hidden state for LSTM"""
        weight = next(self.parameters()).data
        return (weight.new(batch_size, 512).zero_(),
                weight.new(batch_size, 512).zero_())


class A3CConfig:
    """Configuration for A3C"""
    def __init__(self):
        # Network architecture
        self.input_shape = (4, 84, 84)  # 4 stacked frames, 84x84 resolution
        self.num_actions = 7  # AirSim discrete actions
        self.hidden_size = 512
        
        # Training parameters
        self.lr = 1e-4
        self.gamma = 0.99
        self.n_steps = 20
        self.max_episodes = 10000
        self.max_episode_steps = 1000
        self.num_workers = 4
        
        # Loss coefficients
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        # Environment
        self.env_name = 'AirGym'
        
        # Model saving
        self.model_save_path = 'models/a3c/'
        self.log_dir = 'logs/a3c/'


if __name__ == '__main__':
    print("A3C implementation for UAV navigation")
    print("Use this module to train A3C agent in AirSim environment")