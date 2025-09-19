"""
DDPG (Deep Deterministic Policy Gradient) Implementation for UAV Navigation
PyTorch implementation adapted for AirSim environment
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random
from collections import deque
from pathlib import Path
from tensorboardX import SummaryWriter
from tqdm import trange

# Add path to import AirSim environment
from gym_airsim.envs.AirGym import AirSimEnv


class ReplayBuffer:
    """Experience Replay Buffer for DDPG"""
    
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0
    
    def add(self, state, action, next_state, reward, done):
        data = (state, action, next_state, reward, done)
        
        if len(self.storage) == self.max_size:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)
    
    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        states, actions, next_states, rewards, dones = map(np.stack, zip(*batch))
        
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(next_states),
            torch.FloatTensor(rewards),
            torch.FloatTensor(dones)
        )
    
    def size(self):
        return len(self.storage)


class Actor(nn.Module):
    """Actor Network for DDPG"""
    
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=400):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action
    
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    """Twin Critic Network for DDPG (inspired by TD3)"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1


class OUNoise:
    """Ornstein-Uhlenbeck noise for exploration"""
    
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class DDPG:
    """DDPG Agent"""
    
    def __init__(self, state_dim, action_dim, max_action, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        self.max_action = max_action
        self.noise = OUNoise(action_dim)
    
    def select_action(self, state, add_noise=True):
        """Select action using current policy"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            action = action + self.noise.noise()
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def train(self, replay_buffer, batch_size=256):
        """Train the DDPG agent with Twin Critics"""
        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        
        # Compute the target Q value using twin critics
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)  # Take minimum for conservative estimate
            target_Q = reward + (1 - done) * self.config.gamma * target_Q
        
        # Get current Q estimates from both critics
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Compute critic loss for both critics
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute actor loss using Q1 only (like TD3)
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        return actor_loss.item(), critic_loss.item()
    
    def save(self, filename):
        """Save model parameters"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)
    
    def load(self, filename):
        """Load model parameters"""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


class DDPGConfig:
    """Configuration for DDPG"""
    def __init__(self):
        # Network parameters
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.gamma = 0.99
        self.tau = 0.005
        
        # Training parameters
        self.batch_size = 64
        self.start_timesteps = 10000
        self.max_timesteps = 100000
        
        # Environment
        self.env_name = 'AirGym'
        
        # Logging and saving
        self.save_freq = 10000
        self.log_freq = 1000
        self.model_save_path = 'models/ddpg/'
        self.log_dir = 'logs/ddpg/'


def train_ddpg():
    """Main DDPG training function"""
    config = DDPGConfig()
    
    # Create directories
    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Environment
    env = AirSimEnv(need_render=False)
    # AirSimEnv returns [image, inform_vector], we use inform_vector for DDPG
    state_dim = 9  # inform_vector dimension: relative_position(2) + velocity(3) + pry(3) + r_yaw(1)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Max action: {max_action}")
    
    # Agent
    agent = DDPG(state_dim, action_dim, max_action, config)
    
    # Replay buffer
    replay_buffer = ReplayBuffer()
    
    # Logger
    logger = SummaryWriter(config.log_dir)
    
    # Training loop
    obs = env.reset()
    state = obs[1]  # Use only inform_vector for DDPG
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    print(f"Starting DDPG training on {agent.device}")
    
    for t in trange(config.max_timesteps):
        episode_timesteps += 1
        
        # Select action randomly or according to policy
        if t < config.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(np.array(state))
        
        # Ensure action is properly formatted
        action = np.array(action, dtype=np.float32)
        
        # Perform action
        next_obs, reward, done, _ = env.step(action)
        next_state = next_obs[1]  # Use only inform_vector
        
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, float(done))
        
        state = next_state
        episode_reward += reward
        
        # Train agent after collecting sufficient data
        if t >= config.start_timesteps and replay_buffer.size() >= config.batch_size:
            actor_loss, critic_loss = agent.train(replay_buffer, config.batch_size)
            
            if t % config.log_freq == 0:
                logger.add_scalar('loss/actor', actor_loss, t)
                logger.add_scalar('loss/critic', critic_loss, t)
        
        if done:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            logger.add_scalar('episode/reward', episode_reward, episode_num)
            
            # Reset environment and noise
            obs = env.reset()
            state = obs[1]  # Use only inform_vector
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            agent.noise.reset()
        
        # Save model
        if (t + 1) % config.save_freq == 0:
            agent.save(os.path.join(config.model_save_path, f'ddpg_{t+1}.pth'))
    
    # Save final model
    agent.save(os.path.join(config.model_save_path, 'ddpg_final.pth'))
    print('DDPG training completed!')


if __name__ == '__main__':
    train_ddpg()