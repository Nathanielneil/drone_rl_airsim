#!/usr/bin/env python
"""
Improved FuN (FeUdal Networks) Training Script
==============================================

This script provides enhanced training for FuN algorithm with:
- Proper manager-worker hierarchy integration
- Intrinsic motivation through cosine similarity
- Dilated LSTM for temporal abstraction
- Advanced state embedding and goal generation
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import logging
from typing import Dict, List, Tuple, Any
import yaml

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_airsim.envs.AirGym import AirGymEnv
from hierarchical_rl.envs.hierarchical_airsim_env import HierarchicalAirSimEnv
from hierarchical_rl.fun.fun_agent import FuNAgent
from hierarchical_rl.fun.fun_config import FuNConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuNTrainer:
    """Enhanced FuN trainer with feudal architecture."""
    
    def __init__(self, config: FuNConfig, save_dir: str = None):
        self.config = config
        self.save_dir = save_dir or f"results/hierarchical/fun_improved_{int(time.time())}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(self.save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(config), f, default_flow_style=False)
        
        # Initialize environment
        base_env = AirGymEnv()
        self.env = HierarchicalAirSimEnv(
            base_env, 
            goal_dim=config.goal_dim,
            max_episode_steps=config.max_episode_steps
        )
        
        # Initialize agent
        self.agent = FuNAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
            config=config,
            continuous_actions=True
        )
        
        # TensorBoard logging
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'logs'))
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.intrinsic_rewards = []
        self.manager_rewards = []
        self.worker_rewards = []
        self.goal_similarities = []
        
        # Best model tracking
        self.best_reward = float('-inf')
        self.episodes_without_improvement = 0
        
        logger.info(f"FuN trainer initialized. Save directory: {self.save_dir}")
        logger.info(f"Manager horizon: {config.manager_horizon}")
        logger.info(f"Goal dimension: {config.goal_dim}")
        logger.info(f"Embedding dimension: {config.embedding_dim}")
    
    def train(self, num_episodes: int = 1000) -> Dict[str, List[float]]:
        """
        Train FuN agent with manager-worker hierarchy.
        
        Args:
            num_episodes: Number of episodes to train
            
        Returns:
            Training statistics
        """
        logger.info(f"Starting FuN training for {num_episodes} episodes")
        
        # Set random seeds
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_stats = self._run_episode(episode)
            
            # Update statistics
            self.episode_rewards.append(episode_stats['total_reward'])
            self.episode_lengths.append(episode_stats['episode_length'])
            self.intrinsic_rewards.append(episode_stats['intrinsic_reward'])
            self.manager_rewards.append(episode_stats['manager_reward'])
            self.worker_rewards.append(episode_stats['worker_reward'])
            self.goal_similarities.append(episode_stats['avg_goal_similarity'])
            
            # Logging
            if episode % self.config.log_freq == 0:
                self._log_episode_stats(episode, episode_stats)
            
            # Model saving
            if episode % self.config.save_freq == 0:
                self._save_model(episode)
            
            # Save best model
            if episode_stats['total_reward'] > self.best_reward:
                self.best_reward = episode_stats['total_reward']
                self.episodes_without_improvement = 0
                self._save_model(episode, is_best=True)
            else:
                self.episodes_without_improvement += 1
            
            # Early stopping
            if self.episodes_without_improvement > 150:
                logger.info("No improvement for 150 episodes. Stopping training.")
                break
        
        total_time = time.time() - start_time
        logger.info(f"FuN training completed in {total_time:.2f} seconds")
        
        # Save final model and statistics
        self._save_model(episode, is_final=True)
        self._save_training_stats()
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'intrinsic_rewards': self.intrinsic_rewards,
            'manager_rewards': self.manager_rewards,
            'worker_rewards': self.worker_rewards,
            'goal_similarities': self.goal_similarities
        }
    
    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """Run a single episode with feudal hierarchy."""
        state = self.env.reset()
        self.agent.reset_episode()
        
        episode_reward = 0.0
        episode_length = 0
        intrinsic_reward_total = 0.0
        manager_reward_total = 0.0
        worker_reward_total = 0.0
        goal_similarities = []
        
        # Manager decision tracking
        manager_decisions = 0
        goal_consistency_scores = []
        
        while episode_length < self.config.max_episode_steps:
            # Select action using manager-worker hierarchy
            action, info = self.agent.select_action(state, deterministic=False)
            
            # Execute action
            next_state, reward, done, env_info = self.env.step(action)
            
            # Extract hierarchical information
            goal = info.get('goal', np.zeros(self.config.goal_dim))
            intrinsic_reward = info.get('intrinsic_reward', 0.0)
            manager_value = info.get('manager_value', 0.0)
            worker_value = info.get('worker_value', 0.0)
            
            # Calculate goal consistency (cosine similarity)
            if len(goal_similarities) > 0:
                prev_goal = goal_similarities[-1]
                consistency = np.dot(goal, prev_goal) / (np.linalg.norm(goal) * np.linalg.norm(prev_goal) + 1e-8)
                goal_consistency_scores.append(consistency)
            
            goal_similarities.append(goal.copy())
            
            # Track manager decisions
            if episode_length % self.config.manager_horizon == 0:
                manager_decisions += 1
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done, info)
            
            # Update agent (batch learning)
            if episode_length % self.config.buffer_size == 0 and episode_length > 0:
                losses = self.agent.update()
                
                # Log training losses
                if losses:
                    for loss_name, loss_value in losses.items():
                        self.writer.add_scalar(f'losses/{loss_name}', loss_value,
                                             episode * self.config.max_episode_steps + episode_length)
            
            # Update statistics
            state = next_state
            episode_reward += reward
            episode_length += 1
            intrinsic_reward_total += intrinsic_reward
            manager_reward_total += reward if episode_length % self.config.manager_horizon == 0 else 0
            worker_reward_total += intrinsic_reward
            
            if done:
                break
        
        # Final update at end of episode
        if episode_length > 0:
            losses = self.agent.update()
            
            if losses:
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f'losses/{loss_name}', loss_value, episode)
        
        return {
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'intrinsic_reward': intrinsic_reward_total,
            'manager_reward': manager_reward_total,
            'worker_reward': worker_reward_total,
            'avg_goal_similarity': np.mean(goal_similarities) if goal_similarities else 0.0,
            'manager_decisions': manager_decisions,
            'goal_consistency': np.mean(goal_consistency_scores) if goal_consistency_scores else 0.0
        }
    
    def _log_episode_stats(self, episode: int, stats: Dict[str, Any]) -> None:
        """Log episode statistics."""
        # Console logging
        logger.info(
            f"Episode {episode:4d} | "
            f"Reward: {stats['total_reward']:8.2f} | "
            f"Length: {stats['episode_length']:3d} | "
            f"Intrinsic: {stats['intrinsic_reward']:6.2f} | "
            f"Manager: {stats['manager_decisions']:2d} | "
            f"Goal Sim: {stats['avg_goal_similarity']:.3f}"
        )
        
        # TensorBoard logging
        self.writer.add_scalar('episode/total_reward', stats['total_reward'], episode)
        self.writer.add_scalar('episode/length', stats['episode_length'], episode)
        self.writer.add_scalar('episode/intrinsic_reward', stats['intrinsic_reward'], episode)
        self.writer.add_scalar('episode/manager_reward', stats['manager_reward'], episode)
        self.writer.add_scalar('episode/worker_reward', stats['worker_reward'], episode)
        self.writer.add_scalar('episode/avg_goal_similarity', stats['avg_goal_similarity'], episode)
        self.writer.add_scalar('episode/manager_decisions', stats['manager_decisions'], episode)
        self.writer.add_scalar('episode/goal_consistency', stats.get('goal_consistency', 0.0), episode)
        
        # Running averages
        if len(self.episode_rewards) >= 50:
            avg_reward = np.mean(self.episode_rewards[-50:])
            avg_length = np.mean(self.episode_lengths[-50:])
            avg_intrinsic = np.mean(self.intrinsic_rewards[-50:])
            avg_goal_sim = np.mean(self.goal_similarities[-50:])
            
            self.writer.add_scalar('running_avg/reward', avg_reward, episode)
            self.writer.add_scalar('running_avg/length', avg_length, episode)
            self.writer.add_scalar('running_avg/intrinsic_reward', avg_intrinsic, episode)
            self.writer.add_scalar('running_avg/goal_similarity', avg_goal_sim, episode)
    
    def _save_model(self, episode: int, is_best: bool = False, is_final: bool = False) -> None:
        """Save model checkpoint."""
        if is_best:
            filepath = os.path.join(self.save_dir, 'best_model.pth')
            logger.info(f"Saving best FuN model at episode {episode}")
        elif is_final:
            filepath = os.path.join(self.save_dir, 'final_model.pth')
            logger.info(f"Saving final FuN model at episode {episode}")
        else:
            filepath = os.path.join(self.save_dir, f'model_episode_{episode}.pth')
        
        self.agent.save(filepath)
    
    def _save_training_stats(self) -> None:
        """Save training statistics."""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'intrinsic_rewards': self.intrinsic_rewards,
            'manager_rewards': self.manager_rewards,
            'worker_rewards': self.worker_rewards,
            'goal_similarities': self.goal_similarities
        }
        
        stats_path = os.path.join(self.save_dir, 'training_stats.npz')
        np.savez(stats_path, **stats)
        logger.info(f"Training statistics saved to {stats_path}")
    
    def evaluate(self, num_episodes: int = 10, model_path: str = None) -> Dict[str, float]:
        """Evaluate trained FuN agent."""
        if model_path:
            self.agent.load(model_path)
            logger.info(f"Loaded FuN model from {model_path}")
        
        eval_rewards = []
        eval_lengths = []
        eval_intrinsic_rewards = []
        eval_goal_similarities = []
        
        logger.info(f"Evaluating FuN agent for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            self.agent.reset_episode()
            
            episode_reward = 0.0
            episode_length = 0
            intrinsic_reward_total = 0.0
            goal_similarities = []
            
            while episode_length < self.config.max_episode_steps:
                action, info = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, env_info = self.env.step(action)
                
                # Extract information
                intrinsic_reward = info.get('intrinsic_reward', 0.0)
                goal = info.get('goal', np.zeros(self.config.goal_dim))
                
                goal_similarities.append(np.linalg.norm(goal))
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                intrinsic_reward_total += intrinsic_reward
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_intrinsic_rewards.append(intrinsic_reward_total)
            eval_goal_similarities.append(np.mean(goal_similarities) if goal_similarities else 0.0)
            
            logger.info(
                f"Eval Episode {episode:2d} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Intrinsic: {intrinsic_reward_total:6.2f}"
            )
        
        results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'mean_intrinsic_reward': np.mean(eval_intrinsic_rewards),
            'mean_goal_similarity': np.mean(eval_goal_similarities)
        }
        
        logger.info("FuN Evaluation Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return results


def main():
    """Main training function for FuN."""
    # Enhanced configuration for FuN
    config = FuNConfig(
        # Environment
        max_episode_steps=200,
        
        # FuN architecture
        manager_horizon=8,
        embedding_dim=256,
        goal_dim=16,
        
        # Network architectures
        manager_hidden_dims=[512, 256],
        worker_hidden_dims=[256, 256],
        
        # Learning rates
        manager_lr=3e-4,
        worker_lr=3e-4,
        
        # Intrinsic motivation
        alpha=0.5,  # Intrinsic reward coefficient
        dilation=10,
        
        # Training parameters
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        
        # PPO parameters
        ppo_epochs=4,
        clip_param=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        
        # Buffer
        buffer_size=1024,
        
        # Training schedule
        train_freq=1,
        log_freq=10,
        save_freq=50,
        
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42
    )
    
    # Create trainer
    trainer = FuNTrainer(config)
    
    try:
        # Train agent
        training_stats = trainer.train(num_episodes=500)
        
        # Evaluate final performance
        eval_results = trainer.evaluate(num_episodes=20)
        
        print("\nFuN training completed successfully!")
        print(f"Final evaluation results: {eval_results}")
        
    except KeyboardInterrupt:
        logger.info("FuN training interrupted by user")
    except Exception as e:
        logger.error(f"FuN training failed with error: {e}")
        raise
    finally:
        # Cleanup
        trainer.writer.close()


if __name__ == "__main__":
    main()