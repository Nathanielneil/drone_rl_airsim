#!/usr/bin/env python
"""
Improved HIRO (HIerarchical RL with Off-policy correction) Training Script
=========================================================================

This script provides enhanced training for the HIRO algorithm with:
- Proper action space integration for AirSim
- Off-policy correction mechanisms
- Advanced HER (Hindsight Experience Replay)
- Comprehensive logging and monitoring
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
from hierarchical_rl.hiro.hiro_agent import HIROAgent
from hierarchical_rl.hiro.hiro_config import HIROConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HIROTrainer:
    """Enhanced HIRO trainer with improved features."""
    
    def __init__(self, config: HIROConfig, save_dir: str = None):
        self.config = config
        self.save_dir = save_dir or f"results/hierarchical/hiro_improved_{int(time.time())}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(self.save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(config), f, default_flow_style=False)
        
        # Initialize environment with hierarchical wrapper
        base_env = AirGymEnv()
        self.env = HierarchicalAirSimEnv(
            base_env, 
            goal_dim=config.subgoal_dim,
            max_episode_steps=config.max_episode_steps
        )
        
        # Initialize agent
        self.agent = HIROAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
            config=config
        )
        
        # TensorBoard logging
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'logs'))
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.high_level_rewards = []
        self.low_level_rewards = []
        self.subgoal_distances = []
        
        # Best model tracking
        self.best_reward = float('-inf')
        self.episodes_without_improvement = 0
        
        logger.info(f"HIRO trainer initialized. Save directory: {self.save_dir}")
        logger.info(f"Environment: {self.env}")
        logger.info(f"State dim: {self.env.observation_space.shape[0]}")
        logger.info(f"Action dim: {self.env.action_space.shape[0]}")
        logger.info(f"Subgoal dim: {config.subgoal_dim}")
    
    def train(self, num_episodes: int = 1000) -> Dict[str, List[float]]:
        """
        Train HIRO agent with enhanced features.
        
        Args:
            num_episodes: Number of episodes to train
            
        Returns:
            Training statistics
        """
        logger.info(f"Starting HIRO training for {num_episodes} episodes")
        
        # Set random seeds for reproducibility
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
            self.success_rates.append(float(episode_stats['success']))
            self.high_level_rewards.append(episode_stats['high_level_reward'])
            self.low_level_rewards.append(episode_stats['low_level_reward'])
            self.subgoal_distances.append(episode_stats['avg_subgoal_distance'])
            
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
            
            # Early stopping check
            if self.episodes_without_improvement > 100:
                logger.info(f"No improvement for 100 episodes. Stopping training.")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model and statistics
        self._save_model(episode, is_final=True)
        self._save_training_stats()
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates,
            'high_level_rewards': self.high_level_rewards,
            'low_level_rewards': self.low_level_rewards
        }
    
    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """Run a single episode and collect statistics."""
        state = self.env.reset()
        self.agent.reset_episode()
        
        episode_reward = 0.0
        episode_length = 0
        high_level_reward = 0.0
        low_level_reward = 0.0
        subgoal_distances = []
        success = False
        
        # Episode-level statistics
        subgoal_changes = 0
        collision_count = 0
        
        while episode_length < self.config.max_episode_steps:
            # Select action
            action, info = self.agent.select_action(state, deterministic=False)
            
            # Execute action
            next_state, reward, done, env_info = self.env.step(action)
            
            # Calculate subgoal distance
            current_pos = next_state[:self.config.subgoal_dim]
            subgoal = info.get('subgoal', np.zeros(self.config.subgoal_dim))
            subgoal_distance = np.linalg.norm(current_pos - subgoal)
            subgoal_distances.append(subgoal_distance)
            
            # Track subgoal changes
            if info.get('subgoal_step', 0) == self.config.subgoal_freq - 1:
                subgoal_changes += 1
            
            # Track collisions
            if env_info.get('collision', False):
                collision_count += 1
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done, info)
            
            # Update networks
            if episode > 10:  # Start training after some initial episodes
                losses = self.agent.update()
                
                # Log training losses
                if losses and episode_length % 50 == 0:
                    for loss_name, loss_value in losses.items():
                        self.writer.add_scalar(f'losses/{loss_name}', loss_value, 
                                             episode * self.config.max_episode_steps + episode_length)
            
            # Update state and statistics
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Separate high-level and low-level rewards
            if info.get('subgoal_step', 0) == 0:  # High-level reward
                high_level_reward += reward
            else:  # Low-level reward
                low_level_reward += info.get('intrinsic_reward', 0)
            
            # Success check (reaching goal)
            if env_info.get('goal_reached', False):
                success = True
                break
            
            if done:
                break
        
        # End episode
        self.agent.end_episode()
        
        return {
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'high_level_reward': high_level_reward,
            'low_level_reward': low_level_reward,
            'avg_subgoal_distance': np.mean(subgoal_distances) if subgoal_distances else 0.0,
            'subgoal_changes': subgoal_changes,
            'collision_count': collision_count,
            'success': success
        }
    
    def _log_episode_stats(self, episode: int, stats: Dict[str, Any]) -> None:
        """Log episode statistics."""
        # Console logging
        logger.info(
            f"Episode {episode:4d} | "
            f"Reward: {stats['total_reward']:8.2f} | "
            f"Length: {stats['episode_length']:3d} | "
            f"Success: {stats['success']} | "
            f"Subgoals: {stats['subgoal_changes']:2d} | "
            f"Avg Dist: {stats['avg_subgoal_distance']:.2f}"
        )
        
        # TensorBoard logging
        self.writer.add_scalar('episode/total_reward', stats['total_reward'], episode)
        self.writer.add_scalar('episode/length', stats['episode_length'], episode)
        self.writer.add_scalar('episode/success_rate', float(stats['success']), episode)
        self.writer.add_scalar('episode/high_level_reward', stats['high_level_reward'], episode)
        self.writer.add_scalar('episode/low_level_reward', stats['low_level_reward'], episode)
        self.writer.add_scalar('episode/avg_subgoal_distance', stats['avg_subgoal_distance'], episode)
        self.writer.add_scalar('episode/subgoal_changes', stats['subgoal_changes'], episode)
        self.writer.add_scalar('episode/collision_count', stats['collision_count'], episode)
        
        # Running averages (last 100 episodes)
        if len(self.episode_rewards) >= 100:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
            avg_success = np.mean(self.success_rates[-100:])
            
            self.writer.add_scalar('running_avg/reward', avg_reward, episode)
            self.writer.add_scalar('running_avg/length', avg_length, episode)
            self.writer.add_scalar('running_avg/success_rate', avg_success, episode)
    
    def _save_model(self, episode: int, is_best: bool = False, is_final: bool = False) -> None:
        """Save model checkpoint."""
        if is_best:
            filepath = os.path.join(self.save_dir, 'best_model.pth')
            logger.info(f"Saving best model at episode {episode}")
        elif is_final:
            filepath = os.path.join(self.save_dir, 'final_model.pth')
            logger.info(f"Saving final model at episode {episode}")
        else:
            filepath = os.path.join(self.save_dir, f'model_episode_{episode}.pth')
        
        self.agent.save(filepath)
    
    def _save_training_stats(self) -> None:
        """Save training statistics."""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates,
            'high_level_rewards': self.high_level_rewards,
            'low_level_rewards': self.low_level_rewards,
            'subgoal_distances': self.subgoal_distances
        }
        
        stats_path = os.path.join(self.save_dir, 'training_stats.npz')
        np.savez(stats_path, **stats)
        logger.info(f"Training statistics saved to {stats_path}")
    
    def evaluate(self, num_episodes: int = 10, model_path: str = None) -> Dict[str, float]:
        """Evaluate trained agent."""
        if model_path:
            self.agent.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        
        logger.info(f"Evaluating agent for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            self.agent.reset_episode()
            
            episode_reward = 0.0
            episode_length = 0
            success = False
            
            while episode_length < self.config.max_episode_steps:
                action, info = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, env_info = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if env_info.get('goal_reached', False):
                    success = True
                    break
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_successes.append(float(success))
            
            logger.info(f"Eval Episode {episode:2d} | Reward: {episode_reward:8.2f} | Success: {success}")
        
        results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': np.mean(eval_successes)
        }
        
        logger.info("Evaluation Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return results


def main():
    """Main training function."""
    # Configuration
    config = HIROConfig(
        # Environment
        max_episode_steps=200,
        
        # HIRO parameters
        subgoal_dim=3,
        subgoal_freq=10,
        subgoal_scale=8.0,
        
        # HER parameters
        her_ratio=0.8,
        off_policy_correction=True,
        correction_radius=2.0,
        
        # Training
        batch_size=128,
        gamma=0.95,
        tau=0.005,
        buffer_size=500000,
        min_buffer_size=5000,
        
        # Learning rates
        high_level_lr=3e-4,
        low_level_lr=3e-4,
        
        # Exploration
        high_level_noise=0.3,
        low_level_noise=0.2,
        noise_decay=0.9995,
        min_noise=0.05,
        
        # Training schedule
        train_freq=1,
        update_freq=2,
        high_level_train_freq=2,
        
        # Logging and saving
        log_freq=10,
        save_freq=50,
        
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42
    )
    
    # Create trainer
    trainer = HIROTrainer(config)
    
    try:
        # Train agent
        training_stats = trainer.train(num_episodes=500)
        
        # Evaluate final performance
        eval_results = trainer.evaluate(num_episodes=20)
        
        print("\nTraining completed successfully!")
        print(f"Final evaluation results: {eval_results}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Cleanup
        trainer.writer.close()


if __name__ == "__main__":
    main()