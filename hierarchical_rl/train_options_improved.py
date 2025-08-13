#!/usr/bin/env python
"""
Improved Options Framework Training Script
==========================================

This script provides enhanced training for the Options framework with:
- Multiple learned options/skills for temporal abstraction
- Option termination conditions and diversity mechanisms
- Intra-option and option selection policies
- Comprehensive option statistics and analysis
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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_airsim.envs.AirGym import AirGymEnv
from hierarchical_rl.envs.hierarchical_airsim_env import HierarchicalAirSimEnv
from hierarchical_rl.options.options_agent import OptionsAgent
from hierarchical_rl.options.options_config import OptionsConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptionsTrainer:
    """Enhanced Options trainer with skill discovery and analysis."""
    
    def __init__(self, config: OptionsConfig, save_dir: str = None):
        self.config = config
        self.save_dir = save_dir or f"results/hierarchical/options_improved_{int(time.time())}"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'option_analysis'), exist_ok=True)
        
        # Save configuration
        with open(os.path.join(self.save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(config), f, default_flow_style=False)
        
        # Initialize environment
        base_env = AirGymEnv()
        self.env = HierarchicalAirSimEnv(
            base_env, 
            goal_dim=config.num_options,  # Use number of options as goal dimension
            max_episode_steps=config.max_episode_steps
        )
        
        # Initialize agent
        self.agent = OptionsAgent(
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
        self.option_statistics = []
        self.diversity_scores = []
        self.termination_rates = []
        
        # Option analysis
        self.option_trajectories = [[] for _ in range(config.num_options)]
        self.option_performance = [[] for _ in range(config.num_options)]
        
        # Best model tracking
        self.best_reward = float('-inf')
        self.best_diversity = 0.0
        self.episodes_without_improvement = 0
        
        logger.info(f"Options trainer initialized. Save directory: {self.save_dir}")
        logger.info(f"Number of options: {config.num_options}")
        logger.info(f"Option length range: {config.option_min_length}-{config.option_max_length}")
        logger.info(f"Diversity bonus: {config.use_diversity_bonus}")
    
    def train(self, num_episodes: int = 1000) -> Dict[str, List[float]]:
        """
        Train Options agent with skill discovery.
        
        Args:
            num_episodes: Number of episodes to train
            
        Returns:
            Training statistics
        """
        logger.info(f"Starting Options training for {num_episodes} episodes")
        
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
            
            # Get option statistics
            option_stats = self.agent.get_option_statistics()
            self.option_statistics.append(option_stats)
            self.diversity_scores.append(option_stats['option_diversity_score'])
            self.termination_rates.append(episode_stats['termination_rate'])
            
            # Store option trajectories for analysis
            if episode % 20 == 0:  # Sample trajectories periodically
                self._collect_option_trajectories(episode)
            
            # Logging
            if episode % self.config.log_freq == 0:
                self._log_episode_stats(episode, episode_stats, option_stats)
            
            # Option analysis and visualization
            if episode % 100 == 0 and episode > 0:
                self._analyze_options(episode)
            
            # Model saving
            if episode % self.config.save_freq == 0:
                self._save_model(episode)
            
            # Save best model (considering both reward and diversity)
            combined_score = episode_stats['total_reward'] + option_stats['option_diversity_score'] * 100
            if combined_score > self.best_reward:
                self.best_reward = combined_score
                self.episodes_without_improvement = 0
                self._save_model(episode, is_best=True)
            else:
                self.episodes_without_improvement += 1
            
            # Early stopping
            if self.episodes_without_improvement > 200:
                logger.info("No improvement for 200 episodes. Stopping training.")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Options training completed in {total_time:.2f} seconds")
        
        # Final analysis
        self._analyze_options(episode, final=True)
        self._save_model(episode, is_final=True)
        self._save_training_stats()
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'diversity_scores': self.diversity_scores,
            'termination_rates': self.termination_rates
        }
    
    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """Run a single episode with option-based control."""
        state = self.env.reset()
        self.agent.reset_episode()
        
        episode_reward = 0.0
        episode_length = 0
        option_changes = 0
        option_lengths = []
        terminations = []
        current_option_length = 0
        
        # Track option usage this episode
        options_used = set()
        option_rewards = {i: 0.0 for i in range(self.config.num_options)}
        
        while episode_length < self.config.max_episode_steps:
            # Select action using options framework
            action, info = self.agent.select_action(state, deterministic=False)
            
            # Execute action
            next_state, reward, done, env_info = self.env.step(action)
            
            # Extract option information
            current_option = info.get('option', 0)
            option_length = info.get('option_length', 0)
            termination_prob = info.get('termination_prob', 0.0)
            intrinsic_reward = info.get('intrinsic_reward', 0.0)
            diversity_bonus = info.get('diversity_bonus', 0.0)
            
            # Track option changes
            if option_length == 1:  # New option started
                if current_option_length > 0:
                    option_lengths.append(current_option_length)
                current_option_length = 1
                option_changes += 1
                options_used.add(current_option)
            else:
                current_option_length += 1
            
            terminations.append(termination_prob)
            option_rewards[current_option] += reward + intrinsic_reward + diversity_bonus
            
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
            
            if done:
                break
        
        # Final update at end of episode
        if episode_length > 0:
            losses = self.agent.update()
            
            if losses:
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f'losses/{loss_name}', loss_value, episode)
        
        # Add final option length
        if current_option_length > 0:
            option_lengths.append(current_option_length)
        
        return {
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'option_changes': option_changes,
            'avg_option_length': np.mean(option_lengths) if option_lengths else 0,
            'termination_rate': np.mean(terminations) if terminations else 0,
            'options_used': len(options_used),
            'option_rewards': option_rewards
        }
    
    def _collect_option_trajectories(self, episode: int) -> None:
        """Collect trajectories for each option for analysis."""
        state = self.env.reset()
        self.agent.reset_episode()
        
        trajectories = {i: [] for i in range(self.config.num_options)}
        current_trajectories = {i: [] for i in range(self.config.num_options)}
        
        episode_length = 0
        while episode_length < 100:  # Shorter episodes for analysis
            action, info = self.agent.select_action(state, deterministic=True)
            next_state, reward, done, env_info = self.env.step(action)
            
            current_option = info.get('option', 0)
            current_trajectories[current_option].append(state[:3])  # Store position
            
            state = next_state
            episode_length += 1
            
            if done:
                break
        
        # Store completed trajectories
        for option_id, traj in current_trajectories.items():
            if len(traj) > 0:
                trajectories[option_id].extend(traj)
                if len(self.option_trajectories[option_id]) < 1000:  # Limit storage
                    self.option_trajectories[option_id].extend(traj)
    
    def _analyze_options(self, episode: int, final: bool = False) -> None:
        """Analyze option behaviors and create visualizations."""
        try:
            # Get current option statistics
            option_stats = self.agent.get_option_statistics()
            
            # Create option usage visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Options Analysis - Episode {episode}', fontsize=16)
            
            # Option usage frequency
            axes[0, 0].bar(range(self.config.num_options), option_stats['option_frequencies'])
            axes[0, 0].set_title('Option Usage Frequency')
            axes[0, 0].set_xlabel('Option ID')
            axes[0, 0].set_ylabel('Frequency')
            
            # Option visit counts
            axes[0, 1].bar(range(self.config.num_options), option_stats['option_visit_counts'])
            axes[0, 1].set_title('Option Visit Counts')
            axes[0, 1].set_xlabel('Option ID')
            axes[0, 1].set_ylabel('Visits')
            
            # Diversity score over time
            if len(self.diversity_scores) > 0:
                axes[1, 0].plot(self.diversity_scores)
                axes[1, 0].set_title('Option Diversity Score Over Time')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Diversity Score')
            
            # Episode rewards over time
            if len(self.episode_rewards) > 0:
                axes[1, 1].plot(self.episode_rewards)
                axes[1, 1].set_title('Episode Rewards Over Time')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Reward')
            
            plt.tight_layout()
            analysis_path = os.path.join(self.save_dir, 'option_analysis', f'analysis_episode_{episode}.png')
            plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create trajectory visualization if we have enough data
            self._visualize_option_trajectories(episode)
            
            logger.info(f"Option analysis saved for episode {episode}")
            
        except Exception as e:
            logger.warning(f"Failed to create option analysis: {e}")
    
    def _visualize_option_trajectories(self, episode: int) -> None:
        """Visualize option trajectories in 3D space."""
        try:
            fig = plt.figure(figsize=(15, 10))
            
            # 3D trajectory plot
            ax1 = fig.add_subplot(121, projection='3d')
            colors = plt.cm.tab10(np.linspace(0, 1, self.config.num_options))
            
            for option_id, trajectories in enumerate(self.option_trajectories):
                if len(trajectories) > 0:
                    trajectories_array = np.array(trajectories)
                    ax1.scatter(trajectories_array[:, 0], trajectories_array[:, 1], 
                               trajectories_array[:, 2], c=[colors[option_id]], 
                               label=f'Option {option_id}', s=10, alpha=0.6)
            
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.set_zlabel('Z Position')
            ax1.set_title(f'Option Trajectories in 3D Space (Episode {episode})')
            ax1.legend()
            
            # t-SNE visualization of option states (2D projection)
            ax2 = fig.add_subplot(122)
            
            # Collect all states for t-SNE
            all_states = []
            all_labels = []
            for option_id, trajectories in enumerate(self.option_trajectories):
                if len(trajectories) > 10:  # Only include options with sufficient data
                    states_sample = trajectories[-50:]  # Use recent states
                    all_states.extend(states_sample)
                    all_labels.extend([option_id] * len(states_sample))
            
            if len(all_states) > 20:  # Need sufficient points for t-SNE
                try:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_states)//4))
                    states_2d = tsne.fit_transform(np.array(all_states))
                    
                    for option_id in range(self.config.num_options):
                        mask = np.array(all_labels) == option_id
                        if np.any(mask):
                            ax2.scatter(states_2d[mask, 0], states_2d[mask, 1], 
                                       c=[colors[option_id]], label=f'Option {option_id}', 
                                       s=20, alpha=0.7)
                    
                    ax2.set_title('t-SNE Visualization of Option States')
                    ax2.legend()
                except Exception as e:
                    ax2.text(0.5, 0.5, f't-SNE failed: {str(e)[:50]}...', 
                            transform=ax2.transAxes, ha='center', va='center')
            else:
                ax2.text(0.5, 0.5, 'Insufficient data for t-SNE', 
                        transform=ax2.transAxes, ha='center', va='center')
            
            plt.tight_layout()
            trajectory_path = os.path.join(self.save_dir, 'option_analysis', f'trajectories_episode_{episode}.png')
            plt.savefig(trajectory_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to visualize trajectories: {e}")
    
    def _log_episode_stats(self, episode: int, stats: Dict[str, Any], option_stats: Dict[str, Any]) -> None:
        """Log episode and option statistics."""
        # Console logging
        logger.info(
            f"Episode {episode:4d} | "
            f"Reward: {stats['total_reward']:8.2f} | "
            f"Length: {stats['episode_length']:3d} | "
            f"Changes: {stats['option_changes']:2d} | "
            f"Options: {stats['options_used']}/{self.config.num_options} | "
            f"Diversity: {option_stats['option_diversity_score']:.3f}"
        )
        
        # TensorBoard logging
        self.writer.add_scalar('episode/total_reward', stats['total_reward'], episode)
        self.writer.add_scalar('episode/length', stats['episode_length'], episode)
        self.writer.add_scalar('episode/option_changes', stats['option_changes'], episode)
        self.writer.add_scalar('episode/avg_option_length', stats['avg_option_length'], episode)
        self.writer.add_scalar('episode/termination_rate', stats['termination_rate'], episode)
        self.writer.add_scalar('episode/options_used', stats['options_used'], episode)
        
        # Option statistics
        self.writer.add_scalar('options/diversity_score', option_stats['option_diversity_score'], episode)
        self.writer.add_scalar('options/total_steps', option_stats['total_steps'], episode)
        
        # Individual option usage
        for i, freq in enumerate(option_stats['option_frequencies']):
            self.writer.add_scalar(f'option_usage/option_{i}', freq, episode)
        
        # Option rewards
        for option_id, reward in stats['option_rewards'].items():
            self.writer.add_scalar(f'option_rewards/option_{option_id}', reward, episode)
        
        # Running averages
        if len(self.episode_rewards) >= 50:
            avg_reward = np.mean(self.episode_rewards[-50:])
            avg_length = np.mean(self.episode_lengths[-50:])
            avg_diversity = np.mean(self.diversity_scores[-50:])
            
            self.writer.add_scalar('running_avg/reward', avg_reward, episode)
            self.writer.add_scalar('running_avg/length', avg_length, episode)
            self.writer.add_scalar('running_avg/diversity', avg_diversity, episode)
    
    def _save_model(self, episode: int, is_best: bool = False, is_final: bool = False) -> None:
        """Save model checkpoint."""
        if is_best:
            filepath = os.path.join(self.save_dir, 'best_model.pth')
            logger.info(f"Saving best Options model at episode {episode}")
        elif is_final:
            filepath = os.path.join(self.save_dir, 'final_model.pth')
            logger.info(f"Saving final Options model at episode {episode}")
        else:
            filepath = os.path.join(self.save_dir, f'model_episode_{episode}.pth')
        
        self.agent.save(filepath)
    
    def _save_training_stats(self) -> None:
        """Save training statistics."""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'diversity_scores': self.diversity_scores,
            'termination_rates': self.termination_rates,
            'option_statistics': self.option_statistics
        }
        
        stats_path = os.path.join(self.save_dir, 'training_stats.npz')
        np.savez(stats_path, **stats)
        logger.info(f"Training statistics saved to {stats_path}")
    
    def evaluate(self, num_episodes: int = 10, model_path: str = None) -> Dict[str, float]:
        """Evaluate trained Options agent."""
        if model_path:
            self.agent.load(model_path)
            logger.info(f"Loaded Options model from {model_path}")
        
        eval_rewards = []
        eval_lengths = []
        eval_option_changes = []
        eval_diversity_scores = []
        
        logger.info(f"Evaluating Options agent for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            self.agent.reset_episode()
            
            episode_reward = 0.0
            episode_length = 0
            option_changes = 0
            
            while episode_length < self.config.max_episode_steps:
                action, info = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, env_info = self.env.step(action)
                
                # Track option changes
                option_length = info.get('option_length', 0)
                if option_length == 1:
                    option_changes += 1
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            option_stats = self.agent.get_option_statistics()
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_option_changes.append(option_changes)
            eval_diversity_scores.append(option_stats['option_diversity_score'])
            
            logger.info(
                f"Eval Episode {episode:2d} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Changes: {option_changes:2d} | "
                f"Diversity: {option_stats['option_diversity_score']:.3f}"
            )
        
        results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'mean_option_changes': np.mean(eval_option_changes),
            'mean_diversity': np.mean(eval_diversity_scores)
        }
        
        logger.info("Options Evaluation Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return results


def main():
    """Main training function for Options framework."""
    # Enhanced configuration for Options
    config = OptionsConfig(
        # Environment
        max_episode_steps=200,
        
        # Options parameters
        num_options=6,
        option_min_length=4,
        option_max_length=15,
        
        # Option discovery
        use_diversity_bonus=True,
        diversity_coef=0.1,
        use_mutual_info=True,
        mi_coef=0.05,
        
        # Network architectures
        policy_hidden_dims=[256, 256],
        option_policy_hidden_dims=[256, 128],
        termination_hidden_dims=[128, 64],
        
        # Training parameters
        batch_size=128,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        
        # PPO parameters
        ppo_epochs=4,
        clip_param=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        
        # Option-critic parameters
        termination_reg=0.01,
        deliberation_cost=0.0,
        
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
    trainer = OptionsTrainer(config)
    
    try:
        # Train agent
        training_stats = trainer.train(num_episodes=500)
        
        # Evaluate final performance
        eval_results = trainer.evaluate(num_episodes=20)
        
        print("\nOptions training completed successfully!")
        print(f"Final evaluation results: {eval_results}")
        
    except KeyboardInterrupt:
        logger.info("Options training interrupted by user")
    except Exception as e:
        logger.error(f"Options training failed with error: {e}")
        raise
    finally:
        # Cleanup
        trainer.writer.close()


if __name__ == "__main__":
    main()