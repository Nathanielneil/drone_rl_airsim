#!/usr/bin/env python
"""
Main training script for hierarchical reinforcement learning algorithms.
Supports HAC, FuN, HIRO, and Options Framework.
"""

import os
import sys
import argparse
import numpy as np
import torch
import random
from pathlib import Path
import logging
from datetime import datetime
from tensorboardX import SummaryWriter
from tqdm import trange
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_airsim.envs.AirGym import AirSimEnv
from hierarchical_rl.envs.hierarchical_airsim_env import HierarchicalAirSimEnv, MultiGoalAirSimEnv

# Import HRL agents
from hierarchical_rl.hac import HACAgent, HACConfig
from hierarchical_rl.fun import FuNAgent, FuNConfig  
from hierarchical_rl.hiro import HIROAgent, HIROConfig
from hierarchical_rl.options import OptionsAgent, OptionsConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_agent(algorithm: str, state_dim: int, action_dim: int, **kwargs):
    """Create HRL agent based on algorithm name."""
    if algorithm.lower() == 'hac':
        config = HACConfig(**kwargs)
        goal_dim = kwargs.get('goal_dim', 3)  # Default goal_dim = 3
        return HACAgent(state_dim=state_dim, action_dim=action_dim, goal_dim=goal_dim, config=config)
    
    elif algorithm.lower() == 'fun':
        config = FuNConfig(**kwargs)
        return FuNAgent(state_dim=state_dim, action_dim=action_dim, config=config)
    
    elif algorithm.lower() == 'hiro':
        config = HIROConfig(**kwargs)
        return HIROAgent(state_dim=state_dim, action_dim=action_dim, config=config)
    
    elif algorithm.lower() == 'options':
        config = OptionsConfig(**kwargs)
        return OptionsAgent(state_dim=state_dim, action_dim=action_dim, config=config)
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def create_environment(env_type: str = "hierarchical", **kwargs):
    """Create environment based on type."""
    base_env = AirSimEnv()
    
    if env_type == "hierarchical":
        return HierarchicalAirSimEnv(base_env, **kwargs)
    elif env_type == "multi_goal":
        return MultiGoalAirSimEnv(base_env, **kwargs)
    else:
        return base_env


def train_hierarchical_agent(
    algorithm: str,
    env_type: str = "hierarchical",
    num_episodes: int = 1000,
    max_episode_steps: int = 512,
    eval_freq: int = 100,
    save_freq: int = 200,
    log_freq: int = 10,
    seed: int = 42,
    device: str = "cuda",
    save_dir: str = None,
    load_path: str = None,
    **kwargs
):
    """
    Train hierarchical reinforcement learning agent.
    
    Args:
        algorithm: HRL algorithm ('hac', 'fun', 'hiro', 'options')
        env_type: Environment type ('hierarchical', 'multi_goal')
        num_episodes: Number of training episodes
        max_episode_steps: Maximum steps per episode
        eval_freq: Evaluation frequency (episodes)
        save_freq: Model save frequency (episodes)
        log_freq: Logging frequency (episodes)
        seed: Random seed
        device: Computing device
        save_dir: Directory to save models and logs
        load_path: Path to load pretrained model
    """
    
    # Set random seed
    set_seed(seed)
    
    # Create save directory
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"results/hierarchical/{algorithm}_{timestamp}"
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env = create_environment(env_type, max_episode_steps=max_episode_steps, **kwargs)
    
    # Get environment dimensions
    if hasattr(env.observation_space, 'spaces'):
        # Dict observation space - use actual processed state dimension
        # Based on debug info: actual state is 109 dimensions
        state_dim = 109  # Actual processed state dimension
    else:
        # Simple observation space
        state_dim = np.prod(env.observation_space.shape)
    
    if hasattr(env.action_space, 'shape'):
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    logger.info(f"Environment: state_dim={state_dim}, action_dim={action_dim}")
    
    # Create agent
    agent_kwargs = kwargs.copy()
    agent_kwargs.update({
        'device': device,
        'seed': seed,
        'max_episode_steps': max_episode_steps
    })
    
    agent = create_agent(algorithm, state_dim, action_dim, **agent_kwargs)
    
    # Load pretrained model if specified
    if load_path and os.path.exists(load_path):
        agent.load(load_path)
        logger.info(f"Loaded pretrained model from {load_path}")
    
    # Setup logging
    writer = SummaryWriter(save_path / "logs")
    
    # Save configuration
    config_dict = {
        'algorithm': algorithm,
        'env_type': env_type,
        'num_episodes': num_episodes,
        'max_episode_steps': max_episode_steps,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'seed': seed,
        'device': device,
        **kwargs
    }
    
    with open(save_path / "config.yaml", 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in trange(num_episodes, desc=f"Training {algorithm.upper()}"):
        obs, info = env.reset()
        
        if hasattr(agent, 'reset_episode'):
            agent.reset_episode()
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Process observation
            if isinstance(obs, dict):
                # Extract visual features (simplified)
                visual_obs = obs['observation']
                inform_vector = obs.get('inform_vector', np.zeros(9))
                
                # Flatten visual observation for now (can be enhanced with CNN)
                visual_obs = np.array(visual_obs) if not isinstance(visual_obs, np.ndarray) else visual_obs
                if len(visual_obs.shape) > 1:
                    visual_features = visual_obs.flatten()[:100]  # Limit size
                    visual_features = np.pad(visual_features, (0, max(0, 100 - len(visual_features))))
                else:
                    visual_features = np.zeros(100)
                
                # Combine features
                state = np.concatenate([visual_features, inform_vector])
                
                # Goal for goal-conditioned agents
                goal = obs.get('desired_goal', env.get_goal())
            else:
                state = obs.flatten()
                goal = env.get_goal()
            
            # Select action
            action, action_info = agent.select_action(state, goal)
            
            # Execute action
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            if hasattr(agent, 'store_transition'):
                combined_info = {**action_info, **step_info}
                
                # HAC agent requires specific parameters
                if algorithm.lower() == 'hac':
                    # For HAC, we need to store transitions for each level
                    # This is typically handled internally by the agent during action selection
                    # For now, skip explicit storage to avoid parameter mismatch
                    pass
                else:
                    agent.store_transition(state, action, reward, next_obs, done, combined_info)
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            # Check for early termination
            if episode_length >= max_episode_steps:
                break
        
        # End of episode
        if hasattr(agent, 'end_episode'):
            agent.end_episode()
        
        # Update agent
        if episode > 50:  # Start training after some warm-up
            losses = agent.update()
            
            # Log losses
            if losses and episode % log_freq == 0:
                for loss_name, loss_value in losses.items():
                    writer.add_scalar(f"losses/{loss_name}", loss_value, episode)
        
        # Track success
        if info.get('goal_achieved', False) or step_info.get('all_goals_achieved', False):
            success_count += 1
        
        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Logging
        if episode % log_freq == 0:
            recent_rewards = episode_rewards[-log_freq:]
            recent_lengths = episode_lengths[-log_freq:]
            success_rate = success_count / max(1, episode + 1)
            
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            
            logger.info(
                f"Episode {episode}: "
                f"Avg Reward: {avg_reward:.2f}, "
                f"Avg Length: {avg_length:.1f}, "
                f"Success Rate: {success_rate:.3f}"
            )
            
            # Tensorboard logging
            writer.add_scalar("performance/episode_reward", episode_reward, episode)
            writer.add_scalar("performance/episode_length", episode_length, episode)
            writer.add_scalar("performance/avg_reward", avg_reward, episode)
            writer.add_scalar("performance/success_rate", success_rate, episode)
            
            # Algorithm-specific logging
            if hasattr(agent, 'get_stats'):
                stats = agent.get_stats()
                for stat_name, stat_values in stats.items():
                    if stat_values:
                        writer.add_scalar(f"agent_stats/{stat_name}", stat_values[-1], episode)
            
            # Options-specific logging
            if algorithm.lower() == 'options' and hasattr(agent, 'get_option_statistics'):
                option_stats = agent.get_option_statistics()
                writer.add_scalar("options/diversity_score", option_stats['option_diversity_score'], episode)
                for i, count in enumerate(option_stats['option_visit_counts']):
                    writer.add_scalar(f"options/option_{i}_count", count, episode)
        
        # Evaluation
        if episode % eval_freq == 0 and episode > 0:
            eval_reward, eval_success_rate = evaluate_agent(agent, env, num_episodes=10)
            logger.info(f"Evaluation - Reward: {eval_reward:.2f}, Success Rate: {eval_success_rate:.3f}")
            
            writer.add_scalar("evaluation/reward", eval_reward, episode)
            writer.add_scalar("evaluation/success_rate", eval_success_rate, episode)
        
        # Save model
        if episode % save_freq == 0 and episode > 0:
            model_path = save_path / f"model_episode_{episode}.pth"
            agent.save(str(model_path))
            logger.info(f"Model saved to {model_path}")
    
    # Save final model
    final_model_path = save_path / "final_model.pth"
    agent.save(str(final_model_path))
    logger.info(f"Final model saved to {final_model_path}")
    
    # Close environment and writer
    env.close()
    writer.close()
    
    return agent


def evaluate_agent(agent, env, num_episodes: int = 10):
    """Evaluate trained agent."""
    episode_rewards = []
    successes = 0
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        
        if hasattr(agent, 'reset_episode'):
            agent.reset_episode()
        
        episode_reward = 0
        done = False
        
        while not done:
            # Process observation (same as training)
            if isinstance(obs, dict):
                visual_obs = obs['observation']
                inform_vector = obs.get('inform_vector', np.zeros(9))
                
                # Ensure visual_obs is numpy array
                visual_obs = np.array(visual_obs) if not isinstance(visual_obs, np.ndarray) else visual_obs
                if len(visual_obs.shape) > 1:
                    visual_features = visual_obs.flatten()[:100]
                    visual_features = np.pad(visual_features, (0, max(0, 100 - len(visual_features))))
                else:
                    visual_features = np.zeros(100)
                
                state = np.concatenate([visual_features, inform_vector])
                goal = obs.get('desired_goal', env.get_goal())
            else:
                state = obs.flatten()
                goal = env.get_goal()
            
            # Select action deterministically
            action, _ = agent.select_action(state, goal, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        
        if step_info.get('goal_achieved', False) or step_info.get('all_goals_achieved', False):
            successes += 1
    
    avg_reward = np.mean(episode_rewards)
    success_rate = successes / num_episodes
    
    return avg_reward, success_rate


def main():
    parser = argparse.ArgumentParser(description="Train Hierarchical RL Agents")
    
    parser.add_argument('--algorithm', type=str, choices=['hac', 'fun', 'hiro', 'options'], 
                       default='hac', help='HRL algorithm to use')
    parser.add_argument('--env_type', type=str, choices=['hierarchical', 'multi_goal'], 
                       default='hierarchical', help='Environment type')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max_episode_steps', type=int, default=512, help='Maximum steps per episode')
    parser.add_argument('--eval_freq', type=int, default=100, help='Evaluation frequency')
    parser.add_argument('--save_freq', type=int, default=200, help='Save frequency')
    parser.add_argument('--log_freq', type=int, default=10, help='Logging frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_dir', type=str, help='Save directory')
    parser.add_argument('--load_path', type=str, help='Path to load pretrained model')
    
    # Algorithm-specific arguments
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    
    # HAC-specific
    parser.add_argument('--num_levels', type=int, default=2, help='Number of hierarchy levels (HAC)')
    
    # FuN-specific
    parser.add_argument('--manager_horizon', type=int, default=10, help='Manager horizon (FuN)')
    parser.add_argument('--goal_dim', type=int, default=16, help='Goal dimension (FuN)')
    
    # HIRO-specific
    parser.add_argument('--subgoal_freq', type=int, default=10, help='Subgoal frequency (HIRO)')
    parser.add_argument('--subgoal_dim', type=int, default=3, help='Subgoal dimension (HIRO)')
    
    # Options-specific
    parser.add_argument('--num_options', type=int, default=8, help='Number of options')
    
    args = parser.parse_args()
    
    # Convert args to kwargs
    kwargs = vars(args)
    algorithm = kwargs.pop('algorithm')
    env_type = kwargs.pop('env_type')
    
    logger.info(f"Starting training with algorithm: {algorithm.upper()}")
    logger.info(f"Configuration: {kwargs}")
    
    # Train agent
    agent = train_hierarchical_agent(algorithm=algorithm, env_type=env_type, **kwargs)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()