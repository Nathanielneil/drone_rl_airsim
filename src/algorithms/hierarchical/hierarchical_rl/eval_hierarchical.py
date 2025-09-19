#!/usr/bin/env python
"""
Evaluation script for hierarchical reinforcement learning algorithms.
Provides comprehensive evaluation and visualization tools.
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_airsim.envs.AirGym import AirSimEnv
from hierarchical_rl.envs.hierarchical_airsim_env import HierarchicalAirSimEnv, MultiGoalAirSimEnv
from hierarchical_rl.train_hierarchical import create_agent, create_environment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalEvaluator:
    """Comprehensive evaluator for hierarchical RL agents."""
    
    def __init__(
        self,
        agent,
        env,
        algorithm: str,
        save_dir: str = None
    ):
        self.agent = agent
        self.env = env
        self.algorithm = algorithm
        
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"eval_results/{algorithm}_{timestamp}"
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation metrics
        self.episode_data = []
        self.trajectory_data = []
        
    def evaluate(
        self,
        num_episodes: int = 100,
        max_episode_steps: int = 512,
        render: bool = False,
        save_trajectories: bool = True
    ) -> Dict:
        """
        Comprehensive evaluation of the agent.
        
        Args:
            num_episodes: Number of evaluation episodes
            max_episode_steps: Maximum steps per episode
            render: Whether to render episodes
            save_trajectories: Whether to save trajectory data
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Starting evaluation with {num_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        goal_distances = []
        
        # Algorithm-specific metrics
        if self.algorithm.lower() == 'hac':
            subgoal_achievements = []
        elif self.algorithm.lower() == 'options':
            option_usage = {}
        elif self.algorithm.lower() == 'fun':
            manager_decisions = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            
            if hasattr(self.agent, 'reset_episode'):
                self.agent.reset_episode()
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Track trajectory
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'positions': [],
                'goals': [],
                'subgoals': []
            }
            
            while not done and episode_length < max_episode_steps:
                # Process observation
                if isinstance(obs, dict):
                    visual_obs = obs['observation']
                    inform_vector = obs.get('inform_vector', np.zeros(9))
                    
                    if len(visual_obs.shape) > 1:
                        visual_features = visual_obs.flatten()[:100]
                        visual_features = np.pad(visual_features, (0, max(0, 100 - len(visual_features))))
                    else:
                        visual_features = np.zeros(100)
                    
                    state = np.concatenate([visual_features, inform_vector])
                    goal = obs.get('desired_goal', self.env.get_goal())
                else:
                    state = obs.flatten()
                    goal = self.env.get_goal()
                
                # Select action
                action, action_info = self.agent.select_action(state, goal, deterministic=True)
                
                # Store trajectory data
                if save_trajectories:
                    trajectory['states'].append(state.copy())
                    trajectory['actions'].append(action.copy())
                    trajectory['goals'].append(goal.copy())
                    
                    # Get current position
                    if hasattr(self.env, 'env') and hasattr(self.env.env, 'airgym'):
                        current_pos = self.env.env.airgym.drone_pos()[:3]
                        trajectory['positions'].append(current_pos.copy())
                    
                    # Algorithm-specific data
                    if 'subgoal' in action_info:
                        trajectory['subgoals'].append(action_info['subgoal'].copy())
                
                # Execute action
                obs, reward, terminated, truncated, step_info = self.env.step(action)
                done = terminated or truncated
                
                # Store reward
                if save_trajectories:
                    trajectory['rewards'].append(reward)
                
                # Algorithm-specific metrics
                if self.algorithm.lower() == 'hac':
                    if 'subgoals' in action_info:
                        subgoal_achievements.extend([
                            self.agent.is_subgoal_achieved(state, sg) 
                            for sg in action_info['subgoals']
                        ])
                
                elif self.algorithm.lower() == 'options':
                    if 'option' in action_info:
                        option = action_info['option']
                        option_usage[option] = option_usage.get(option, 0) + 1
                
                elif self.algorithm.lower() == 'fun':
                    if 'goal' in action_info:
                        manager_decisions.append(action_info['goal'])
                
                episode_reward += reward
                episode_length += 1
                
                if render and episode < 5:  # Render first few episodes
                    self.env.render()
            
            # Episode finished
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Check success
            if step_info.get('goal_achieved', False) or step_info.get('all_goals_achieved', False):
                success_count += 1
            
            # Track goal distance
            if 'goal_distance' in step_info:
                goal_distances.append(step_info['goal_distance'])
            
            # Store episode data
            episode_data = {
                'episode': episode,
                'reward': episode_reward,
                'length': episode_length,
                'success': step_info.get('goal_achieved', False),
                'final_goal_distance': step_info.get('goal_distance', 0)
            }
            
            self.episode_data.append(episode_data)
            
            if save_trajectories:
                self.trajectory_data.append(trajectory)
            
            if (episode + 1) % 20 == 0:
                logger.info(f"Completed {episode + 1}/{num_episodes} episodes")
        
        # Compute final metrics
        results = {
            'num_episodes': num_episodes,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': success_count / num_episodes,
            'avg_final_distance': np.mean(goal_distances) if goal_distances else 0,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        # Algorithm-specific results
        if self.algorithm.lower() == 'hac' and subgoal_achievements:
            results['subgoal_success_rate'] = np.mean(subgoal_achievements)
        
        elif self.algorithm.lower() == 'options' and option_usage:
            total_usage = sum(option_usage.values())
            option_probs = {k: v/total_usage for k, v in option_usage.items()}
            results['option_usage'] = option_usage
            results['option_probabilities'] = option_probs
            results['option_entropy'] = -sum(p * np.log(p + 1e-8) for p in option_probs.values())
        
        elif self.algorithm.lower() == 'fun' and manager_decisions:
            results['avg_manager_decisions'] = len(manager_decisions) / num_episodes
        
        # Save results
        self._save_results(results)
        
        logger.info(f"Evaluation completed!")
        logger.info(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        logger.info(f"Success Rate: {results['success_rate']:.3f}")
        logger.info(f"Average Length: {results['avg_length']:.1f} ± {results['std_length']:.1f}")
        
        return results
    
    def _save_results(self, results: Dict) -> None:
        """Save evaluation results."""
        # Save numerical results
        with open(self.save_dir / "results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    json_results[key] = float(value)
                else:
                    json_results[key] = value
            json.dump(json_results, f, indent=2)
        
        # Save episode data
        with open(self.save_dir / "episode_data.json", 'w') as f:
            json.dump(self.episode_data, f, indent=2)
        
        # Save trajectory data if available
        if self.trajectory_data:
            np.save(self.save_dir / "trajectories.npy", self.trajectory_data)
        
        logger.info(f"Results saved to {self.save_dir}")
    
    def visualize_results(self, show_plots: bool = True) -> None:
        """Create visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.algorithm.upper()} Evaluation Results', fontsize=16)
        
        # Episode rewards
        episode_rewards = [ep['reward'] for ep in self.episode_data]
        axes[0, 0].plot(episode_rewards)
        axes[0, 0].axhline(y=np.mean(episode_rewards), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(episode_rewards):.2f}')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Episode lengths
        episode_lengths = [ep['length'] for ep in self.episode_data]
        axes[0, 1].plot(episode_lengths)
        axes[0, 1].axhline(y=np.mean(episode_lengths), color='r', linestyle='--',
                          label=f'Mean: {np.mean(episode_lengths):.1f}')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Success rate over time
        successes = [ep['success'] for ep in self.episode_data]
        window_size = 10
        success_rate_smooth = []\n        for i in range(len(successes)):\n            start = max(0, i - window_size + 1)\n            success_rate_smooth.append(np.mean(successes[start:i+1]))\n        \n        axes[1, 0].plot(success_rate_smooth)\n        axes[1, 0].axhline(y=np.mean(successes), color='r', linestyle='--',\n                          label=f'Overall: {np.mean(successes):.3f}')\n        axes[1, 0].set_title(f'Success Rate (Window: {window_size})')\n        axes[1, 0].set_xlabel('Episode')\n        axes[1, 0].set_ylabel('Success Rate')\n        axes[1, 0].legend()\n        axes[1, 0].grid(True)\n        \n        # Final distances\n        final_distances = [ep['final_goal_distance'] for ep in self.episode_data]\n        axes[1, 1].hist(final_distances, bins=20, alpha=0.7)\n        axes[1, 1].axvline(x=np.mean(final_distances), color='r', linestyle='--',\n                          label=f'Mean: {np.mean(final_distances):.2f}')\n        axes[1, 1].set_title('Final Goal Distances')\n        axes[1, 1].set_xlabel('Distance')\n        axes[1, 1].set_ylabel('Frequency')\n        axes[1, 1].legend()\n        axes[1, 1].grid(True)\n        \n        plt.tight_layout()\n        plt.savefig(self.save_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')\n        \n        if show_plots:\n            plt.show()\n        \n        logger.info(f\"Plots saved to {self.save_dir / 'evaluation_plots.png'}\")\n    \n    def visualize_trajectories(self, num_trajectories: int = 5, show_plots: bool = True) -> None:\n        \"\"\"Visualize agent trajectories.\"\"\"\n        if not self.trajectory_data:\n            logger.warning(\"No trajectory data available for visualization\")\n            return\n        \n        fig = plt.figure(figsize=(15, 10))\n        \n        # Select best trajectories (highest rewards)\n        trajectory_rewards = [sum(traj['rewards']) for traj in self.trajectory_data]\n        best_indices = np.argsort(trajectory_rewards)[-num_trajectories:]\n        \n        for i, idx in enumerate(best_indices):\n            trajectory = self.trajectory_data[idx]\n            \n            if not trajectory['positions']:\n                continue\n                \n            positions = np.array(trajectory['positions'])\n            goals = np.array(trajectory['goals'])\n            \n            # 3D trajectory plot\n            ax = fig.add_subplot(2, 3, i + 1, projection='3d')\n            \n            # Plot trajectory\n            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], \n                   'b-', alpha=0.7, linewidth=2, label='Trajectory')\n            \n            # Plot start and end points\n            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], \n                      c='green', s=100, marker='o', label='Start')\n            ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], \n                      c='red', s=100, marker='s', label='End')\n            \n            # Plot goal\n            if len(goals) > 0:\n                goal = goals[0]  # Use first goal\n                ax.scatter(goal[0], goal[1], goal[2] if len(goal) > 2 else 0, \n                          c='gold', s=150, marker='*', label='Goal')\n            \n            ax.set_title(f'Trajectory {idx + 1} (Reward: {sum(trajectory[\"rewards\"]):.1f})')\n            ax.set_xlabel('X')\n            ax.set_ylabel('Y')\n            ax.set_zlabel('Z')\n            ax.legend()\n        \n        plt.tight_layout()\n        plt.savefig(self.save_dir / 'trajectory_plots.png', dpi=300, bbox_inches='tight')\n        \n        if show_plots:\n            plt.show()\n        \n        logger.info(f\"Trajectory plots saved to {self.save_dir / 'trajectory_plots.png'}\")\n\n\ndef main():\n    parser = argparse.ArgumentParser(description=\"Evaluate Hierarchical RL Agents\")\n    \n    parser.add_argument('model_path', type=str, help='Path to trained model')\n    parser.add_argument('--algorithm', type=str, choices=['hac', 'fun', 'hiro', 'options'], \n                       required=True, help='HRL algorithm')\n    parser.add_argument('--env_type', type=str, choices=['hierarchical', 'multi_goal'], \n                       default='hierarchical', help='Environment type')\n    parser.add_argument('--num_episodes', type=int, default=100, help='Number of evaluation episodes')\n    parser.add_argument('--max_episode_steps', type=int, default=512, help='Maximum steps per episode')\n    parser.add_argument('--render', action='store_true', help='Render episodes')\n    parser.add_argument('--save_trajectories', action='store_true', help='Save trajectory data')\n    parser.add_argument('--visualize', action='store_true', help='Create visualization plots')\n    parser.add_argument('--save_dir', type=str, help='Save directory for results')\n    parser.add_argument('--seed', type=int, default=42, help='Random seed')\n    parser.add_argument('--device', type=str, default='cuda', help='Device to use')\n    \n    # Environment and agent parameters\n    parser.add_argument('--goal_dim', type=int, default=3, help='Goal dimension')\n    parser.add_argument('--num_options', type=int, default=8, help='Number of options')\n    parser.add_argument('--num_levels', type=int, default=2, help='Number of hierarchy levels')\n    \n    args = parser.parse_args()\n    \n    # Set seed\n    torch.manual_seed(args.seed)\n    np.random.seed(args.seed)\n    \n    # Create environment\n    env = create_environment(args.env_type, max_episode_steps=args.max_episode_steps)\n    \n    # Get environment dimensions\n    if hasattr(env.observation_space, 'spaces'):\n        obs_space = env.observation_space['observation']\n        if hasattr(obs_space, 'shape'):\n            state_dim = np.prod(obs_space.shape) + 9\n        else:\n            state_dim = 9\n    else:\n        state_dim = np.prod(env.observation_space.shape)\n    \n    if hasattr(env.action_space, 'shape'):\n        action_dim = env.action_space.shape[0]\n    else:\n        action_dim = env.action_space.n\n    \n    # Create agent\n    agent_kwargs = {\n        'device': args.device,\n        'seed': args.seed,\n        'max_episode_steps': args.max_episode_steps\n    }\n    \n    if args.algorithm == 'options':\n        agent_kwargs['num_options'] = args.num_options\n    elif args.algorithm == 'hac':\n        agent_kwargs['num_levels'] = args.num_levels\n    elif args.algorithm in ['fun', 'hiro']:\n        agent_kwargs['goal_dim'] = args.goal_dim\n    \n    agent = create_agent(args.algorithm, state_dim, action_dim, **agent_kwargs)\n    \n    # Load trained model\n    if os.path.exists(args.model_path):\n        agent.load(args.model_path)\n        logger.info(f\"Loaded model from {args.model_path}\")\n    else:\n        raise FileNotFoundError(f\"Model not found: {args.model_path}\")\n    \n    # Create evaluator\n    evaluator = HierarchicalEvaluator(agent, env, args.algorithm, args.save_dir)\n    \n    # Run evaluation\n    results = evaluator.evaluate(\n        num_episodes=args.num_episodes,\n        max_episode_steps=args.max_episode_steps,\n        render=args.render,\n        save_trajectories=args.save_trajectories\n    )\n    \n    # Create visualizations\n    if args.visualize:\n        evaluator.visualize_results(show_plots=False)\n        if args.save_trajectories:\n            evaluator.visualize_trajectories(show_plots=False)\n    \n    logger.info(\"Evaluation completed successfully!\")\n\n\nif __name__ == \"__main__\":\n    main()