#!/usr/bin/env python
"""
Hierarchical AirSim Environment Wrapper
=======================================

Enhanced AirSim environment wrapper designed for hierarchical reinforcement learning.
Provides goal-conditioned observations, subgoal management, and HRL-specific features.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, List
import logging

from gym_airsim.envs.AirGym import AirSimEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalAirSimEnv(gym.Wrapper):
    """
    Hierarchical wrapper for AirSim environment.
    
    Features:
    - Goal-conditioned observations
    - Subgoal management and tracking
    - Hierarchical reward computation
    - Multi-level state representations
    """
    
    def __init__(
        self,
        env: Optional[AirSimEnv] = None,
        goal_dim: int = 3,
        subgoal_bounds: Tuple[float, float] = (-10.0, 10.0),
        goal_threshold: float = 2.0,
        sparse_reward: bool = False,
        include_goal_in_obs: bool = True,
        max_episode_steps: int = 512,
        **kwargs
    ):
        """
        Initialize hierarchical AirSim environment.
        
        Args:
            env: Base AirSim environment (if None, will create one)
            goal_dim: Dimension of goal space
            subgoal_bounds: Bounds for subgoal generation
            goal_threshold: Distance threshold for goal achievement
            sparse_reward: Whether to use sparse or dense rewards
            include_goal_in_obs: Include goal in observation space
            max_episode_steps: Maximum episode length
        """
        if env is None:
            env = AirSimEnv()
        
        super().__init__(env)
        
        self.goal_dim = goal_dim
        self.subgoal_bounds = np.array(subgoal_bounds)
        self.goal_threshold = goal_threshold
        self.sparse_reward = sparse_reward
        self.include_goal_in_obs = include_goal_in_obs
        self.max_episode_steps = max_episode_steps
        
        # Get original observation space
        self.original_obs_space = env.observation_space
        self.original_action_space = env.action_space
        
        # Enhanced observation space
        if include_goal_in_obs:
            # Add goal to observation space
            if isinstance(self.original_obs_space, spaces.Box):
                # Visual observations + inform vector + goal
                self.inform_dim = 9  # From original AirSim environment
                
                # Create composite observation space
                self.observation_space = spaces.Dict({
                    'observation': self.original_obs_space,
                    'achieved_goal': spaces.Box(-np.inf, np.inf, (goal_dim,), dtype=np.float32),
                    'desired_goal': spaces.Box(-np.inf, np.inf, (goal_dim,), dtype=np.float32),
                    'inform_vector': spaces.Box(-np.inf, np.inf, (self.inform_dim,), dtype=np.float32)
                })
            else:
                # Fallback to simple concatenation
                obs_dim = np.prod(self.original_obs_space.shape)
                total_dim = obs_dim + 2 * goal_dim + self.inform_dim
                self.observation_space = spaces.Box(
                    -np.inf, np.inf, (total_dim,), dtype=np.float32
                )
        else:
            self.observation_space = self.original_obs_space
        
        # Current episode state
        self.current_goal = None
        self.episode_step = 0
        self.episode_reward = 0.0
        self.goal_achieved = False
        
        # Subgoal tracking
        self.subgoal_history = []
        self.achievement_history = []
        
        logger.info(f"Initialized HierarchicalAirSimEnv with goal_dim={goal_dim}")
    
    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and sample new goal."""
        obs, info = self.env.reset(**kwargs)
        
        # Sample new goal
        self.current_goal = self._sample_goal()
        
        # Reset episode state
        self.episode_step = 0
        self.episode_reward = 0.0
        self.goal_achieved = False
        self.subgoal_history.clear()
        self.achievement_history.clear()
        
        # Create hierarchical observation
        hierarchical_obs = self._create_hierarchical_obs(obs, info)
        
        # Enhanced info - 安全地处理基础环境的info
        if isinstance(info, dict):
            enhanced_info = dict(info)
        else:
            # 如果基础环境不返回字典，创建一个空字典
            enhanced_info = {}
        
        enhanced_info.update({
            'goal': self.current_goal.copy(),
            'goal_achieved': self.goal_achieved,
            'episode_step': self.episode_step
        })
        
        return hierarchical_obs, enhanced_info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute action and compute hierarchical rewards."""
        # 处理基础环境返回值数量不一致的问题
        step_result = self.env.step(action)
        if len(step_result) == 4:
            # 老版本gym格式：obs, reward, done, info
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        else:
            # 新版本gymnasium格式：obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = step_result
        
        self.episode_step += 1
        
        # Get current position (achieved goal)
        achieved_goal = self._extract_achieved_goal(obs, info)
        
        # Compute goal-conditioned reward
        goal_reward = self._compute_goal_reward(achieved_goal, self.current_goal)
        
        # Check goal achievement
        goal_distance = np.linalg.norm(achieved_goal - self.current_goal)
        self.goal_achieved = goal_distance <= self.goal_threshold
        
        # Combine rewards
        if self.sparse_reward:
            total_reward = 1.0 if self.goal_achieved else 0.0
        else:
            total_reward = reward + goal_reward
        
        self.episode_reward += total_reward
        
        # Episode termination conditions
        episode_terminated = terminated or self.goal_achieved
        episode_truncated = truncated or (self.episode_step >= self.max_episode_steps)
        
        # Create hierarchical observation
        hierarchical_obs = self._create_hierarchical_obs(obs, info)
        
        # Enhanced info - 安全地处理基础环境的info
        if isinstance(info, dict):
            enhanced_info = dict(info)
        else:
            # 如果基础环境不返回字典，创建一个空字典
            enhanced_info = {}
        enhanced_info.update({
            'goal': self.current_goal.copy(),
            'achieved_goal': achieved_goal.copy(),
            'goal_distance': goal_distance,
            'goal_achieved': self.goal_achieved,
            'goal_reward': goal_reward,
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward
        })
        
        return hierarchical_obs, total_reward, episode_terminated, episode_truncated, enhanced_info
    
    def _sample_goal(self) -> np.ndarray:
        """Sample a random goal within the environment bounds."""
        # Get current position from the environment
        current_pos = self.env.airgym.drone_pos()[:self.goal_dim]
        
        # Sample goal relative to current position
        goal_offset = np.random.uniform(
            self.subgoal_bounds[0], 
            self.subgoal_bounds[1], 
            size=self.goal_dim
        )
        goal = current_pos + goal_offset
        
        # 修复高度问题：确保目标Z坐标在合理的飞行高度范围内
        if self.goal_dim >= 3:
            # 限制Z坐标在飞行高度范围内 (AirSim坐标系中负值表示高度)
            min_flight_height = -5.0  # 最高5米
            max_flight_height = -0.5  # 最低0.5米
            goal[2] = np.clip(goal[2], min_flight_height, max_flight_height)
            print(f"Goal sampled: X={goal[0]:.2f}, Y={goal[1]:.2f}, Z={goal[2]:.2f}")
        
        return goal.astype(np.float32)
    
    def _extract_achieved_goal(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """Extract achieved goal (current position) from observation."""
        # Get current position from drone
        if hasattr(self.env, 'airgym'):
            current_pos = self.env.airgym.drone_pos()[:self.goal_dim]
        else:
            # Fallback: use info or approximate from observation
            current_pos = info.get('position', np.zeros(self.goal_dim))
        
        return current_pos.astype(np.float32)
    
    def _compute_goal_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """Compute reward based on goal achievement."""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        
        if self.sparse_reward:
            return 1.0 if distance <= self.goal_threshold else 0.0
        else:
            # Dense reward: negative distance with bonus for achievement
            reward = -distance
            if distance <= self.goal_threshold:
                reward += 10.0  # Achievement bonus
            return reward
    
    def _create_hierarchical_obs(self, obs: np.ndarray, info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Create hierarchical observation with goal information."""
        achieved_goal = self._extract_achieved_goal(obs, info)
        
        if self.include_goal_in_obs:
            # Get inform vector from environment
            if hasattr(self.env, 'airgym') and hasattr(self.env.airgym, 'get_inform_vector'):
                inform_vector = self.env.airgym.get_inform_vector()
            else:
                # 创建基本的inform vector，包含当前位置、目标位置等信息
                current_pos = self._extract_achieved_goal(obs, info)
                goal_direction = self.current_goal - current_pos
                goal_distance = np.linalg.norm(goal_direction)
                
                # 构建inform vector: [goal_direction(3), goal_distance(1), current_pos(3), goal(3)]
                # 总共10维，如果inform_dim不同则调整
                basic_inform = np.concatenate([
                    goal_direction,                    # 3D: 到目标的方向向量
                    [goal_distance],                   # 1D: 到目标的距离
                    current_pos,                       # 3D: 当前位置
                    self.current_goal                  # 3D: 目标位置
                ])
                
                # 确保维度匹配
                if len(basic_inform) != self.inform_dim:
                    if len(basic_inform) > self.inform_dim:
                        inform_vector = basic_inform[:self.inform_dim]
                    else:
                        inform_vector = np.zeros(self.inform_dim)
                        inform_vector[:len(basic_inform)] = basic_inform
                else:
                    inform_vector = basic_inform
            
            hierarchical_obs = {
                'observation': obs,
                'achieved_goal': achieved_goal,
                'desired_goal': self.current_goal.copy(),
                'inform_vector': inform_vector.astype(np.float32)
            }
        else:
            hierarchical_obs = obs
        
        return hierarchical_obs
    
    def set_goal(self, goal: np.ndarray) -> None:
        """Set a specific goal for the environment."""
        self.current_goal = goal.astype(np.float32)
    
    def get_goal(self) -> np.ndarray:
        """Get current goal."""
        return self.current_goal.copy() if self.current_goal is not None else np.zeros(self.goal_dim)
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """Compute reward for HER-style goal relabeling."""
        return self._compute_goal_reward(achieved_goal, desired_goal)
    
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """Check if goal is achieved (for HER)."""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance <= self.goal_threshold
    
    def add_subgoal(self, subgoal: np.ndarray, achieved: bool = False) -> None:
        """Add subgoal to tracking history."""
        self.subgoal_history.append(subgoal.copy())
        self.achievement_history.append(achieved)
    
    def get_subgoal_statistics(self) -> Dict[str, Any]:
        """Get subgoal achievement statistics."""
        if len(self.achievement_history) == 0:
            return {'subgoal_success_rate': 0.0, 'num_subgoals': 0}
        
        success_rate = np.mean(self.achievement_history)
        return {
            'subgoal_success_rate': success_rate,
            'num_subgoals': len(self.subgoal_history),
            'subgoal_history': self.subgoal_history.copy(),
            'achievement_history': self.achievement_history.copy()
        }
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render environment with goal visualization."""
        # Call base environment render
        img = self.env.render(mode)
        
        # TODO: Add goal visualization overlay
        # This would require access to the AirSim camera and rendering pipeline
        
        return img


class MultiGoalAirSimEnv(HierarchicalAirSimEnv):
    """
    Multi-goal version of hierarchical AirSim environment.
    Supports multiple simultaneous goals and complex mission structures.
    """
    
    def __init__(
        self,
        num_goals: int = 3,
        goal_curriculum: bool = True,
        **kwargs
    ):
        """
        Initialize multi-goal environment.
        
        Args:
            num_goals: Number of goals to manage simultaneously
            goal_curriculum: Whether to use curriculum learning for goals
        """
        super().__init__(**kwargs)
        
        self.num_goals = num_goals
        self.goal_curriculum = goal_curriculum
        
        # Multi-goal state
        self.goals = []
        self.goal_priorities = []
        self.goals_achieved = []
        
        # Curriculum parameters
        self.curriculum_level = 1
        self.success_threshold = 0.8
        self.success_window = 100
        self.success_history = []
        
        logger.info(f"Initialized MultiGoalAirSimEnv with {num_goals} goals")
    
    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset with multiple goals."""
        obs, info = super().reset(**kwargs)
        
        # Sample multiple goals
        self._sample_multiple_goals()
        
        return obs, info
    
    def _sample_multiple_goals(self) -> None:
        """Sample multiple goals based on curriculum level."""
        self.goals.clear()
        self.goal_priorities.clear()
        self.goals_achieved.clear()
        
        # Determine difficulty based on curriculum level
        if self.goal_curriculum:
            max_distance = min(5.0 * self.curriculum_level, 20.0)
        else:
            max_distance = 20.0
        
        current_pos = self.env.airgym.drone_pos()[:self.goal_dim]
        
        for i in range(self.num_goals):
            # Sample goal at increasing distances
            goal_distance = np.random.uniform(
                2.0 + i * 2.0, 
                min(max_distance, 2.0 + (i + 1) * 4.0)
            )
            
            # Random direction
            direction = np.random.randn(self.goal_dim)
            direction /= np.linalg.norm(direction)
            
            goal = current_pos + direction * goal_distance
            priority = self.num_goals - i  # Higher priority for closer goals
            
            self.goals.append(goal.astype(np.float32))
            self.goal_priorities.append(priority)
            self.goals_achieved.append(False)
        
        # Set primary goal
        self.current_goal = self.goals[0]
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Step with multi-goal logic."""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Check all goals for achievement
        achieved_goal = info['achieved_goal']
        
        for i, goal in enumerate(self.goals):
            if not self.goals_achieved[i]:
                distance = np.linalg.norm(achieved_goal - goal)
                if distance <= self.goal_threshold:
                    self.goals_achieved[i] = True
                    reward += 5.0 * self.goal_priorities[i]  # Bonus based on priority
        
        # Update current goal to next unachieved goal
        for i, achieved in enumerate(self.goals_achieved):
            if not achieved:
                self.current_goal = self.goals[i]
                break
        
        # Check if all goals achieved
        all_goals_achieved = all(self.goals_achieved)
        if all_goals_achieved:
            reward += 20.0  # Big bonus for completing all goals
            terminated = True
        
        # Update curriculum
        if self.goal_curriculum and (terminated or truncated):
            self._update_curriculum(all_goals_achieved)
        
        # Enhanced info
        info.update({
            'goals': self.goals.copy(),
            'goals_achieved': self.goals_achieved.copy(),
            'goal_priorities': self.goal_priorities.copy(),
            'all_goals_achieved': all_goals_achieved,
            'curriculum_level': self.curriculum_level
        })
        
        return obs, reward, terminated, truncated, info
    
    def _update_curriculum(self, success: bool) -> None:
        """Update curriculum based on performance."""
        self.success_history.append(success)
        
        # Keep only recent history
        if len(self.success_history) > self.success_window:
            self.success_history.pop(0)
        
        # Check for curriculum progression
        if len(self.success_history) >= self.success_window:
            success_rate = np.mean(self.success_history)
            
            if success_rate >= self.success_threshold and self.curriculum_level < 5:
                self.curriculum_level += 1
                self.success_history.clear()  # Reset history
                logger.info(f"Curriculum level increased to {self.curriculum_level}")
            elif success_rate < 0.3 and self.curriculum_level > 1:
                self.curriculum_level -= 1
                self.success_history.clear()  # Reset history
                logger.info(f"Curriculum level decreased to {self.curriculum_level}")