"""
基于目标点的AirSim环境
实现点到点导航任务，具有明确的目标设置和到达判定
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
import random
import math
from gymnasium import spaces

from src.environments.airsim_env.improved_reward_env import ImprovedRewardAirSimEnv
from src.utils.goal_validator import GoalValidator

logger = logging.getLogger(__name__)


class GoalBasedAirSimEnv(ImprovedRewardAirSimEnv):
    """基于目标点的AirSim环境"""
    
    def __init__(self, config=None, **kwargs):
        # 添加目标相关的默认配置
        goal_config = self._get_goal_default_config()
        if config:
            goal_config.update(config)
        
        super().__init__(config=goal_config, **kwargs)
        
        # 目标点管理
        self.current_goal = None
        self.goal_history = []
        self.goals_reached = 0
        self.goal_tolerance = self.config.get("goal_tolerance", 3.0)  # 到达目标的距离阈值
        
        # 目标生成参数
        self.goal_generation_mode = self.config.get("goal_generation_mode", "random")  # random, fixed, sequential
        self.fixed_goals = self.config.get("fixed_goals", [])
        self.goal_range = self.config.get("goal_range", {
            "x": [10, 50],
            "y": [-20, 20], 
            "z": [2, 8]  # 高度范围
        })
        
        # 目标验证器
        self.goal_validator = None
        self.validation_enabled = self.config.get("enable_goal_validation", True)
        self.validation_stats = {"generated": 0, "validated": 0, "failed": 0}
        
        logger.info(f"目标环境初始化: 模式={self.goal_generation_mode}, 容忍度={self.goal_tolerance}cm({self.goal_tolerance/100:.1f}m), 验证={self.validation_enabled}")
    
    def _get_goal_default_config(self) -> Dict:
        """获取目标相关的默认配置"""
        config = super()._get_improved_default_config()
        
        # 目标相关配置
        config.update({
            # 目标奖励
            "goal_reached_reward": 100.0,      # 到达目标的大奖励
            "goal_distance_reward_scale": 5.0,  # 接近目标的奖励比例
            "goal_progress_reward": 2.0,       # 朝目标前进的奖励
            
            # 目标设置
            "goal_tolerance": 3.0,             # 到达目标的距离阈值
            "goal_generation_mode": "random",   # 目标生成模式
            "goal_range": {
                "x": [15, 40],
                "y": [-15, 15],
                "z": [3, 6]
            },
            
            # 任务配置
            "max_goals_per_episode": 3,        # 每episode最多目标数
            "goal_timeout_steps": 1000,        # 单个目标的超时步数
            
            # 可视化
            "visualize_goal": True,            # 是否在仿真中显示目标
            "goal_marker_size": 2.0,           # 目标标记大小
            
            # 目标验证
            "enable_goal_validation": True,    # 启用目标验证
            "min_clearance": 3.0,             # 最小安全距离
            "max_validation_attempts": 20,     # 最大验证尝试次数
            "validation_resolution": 1.0,      # 路径检查分辨率
        })
        
        return config
    
    def reset(self, seed=None, options=None):
        """重置环境并生成新目标"""
        observation, info = super().reset(seed=seed, options=options)
        
        # 初始化目标验证器（如果启用）
        if self.validation_enabled and self.goal_validator is None:
            try:
                validation_config = {
                    "min_clearance": self.config.get("min_clearance", 3.0),
                    "max_validation_attempts": self.config.get("max_validation_attempts", 20),
                    "validation_resolution": self.config.get("validation_resolution", 1.0),
                    "enable_caching": True
                }
                self.goal_validator = GoalValidator(self.client, validation_config)
                self.goal_validator.initialize_scene_analysis(self.vehicle_name)
                logger.info("目标验证器初始化完成")
            except Exception as e:
                logger.warning(f"目标验证器初始化失败，禁用验证: {e}")
                self.validation_enabled = False
        
        # 重置目标相关状态
        self.goals_reached = 0
        self.goal_history = []
        
        # 生成第一个目标
        self.current_goal = self._generate_goal()
        self.goal_start_step = 0
        
        # 在仿真中可视化目标（如果支持）
        if self.config.get("visualize_goal", False):
            self._visualize_goal()
        
        # 添加目标信息到观察和info
        observation = self._add_goal_to_observation(observation)
        info.update(self._get_goal_info())
        
        logger.info(f"新目标生成: {self.current_goal}")
        
        return observation, info
    
    def step(self, action):
        """执行动作，包括目标检查和奖励计算"""
        observation, reward, terminated, truncated, info = super().step(action)
        
        if self.current_goal is not None:
            # 检查是否到达目标
            current_pos = self._get_position()
            distance_to_goal = self._calculate_distance_to_goal(current_pos)
            
            # 目标到达检查
            if distance_to_goal <= self.goal_tolerance:
                self._handle_goal_reached()
                
                # 生成下一个目标（如果还有）
                if self.goals_reached < self.config["max_goals_per_episode"]:
                    self.current_goal = self._generate_goal()
                    self.goal_start_step = self.current_step
                    if self.config.get("visualize_goal", False):
                        self._visualize_goal()
                else:
                    self.current_goal = None
                    terminated = True  # 完成所有目标，episode结束
            
            # 检查目标超时
            if (self.current_step - self.goal_start_step) >= self.config["goal_timeout_steps"]:
                logger.info(f"目标超时，生成新目标")
                self.current_goal = self._generate_goal()
                self.goal_start_step = self.current_step
        
        # 更新观察和信息
        observation = self._add_goal_to_observation(observation)
        info.update(self._get_goal_info())
        
        return observation, reward, terminated, truncated, info
    
    def _generate_goal(self) -> np.ndarray:
        """生成新的目标点"""
        self.validation_stats["generated"] += 1
        
        # 如果启用验证，使用验证器生成安全目标
        if self.validation_enabled and self.goal_validator is not None:
            try:
                current_pos = self._get_position()
                goal_range = self.config["goal_range"]
                
                # 使用验证器生成安全目标
                safe_goal = self.goal_validator.generate_safe_goal(goal_range, current_pos, self.vehicle_name)
                
                if safe_goal is not None:
                    self.validation_stats["validated"] += 1
                    logger.info(f"生成验证安全目标: {safe_goal}")
                    return safe_goal
                else:
                    logger.warning("无法生成安全目标，使用传统方法")
                    self.validation_stats["failed"] += 1
            except Exception as e:
                logger.error(f"目标验证过程失败: {e}")
                self.validation_stats["failed"] += 1
        
        # 传统目标生成方法（作为备选）
        return self._generate_traditional_goal()
    
    def _generate_traditional_goal(self) -> np.ndarray:
        """传统的目标生成方法"""
        if self.goal_generation_mode == "fixed" and self.fixed_goals:
            # 固定目标序列
            goal_idx = self.goals_reached % len(self.fixed_goals)
            goal = np.array(self.fixed_goals[goal_idx])
        
        elif self.goal_generation_mode == "sequential":
            # 顺序生成目标
            goal = self._generate_sequential_goal()
        
        else:
            # 随机生成目标
            goal_range = self.config["goal_range"]
            goal = np.array([
                random.uniform(goal_range["x"][0], goal_range["x"][1]),
                random.uniform(goal_range["y"][0], goal_range["y"][1]),
                -random.uniform(goal_range["z"][0], goal_range["z"][1])  # AirSim Z轴向下为正
            ])
        
        # 确保目标不会太接近起始点
        current_pos = self._get_position()
        min_distance = 10.0
        if np.linalg.norm(goal - current_pos) < min_distance:
            # 重新生成
            return self._generate_traditional_goal()
        
        return goal
    
    def _generate_sequential_goal(self) -> np.ndarray:
        """生成顺序目标点"""
        # 创建一个渐进的目标序列
        base_distance = 15 + self.goals_reached * 10
        angle = (self.goals_reached * 60) % 360  # 每个目标旋转60度
        
        x = base_distance * math.cos(math.radians(angle))
        y = base_distance * math.sin(math.radians(angle))
        z = 3 + self.goals_reached * 1  # 逐渐升高
        
        return np.array([x, y, z])
    
    def _calculate_distance_to_goal(self, position: np.ndarray) -> float:
        """计算到目标的距离"""
        if self.current_goal is None:
            return float('inf')
        return np.linalg.norm(position - self.current_goal)
    
    def _handle_goal_reached(self):
        """处理到达目标的逻辑"""
        self.goals_reached += 1
        self.goal_history.append({
            "goal": self.current_goal.copy(),
            "reached_at_step": self.current_step,
            "time_taken": self.current_step - self.goal_start_step
        })
        
        logger.info(f"目标到达! 第{self.goals_reached}个目标，用时{self.current_step - self.goal_start_step}步")
    
    def _calculate_reward(self) -> float:
        """计算包含目标的奖励"""
        # 获取基础奖励
        base_reward = super()._calculate_reward()
        
        if self.current_goal is None:
            return base_reward
        
        goal_reward = 0.0
        current_pos = self._get_position()
        distance_to_goal = self._calculate_distance_to_goal(current_pos)
        
        # 1. 到达目标的巨大奖励
        if distance_to_goal <= self.goal_tolerance:
            goal_reward += self.config["goal_reached_reward"]
            self.reward_components["goal_reached"] = self.config["goal_reached_reward"]
        
        # 2. 距离奖励（距离目标越近奖励越大）
        max_distance = 100.0  # 最大可能距离
        normalized_distance = min(distance_to_goal / max_distance, 1.0)
        distance_reward = (1.0 - normalized_distance) * self.config["goal_distance_reward_scale"]
        goal_reward += distance_reward
        
        # 3. 进度奖励（朝目标移动）
        if hasattr(self, 'last_distance_to_goal'):
            progress = self.last_distance_to_goal - distance_to_goal
            if progress > 0:  # 正在接近目标
                goal_reward += progress * self.config["goal_progress_reward"]
        
        self.last_distance_to_goal = distance_to_goal
        
        # 记录目标相关奖励组件
        self.reward_components["goal_distance"] = distance_reward
        self.reward_components["goal_progress"] = goal_reward - distance_reward - self.reward_components.get("goal_reached", 0)
        
        return base_reward + goal_reward
    
    def _add_goal_to_observation(self, observation: Dict) -> Dict:
        """在观察中添加目标信息"""
        if self.current_goal is not None:
            current_pos = self._get_position()
            
            # 相对目标位置
            relative_goal = self.current_goal - current_pos
            distance_to_goal = np.linalg.norm(relative_goal)
            
            # 添加目标相关信息到状态
            goal_state = np.array([
                relative_goal[0],  # 相对X
                relative_goal[1],  # 相对Y  
                relative_goal[2],  # 相对Z
                distance_to_goal,  # 距离
                self.goals_reached,  # 已完成目标数
            ])
            
            # 扩展原有状态
            extended_state = np.concatenate([observation["state"], goal_state])
            observation["state"] = extended_state.astype(np.float32)
        
        return observation
    
    def _get_goal_info(self) -> Dict:
        """获取目标相关信息"""
        info = {
            "current_goal": self.current_goal.tolist() if self.current_goal is not None else None,
            "goals_reached": self.goals_reached,
            "goal_history": self.goal_history,
        }
        
        if self.current_goal is not None:
            current_pos = self._get_position()
            info["distance_to_goal"] = self._calculate_distance_to_goal(current_pos)
            info["goal_direction"] = (self.current_goal - current_pos).tolist()
        
        return info
    
    def _visualize_goal(self):
        """在AirSim中可视化目标点（简化版本）"""
        try:
            # 这里可以添加在AirSim中显示目标标记的代码
            # 例如使用simPlotPoints或其他可视化API
            if hasattr(self.client, 'simPlotPoints'):
                self.client.simPlotPoints(
                    [self.current_goal.tolist()], 
                    color_rgba=[1.0, 0.0, 0.0, 1.0],  # 红色
                    size=self.config.get("goal_marker_size", 2.0),
                    duration=30.0,  # 30秒显示时间
                    is_persistent=False
                )
        except Exception as e:
            logger.debug(f"目标可视化失败: {e}")
    
    def _setup_spaces(self):
        """设置包含目标信息的观察空间"""
        super()._setup_spaces()
        
        # 扩展状态空间以包含目标信息
        original_state_dim = self.observation_space["state"].shape[0]
        goal_state_dim = 5  # 相对位置(3) + 距离(1) + 已完成目标数(1)
        
        new_state_dim = original_state_dim + goal_state_dim
        
        self.observation_space.spaces["state"] = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(new_state_dim,),
            dtype=np.float32
        )
    
    def get_goal_info(self) -> Dict:
        """获取详细的目标信息用于分析"""
        info = super().get_reward_info()
        info.update({
            "current_goal": self.current_goal.tolist() if self.current_goal is not None else None,
            "goals_reached": self.goals_reached,
            "goal_completion_rate": self.goals_reached / max(self.config["max_goals_per_episode"], 1),
            "average_goal_time": np.mean([g["time_taken"] for g in self.goal_history]) if self.goal_history else 0,
            "validation_stats": self.validation_stats.copy(),
            "validation_enabled": self.validation_enabled
        })
        
        # 添加验证器统计信息
        if self.goal_validator is not None:
            validator_stats = self.goal_validator.get_statistics()
            info["validator_stats"] = validator_stats
        
        return info