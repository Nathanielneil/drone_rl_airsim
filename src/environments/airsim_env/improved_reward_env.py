"""
改进的AirSim环境奖励函数
解决负奖励过大和学习困难问题
"""

import numpy as np
import logging
from typing import Dict, Tuple
from .modern_airsim_env import ModernAirSimEnv

logger = logging.getLogger(__name__)


class ImprovedRewardAirSimEnv(ModernAirSimEnv):
    """改进奖励函数的AirSim环境"""
    
    def __init__(self, config=None, **kwargs):
        # 设置改进的默认配置
        improved_config = self._get_improved_default_config()
        if config:
            improved_config.update(config)
        
        super().__init__(config=improved_config, **kwargs)
        
        # 奖励追踪
        self.reward_components = {
            "progress": 0.0,
            "safety": 0.0,
            "efficiency": 0.0,
            "collision": 0.0,
            "exploration": 0.0
        }
        
        # 安全距离追踪
        self.min_distance_to_obstacle = float('inf')
        self.consecutive_safe_steps = 0
        self.progress_history = []
        
    def _get_improved_default_config(self) -> Dict:
        """获取改进的默认配置"""
        config = super()._get_default_config()
        
        # 改进的奖励参数
        config.update({
            # 基础奖励缩放
            "reward_scale": 0.1,  # 整体奖励缩放因子
            
            # 进度奖励（鼓励向前探索）
            "progress_reward_scale": 10.0,  # 前进奖励
            "goal_reward": 50.0,  # 到达目标奖励（降低）
            
            # 安全奖励（鼓励避障）
            "safe_distance_threshold": 3.0,  # 安全距离阈值
            "safety_reward_scale": 2.0,  # 保持安全距离奖励
            "near_miss_penalty": -2.0,  # 险象环生惩罚（降低）
            "collision_penalty": -10.0,  # 碰撞惩罚（大幅降低）
            
            # 效率奖励
            "time_penalty": -0.02,  # 时间惩罚（轻微增加）
            "velocity_reward_scale": 1.0,  # 合理速度奖励
            "altitude_reward_scale": 0.5,  # 高度维持奖励
            
            # 探索奖励
            "exploration_reward_scale": 0.5,  # 新区域探索奖励
            "hover_penalty": -0.5,  # 悬停惩罚
            
            # 课程学习参数
            "curriculum_enabled": True,
            "difficulty_level": 1,  # 1-5难度等级
            "collision_forgiveness": 3,  # 允许的碰撞次数
            
            # 形状奖励（引导学习）
            "distance_shaping": True,  # 距离形状奖励
            "velocity_shaping": True,  # 速度形状奖励
        })
        
        return config
    
    def _calculate_reward(self) -> float:
        """计算改进的奖励"""
        # 重置奖励组件
        self.reward_components = {key: 0.0 for key in self.reward_components}
        
        try:
            # 获取当前状态
            current_position = self._get_position()
            state = self._get_state()
            velocity = state[3:6]
            
            # 检查碰撞
            collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
            if collision_info.has_collided:
                return self._handle_collision_reward()
            
            # 1. 进度奖励
            progress_reward = self._calculate_progress_reward(current_position)
            self.reward_components["progress"] = progress_reward
            
            # 2. 安全奖励
            safety_reward = self._calculate_safety_reward(current_position)
            self.reward_components["safety"] = safety_reward
            
            # 3. 效率奖励
            efficiency_reward = self._calculate_efficiency_reward(velocity, current_position)
            self.reward_components["efficiency"] = efficiency_reward
            
            # 4. 探索奖励
            exploration_reward = self._calculate_exploration_reward(current_position)
            self.reward_components["exploration"] = exploration_reward
            
            # 总奖励
            total_reward = sum(self.reward_components.values())
            
            # 应用课程学习调整
            total_reward = self._apply_curriculum_adjustment(total_reward)
            
            # 应用奖励缩放
            total_reward *= self.config["reward_scale"]
            
            # 更新历史
            self.last_position = current_position
            self.consecutive_safe_steps += 1
            
            return float(total_reward)
            
        except Exception as e:
            logger.error(f"奖励计算失败: {e}")
            return -0.1
    
    def _calculate_progress_reward(self, current_position: np.ndarray) -> float:
        """计算进度奖励"""
        reward = 0.0
        
        if self.last_position is not None:
            # 前进距离奖励
            forward_progress = current_position[0] - self.last_position[0]
            reward += forward_progress * self.config["progress_reward_scale"]
            
            # 记录进度历史
            self.progress_history.append(forward_progress)
            if len(self.progress_history) > 10:
                self.progress_history.pop(0)
            
            # 持续前进奖励
            if len(self.progress_history) >= 5:
                avg_progress = np.mean(self.progress_history)
                if avg_progress > 0.1:  # 持续前进
                    reward += 1.0
        
        return reward
    
    def _calculate_safety_reward(self, current_position: np.ndarray) -> float:
        """计算安全奖励"""
        reward = 0.0
        
        try:
            # 获取距离传感器信息（模拟）
            # 在实际实现中，这里应该使用LiDAR或深度图像计算最近障碍物距离
            min_distance = self._estimate_min_obstacle_distance()
            self.min_distance_to_obstacle = min(self.min_distance_to_obstacle, min_distance)
            
            # 安全距离奖励
            safe_threshold = self.config["safe_distance_threshold"]
            if min_distance > safe_threshold:
                reward += self.config["safety_reward_scale"]
            elif min_distance > safe_threshold * 0.5:
                # 部分安全奖励
                ratio = min_distance / safe_threshold
                reward += self.config["safety_reward_scale"] * ratio
            else:
                # 险象环生惩罚
                reward += self.config["near_miss_penalty"]
            
            # 连续安全飞行奖励
            if min_distance > safe_threshold:
                bonus = min(self.consecutive_safe_steps * 0.01, 1.0)
                reward += bonus
            else:
                self.consecutive_safe_steps = 0
            
        except Exception as e:
            logger.warning(f"安全奖励计算失败: {e}")
        
        return reward
    
    def _calculate_efficiency_reward(self, velocity: np.ndarray, position: np.ndarray) -> float:
        """计算效率奖励"""
        reward = 0.0
        
        # 时间惩罚
        reward += self.config["time_penalty"]
        
        # 速度奖励（鼓励合理速度）
        speed = np.linalg.norm(velocity)
        optimal_speed = self.config["max_velocity"] * 0.6  # 60%最大速度为最优
        
        if 0.3 * optimal_speed <= speed <= optimal_speed:
            # 在合理速度范围内
            reward += self.config["velocity_reward_scale"]
        elif speed > optimal_speed:
            # 速度过快惩罚
            excess_speed = speed - optimal_speed
            reward -= excess_speed * 0.1
        elif speed < 0.1:
            # 悬停惩罚
            reward += self.config["hover_penalty"]
        
        # 高度维持奖励
        altitude = -position[2]
        target_altitude = self.config["takeoff_height"]
        altitude_diff = abs(altitude - target_altitude)
        
        if altitude_diff < 1.0:
            reward += self.config["altitude_reward_scale"]
        elif altitude_diff > 5.0:
            reward -= altitude_diff * 0.1
        
        return reward
    
    def _calculate_exploration_reward(self, position: np.ndarray) -> float:
        """计算探索奖励"""
        reward = 0.0
        
        # 简单的探索奖励：鼓励到达新区域
        distance_from_origin = np.linalg.norm(position[:2])
        
        if distance_from_origin > 5.0:  # 探索远离起点
            reward += self.config["exploration_reward_scale"]
        
        return reward
    
    def _handle_collision_reward(self) -> float:
        """处理碰撞奖励"""
        self.collision_count += 1
        
        # 基础碰撞惩罚
        penalty = self.config["collision_penalty"]
        
        # 课程学习：早期碰撞惩罚更轻
        if self.config["curriculum_enabled"]:
            difficulty = self.config["difficulty_level"]
            if difficulty <= 2 and self.collision_count <= self.config["collision_forgiveness"]:
                penalty *= 0.5  # 早期学习阶段减少惩罚
        
        self.reward_components["collision"] = penalty
        self.consecutive_safe_steps = 0
        
        return penalty * self.config["reward_scale"]
    
    def _estimate_min_obstacle_distance(self) -> float:
        """估算到最近障碍物的距离（简化版本）"""
        # 这里应该使用深度图像或LiDAR数据
        # 目前使用简化的距离估算
        try:
            # 获取深度图像
            image_request = airsim.ImageRequest(
                self.camera_name, 
                airsim.ImageType.DepthPerspective, 
                True, 
                False
            )
            responses = self.client.simGetImages([image_request], vehicle_name=self.vehicle_name)
            
            if responses and len(responses) > 0:
                # 处理深度图像
                depth_data = np.array(responses[0].image_data_float, dtype=np.float32)
                depth_data = depth_data.reshape(responses[0].height, responses[0].width)
                
                # 过滤无效值
                valid_depth = depth_data[(depth_data > 0) & (depth_data < 100)]
                
                if len(valid_depth) > 0:
                    return float(np.min(valid_depth))
            
            return 10.0  # 默认安全距离
            
        except Exception as e:
            logger.warning(f"深度估算失败: {e}")
            return 5.0  # 保守估计
    
    def _apply_curriculum_adjustment(self, reward: float) -> float:
        """应用课程学习调整"""
        if not self.config["curriculum_enabled"]:
            return reward
        
        difficulty = self.config["difficulty_level"]
        
        # 早期阶段给予更多正奖励
        if difficulty <= 2:
            if reward > 0:
                reward *= 1.5  # 增强正奖励
            elif reward < 0:
                reward *= 0.7  # 减少负奖励
        
        return reward
    
    def get_reward_info(self) -> Dict:
        """获取详细奖励信息"""
        return {
            "reward_components": self.reward_components.copy(),
            "consecutive_safe_steps": self.consecutive_safe_steps,
            "min_distance": self.min_distance_to_obstacle,
            "collision_count": self.collision_count,
            "difficulty_level": self.config["difficulty_level"]
        }
    
    def increase_difficulty(self):
        """增加训练难度"""
        if self.config["difficulty_level"] < 5:
            self.config["difficulty_level"] += 1
            
            # 调整奖励参数
            self.config["collision_penalty"] *= 1.2
            self.config["safe_distance_threshold"] += 0.5
            self.config["collision_forgiveness"] = max(1, self.config["collision_forgiveness"] - 1)
            
            logger.info(f"难度提升到等级 {self.config['difficulty_level']}")
    
    def reset(self, seed=None, options=None):
        """重置环境时清理状态"""
        observation, info = super().reset(seed=seed, options=options)
        
        # 重置奖励追踪
        self.reward_components = {key: 0.0 for key in self.reward_components}
        self.min_distance_to_obstacle = float('inf')
        self.consecutive_safe_steps = 0
        self.progress_history = []
        
        # 添加奖励信息到info中
        info.update(self.get_reward_info())
        
        return observation, info