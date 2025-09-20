"""
课程学习管理器
自动调整训练难度和奖励参数
"""

import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)


class CurriculumManager:
    """课程学习管理器"""
    
    def __init__(self, env, config: Dict = None):
        self.env = env
        self.config = config or {}
        
        # 性能追踪
        self.performance_history = []
        self.collision_rate_history = []
        self.success_rate_history = []
        
        # 难度控制
        self.current_difficulty = 1
        self.max_difficulty = 5
        self.evaluation_window = 50  # 评估窗口（episodes）
        
        # 升级条件
        self.success_threshold = 0.7  # 成功率阈值
        self.collision_threshold = 0.2  # 碰撞率阈值
        self.reward_threshold = -5.0  # 平均奖励阈值
        
        # 稳定性要求
        self.stable_episodes = 100  # 需要稳定的episode数
        
        logger.info("课程学习管理器初始化完成")
    
    def update_performance(self, episode_reward: float, episode_length: int, collision_occurred: bool):
        """更新性能统计"""
        # 计算成功率（基于奖励）
        success = episode_reward > self.reward_threshold
        self.performance_history.append({
            'reward': episode_reward,
            'length': episode_length,
            'success': success,
            'collision': collision_occurred
        })
        
        # 保持窗口大小
        if len(self.performance_history) > self.evaluation_window * 2:
            self.performance_history = self.performance_history[-self.evaluation_window * 2:]
        
        # 计算统计指标
        if len(self.performance_history) >= self.evaluation_window:
            recent_performance = self.performance_history[-self.evaluation_window:]
            
            # 成功率
            success_rate = sum(1 for p in recent_performance if p['success']) / len(recent_performance)
            self.success_rate_history.append(success_rate)
            
            # 碰撞率
            collision_rate = sum(1 for p in recent_performance if p['collision']) / len(recent_performance)
            self.collision_rate_history.append(collision_rate)
            
            # 检查是否可以提升难度
            if self._should_increase_difficulty():
                self._increase_difficulty()
            elif self._should_decrease_difficulty():
                self._decrease_difficulty()
    
    def _should_increase_difficulty(self) -> bool:
        """检查是否应该提升难度"""
        if self.current_difficulty >= self.max_difficulty:
            return False
        
        if len(self.success_rate_history) < 3:
            return False
        
        # 检查最近的性能
        recent_success = self.success_rate_history[-3:]
        recent_collision = self.collision_rate_history[-3:]
        
        # 稳定的高性能
        avg_success = np.mean(recent_success)
        avg_collision = np.mean(recent_collision)
        
        success_stable = all(s >= self.success_threshold for s in recent_success)
        collision_low = all(c <= self.collision_threshold for c in recent_collision)
        
        return success_stable and collision_low and avg_success >= self.success_threshold
    
    def _should_decrease_difficulty(self) -> bool:
        """检查是否应该降低难度"""
        if self.current_difficulty <= 1:
            return False
        
        if len(self.success_rate_history) < 3:
            return False
        
        # 检查是否性能太差
        recent_success = self.success_rate_history[-3:]
        recent_collision = self.collision_rate_history[-3:]
        
        avg_success = np.mean(recent_success)
        avg_collision = np.mean(recent_collision)
        
        # 如果成功率太低或碰撞率太高
        poor_performance = avg_success < 0.3 or avg_collision > 0.5
        
        return poor_performance
    
    def _increase_difficulty(self):
        """提升难度"""
        if hasattr(self.env, 'increase_difficulty'):
            self.env.increase_difficulty()
            self.current_difficulty = min(self.max_difficulty, self.current_difficulty + 1)
            
            logger.info(f"课程学习：难度提升到等级 {self.current_difficulty}")
            
            # 记录难度变化
            self._log_difficulty_change("increased")
    
    def _decrease_difficulty(self):
        """降低难度"""
        if hasattr(self.env, 'config'):
            self.current_difficulty = max(1, self.current_difficulty - 1)
            
            # 直接修改环境配置
            self.env.config["difficulty_level"] = self.current_difficulty
            self.env.config["collision_penalty"] *= 0.8  # 减少碰撞惩罚
            self.env.config["collision_forgiveness"] += 1  # 增加宽容度
            
            logger.info(f"课程学习：难度降低到等级 {self.current_difficulty}")
            
            # 记录难度变化
            self._log_difficulty_change("decreased")
    
    def _log_difficulty_change(self, change_type: str):
        """记录难度变化"""
        if self.success_rate_history and self.collision_rate_history:
            recent_success = np.mean(self.success_rate_history[-3:])
            recent_collision = np.mean(self.collision_rate_history[-3:])
            
            logger.info(f"课程学习统计 - 成功率: {recent_success:.2f}, "
                       f"碰撞率: {recent_collision:.2f}, "
                       f"难度变化: {change_type}")
    
    def get_current_stats(self) -> Dict:
        """获取当前统计信息"""
        if not self.performance_history:
            return {}
        
        recent_performance = self.performance_history[-self.evaluation_window:]
        
        stats = {
            "current_difficulty": self.current_difficulty,
            "total_episodes": len(self.performance_history),
            "recent_episodes": len(recent_performance),
        }
        
        if recent_performance:
            rewards = [p['reward'] for p in recent_performance]
            stats.update({
                "avg_reward": np.mean(rewards),
                "success_rate": sum(1 for p in recent_performance if p['success']) / len(recent_performance),
                "collision_rate": sum(1 for p in recent_performance if p['collision']) / len(recent_performance),
                "avg_episode_length": np.mean([p['length'] for p in recent_performance])
            })
        
        return stats
    
    def should_save_checkpoint(self) -> bool:
        """判断是否应该保存检查点（在难度提升时）"""
        # 在难度刚提升后保存检查点
        return len(self.success_rate_history) > 0 and self._should_increase_difficulty()
    
    def get_adaptive_hyperparameters(self) -> Dict:
        """获取自适应超参数"""
        # 根据当前难度调整学习参数
        base_lr = 0.0003
        base_exploration = 0.1
        
        # 早期阶段使用更高的学习率和探索率
        if self.current_difficulty <= 2:
            lr_multiplier = 1.5
            exploration_multiplier = 2.0
        elif self.current_difficulty <= 3:
            lr_multiplier = 1.2
            exploration_multiplier = 1.5
        else:
            lr_multiplier = 1.0
            exploration_multiplier = 1.0
        
        return {
            "learning_rate": base_lr * lr_multiplier,
            "exploration_rate": base_exploration * exploration_multiplier,
            "difficulty_level": self.current_difficulty
        }


class RewardShaper:
    """奖励塑形器 - 动态调整奖励函数"""
    
    def __init__(self, initial_config: Dict):
        self.config = initial_config.copy()
        self.performance_buffer = []
        self.adjustment_history = []
        
    def update_rewards_based_on_performance(self, performance_stats: Dict):
        """基于性能统计调整奖励参数"""
        collision_rate = performance_stats.get('collision_rate', 0.5)
        success_rate = performance_stats.get('success_rate', 0.0)
        avg_reward = performance_stats.get('avg_reward', -10.0)
        
        adjustments = {}
        
        # 如果碰撞率太高，减少碰撞惩罚
        if collision_rate > 0.4:
            new_penalty = self.config['collision_penalty'] * 0.9
            if new_penalty > -50.0:  # 设置下限
                adjustments['collision_penalty'] = new_penalty
        
        # 如果成功率太低，增加正向奖励
        if success_rate < 0.3:
            adjustments['progress_reward_scale'] = self.config['progress_reward_scale'] * 1.1
            adjustments['safety_reward_scale'] = self.config['safety_reward_scale'] * 1.1
        
        # 如果平均奖励太低，整体增加奖励
        if avg_reward < -20.0:
            adjustments['reward_scale'] = self.config['reward_scale'] * 1.05
        
        # 应用调整
        if adjustments:
            self.config.update(adjustments)
            self.adjustment_history.append({
                'performance': performance_stats,
                'adjustments': adjustments
            })
            
            logger.info(f"奖励塑形调整: {adjustments}")
        
        return adjustments
    
    def get_current_config(self) -> Dict:
        """获取当前奖励配置"""
        return self.config.copy()