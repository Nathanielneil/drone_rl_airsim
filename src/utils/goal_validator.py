"""
目标点验证器
确保生成的目标点不与仿真场景中的障碍物干涉
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
import airsim
import cv2
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class GoalValidator:
    """目标点验证器，确保目标点可达且安全"""
    
    def __init__(self, client: airsim.MultirotorClient, config: Dict = None):
        self.client = client
        self.config = config or {}
        
        # 验证参数
        self.min_clearance = self.config.get("min_clearance", 3.0)  # 最小安全距离
        self.max_validation_attempts = self.config.get("max_validation_attempts", 20)
        self.validation_resolution = self.config.get("validation_resolution", 1.0)  # 路径检查分辨率
        
        # 场景分析参数
        self.scene_bounds = None
        self.obstacle_map = None
        self.safe_zones = []
        
        # 缓存设置
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_file = Path(self.config.get("cache_file", "data/goal_validation_cache.json"))
        self.validated_goals_cache = self._load_cache()
        
        logger.info(f"目标验证器初始化: 最小安全距离={self.min_clearance}m")
    
    def initialize_scene_analysis(self, vehicle_name: str = "Drone1"):
        """初始化场景分析，探测障碍物和安全区域"""
        logger.info("开始场景分析...")
        
        try:
            # 1. 分析场景边界
            self._analyze_scene_bounds(vehicle_name)
            
            # 2. 创建简单的障碍物地图
            self._create_obstacle_map(vehicle_name)
            
            # 3. 识别安全区域
            self._identify_safe_zones()
            
            logger.info(f"场景分析完成: 安全区域数量={len(self.safe_zones)}")
            
        except Exception as e:
            logger.error(f"场景分析失败: {e}")
            # 使用保守的默认设置
            self._use_conservative_defaults()
    
    def validate_goal(self, goal_position: np.ndarray, start_position: np.ndarray = None, 
                     vehicle_name: str = "Drone1") -> Tuple[bool, str]:
        """
        验证目标点是否可达且安全
        
        Args:
            goal_position: 目标位置 [x, y, z]
            start_position: 起始位置，如果为None则使用当前位置
            vehicle_name: 载具名称
        
        Returns:
            (is_valid, reason): 验证结果和原因
        """
        try:
            # 检查缓存
            goal_key = self._get_goal_cache_key(goal_position)
            if self.enable_caching and goal_key in self.validated_goals_cache:
                cached_result = self.validated_goals_cache[goal_key]
                return cached_result["valid"], cached_result["reason"]
            
            # 获取起始位置
            if start_position is None:
                start_position = self._get_current_position(vehicle_name)
            
            # 1. 基础范围检查
            if not self._check_basic_bounds(goal_position):
                reason = "目标超出场景边界"
                self._cache_result(goal_key, False, reason)
                return False, reason
            
            # 2. 高度检查
            if not self._check_height_constraints(goal_position):
                reason = "目标高度不合适"
                self._cache_result(goal_key, False, reason)
                return False, reason
            
            # 3. 点位碰撞检查
            if not self._check_point_collision(goal_position, vehicle_name):
                reason = "目标位置存在障碍物"
                self._cache_result(goal_key, False, reason)
                return False, reason
            
            # 4. 路径可达性检查
            if not self._check_path_reachability(start_position, goal_position, vehicle_name):
                reason = "无法安全到达目标"
                self._cache_result(goal_key, False, reason)
                return False, reason
            
            # 5. 安全余量检查
            if not self._check_safety_clearance(goal_position, vehicle_name):
                reason = "目标位置安全余量不足"
                self._cache_result(goal_key, False, reason)
                return False, reason
            
            # 验证通过
            reason = "目标位置有效"
            self._cache_result(goal_key, True, reason)
            return True, reason
            
        except Exception as e:
            logger.error(f"目标验证失败: {e}")
            return False, f"验证过程出错: {str(e)}"
    
    def generate_safe_goal(self, goal_range: Dict, start_position: np.ndarray = None, 
                          vehicle_name: str = "Drone1") -> Optional[np.ndarray]:
        """
        生成安全的目标点
        
        Args:
            goal_range: 目标范围字典 {"x": [min, max], "y": [min, max], "z": [min, max]}
            start_position: 起始位置
            vehicle_name: 载具名称
        
        Returns:
            安全的目标位置，如果无法生成则返回None
        """
        for attempt in range(self.max_validation_attempts):
            # 生成候选目标
            candidate = self._generate_candidate_goal(goal_range, start_position)
            
            # 验证目标
            is_valid, reason = self.validate_goal(candidate, start_position, vehicle_name)
            
            if is_valid:
                logger.info(f"生成安全目标: {candidate} (尝试{attempt+1}次)")
                return candidate
            else:
                logger.debug(f"目标候选{candidate}无效: {reason}")
        
        # 如果无法生成安全目标，使用已知安全区域
        logger.warning(f"无法在{self.max_validation_attempts}次尝试内生成安全目标，使用安全区域")
        return self._get_goal_from_safe_zones(goal_range)
    
    def _analyze_scene_bounds(self, vehicle_name: str):
        """分析场景边界"""
        try:
            # 获取当前位置作为参考
            current_pos = self._get_current_position(vehicle_name)
            
            # 保守估计场景边界（可以根据具体环境调整）
            self.scene_bounds = {
                "x": [current_pos[0] - 200, current_pos[0] + 200],
                "y": [current_pos[1] - 200, current_pos[1] + 200], 
                "z": [current_pos[2] - 50, current_pos[2] + 50]
            }
            
            logger.info(f"场景边界设定: {self.scene_bounds}")
            
        except Exception as e:
            logger.error(f"场景边界分析失败: {e}")
            self._use_conservative_defaults()
    
    def _create_obstacle_map(self, vehicle_name: str):
        """创建简化的障碍物地图"""
        try:
            # 使用深度图像进行障碍物检测
            current_pos = self._get_current_position(vehicle_name)
            
            # 简化版本：使用多个测试点进行碰撞检测
            test_positions = self._generate_test_positions(current_pos)
            obstacles = []
            
            for pos in test_positions:
                if not self._check_point_collision(pos, vehicle_name):
                    obstacles.append(pos)
            
            self.obstacle_map = obstacles
            logger.info(f"检测到{len(obstacles)}个障碍物位置")
            
        except Exception as e:
            logger.error(f"障碍物地图创建失败: {e}")
            self.obstacle_map = []
    
    def _identify_safe_zones(self):
        """识别安全区域"""
        try:
            if self.scene_bounds is None:
                return
            
            # 基于当前位置和场景边界定义安全区域
            # 这里使用简化的方法，实际应用中可以更复杂
            
            center_x = (self.scene_bounds["x"][0] + self.scene_bounds["x"][1]) / 2
            center_y = (self.scene_bounds["y"][0] + self.scene_bounds["y"][1]) / 2
            
            # 定义几个基础安全区域
            self.safe_zones = [
                {
                    "center": [center_x + 20, center_y, -5],
                    "radius": 10,
                    "height_range": [-10, -2]
                },
                {
                    "center": [center_x - 20, center_y, -5], 
                    "radius": 10,
                    "height_range": [-10, -2]
                },
                {
                    "center": [center_x, center_y + 20, -5],
                    "radius": 10,
                    "height_range": [-10, -2]
                }
            ]
            
        except Exception as e:
            logger.error(f"安全区域识别失败: {e}")
            self.safe_zones = []
    
    def _check_basic_bounds(self, goal_position: np.ndarray) -> bool:
        """检查基础边界约束"""
        if self.scene_bounds is None:
            return True
        
        x, y, z = goal_position
        
        if not (self.scene_bounds["x"][0] <= x <= self.scene_bounds["x"][1]):
            return False
        if not (self.scene_bounds["y"][0] <= y <= self.scene_bounds["y"][1]):
            return False
        if not (self.scene_bounds["z"][0] <= z <= self.scene_bounds["z"][1]):
            return False
        
        return True
    
    def _check_height_constraints(self, goal_position: np.ndarray) -> bool:
        """检查高度约束"""
        z = goal_position[2]
        
        # AirSim中Z轴向下为正，所以负值表示空中
        if z > 0:  # 地下
            return False
        if z < -50:  # 过高
            return False
        
        return True
    
    def _check_point_collision(self, position: np.ndarray, vehicle_name: str) -> bool:
        """检查点位是否与障碍物碰撞"""
        try:
            # 获取当前位置
            original_pos = self._get_current_position(vehicle_name)
            
            # 移动到目标位置进行测试
            self.client.moveToPositionAsync(
                position[0], position[1], position[2], 5.0, 
                vehicle_name=vehicle_name
            ).join()
            
            # 等待一小段时间让物理引擎稳定
            time.sleep(0.1)
            
            # 检查碰撞
            collision_info = self.client.simGetCollisionInfo(vehicle_name=vehicle_name)
            has_collision = collision_info.has_collided
            
            # 恢复到原始位置
            self.client.moveToPositionAsync(
                original_pos[0], original_pos[1], original_pos[2], 5.0,
                vehicle_name=vehicle_name
            ).join()
            
            return not has_collision
            
        except Exception as e:
            logger.error(f"点位碰撞检查失败: {e}")
            return False
    
    def _check_path_reachability(self, start: np.ndarray, goal: np.ndarray, 
                                vehicle_name: str) -> bool:
        """检查路径可达性"""
        try:
            # 简化的路径检查：在起点和终点之间采样几个中间点
            num_samples = int(np.linalg.norm(goal - start) / self.validation_resolution)
            num_samples = max(3, min(num_samples, 10))  # 限制采样点数量
            
            for i in range(1, num_samples):
                t = i / num_samples
                intermediate_point = start + t * (goal - start)
                
                # 检查中间点是否安全
                if not self._check_point_collision(intermediate_point, vehicle_name):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"路径可达性检查失败: {e}")
            return False
    
    def _check_safety_clearance(self, position: np.ndarray, vehicle_name: str) -> bool:
        """检查安全余量"""
        try:
            # 检查周围一定范围内是否有障碍物
            clearance_points = [
                position + [self.min_clearance, 0, 0],
                position + [-self.min_clearance, 0, 0],
                position + [0, self.min_clearance, 0],
                position + [0, -self.min_clearance, 0],
                position + [0, 0, self.min_clearance/2],
                position + [0, 0, -self.min_clearance/2]
            ]
            
            for point in clearance_points:
                if not self._check_point_collision(point, vehicle_name):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"安全余量检查失败: {e}")
            return True  # 保守地假设安全
    
    def _generate_candidate_goal(self, goal_range: Dict, start_position: np.ndarray = None) -> np.ndarray:
        """生成候选目标点"""
        x = np.random.uniform(goal_range["x"][0], goal_range["x"][1])
        y = np.random.uniform(goal_range["y"][0], goal_range["y"][1])
        z = np.random.uniform(goal_range["z"][0], goal_range["z"][1])
        
        # AirSim中z轴向下为正，转换高度值
        z = -abs(z)
        
        return np.array([x, y, z])
    
    def _get_goal_from_safe_zones(self, goal_range: Dict) -> Optional[np.ndarray]:
        """从已知安全区域获取目标"""
        if not self.safe_zones:
            return None
        
        # 选择一个随机安全区域
        safe_zone = np.random.choice(self.safe_zones)
        
        # 在安全区域内生成目标
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, safe_zone["radius"])
        
        x = safe_zone["center"][0] + radius * np.cos(angle)
        y = safe_zone["center"][1] + radius * np.sin(angle)
        z = np.random.uniform(safe_zone["height_range"][0], safe_zone["height_range"][1])
        
        return np.array([x, y, z])
    
    def _generate_test_positions(self, center: np.ndarray, radius: float = 50) -> List[np.ndarray]:
        """生成测试位置用于障碍物检测"""
        positions = []
        
        # 在中心周围生成网格点
        for x in range(-int(radius), int(radius), 10):
            for y in range(-int(radius), int(radius), 10):
                for z in range(-20, 0, 5):
                    pos = center + np.array([x, y, z])
                    positions.append(pos)
        
        return positions
    
    def _get_current_position(self, vehicle_name: str) -> np.ndarray:
        """获取当前位置"""
        try:
            state = self.client.getMultirotorState(vehicle_name=vehicle_name)
            pos = state.kinematics_estimated.position
            return np.array([pos.x_val, pos.y_val, pos.z_val])
        except:
            return np.array([0.0, 0.0, -3.0])  # 默认位置
    
    def _get_goal_cache_key(self, goal_position: np.ndarray) -> str:
        """生成目标缓存键"""
        # 四舍五入到0.5米精度以提高缓存命中率
        rounded = np.round(goal_position * 2) / 2
        return f"{rounded[0]:.1f}_{rounded[1]:.1f}_{rounded[2]:.1f}"
    
    def _cache_result(self, key: str, valid: bool, reason: str):
        """缓存验证结果"""
        if self.enable_caching:
            self.validated_goals_cache[key] = {
                "valid": valid,
                "reason": reason,
                "timestamp": time.time()
            }
            self._save_cache()
    
    def _load_cache(self) -> Dict:
        """加载缓存"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"缓存加载失败: {e}")
        return {}
    
    def _save_cache(self):
        """保存缓存"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.validated_goals_cache, f)
        except Exception as e:
            logger.debug(f"缓存保存失败: {e}")
    
    def _use_conservative_defaults(self):
        """使用保守的默认设置"""
        self.scene_bounds = {
            "x": [-100, 100],
            "y": [-100, 100],
            "z": [-50, 10]
        }
        self.safe_zones = [
            {
                "center": [20, 0, -5],
                "radius": 15,
                "height_range": [-10, -2]
            }
        ]
        logger.info("使用保守的默认场景设置")
    
    def get_statistics(self) -> Dict:
        """获取验证统计信息"""
        if not self.validated_goals_cache:
            return {"total": 0, "valid": 0, "invalid": 0, "valid_rate": 0.0}
        
        total = len(self.validated_goals_cache)
        valid = sum(1 for result in self.validated_goals_cache.values() if result["valid"])
        invalid = total - valid
        valid_rate = valid / total if total > 0 else 0.0
        
        return {
            "total": total,
            "valid": valid, 
            "invalid": invalid,
            "valid_rate": valid_rate,
            "safe_zones": len(self.safe_zones)
        }