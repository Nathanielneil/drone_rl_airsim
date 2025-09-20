"""
现代化的AirSim环境接口
针对 Windows 10 + AirSim 1.8.1 + UE4.7.2 + CUDA 12.1 优化
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json

import gymnasium as gym
import numpy as np
import cv2
import torch
import airsim
from airsim import MultirotorClient, ImageRequest, ImageType, YawMode
from gymnasium import spaces
from gymnasium.utils import seeding

# 设置日志
logger = logging.getLogger(__name__)


class ModernAirSimEnv(gym.Env):
    """
    现代化的AirSim环境，支持：
    - AirSim 1.8.1 API
    - 异步操作
    - GPU优化的图像处理
    - Windows路径兼容
    - 现代Gymnasium接口
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        
        # 默认配置
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
            
        self.render_mode = render_mode
        
        # AirSim连接配置
        self.host = self.config.get("host", "127.0.0.1")
        self.port = self.config.get("port", 41451)
        self.vehicle_name = self.config.get("vehicle_name", "Drone1")
        
        # 相机配置
        self.camera_name = self.config.get("camera_name", "front_center")
        self.image_type = self.config.get("image_type", "DepthVis")
        self.image_width = int(self.config.get("image_width", 84))
        self.image_height = int(self.config.get("image_height", 84))
        
        # 环境配置
        self.max_episode_steps = int(self.config.get("max_episode_steps", 1000))
        self.action_space_type = self.config.get("action_space_type", "continuous")
        
        # GPU配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gpu_processing = self.config.get("use_gpu_processing", torch.cuda.is_available())
        
        # 设置观察空间和动作空间
        self._setup_spaces()
        
        # AirSim客户端
        self.client = None
        self.connected = False
        
        # 状态跟踪
        self.current_step = 0
        self.episode_reward = 0.0
        self.collision_count = 0
        self.last_position = None
        
        # 性能监控
        self.frame_times = []
        self.action_times = []
        
        # 连接到AirSim
        self._connect_airsim()
        
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "host": "127.0.0.1",
            "port": 41451,
            "vehicle_name": "Drone1",
            "camera_name": "front_center",
            "image_type": "DepthVis",
            "image_width": 84,
            "image_height": 84,
            "max_episode_steps": 1000,
            "action_space_type": "continuous",
            "use_gpu_processing": True,
            "max_velocity": 10.0,
            "max_altitude": 50.0,
            "min_altitude": -10.0,
            "takeoff_height": 2.0,
            "collision_penalty": -100.0,
            "goal_reward": 100.0,
            "distance_reward_scale": 1.0,
            "velocity_penalty_scale": -0.1,
            "time_penalty": -0.01,
            "safe_distance": 1.0,
        }
    
    def _setup_spaces(self):
        """设置观察空间和动作空间"""
        # 观察空间：深度图像 + 状态信息
        image_shape = (self.image_height, self.image_width, 1)
        state_dim = 9  # position(3) + velocity(3) + orientation(3)
        
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255, 
                shape=image_shape, 
                dtype=np.uint8
            ),
            "state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(state_dim,),
                dtype=np.float32
            )
        })
        
        # 动作空间
        if self.action_space_type == "continuous":
            # 连续动作：[vx, vy, vz] 速度控制
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0]),
                high=np.array([1.0, 1.0, 1.0]),
                dtype=np.float32
            )
        else:
            # 离散动作：8个方向 + 停止
            self.action_space = spaces.Discrete(9)
    
    def _connect_airsim(self):
        """连接到AirSim"""
        try:
            logger.info(f"连接到AirSim: {self.host}:{self.port}")
            
            # 创建客户端
            self.client = MultirotorClient(ip=self.host, port=self.port)
            
            # 确认连接
            self.client.confirmConnection()
            
            # 启用API控制
            self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
            
            # 解锁无人机
            self.client.armDisarm(True, vehicle_name=self.vehicle_name)
            
            self.connected = True
            logger.info("AirSim连接成功")
            
        except Exception as e:
            logger.error(f"AirSim连接失败: {e}")
            raise ConnectionError(f"无法连接到AirSim: {e}")
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        if not self.connected:
            self._connect_airsim()
        
        try:
            # 重置无人机
            self.client.reset()
            
            # 启用API控制
            self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
            self.client.armDisarm(True, vehicle_name=self.vehicle_name)
            
            # 起飞到指定高度
            takeoff_height = self.config["takeoff_height"]
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
            self.client.moveToZAsync(-takeoff_height, 1, vehicle_name=self.vehicle_name).join()
            
            # 等待稳定
            time.sleep(1.0)
            
            # 重置状态
            self.current_step = 0
            self.episode_reward = 0.0
            self.collision_count = 0
            self.last_position = self._get_position()
            
            # 获取初始观察
            observation = self._get_observation()
            info = self._get_info()
            
            return observation, info
            
        except Exception as e:
            logger.error(f"环境重置失败: {e}")
            raise RuntimeError(f"环境重置失败: {e}")
    
    def step(self, action: Union[np.ndarray, int]) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行动作"""
        start_time = time.time()
        
        try:
            # 检查episode是否结束
            if self.current_step >= self.max_episode_steps:
                observation = self._get_observation()
                info = self._get_info()
                return observation, 0.0, False, True, info
            
            # 执行动作
            self._execute_action(action)
            
            # 获取新状态
            observation = self._get_observation()
            
            # 计算奖励
            reward = self._calculate_reward()
            
            # 检查是否终止
            terminated = self._check_terminated()
            truncated = self.current_step >= self.max_episode_steps
            
            # 更新状态
            self.current_step += 1
            self.episode_reward += reward
            
            # 记录性能
            action_time = time.time() - start_time
            self.action_times.append(action_time)
            
            info = self._get_info()
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"步骤执行失败: {e}")
            # 返回安全状态
            observation = self._get_observation()
            info = self._get_info()
            return observation, -10.0, True, False, info
    
    def _execute_action(self, action: Union[np.ndarray, int]):
        """执行动作"""
        if self.action_space_type == "continuous":
            # 连续动作：速度控制
            vx, vy, vz = action
            
            # 缩放到实际速度范围
            max_vel = self.config["max_velocity"]
            vx = float(vx * max_vel)
            vy = float(vy * max_vel)
            vz = float(vz * max_vel)
            
            # 发送速度命令
            self.client.moveByVelocityAsync(
                vx, vy, vz, 
                duration=0.1,  # 100ms控制周期
                vehicle_name=self.vehicle_name
            )
            
        else:
            # 离散动作
            action_map = {
                0: (0, 0, 0),      # 停止
                1: (1, 0, 0),      # 前进
                2: (-1, 0, 0),     # 后退
                3: (0, 1, 0),      # 左移
                4: (0, -1, 0),     # 右移
                5: (0, 0, -1),     # 上升
                6: (0, 0, 1),      # 下降
                7: (1, 1, 0),      # 前左
                8: (1, -1, 0),     # 前右
            }
            
            if action in action_map:
                vx, vy, vz = action_map[action]
                max_vel = self.config["max_velocity"]
                vx, vy, vz = vx * max_vel, vy * max_vel, vz * max_vel
                
                self.client.moveByVelocityAsync(
                    vx, vy, vz,
                    duration=0.1,
                    vehicle_name=self.vehicle_name
                )
    
    def _get_observation(self) -> Dict:
        """获取观察数据"""
        start_time = time.time()
        
        try:
            # 获取图像
            image = self._get_image()
            
            # 获取状态信息
            state = self._get_state()
            
            observation = {
                "image": image,
                "state": state
            }
            
            # 记录性能
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            
            return observation
            
        except Exception as e:
            logger.error(f"获取观察失败: {e}")
            # 返回默认观察
            return {
                "image": np.zeros((self.image_height, self.image_width, 1), dtype=np.uint8),
                "state": np.zeros(9, dtype=np.float32)
            }
    
    def _get_image(self) -> np.ndarray:
        """获取深度图像"""
        try:
            # 请求图像
            image_request = ImageRequest(
                camera_name=self.camera_name,
                image_type=getattr(ImageType, self.image_type),
                pixels_as_float=False,
                compress=False
            )
            
            responses = self.client.simGetImages(
                [image_request], 
                vehicle_name=self.vehicle_name
            )
            
            if not responses:
                raise ValueError("未收到图像响应")
            
            response = responses[0]
            
            # 处理图像数据
            if response.image_type == ImageType.DepthVis:
                # 深度可视化图像
                img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img_1d.reshape(response.height, response.width, 3)
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            else:
                # 其他类型图像
                img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_gray = img_1d.reshape(response.height, response.width)
            
            # 调整大小
            if img_gray.shape != (self.image_height, self.image_width):
                img_gray = cv2.resize(
                    img_gray, 
                    (self.image_width, self.image_height),
                    interpolation=cv2.INTER_LINEAR
                )
            
            # 添加通道维度
            image = np.expand_dims(img_gray, axis=-1)
            
            return image.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"获取图像失败: {e}")
            return np.zeros((self.image_height, self.image_width, 1), dtype=np.uint8)
    
    def _get_state(self) -> np.ndarray:
        """获取状态信息"""
        try:
            # 获取位置和速度
            kinematics = self.client.getMultirotorState(
                vehicle_name=self.vehicle_name
            ).kinematics_estimated
            
            # 位置 (x, y, z)
            position = kinematics.position
            pos = np.array([position.x_val, position.y_val, position.z_val])
            
            # 速度 (vx, vy, vz)
            velocity = kinematics.linear_velocity
            vel = np.array([velocity.x_val, velocity.y_val, velocity.z_val])
            
            # 方向 (roll, pitch, yaw)
            orientation = kinematics.orientation
            # 转换四元数到欧拉角
            roll, pitch, yaw = airsim.to_eularian_angles(orientation)
            ori = np.array([roll, pitch, yaw])
            
            # 组合状态
            state = np.concatenate([pos, vel, ori]).astype(np.float32)
            
            return state
            
        except Exception as e:
            logger.error(f"获取状态失败: {e}")
            return np.zeros(9, dtype=np.float32)
    
    def _get_position(self) -> np.ndarray:
        """获取当前位置"""
        try:
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            position = state.kinematics_estimated.position
            return np.array([position.x_val, position.y_val, position.z_val])
        except:
            return np.array([0.0, 0.0, 0.0])
    
    def _calculate_reward(self) -> float:
        """计算奖励"""
        reward = 0.0
        
        try:
            # 基础时间惩罚
            reward += self.config["time_penalty"]
            
            # 检查碰撞
            collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
            if collision_info.has_collided:
                reward += self.config["collision_penalty"]
                self.collision_count += 1
                return reward
            
            # 位置相关奖励
            current_position = self._get_position()
            if self.last_position is not None:
                # 距离奖励（向前移动给予奖励）
                distance_moved = current_position[0] - self.last_position[0]  # x轴为前进方向
                reward += distance_moved * self.config["distance_reward_scale"]
            
            # 速度惩罚（过快飞行）
            velocity = self._get_state()[3:6]  # 提取速度部分
            speed = np.linalg.norm(velocity)
            if speed > self.config["max_velocity"] * 0.8:
                reward += self.config["velocity_penalty_scale"] * (speed - self.config["max_velocity"] * 0.8)
            
            # 高度限制
            altitude = -current_position[2]  # AirSim中z轴向下为正
            if altitude > self.config["max_altitude"] or altitude < self.config["min_altitude"]:
                reward -= 10.0
            
            self.last_position = current_position
            
            return float(reward)
            
        except Exception as e:
            logger.error(f"奖励计算失败: {e}")
            return -1.0
    
    def _check_terminated(self) -> bool:
        """检查是否终止"""
        try:
            # 检查碰撞
            collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
            if collision_info.has_collided:
                return True
            
            # 检查高度限制
            position = self._get_position()
            altitude = -position[2]
            if altitude > self.config["max_altitude"] or altitude < self.config["min_altitude"]:
                return True
            
            # 检查距离限制（可选）
            distance_from_origin = np.linalg.norm(position[:2])
            if distance_from_origin > 100.0:  # 100米限制
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"终止检查失败: {e}")
            return True
    
    def _get_info(self) -> Dict:
        """获取信息字典"""
        return {
            "episode_step": self.current_step,
            "episode_reward": self.episode_reward,
            "collision_count": self.collision_count,
            "avg_frame_time": np.mean(self.frame_times[-100:]) if self.frame_times else 0.0,
            "avg_action_time": np.mean(self.action_times[-100:]) if self.action_times else 0.0,
            "connected": self.connected,
        }
    
    def render(self, mode: str = "human"):
        """渲染环境"""
        if mode == "human":
            # 可以实现窗口显示
            pass
        elif mode == "rgb_array":
            # 返回RGB数组
            observation = self._get_observation()
            return observation["image"]
    
    def close(self):
        """关闭环境"""
        if self.client and self.connected:
            try:
                # 停止无人机
                self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
                # 着陆
                self.client.landAsync(vehicle_name=self.vehicle_name).join()
                # 禁用API控制
                self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
                self.connected = False
                logger.info("AirSim环境已关闭")
            except Exception as e:
                logger.error(f"关闭环境失败: {e}")


# 注册环境
gym.register(
    id="ModernAirSim-v1",
    entry_point="src.environments.airsim_env.modern_airsim_env:ModernAirSimEnv",
    max_episode_steps=1000,
)