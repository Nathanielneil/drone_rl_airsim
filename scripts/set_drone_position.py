#!/usr/bin/env python3
"""
AirSim无人机位置设置脚本
支持通过命令行或配置文件设置无人机初始位置
"""

import airsim
import numpy as np
import argparse
import json
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DronePositionManager:
    """无人机位置管理器"""
    
    def __init__(self, host="127.0.0.1", port=41451):
        self.host = host
        self.port = port
        self.client = None
        self.connected = False
        
    def connect(self):
        """连接到AirSim"""
        try:
            self.client = airsim.MultirotorClient(ip=self.host, port=self.port)
            self.client.confirmConnection()
            self.connected = True
            logger.info(f"已连接到AirSim: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接AirSim失败: {e}")
            raise
    
    def set_single_position(self, vehicle_name: str, x: float, y: float, z: float, 
                           yaw: float = 0.0, reset_physics: bool = True):
        """
        设置单个无人机位置
        
        Args:
            vehicle_name: 载具名称
            x, y, z: 位置坐标
            yaw: 朝向角度(度)
            reset_physics: 是否重置物理状态
        """
        if not self.connected:
            self.connect()
        
        try:
            # 创建位置和方向
            position = airsim.Vector3r(float(x), float(y), float(z))
            orientation = airsim.to_quaternion(0, 0, np.radians(yaw))
            pose = airsim.Pose(position, orientation)
            
            # 设置位置
            self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=vehicle_name)
            
            if reset_physics:
                # 重置物理状态
                self.client.reset()
                
                # 重新启用API控制
                self.client.enableApiControl(True, vehicle_name=vehicle_name)
                self.client.armDisarm(True, vehicle_name=vehicle_name)
            
            logger.info(f"无人机 {vehicle_name} 位置设置为: ({x:.2f}, {y:.2f}, {z:.2f}), 朝向: {yaw:.1f}°")
            
        except Exception as e:
            logger.error(f"设置位置失败: {e}")
            raise
    
    def set_multiple_positions(self, positions: List[Dict]):
        """
        设置多个无人机位置
        
        Args:
            positions: 位置配置列表
            [{"vehicle": "Drone1", "x": 0, "y": 0, "z": -3, "yaw": 0}, ...]
        """
        if not self.connected:
            self.connect()
        
        for pos_config in positions:
            vehicle = pos_config.get("vehicle", "Drone1")
            x = pos_config.get("x", 0.0)
            y = pos_config.get("y", 0.0) 
            z = pos_config.get("z", -3.0)
            yaw = pos_config.get("yaw", 0.0)
            
            self.set_single_position(vehicle, x, y, z, yaw, reset_physics=False)
        
        # 最后统一重置物理状态
        self.client.reset()
        for pos_config in positions:
            vehicle = pos_config.get("vehicle", "Drone1")
            self.client.enableApiControl(True, vehicle_name=vehicle)
            self.client.armDisarm(True, vehicle_name=vehicle)
    
    def set_random_position(self, vehicle_name: str = "Drone1", 
                           x_range: Tuple[float, float] = (-50, 50),
                           y_range: Tuple[float, float] = (-50, 50),
                           z_range: Tuple[float, float] = (-10, -2),
                           yaw_range: Tuple[float, float] = (0, 360)) -> Dict:
        """
        随机设置无人机位置
        
        Returns:
            设置的位置信息字典
        """
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        z = np.random.uniform(z_range[0], z_range[1])
        yaw = np.random.uniform(yaw_range[0], yaw_range[1])
        
        self.set_single_position(vehicle_name, x, y, z, yaw)
        
        return {
            "vehicle": vehicle_name,
            "x": x, "y": y, "z": z, "yaw": yaw,
            "timestamp": np.datetime64('now').astype(str)
        }
    
    def get_current_position(self, vehicle_name: str = "Drone1") -> Dict:
        """获取当前位置"""
        if not self.connected:
            self.connect()
        
        try:
            state = self.client.getMultirotorState(vehicle_name=vehicle_name)
            pos = state.kinematics_estimated.position
            ori = state.kinematics_estimated.orientation
            
            # 转换四元数到欧拉角
            pitch, roll, yaw = airsim.to_eularian_angles(ori)
            
            return {
                "vehicle": vehicle_name,
                "x": pos.x_val,
                "y": pos.y_val, 
                "z": pos.z_val,
                "yaw": np.degrees(yaw),
                "pitch": np.degrees(pitch),
                "roll": np.degrees(roll)
            }
        except Exception as e:
            logger.error(f"获取位置失败: {e}")
            raise
    
    def save_position_preset(self, positions: List[Dict], filename: str):
        """保存位置预设"""
        preset_path = Path("configs/position_presets") / f"{filename}.yaml"
        preset_path.parent.mkdir(parents=True, exist_ok=True)
        
        preset_data = {
            "description": f"位置预设: {filename}",
            "created_at": np.datetime64('now').astype(str),
            "positions": positions
        }
        
        with open(preset_path, 'w', encoding='utf-8') as f:
            yaml.dump(preset_data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"位置预设已保存: {preset_path}")
    
    def load_position_preset(self, filename: str) -> List[Dict]:
        """加载位置预设"""
        preset_path = Path("configs/position_presets") / f"{filename}.yaml"
        
        if not preset_path.exists():
            raise FileNotFoundError(f"预设文件不存在: {preset_path}")
        
        with open(preset_path, 'r', encoding='utf-8') as f:
            preset_data = yaml.safe_load(f)
        
        logger.info(f"已加载位置预设: {filename}")
        return preset_data.get("positions", [])


def create_preset_examples():
    """创建示例预设"""
    manager = DronePositionManager()
    
    # 单点起飞预设
    single_preset = [
        {"vehicle": "Drone1", "x": 0, "y": 0, "z": -3, "yaw": 0}
    ]
    manager.save_position_preset(single_preset, "single_takeoff")
    
    # 多点编队预设
    formation_preset = [
        {"vehicle": "Drone1", "x": 0, "y": 0, "z": -3, "yaw": 0},
        {"vehicle": "Drone2", "x": 5, "y": 0, "z": -3, "yaw": 0},
        {"vehicle": "Drone3", "x": 0, "y": 5, "z": -3, "yaw": 90},
        {"vehicle": "Drone4", "x": 5, "y": 5, "z": -3, "yaw": 45}
    ]
    manager.save_position_preset(formation_preset, "formation_square")
    
    # 训练起点预设
    training_preset = [
        {"vehicle": "Drone1", "x": 10, "y": 0, "z": -5, "yaw": 0}
    ]
    manager.save_position_preset(training_preset, "training_start")
    
    logger.info("示例预设已创建")


def main():
    parser = argparse.ArgumentParser(description='AirSim无人机位置设置工具')
    
    # 基本参数
    parser.add_argument('--host', default='127.0.0.1', help='AirSim主机地址')
    parser.add_argument('--port', type=int, default=41451, help='AirSim端口')
    parser.add_argument('--vehicle', default='Drone1', help='载具名称')
    
    # 命令组
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 设置位置命令
    pos_parser = subparsers.add_parser('set', help='设置位置')
    pos_parser.add_argument('--x', type=float, default=0.0, help='X坐标')
    pos_parser.add_argument('--y', type=float, default=0.0, help='Y坐标')
    pos_parser.add_argument('--z', type=float, default=-3.0, help='Z坐标')
    pos_parser.add_argument('--yaw', type=float, default=0.0, help='朝向角度')
    
    # 随机位置命令
    rand_parser = subparsers.add_parser('random', help='随机位置')
    rand_parser.add_argument('--x-range', nargs=2, type=float, default=[-50, 50], help='X范围')
    rand_parser.add_argument('--y-range', nargs=2, type=float, default=[-50, 50], help='Y范围')
    rand_parser.add_argument('--z-range', nargs=2, type=float, default=[-10, -2], help='Z范围')
    rand_parser.add_argument('--yaw-range', nargs=2, type=float, default=[0, 360], help='朝向范围')
    
    # 预设命令
    preset_parser = subparsers.add_parser('preset', help='使用预设')
    preset_parser.add_argument('name', help='预设名称')
    
    # 获取位置命令
    subparsers.add_parser('get', help='获取当前位置')
    
    # 创建示例预设命令
    subparsers.add_parser('create-examples', help='创建示例预设')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 创建管理器
    manager = DronePositionManager(args.host, args.port)
    
    try:
        if args.command == 'set':
            manager.set_single_position(args.vehicle, args.x, args.y, args.z, args.yaw)
            
        elif args.command == 'random':
            result = manager.set_random_position(
                args.vehicle, 
                tuple(args.x_range), 
                tuple(args.y_range),
                tuple(args.z_range), 
                tuple(args.yaw_range)
            )
            print(f"随机位置: {result}")
            
        elif args.command == 'preset':
            positions = manager.load_position_preset(args.name)
            manager.set_multiple_positions(positions)
            
        elif args.command == 'get':
            position = manager.get_current_position(args.vehicle)
            print(f"当前位置: {position}")
            
        elif args.command == 'create-examples':
            create_preset_examples()
            
    except Exception as e:
        logger.error(f"执行命令失败: {e}")


if __name__ == "__main__":
    main()