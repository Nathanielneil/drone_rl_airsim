#!/usr/bin/env python
"""
AirSim Configuration Setup Script for Hierarchical RL
====================================================

This script helps backup and configure AirSim settings.json for optimal HRL training.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path


def find_airsim_settings():
    """Find AirSim settings.json file."""
    possible_paths = [
        Path.home() / "Documents" / "AirSim" / "settings.json",
        Path.home() / ".airsim" / "settings.json",
        Path("/usr/local/AirSim/settings.json"),
        Path("/opt/AirSim/settings.json")
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"Found AirSim settings at: {path}")
            return path
    
    print("AirSim settings.json not found in common locations.")
    print("Please specify the path manually or create a new settings.json")
    return None


def backup_settings(settings_path):
    """Create timestamped backup of settings.json."""
    if not settings_path.exists():
        print(f"Settings file not found at: {settings_path}")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = settings_path.parent / f"settings_backup_{timestamp}.json"
    
    try:
        shutil.copy2(settings_path, backup_path)
        print(f"✓ Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return None


def create_hrl_settings():
    """Create optimized settings for hierarchical RL."""
    settings = {
        "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
        "SettingsVersion": 1.2,
        
        # 仿真设置
        "SimMode": "Multirotor",
        "ClockSpeed": 1.0,  # 正常速度，可以加速训练时调整为更大值
        
        # 渲染设置 - 针对RL优化
        "ViewMode": "",
        "RenderSettings": {
            "EnableMultipleWindows": False,  # 减少资源消耗
            "WindowsMode": "Windowed",  # 窗口模式便于调试
            "WindowWidth": 640,  # 较小窗口节省资源
            "WindowHeight": 480
        },
        
        # 车辆设置
        "Vehicles": {
            "Drone1": {
                "VehicleType": "SimpleFlight",
                "X": 0, "Y": 0, "Z": -2,  # 起始位置
                "Yaw": 0,
                
                # 传感器配置 - HRL需要的传感器
                "Cameras": {
                    "front_center": {
                        "CameraName": "front_center",
                        "ImageType": 0,  # Scene
                        "FOV_Degrees": 90,
                        "X": 0.50, "Y": 0.00, "Z": 0.10,
                        "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
                    },
                    "front_depth": {
                        "CameraName": "front_depth", 
                        "ImageType": 1,  # DepthPerspective
                        "FOV_Degrees": 90,
                        "X": 0.50, "Y": 0.00, "Z": 0.10,
                        "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
                    }
                },
                
                # IMU传感器
                "Sensors": {
                    "imu": {
                        "SensorType": 2,
                        "Enabled": True
                    },
                    "gps": {
                        "SensorType": 3,
                        "Enabled": True
                    }
                }
            }
        },
        
        # 物理引擎设置
        "PhysicsEngineName": "FastPhysicsEngine",  # 快速物理引擎，适合RL训练
        
        # API设置
        "ApiServerPort": 41451,
        "LogMessagesVisible": False,  # 减少日志输出
        
        # 环境设置 - 适合HRL训练
        "Environment": {
            "SunSize": 50,
            "UpdateWeatherTimeIntervalSecs": 60,
            "StartDateTime": "2018-02-12 15:20:00",
            "IsEnabled": False,  # 禁用天气变化，保持环境一致性
            
            # 风力设置 - 可以增加训练难度
            "Wind": {
                "X": 0, "Y": 0, "Z": 0
            }
        },
        
        # 录制设置 - 用于调试和可视化
        "Recording": {
            "RecordOnMove": False,
            "RecordInterval": 0.05,
            "Cameras": ["front_center", "front_depth"]
        },
        
        # HRL特定设置
        "HRL": {
            "Comment": "Settings optimized for Hierarchical Reinforcement Learning",
            "GoalVisualization": True,
            "SubgoalVisualization": True,
            "TrajectoryVisualization": False,  # 可能影响性能
            "EnableCollisionRecovery": True,
            "CollisionPenalty": -10.0,
            "GoalReachReward": 100.0,
            "StepPenalty": -0.1
        }
    }
    
    return settings


def write_settings(settings_path, settings):
    """Write settings to file."""
    try:
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        print(f"✓ Settings written to: {settings_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to write settings: {e}")
        return False


def main():
    """Main function to setup AirSim configuration."""
    print("=" * 60)
    print("AirSim Configuration Setup for Hierarchical RL")
    print("=" * 60)
    
    # 尝试找到现有的settings.json
    settings_path = find_airsim_settings()
    
    if settings_path is None:
        # 如果没找到，创建默认路径
        airsim_dir = Path.home() / "Documents" / "AirSim"
        airsim_dir.mkdir(parents=True, exist_ok=True)
        settings_path = airsim_dir / "settings.json"
        print(f"Will create new settings at: {settings_path}")
    else:
        # 备份现有设置
        backup_path = backup_settings(settings_path)
        if backup_path:
            print(f"Original settings backed up to: {backup_path}")
    
    # 创建HRL优化的设置
    hrl_settings = create_hrl_settings()
    
    # 写入新设置
    if write_settings(settings_path, hrl_settings):
        print("\n✅ AirSim settings configured for Hierarchical RL!")
        print("\nKey optimizations made:")
        print("- Fast physics engine for RL training")
        print("- Depth camera for navigation")
        print("- Disabled weather changes for consistency")
        print("- Optimized rendering settings")
        print("- GPS and IMU sensors enabled")
        print("- Added HRL-specific configuration section")
        
        print(f"\n📁 Settings location: {settings_path}")
        print("\nRestart AirSim for changes to take effect.")
    else:
        print("❌ Failed to configure AirSim settings")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("1. Restart AirSim simulation")
    print("2. Run: python hierarchical_rl/test_hrl_components_fixed.py") 
    print("3. Start HRL training: python train_hac.py")
    print("=" * 60)