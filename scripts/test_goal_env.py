#!/usr/bin/env python3
"""
测试目标导航环境
验证环境是否正常工作
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试导入"""
    print("🔍 测试模块导入...")
    
    try:
        import airsim
        print("✅ AirSim导入成功")
    except ImportError as e:
        print(f"❌ AirSim导入失败: {e}")
        return False
    
    try:
        import stable_baselines3
        print("✅ Stable-Baselines3导入成功")
    except ImportError as e:
        print(f"❌ Stable-Baselines3导入失败: {e}")
        return False
    
    try:
        from src.environments.airsim_env.modern_airsim_env import ModernAirSimEnv
        print("✅ ModernAirSimEnv导入成功")
    except ImportError as e:
        print(f"❌ ModernAirSimEnv导入失败: {e}")
        return False
    
    try:
        from src.environments.airsim_env.improved_reward_env import ImprovedRewardAirSimEnv
        print("✅ ImprovedRewardAirSimEnv导入成功")
    except ImportError as e:
        print(f"❌ ImprovedRewardAirSimEnv导入失败: {e}")
        return False
    
    try:
        from src.environments.airsim_env.goal_based_env import GoalBasedAirSimEnv
        print("✅ GoalBasedAirSimEnv导入成功")
    except ImportError as e:
        print(f"❌ GoalBasedAirSimEnv导入失败: {e}")
        return False
    
    return True

def test_airsim_connection():
    """测试AirSim连接"""
    print("\n🔌 测试AirSim连接...")
    
    try:
        import airsim
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✅ AirSim连接成功")
        
        # 获取无人机状态
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        print(f"📍 当前位置: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
        
        return True
    except Exception as e:
        print(f"❌ AirSim连接失败: {e}")
        print("请确保:")
        print("  1. AirSim环境正在运行")
        print("  2. 端口设置正确 (默认41451)")
        print("  3. 防火墙没有阻止连接")
        return False

def test_environment():
    """测试环境创建"""
    print("\n🌍 测试环境创建...")
    
    try:
        from src.environments.airsim_env.goal_based_env import GoalBasedAirSimEnv
        
        # 创建环境配置
        config = {
            "host": "127.0.0.1",
            "port": 41451,
            "vehicle_name": "Drone1",
            "max_episode_steps": 100,
            "takeoff_height": 3.0
        }
        
        # 创建环境
        env = GoalBasedAirSimEnv(config=config)
        print("✅ 目标导航环境创建成功")
        
        # 测试重置
        print("🔄 测试环境重置...")
        obs, info = env.reset()
        print("✅ 环境重置成功")
        
        print(f"📊 观察空间: {env.observation_space}")
        print(f"🎮 动作空间: {env.action_space}")
        
        if 'current_goal' in info:
            goal = info['current_goal']
            print(f"🎯 当前目标: [{goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}]")
        
        # 测试一个动作
        print("🎮 测试动作执行...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ 动作执行成功，奖励: {reward:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_file():
    """测试配置文件"""
    print("\n📄 测试配置文件...")
    
    config_path = Path("configs/goal_based_training_config.yaml")
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ 配置文件加载成功")
        
        # 检查关键配置
        if 'environment' in config:
            env_config = config['environment']
            goal_range = env_config.get('goal_range', {})
            print(f"🎯 目标范围: X={goal_range.get('x')}, Y={goal_range.get('y')}, Z={goal_range.get('z')}")
        
        return True
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 目标导航环境测试")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_imports),
        ("配置文件", test_config_file),
        ("AirSim连接", test_airsim_connection),
        ("环境创建", test_environment)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*50)
    print("📋 测试结果总结:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 测试通过率: {passed}/{len(results)} ({100*passed/len(results):.1f}%)")
    
    if passed == len(results):
        print("\n🎉 所有测试通过! 可以开始训练了")
        print("运行训练命令:")
        print("   python scripts/train_goal_based_fixed.py")
    else:
        print("\n⚠️  部分测试失败，请先解决问题")
        print("常见解决方案:")
        print("1. 确保AirSim环境正在运行")
        print("2. 检查依赖包安装: pip install -r requirements_windows_cuda121.txt")
        print("3. 检查Python路径和虚拟环境")

if __name__ == "__main__":
    main()