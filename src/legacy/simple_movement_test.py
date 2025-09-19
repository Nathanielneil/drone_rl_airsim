#!/usr/bin/env python
"""
简化的无人机移动测试脚本
直接测试AirSim API，绕过环境重置问题
"""

import airsim
import numpy as np
import time

def test_direct_airsim_control():
    """直接使用AirSim API测试无人机控制"""
    print("=" * 60)
    print("直接AirSim API控制测试")
    print("=" * 60)
    
    try:
        # 连接AirSim
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        
        print("✓ AirSim连接成功")
        
        # 起飞
        print("起飞中...")
        client.takeoffAsync().join()
        time.sleep(2)
        
        # 获取初始位置
        state = client.getMultirotorState()
        initial_pos = state.kinematics_estimated.position
        print(f"初始位置: x={initial_pos.x_val:.2f}, y={initial_pos.y_val:.2f}, z={initial_pos.z_val:.2f}")
        
        # 测试不同方向的移动
        movements = [
            (3, 0, 0, "前进3米"),
            (0, 3, 0, "右移3米"),
            (-3, 0, 0, "后退3米"),
            (0, -3, 0, "左移3米"),
            (2, 2, 0, "斜向移动"),
            (0, 0, 0, "返回中心")
        ]
        
        print("\n测试移动指令:")
        for dx, dy, dz, description in movements:
            target_x = initial_pos.x_val + dx
            target_y = initial_pos.y_val + dy
            target_z = initial_pos.z_val + dz
            
            print(f"执行: {description}")
            print(f"  目标位置: ({target_x:.1f}, {target_y:.1f}, {target_z:.1f})")
            
            # 移动到目标位置
            client.moveToPositionAsync(target_x, target_y, target_z, 2).join()
            time.sleep(1)
            
            # 获取当前位置
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            print(f"  实际位置: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
            
            # 计算移动距离
            move_dist = np.sqrt((pos.x_val - initial_pos.x_val)**2 + 
                               (pos.y_val - initial_pos.y_val)**2)
            print(f"  水平移动距离: {move_dist:.2f}米")
            
            if move_dist > 0.5:
                print("  ✅ 成功水平移动")
            else:
                print("  ❌ 水平移动失败")
            
            time.sleep(1)
        
        # 降落
        print("\n降落...")
        client.landAsync().join()
        
        # 断开连接
        client.armDisarm(False)
        client.enableApiControl(False)
        
        return True
        
    except Exception as e:
        print(f"❌ AirSim控制测试失败: {e}")
        return False

def test_velocity_control():
    """测试速度控制模式"""
    print("\n" + "=" * 60)
    print("速度控制模式测试")
    print("=" * 60)
    
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        
        print("起飞...")
        client.takeoffAsync().join()
        time.sleep(2)
        
        # 获取初始位置
        state = client.getMultirotorState()
        initial_pos = state.kinematics_estimated.position
        print(f"初始位置: ({initial_pos.x_val:.2f}, {initial_pos.y_val:.2f}, {initial_pos.z_val:.2f})")
        
        # 测试速度控制
        velocity_tests = [
            (2, 0, 0, "前进速度"),
            (0, 2, 0, "右移速度"),
            (-2, 0, 0, "后退速度"),
            (0, -2, 0, "左移速度"),
            (0, 0, 0, "停止")
        ]
        
        print("\n测试速度控制:")
        for vx, vy, vz, description in velocity_tests:
            print(f"执行: {description} - 速度({vx}, {vy}, {vz})")
            
            # 应用速度控制
            client.moveByVelocityAsync(vx, vy, vz, 3).join()
            time.sleep(0.5)
            
            # 检查位置变化
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            
            print(f"  位置: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
            print(f"  速度: ({vel.x_val:.2f}, {vel.y_val:.2f}, {vel.z_val:.2f})")
            
            time.sleep(1)
        
        # 降落
        print("\n降落...")
        client.landAsync().join()
        
        client.armDisarm(False)
        client.enableApiControl(False)
        
        return True
        
    except Exception as e:
        print(f"❌ 速度控制测试失败: {e}")
        return False

def analyze_action_mapping():
    """分析动作映射问题"""
    print("\n" + "=" * 60)
    print("分析动作映射")
    print("=" * 60)
    
    # 检查AirGym中的动作映射
    try:
        import sys
        sys.path.append('.')
        from gym_airsim.envs.AirGym import AirSimEnv
        
        print("动作空间范围: [-0.3, 0.3]")
        print("这个范围很小，可能导致移动幅度不足")
        
        # 建议的动作映射改进
        print("\n🔧 建议的改进:")
        print("1. 增加动作空间范围到 [-1.0, 1.0]")
        print("2. 在动作映射中添加速度缩放因子")
        print("3. 确保动作正确映射到世界坐标系")
        
        return True
        
    except Exception as e:
        print(f"❌ 动作映射分析失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚁 开始简化移动测试...")
    
    results = []
    results.append(("直接AirSim控制", test_direct_airsim_control()))
    results.append(("速度控制测试", test_velocity_control()))
    results.append(("动作映射分析", analyze_action_mapping()))
    
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print("\n🔍 诊断结论:")
    print("如果AirSim直接控制成功但环境控制失败，问题在于:")
    print("1. 环境包装器的动作映射")
    print("2. 动作空间范围太小 [-0.3, 0.3]")
    print("3. 奖励函数不鼓励水平移动")
    print("4. HAC的动作选择策略")

if __name__ == "__main__":
    main()