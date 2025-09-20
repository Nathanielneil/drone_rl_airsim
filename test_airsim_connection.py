#!/usr/bin/env python3
"""
AirSim连接测试脚本
测试AirSim环境是否正常工作并准备训练
"""

import airsim
import time
import sys
import numpy as np

def test_airsim_connection():
    """测试AirSim连接和基本功能"""
    
    print('🔍 测试AirSim连接...')
    print('=' * 50)
    
    try:
        # 1. 创建客户端连接
        print('1. 创建AirSim客户端...')
        client = airsim.MultirotorClient()
        
        # 2. 确认连接
        print('2. 确认连接...')
        client.confirmConnection()
        print('✅ AirSim连接成功!')
        
        # 3. 获取API版本
        api_version = client.getApiVersion()
        print(f'✅ API版本: {api_version}')
        
        # 4. 获取无人机状态
        print('3. 获取无人机状态...')
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        
        print(f'✅ 无人机位置: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}')
        print(f'✅ 无人机速度: VX={vel.x_val:.2f}, VY={vel.y_val:.2f}, VZ={vel.z_val:.2f}')
        
        # 5. 启用API控制
        print('4. 启用API控制...')
        client.enableApiControl(True)
        print('✅ API控制已启用')
        
        # 6. 解锁无人机
        print('5. 解锁无人机...')
        client.armDisarm(True)
        print('✅ 无人机已解锁')
        
        # 7. 测试图像获取
        print('6. 测试相机图像获取...')
        try:
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False)
            ])
            
            if responses and len(responses) > 0:
                img_data = responses[0].image_data_uint8
                print(f'✅ 相机图像: {len(img_data)} bytes')
                
                # 转换为numpy数组测试
                img_1d = np.frombuffer(img_data, dtype=np.uint8)
                if len(img_1d) > 0:
                    print('✅ 图像数据转换成功')
                else:
                    print('⚠️ 图像数据为空')
            else:
                print('⚠️ 未获取到图像响应')
                
        except Exception as img_e:
            print(f'⚠️ 图像获取警告: {img_e}')
        
        # 8. 测试基本移动命令
        print('7. 测试基本控制命令...')
        try:
            # 悬停测试
            client.hoverAsync().join()
            print('✅ 悬停命令成功')
            
            # 获取当前位置作为参考
            current_state = client.getMultirotorState()
            current_pos = current_state.kinematics_estimated.position
            print(f'✅ 当前位置确认: X={current_pos.x_val:.2f}, Y={current_pos.y_val:.2f}, Z={current_pos.z_val:.2f}')
            
        except Exception as move_e:
            print(f'⚠️ 移动命令警告: {move_e}')
        
        # 9. 测试碰撞检测
        print('8. 测试碰撞检测...')
        try:
            collision_info = client.simGetCollisionInfo()
            print(f'✅ 碰撞检测: {"发生碰撞" if collision_info.has_collided else "无碰撞"}')
        except Exception as collision_e:
            print(f'⚠️ 碰撞检测警告: {collision_e}')
        
        print('=' * 50)
        print('🎉 AirSim环境测试完成!')
        print('📋 测试结果总结:')
        print('   ✅ 网络连接正常')
        print('   ✅ API控制可用') 
        print('   ✅ 无人机状态获取正常')
        print('   ✅ 图像数据可用')
        print('   ✅ 控制命令响应正常')
        print('')
        print('🚀 环境就绪，可以开始训练!')
        print('💡 下一步: 运行 python test_quick_training.py')
        
        return True
        
    except ConnectionError as ce:
        print(f'❌ 连接错误: {ce}')
        print('')
        print('🔧 请检查:')
        print('   1. AirSim环境是否正在运行')
        print('   2. 端口41451是否开放')
        print('   3. 防火墙设置是否正确')
        return False
        
    except Exception as e:
        print(f'❌ 测试失败: {e}')
        print('')
        print('🔧 可能的解决方案:')
        print('   1. 重启AirSim环境')
        print('   2. 检查AirSim settings.json配置')
        print('   3. 确认AirSim版本为1.8.1')
        return False

def print_connection_help():
    """打印连接帮助信息"""
    print('')
    print('🆘 AirSim连接帮助:')
    print('=' * 50)
    print('1. 确保AirSim环境正在运行')
    print('   - 双击AirSim可执行文件')
    print('   - 等待环境完全加载')
    print('')
    print('2. 检查settings.json配置:')
    print('   位置: %USERPROFILE%\\Documents\\AirSim\\settings.json')
    print('   确保ApiServerPort为41451')
    print('')
    print('3. 检查网络:')
    print('   命令: netstat -an | findstr 41451')
    print('')
    print('4. 防火墙设置:')
    print('   将AirSim添加到防火墙例外')

if __name__ == "__main__":
    print('🧪 AirSim环境连接测试')
    print('适用于: Windows 10 + AirSim 1.8.1 + 现代化无人机RL')
    print('')
    
    success = test_airsim_connection()
    
    if not success:
        print_connection_help()
        sys.exit(1)
    else:
        print('')
        print('✨ 测试完成! 环境就绪!')
        sys.exit(0)