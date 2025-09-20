#!/usr/bin/env python3
"""
简化的AirSim连接测试
兼容多个AirSim版本
"""

import sys

def test_airsim_simple():
    """简化的AirSim连接测试"""
    
    print('🔍 简化AirSim连接测试...')
    print('=' * 40)
    
    try:
        print('1. 导入AirSim...')
        import airsim
        print('✅ AirSim库导入成功')
        
        print('2. 创建客户端连接...')
        # 尝试不同的连接方式
        try:
            # 方式1: 最新的显式参数方式
            client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
            print('✅ 使用显式参数创建客户端')
        except TypeError:
            try:
                # 方式2: 较旧版本的方式
                client = airsim.MultirotorClient()
                print('✅ 使用默认参数创建客户端')
            except Exception as e:
                print(f'❌ 客户端创建失败: {e}')
                return False
        
        print('3. 测试连接...')
        client.confirmConnection()
        print('✅ AirSim连接成功!')
        
        print('4. 获取基本信息...')
        try:
            # 获取API版本
            api_version = client.getApiVersion()
            print(f'✅ API版本: {api_version}')
        except:
            print('⚠️ API版本获取失败，但连接正常')
        
        try:
            # 获取无人机状态
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            print(f'✅ 无人机位置: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}')
        except Exception as e:
            print(f'⚠️ 状态获取失败: {e}')
        
        print('')
        print('🎉 基本连接测试成功!')
        print('💡 可以继续运行完整的训练流程')
        return True
        
    except ImportError as ie:
        print(f'❌ AirSim导入失败: {ie}')
        print('')
        print('🔧 解决方案:')
        print('   pip install airsim')
        return False
        
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
        print(f'错误类型: {type(e).__name__}')
        print('')
        print('🔧 可能的解决方案:')
        print('   1. 重启AirSim环境')
        print('   2. 检查AirSim版本兼容性')
        print('   3. 重装airsim包: pip install --upgrade airsim')
        return False

if __name__ == "__main__":
    print('🧪 简化AirSim连接测试')
    print('适用于: 多版本AirSim兼容性测试')
    print('')
    
    success = test_airsim_simple()
    
    if success:
        print('')
        print('✨ 测试完成! 可以开始训练!')
        print('💡 下一步: python scripts/modern_train.py --total-timesteps 1000')
        sys.exit(0)
    else:
        print('')
        print('❌ 连接测试失败，请按提示解决问题')
        sys.exit(1)