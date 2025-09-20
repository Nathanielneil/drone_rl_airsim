#!/usr/bin/env python3
"""
快速训练测试脚本
验证现代化训练环境是否正常工作
"""

import sys
import time
import subprocess
from pathlib import Path

def test_environment_imports():
    """测试所有必要的环境导入"""
    print('🔍 测试环境导入...')
    print('=' * 50)
    
    # 测试基础包
    try:
        import torch
        print(f'✅ PyTorch: {torch.__version__}')
        print(f'✅ CUDA可用: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'✅ GPU: {torch.cuda.get_device_name()}')
            print(f'✅ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
    except Exception as e:
        print(f'❌ PyTorch导入失败: {e}')
        return False
    
    try:
        import gymnasium
        print(f'✅ Gymnasium: {gymnasium.__version__}')
    except Exception as e:
        print(f'❌ Gymnasium导入失败: {e}')
        return False
    
    try:
        import stable_baselines3
        print(f'✅ Stable-Baselines3: {stable_baselines3.__version__}')
    except Exception as e:
        print(f'❌ Stable-Baselines3导入失败: {e}')
        return False
    
    try:
        import omegaconf
        print(f'✅ OmegaConf: {omegaconf.__version__}')
    except Exception as e:
        print(f'❌ OmegaConf导入失败: {e}')
        return False
    
    try:
        import airsim
        print('✅ AirSim: 已安装')
    except Exception as e:
        print(f'❌ AirSim导入失败: {e}')
        return False
    
    return True

def test_project_components():
    """测试项目组件"""
    print('')
    print('🔍 测试项目组件...')
    print('=' * 50)
    
    try:
        from src.core.config_manager import ConfigManager
        config_manager = ConfigManager()
        print('✅ 配置管理系统: 正常')
        
        # 测试配置创建
        config = config_manager.create_modern_config()
        print('✅ 现代化配置创建: 成功')
        
    except Exception as e:
        print(f'❌ 配置管理系统: {e}')
        return False
    
    try:
        from src.utils.performance.gpu_manager import GPUMemoryManager
        gpu_manager = GPUMemoryManager()
        gpu_info = gpu_manager.get_memory_info()
        print(f'✅ GPU性能管理: 正常 (显存: {gpu_info.total:.1f}GB)')
        
    except Exception as e:
        print(f'❌ GPU性能管理: {e}')
        return False
    
    try:
        from src.environments.airsim_env.modern_airsim_env import ModernAirSimEnv
        print('✅ 现代化AirSim环境: 导入成功')
        
    except Exception as e:
        print(f'❌ 现代化AirSim环境: {e}')
        return False
    
    try:
        from src.algorithms.actor_critic.sac.modern_sac import ModernSAC
        print('✅ 现代化SAC算法: 导入成功')
        
    except Exception as e:
        print(f'❌ 现代化SAC算法: {e}')
        return False
    
    return True

def test_airsim_connection():
    """快速测试AirSim连接"""
    print('')
    print('🔍 测试AirSim连接...')
    print('=' * 50)
    
    try:
        import airsim
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print('✅ AirSim连接: 成功')
        return True
    except Exception as e:
        print(f'❌ AirSim连接: {e}')
        print('💡 请先运行: python test_airsim_connection.py')
        return False

def run_quick_training():
    """运行快速训练测试"""
    print('')
    print('🚀 运行快速训练测试...')
    print('=' * 50)
    
    try:
        # 检查训练脚本是否存在
        script_path = Path('scripts/modern_train.py')
        if not script_path.exists():
            print(f'❌ 训练脚本不存在: {script_path}')
            return False
        
        print('📋 训练参数:')
        print('   - 算法: SAC')
        print('   - 总步数: 20')
        print('   - 批次大小: 32') 
        print('   - 实验名称: quick_test')
        print('')
        
        # 运行训练
        print('🎯 开始训练...')
        cmd = [
            sys.executable, 'scripts/modern_train.py',
            '--algorithm', 'sac',
            '--total-timesteps', '20',
            '--batch-size', '32',
            '--experiment-name', 'quick_test',
            '--log-level', 'INFO'
        ]
        
        print(f'命令: {" ".join(cmd)}')
        print('')
        
        # 设置超时时间
        start_time = time.time()
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              timeout=300,  # 5分钟超时
                              encoding='utf-8')
        
        elapsed_time = time.time() - start_time
        
        print(f'⏱️ 训练耗时: {elapsed_time:.1f}秒')
        print('')
        
        # 显示输出
        if result.stdout:
            print('📊 训练输出:')
            print('-' * 30)
            print(result.stdout)
            print('-' * 30)
        
        if result.stderr:
            print('⚠️ 错误/警告信息:')
            print('-' * 30)
            print(result.stderr)
            print('-' * 30)
        
        if result.returncode == 0:
            print('✅ 快速训练测试成功!')
            print('')
            print('🎉 环境完全就绪!')
            print('💡 现在可以运行正式训练:')
            print('   python scripts/modern_train.py --total-timesteps 10000')
            return True
        else:
            print(f'❌ 训练失败，返回码: {result.returncode}')
            return False
            
    except subprocess.TimeoutExpired:
        print('⚠️ 训练超时（超过5分钟）')
        print('这可能是正常的，环境加载可能需要更长时间')
        return False
        
    except Exception as e:
        print(f'❌ 训练测试异常: {e}')
        return False

def main():
    """主测试流程"""
    print('🧪 现代化无人机RL环境 - 快速训练测试')
    print('适用于: Windows 10 + AirSim 1.8.1 + CUDA 12.1')
    print('')
    
    # 步骤1: 测试环境导入
    if not test_environment_imports():
        print('')
        print('❌ 环境导入测试失败')
        print('💡 请检查conda环境和包安装')
        return False
    
    # 步骤2: 测试项目组件
    if not test_project_components():
        print('')
        print('❌ 项目组件测试失败')
        print('💡 请检查项目文件完整性')
        return False
    
    # 步骤3: 测试AirSim连接
    if not test_airsim_connection():
        print('')
        print('❌ AirSim连接测试失败')
        print('💡 请先确保AirSim环境正在运行')
        return False
    
    # 步骤4: 运行快速训练
    if not run_quick_training():
        print('')
        print('❌ 快速训练测试失败')
        return False
    
    print('')
    print('🎊 所有测试通过!')
    print('=' * 50)
    print('🚀 建议的下一步操作:')
    print('')
    print('1. 运行完整训练:')
    print('   python scripts/modern_train.py --total-timesteps 10000')
    print('')
    print('2. 启动训练监控:')
    print('   tensorboard --logdir experiments/logs')
    print('')
    print('3. 查看训练文档:')
    print('   - MODERN_SETUP.md')
    print('   - WINDOWS_DEPLOYMENT.md')
    print('')
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)