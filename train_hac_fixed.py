#!/usr/bin/env python
"""
修复后的HAC训练脚本 - 移除不支持的参数
"""

import sys
import os
sys.path.append('hierarchical_rl')

from hierarchical_rl.train_hierarchical import train_hierarchical_agent

def main():
    # 简化的HAC训练配置 - 只使用支持的参数
    config = {
        'algorithm': 'hac',
        'env_type': 'hierarchical',
        'num_episodes': 100,  # 减少训练轮数进行测试
        'max_episode_steps': 100,
        'eval_freq': 25,
        'save_freq': 50,
        'log_freq': 5,
        'seed': 42,
        'device': 'cuda',
        'save_dir': 'results/hierarchical/hac_fixed',
        
        # HAC-specific parameters - 只用支持的参数
        'num_levels': 2,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'gamma': 0.95,
        'subgoal_test_perc': 0.5,
        'her_ratio': 0.6,
        'buffer_size': 3000,  # 适中的buffer大小
        
        # 探索参数 - 使用默认支持的参数名
        'atomic_noise': 0.5,   # 增加底层噪声促进探索
        'subgoal_noise': 0.2,  # 增加子目标噪声
        'max_actions': 15,
        
        # 基础参数
        'goal_dim': 3,
        'subgoal_bounds': [-8.0, 8.0]
    }
    
    print("启动修复版HAC训练...")
    print("关键修复：")
    print("- 动作空间扩大到 [-2.0, 2.0]")
    print("- 移除不支持的curriculum_learning参数") 
    print("- 增加探索噪声促进水平移动")
    print("- 优化内存使用")
    print()
    print(f"配置参数: {config}")
    
    try:
        # 训练智能体
        agent = train_hierarchical_agent(**config)
        
        print("HAC修复版训练完成!")
        print(f"模型保存至: {config['save_dir']}")
        
        # 训练完成后的建议
        print("\n观察建议：")
        print("1. 查看训练过程中的动作输出")
        print("2. 观察无人机是否开始水平移动")
        print("3. 检查奖励变化趋势")
        print("4. 如果移动效果好，可以增加训练轮数")
        
        return agent
        
    except Exception as e:
        print(f"训练失败: {e}")
        print("\n可能的解决方案：")
        print("1. 检查内存使用情况")
        print("2. 进一步减小buffer_size到1000")
        print("3. 确认AirSim连接正常")
        
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()