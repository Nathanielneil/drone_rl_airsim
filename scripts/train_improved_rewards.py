#!/usr/bin/env python3
"""
使用改进奖励函数的训练脚本
解决负奖励过大和碰撞过多的问题
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    """主函数"""
    print("🚁 启动改进奖励函数的无人机强化学习训练")
    print("=" * 60)
    
    # 导入训练脚本
    from scripts.modern_train import main as train_main
    
    # 设置命令行参数模拟
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="sac")
    parser.add_argument("--config", default="configs/improved_training_config.yaml")
    parser.add_argument("--experiment-name", default="improved_rewards_test")
    parser.add_argument("--total-timesteps", type=int, default=50000)  # 较短的测试训练
    parser.add_argument("--log-level", default="INFO")
    
    # 解析参数
    args = parser.parse_args([
        "--algorithm", "sac",
        "--config", "configs/improved_training_config.yaml", 
        "--experiment-name", "improved_rewards_test",
        "--total-timesteps", "50000",
        "--log-level", "INFO"
    ])
    
    # 显示配置信息
    print(f"🔧 配置信息:")
    print(f"   算法: {args.algorithm}")
    print(f"   配置文件: {args.config}")
    print(f"   实验名称: {args.experiment_name}")
    print(f"   训练步数: {args.total_timesteps:,}")
    print()
    
    print("📋 改进的奖励函数特性:")
    print("   ✅ 碰撞惩罚从 -100.0 降低到 -8.0")
    print("   ✅ 增加安全距离奖励和避障引导")
    print("   ✅ 添加课程学习，从简单到困难")
    print("   ✅ 平衡的多组件奖励系统")
    print("   ✅ 自动难度调整和性能监控")
    print()
    
    print("🎯 预期改进效果:")
    print("   📈 减少过度保守行为")
    print("   📉 降低训练初期的挫败感")
    print("   🎓 渐进式学习避障技能") 
    print("   📊 更稳定的训练收敛")
    print()
    
    # 修改sys.argv来传递参数
    original_argv = sys.argv[:]
    sys.argv = [
        "train_improved_rewards.py",
        "--algorithm", "sac",
        "--config", "configs/improved_training_config.yaml",
        "--experiment-name", "improved_rewards_test", 
        "--total-timesteps", "50000",
        "--log-level", "INFO"
    ]
    
    try:
        print("🚀 开始训练...")
        print("-" * 60)
        
        # 调用主训练函数
        train_main()
        
        print()
        print("✅ 训练完成!")
        print("📊 查看结果:")
        print(f"   TensorBoard: tensorboard --logdir data/experiments")
        print(f"   模型文件: models/sac/development/")
        print(f"   训练数据: data/experiments/")
        
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        raise
    finally:
        # 恢复原始argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()