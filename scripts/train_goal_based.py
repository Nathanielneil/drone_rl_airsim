#!/usr/bin/env python3
"""
基于目标点的训练脚本
实现点到点导航任务的强化学习训练
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
    print("无人机点到点导航训练")
    print("=" * 50)
    
    # 导入训练脚本
    from scripts.modern_train import main as train_main
    
    # 设置命令行参数模拟
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="sac")
    parser.add_argument("--config", default="configs/goal_based_training_config.yaml")
    parser.add_argument("--experiment-name", default="goal_navigation_test")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--log-level", default="INFO")
    
    # 解析参数
    args = parser.parse_args([
        "--algorithm", "sac",
        "--config", "configs/goal_based_training_config.yaml", 
        "--experiment-name", "goal_navigation_test",
        "--total-timesteps", "100000",
        "--log-level", "INFO"
    ])
    
    # 显示配置信息
    print(f"配置信息:")
    print(f"   任务类型: 点到点导航")
    print(f"   算法: {args.algorithm}")
    print(f"   配置文件: {args.config}")
    print(f"   实验名称: {args.experiment_name}")
    print(f"   训练步数: {args.total_timesteps:,}")
    print()
    
    print("目标导航特性:")
    print("   明确的目标点设置")
    print("   距离奖励和到达奖励")
    print("   多目标序列任务")
    print("   渐进式难度调整")
    print("   目标可视化支持")
    print()
    
    print("预期学习效果:")
    print("   学会飞向指定目标点")
    print("   优化飞行路径规划")
    print("   提高避障导航能力")
    print("   增强任务完成率")
    print()
    
    # 修改sys.argv来传递参数
    original_argv = sys.argv[:]
    sys.argv = [
        "train_goal_based.py",
        "--algorithm", "sac",
        "--config", "configs/goal_based_training_config.yaml",
        "--experiment-name", "goal_navigation_test", 
        "--total-timesteps", "100000",
        "--log-level", "INFO"
    ]
    
    try:
        print("开始训练...")
        print("-" * 50)
        
        # 调用主训练函数
        train_main()
        
        print()
        print("训练完成!")
        print("查看结果:")
        print(f"   TensorBoard: tensorboard --logdir data/experiments")
        print(f"   模型文件: models/sac/development/")
        print(f"   目标完成率和导航分析: scripts/analyze_experiments.py")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练失败: {e}")
        raise
    finally:
        # 恢复原始argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()