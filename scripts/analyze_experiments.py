#!/usr/bin/env python3
"""
实验数据分析工具
提供实验数据查看、比较和分析功能
"""

import sys
import argparse
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.data_manager import DataManager


class ExperimentAnalyzer:
    """实验数据分析器"""
    
    def __init__(self):
        self.data_manager = DataManager()
        
    def list_experiments(self, algorithm: str = None, status: str = None, limit: int = 20):
        """列出实验"""
        experiments = self.data_manager.list_experiments(
            algorithm=algorithm,
            status=status,
            limit=limit
        )
        
        print(f"\n{'实验ID':<40} {'状态':<10} {'算法':<8} {'最佳奖励':<12} {'训练时长':<12}")
        print("-" * 90)
        
        for exp in experiments:
            exp_id = exp.get("experiment_id", "")[:38]
            status = exp.get("status", "unknown")
            algorithm = exp.get("algorithm", {}).get("name", "unknown")
            best_reward = exp.get("results", {}).get("best_reward", 0.0)
            duration = exp.get("duration", "00:00:00")
            
            print(f"{exp_id:<40} {status:<10} {algorithm:<8} {best_reward:<12.2f} {duration:<12}")
    
    def show_experiment_details(self, experiment_id: str):
        """显示实验详情"""
        exp_path = self.data_manager.get_experiment_path(experiment_id)
        metadata_path = exp_path / "metadata.json"
        
        if not metadata_path.exists():
            print(f"实验不存在: {experiment_id}")
            return
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"\n实验详情: {experiment_id}")
        print("=" * 80)
        print(f"名称: {metadata.get('name', 'N/A')}")
        print(f"描述: {metadata.get('description', 'N/A')}")
        print(f"状态: {metadata.get('status', 'unknown')}")
        print(f"创建时间: {metadata.get('created_date', 'N/A')}")
        print(f"训练时长: {metadata.get('duration', 'N/A')}")
        print(f"算法: {metadata.get('algorithm', {}).get('name', 'N/A')}")
        print(f"环境: {metadata.get('environment', {}).get('name', 'N/A')}")
        print(f"标签: {', '.join(metadata.get('tags', []))}")
        
        # 超参数
        print("\n超参数:")
        for key, value in metadata.get('hyperparameters', {}).items():
            print(f"  {key}: {value}")
        
        # 结果
        results = metadata.get('results', {})
        if results:
            print("\n训练结果:")
            for key, value in results.items():
                print(f"  {key}: {value}")
        
        # 文件
        print(f"\n数据文件:")
        print(f"  TensorBoard: {exp_path / 'logs/tensorboard'}")
        print(f"  指标数据: {exp_path / 'metrics'}")
        print(f"  检查点: {exp_path / 'checkpoints'}")
    
    def plot_training_curves(self, experiment_ids: List[str], output_dir: str = "analysis"):
        """绘制训练曲线"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 收集所有实验数据
        all_episode_data = []
        all_loss_data = []
        
        for exp_id in experiment_ids:
            metrics_dir = self.data_manager.get_metrics_path(exp_id)
            
            # 加载episode数据
            episode_file = metrics_dir / "episode_rewards.csv"
            if episode_file.exists():
                df = pd.read_csv(episode_file)
                df['experiment_id'] = exp_id[:20]  # 截断ID用于显示
                all_episode_data.append(df)
            
            # 加载损失数据
            loss_file = metrics_dir / "loss_curves.csv"
            if loss_file.exists():
                df = pd.read_csv(loss_file)
                df['experiment_id'] = exp_id[:20]
                all_loss_data.append(df)
        
        # 绘制episode奖励曲线
        if all_episode_data:
            episode_df = pd.concat(all_episode_data, ignore_index=True)
            
            plt.figure(figsize=(12, 8))
            
            # 绘制原始奖励
            plt.subplot(2, 2, 1)
            for exp_id in episode_df['experiment_id'].unique():
                exp_data = episode_df[episode_df['experiment_id'] == exp_id]
                plt.plot(exp_data['episode'], exp_data['reward'], alpha=0.7, label=exp_id)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Episode Rewards')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 绘制滑动平均
            plt.subplot(2, 2, 2)
            for exp_id in episode_df['experiment_id'].unique():
                exp_data = episode_df[episode_df['experiment_id'] == exp_id].copy()
                exp_data['reward_smooth'] = exp_data['reward'].rolling(window=10, min_periods=1).mean()
                plt.plot(exp_data['episode'], exp_data['reward_smooth'], label=f"{exp_id} (smooth)")
            plt.xlabel('Episode')
            plt.ylabel('Smoothed Reward')
            plt.title('Smoothed Episode Rewards (10-episode MA)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 绘制episode长度
            plt.subplot(2, 2, 3)
            for exp_id in episode_df['experiment_id'].unique():
                exp_data = episode_df[episode_df['experiment_id'] == exp_id]
                plt.plot(exp_data['episode'], exp_data['length'], alpha=0.7, label=exp_id)
            plt.xlabel('Episode')
            plt.ylabel('Episode Length')
            plt.title('Episode Lengths')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 绘制FPS
            plt.subplot(2, 2, 4)
            for exp_id in episode_df['experiment_id'].unique():
                exp_data = episode_df[episode_df['experiment_id'] == exp_id]
                if 'fps' in exp_data.columns:
                    plt.plot(exp_data['episode'], exp_data['fps'], alpha=0.7, label=exp_id)
            plt.xlabel('Episode')
            plt.ylabel('FPS')
            plt.title('Training FPS')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'training_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"训练曲线已保存: {output_path / 'training_curves.png'}")
        
        # 绘制损失曲线
        if all_loss_data:
            loss_df = pd.concat(all_loss_data, ignore_index=True)
            
            # 获取所有损失类型
            loss_columns = [col for col in loss_df.columns if 'loss' in col.lower() or 'error' in col.lower()]
            
            if loss_columns:
                n_plots = len(loss_columns)
                n_cols = 2
                n_rows = (n_plots + 1) // 2
                
                plt.figure(figsize=(15, 5 * n_rows))
                
                for i, loss_col in enumerate(loss_columns):
                    plt.subplot(n_rows, n_cols, i + 1)
                    for exp_id in loss_df['experiment_id'].unique():
                        exp_data = loss_df[loss_df['experiment_id'] == exp_id]
                        if loss_col in exp_data.columns:
                            plt.plot(exp_data['step'], exp_data[loss_col], alpha=0.7, label=exp_id)
                    plt.xlabel('Training Step')
                    plt.ylabel(loss_col)
                    plt.title(f'{loss_col.replace("_", " ").title()}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_path / 'loss_curves.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"损失曲线已保存: {output_path / 'loss_curves.png'}")
    
    def compare_experiments(self, experiment_ids: List[str], output_file: str = "comparison_report.json"):
        """比较多个实验"""
        comparison = self.data_manager.compare_experiments(experiment_ids)
        
        # 保存详细对比
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        print(f"\n实验对比摘要 (共{len(experiment_ids)}个实验):")
        print("=" * 80)
        
        for exp in comparison['experiments']:
            exp_id = exp.get('experiment_id', '')[:40]
            best_reward = exp.get('results', {}).get('best_reward', 0.0)
            duration = exp.get('duration', '00:00:00')
            status = exp.get('status', 'unknown')
            
            print(f"{exp_id:<40} | {best_reward:>10.2f} | {duration:>10} | {status}")
        
        if 'summary' in comparison:
            summary = comparison['summary']
            print(f"\n最佳奖励: {summary.get('best_reward', 0.0):.2f}")
        
        print(f"\n详细对比报告已保存: {output_file}")
        
        return comparison
    
    def export_experiment_data(self, experiment_id: str, output_dir: str = "export"):
        """导出实验数据"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 导出指标数据
        metrics = self.data_manager.export_metrics(experiment_id)
        
        # 保存为JSON
        json_file = output_path / f"{experiment_id}_metrics.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 保存CSV文件
        for key, data in metrics.items():
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                csv_file = output_path / f"{experiment_id}_{key}.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"实验数据已导出到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="实验数据分析工具")
    
    parser.add_argument("command", choices=[
        "list", "show", "plot", "compare", "export"
    ], help="分析命令")
    
    parser.add_argument("--experiment-id", "-e", type=str, 
                       help="实验ID")
    parser.add_argument("--experiment-ids", "-es", nargs="+",
                       help="多个实验ID（用于比较）")
    parser.add_argument("--algorithm", "-a", type=str,
                       help="算法筛选")
    parser.add_argument("--status", "-s", type=str,
                       help="状态筛选")
    parser.add_argument("--limit", "-l", type=int, default=20,
                       help="结果数量限制")
    parser.add_argument("--output", "-o", type=str, default="analysis",
                       help="输出目录")
    
    args = parser.parse_args()
    
    analyzer = ExperimentAnalyzer()
    
    if args.command == "list":
        analyzer.list_experiments(
            algorithm=args.algorithm,
            status=args.status,
            limit=args.limit
        )
    
    elif args.command == "show":
        if not args.experiment_id:
            print("错误: 需要指定 --experiment-id")
            return
        analyzer.show_experiment_details(args.experiment_id)
    
    elif args.command == "plot":
        if not args.experiment_ids:
            print("错误: 需要指定 --experiment-ids")
            return
        analyzer.plot_training_curves(args.experiment_ids, args.output)
    
    elif args.command == "compare":
        if not args.experiment_ids or len(args.experiment_ids) < 2:
            print("错误: 需要指定至少2个 --experiment-ids")
            return
        analyzer.compare_experiments(
            args.experiment_ids,
            f"{args.output}/comparison_report.json"
        )
    
    elif args.command == "export":
        if not args.experiment_id:
            print("错误: 需要指定 --experiment-id")
            return
        analyzer.export_experiment_data(args.experiment_id, args.output)


if __name__ == "__main__":
    main()