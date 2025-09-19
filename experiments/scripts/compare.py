#!/usr/bin/env python3
"""
Compare multiple trained models
"""
import argparse
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.core.evaluator import Evaluator
from src.core.config_manager import ConfigManager
from src.utils.common.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Compare multiple trained RL agents")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                       help="Paths to trained models")
    parser.add_argument("--algorithms", type=str, nargs="+", required=True,
                       help="Algorithms corresponding to each model")
    parser.add_argument("--names", type=str, nargs="+", default=None,
                       help="Custom names for each model")
    parser.add_argument("--env-config", type=str, default="config/environments/airsim.yaml",
                       help="Environment config file")
    parser.add_argument("--num-episodes", type=int, default=20,
                       help="Number of evaluation episodes per model")
    parser.add_argument("--output-dir", type=str, default="experiments/results/comparison",
                       help="Output directory for comparison results")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.models) != len(args.algorithms):
        raise ValueError("Number of models must match number of algorithms")
    
    if args.names and len(args.names) != len(args.models):
        raise ValueError("Number of names must match number of models")
    
    # Setup logging
    logger = setup_logger("comparison", level=args.log_level)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config_manager = ConfigManager()
    env_config = config_manager.load_config(args.env_config)
    
    # Generate model names
    if not args.names:
        args.names = [f"{algo}_{i}" for i, algo in enumerate(args.algorithms)]
    
    logger.info(f"Comparing {len(args.models)} models:")
    for name, model, algo in zip(args.names, args.models, args.algorithms):
        logger.info(f"  {name}: {algo} - {model}")
    
    # Results storage
    all_results = {}
    
    # Evaluate each model
    for name, model_path, algorithm in zip(args.names, args.models, args.algorithms):
        logger.info(f"Evaluating {name}...")
        
        # Load algorithm config
        default_config = config_manager.load_config("config/default.yaml")
        algo_config = config_manager.load_config(f"config/algorithms/{algorithm}.yaml")
        config = config_manager.merge_configs(default_config, algo_config, env_config)
        
        config["algorithm"] = algorithm
        config["model_path"] = model_path
        config["num_episodes"] = args.num_episodes
        
        # Initialize evaluator
        evaluator = Evaluator(config, logger)
        evaluator.load_model(model_path)
        
        # Run evaluation
        try:
            results = evaluator.evaluate()
            all_results[name] = results
            logger.info(f"{name} - Average Return: {results['mean_return']:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {name}: {str(e)}")
            continue
    
    # Generate comparison report
    logger.info("Generating comparison report...")
    
    # Create comparison plots
    create_comparison_plots(all_results, args.output_dir)
    
    # Create summary table
    create_summary_table(all_results, args.output_dir)
    
    # Save raw results
    results_file = os.path.join(args.output_dir, "comparison_results.yaml")
    with open(results_file, 'w') as f:
        yaml.dump(all_results, f, default_flow_style=False)
    
    logger.info(f"Comparison results saved to: {args.output_dir}")


def create_comparison_plots(results, output_dir):
    """Create comparison plots"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Bar plot of average returns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    names = list(results.keys())
    mean_returns = [results[name]['mean_return'] for name in names]
    std_returns = [results[name]['std_return'] for name in names]
    
    ax1.bar(names, mean_returns, yerr=std_returns, capsize=5)
    ax1.set_title('Average Episode Return')
    ax1.set_ylabel('Return')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Bar plot of success rates
    success_rates = [results[name]['success_rate'] * 100 for name in names]
    ax2.bar(names, success_rates)
    ax2.set_title('Success Rate')
    ax2.set_ylabel('Success Rate (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Bar plot of episode lengths
    mean_lengths = [results[name]['mean_episode_length'] for name in names]
    std_lengths = [results[name]['std_episode_length'] for name in names]
    
    ax3.bar(names, mean_lengths, yerr=std_lengths, capsize=5)
    ax3.set_title('Average Episode Length')
    ax3.set_ylabel('Steps')
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plots for detailed distribution comparison
    if all('episode_returns' in results[name] for name in names):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = [results[name]['episode_returns'] for name in names]
        ax.boxplot(data, labels=names)
        ax.set_title('Episode Return Distribution')
        ax.set_ylabel('Return')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'return_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()


def create_summary_table(results, output_dir):
    """Create summary table"""
    
    # Create markdown table
    table_lines = [
        "# Model Comparison Summary\n",
        "| Model | Algorithm | Avg Return | Std Return | Success Rate | Avg Length | Std Length |",
        "|-------|-----------|------------|------------|--------------|------------|------------|"
    ]
    
    for name in results.keys():
        r = results[name]
        line = f"| {name} | {r.get('algorithm', 'Unknown')} | {r['mean_return']:.2f} | {r['std_return']:.2f} | {r['success_rate']:.2%} | {r['mean_episode_length']:.1f} | {r['std_episode_length']:.1f} |"
        table_lines.append(line)
    
    # Save table
    table_file = os.path.join(output_dir, "comparison_summary.md")
    with open(table_file, 'w') as f:
        f.write('\n'.join(table_lines))


if __name__ == "__main__":
    main()