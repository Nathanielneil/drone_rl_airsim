#!/usr/bin/env python3
"""
Evaluation script for trained RL agents
"""
import argparse
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.core.evaluator import Evaluator
from src.core.config_manager import ConfigManager
from src.utils.common.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--algorithm", type=str, required=True,
                       choices=["sac", "ppo", "ddpg", "td3", "dqn", "rainbow"],
                       help="RL algorithm used")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    parser.add_argument("--env-config", type=str, default="config/environments/airsim.yaml",
                       help="Environment config file")
    parser.add_argument("--num-episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                       help="Render evaluation episodes")
    parser.add_argument("--save-video", action="store_true",
                       help="Save evaluation videos")
    parser.add_argument("--output-dir", type=str, default="experiments/results",
                       help="Output directory for results")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("evaluation", level=args.log_level)
    
    # Load configuration
    config_manager = ConfigManager()
    
    # Load configs
    if args.config:
        config = config_manager.load_config(args.config)
    else:
        default_config = config_manager.load_config("config/default.yaml")
        algo_config = config_manager.load_config(f"config/algorithms/{args.algorithm}.yaml")
        env_config = config_manager.load_config(args.env_config)
        config = config_manager.merge_configs(default_config, algo_config, env_config)
    
    # Override with command line arguments
    config["algorithm"] = args.algorithm
    config["model_path"] = args.model_path
    config["num_episodes"] = args.num_episodes
    config["render"] = args.render
    config["save_video"] = args.save_video
    config["output_dir"] = args.output_dir
    
    logger.info(f"Evaluating {args.algorithm} model: {args.model_path}")
    logger.info(f"Number of episodes: {args.num_episodes}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = Evaluator(config, logger)
    
    # Load model
    evaluator.load_model(args.model_path)
    
    # Run evaluation
    try:
        results = evaluator.evaluate()
        
        # Print results
        logger.info("Evaluation Results:")
        logger.info(f"Average Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
        logger.info(f"Average Episode Length: {results['mean_episode_length']:.2f} ± {results['std_episode_length']:.2f}")
        logger.info(f"Success Rate: {results['success_rate']:.2%}")
        
        # Save results
        results_file = os.path.join(args.output_dir, f"evaluation_results_{args.algorithm}.yaml")
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Generate plots
        evaluator.plot_results(args.output_dir)
        logger.info(f"Plots saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()