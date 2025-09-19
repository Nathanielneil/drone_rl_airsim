#!/usr/bin/env python3
"""
Unified training script for all RL algorithms
"""
import argparse
import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.core.trainer import Trainer
from src.core.config_manager import ConfigManager
from src.utils.common.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for drone navigation")
    parser.add_argument("--algorithm", type=str, required=True, 
                       choices=["sac", "ppo", "ddpg", "td3", "dqn", "rainbow"],
                       help="RL algorithm to use")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (optional, will use default)")
    parser.add_argument("--env-config", type=str, default="config/environments/airsim.yaml",
                       help="Environment config file")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Name for this experiment")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation after training")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("training", level=args.log_level)
    
    # Load configuration
    config_manager = ConfigManager()
    
    # Load default config
    default_config = config_manager.load_config("config/default.yaml")
    
    # Load algorithm config
    if args.config:
        algo_config = config_manager.load_config(args.config)
    else:
        algo_config = config_manager.load_config(f"config/algorithms/{args.algorithm}.yaml")
    
    # Load environment config
    env_config = config_manager.load_config(args.env_config)
    
    # Merge configurations
    config = config_manager.merge_configs(default_config, algo_config, env_config)
    
    # Override with command line arguments
    config["algorithm"] = args.algorithm
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    else:
        config["experiment_name"] = f"{args.algorithm}_drone_navigation"
    
    logger.info(f"Starting training with algorithm: {args.algorithm}")
    logger.info(f"Experiment name: {config['experiment_name']}")
    
    # Initialize trainer
    trainer = Trainer(config, logger)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from: {args.resume}")
    
    # Start training
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Run evaluation if requested
        if args.evaluate:
            logger.info("Starting evaluation...")
            trainer.evaluate()
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    

if __name__ == "__main__":
    main()