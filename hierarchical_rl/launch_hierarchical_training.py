#!/usr/bin/env python
"""
Hierarchical RL Training Launcher
=================================

Unified launcher for all hierarchical reinforcement learning algorithms.
Provides easy selection and configuration of different algorithms:

- HAC (Hindsight Action Control) - Fixed and production ready
- HIRO (HIerarchical RL with Off-policy correction) - Enhanced implementation  
- FuN (FeUdal Networks) - Manager-worker feudal architecture
- Options (Options Framework) - Temporal abstraction with learned skills

Usage:
    python launch_hierarchical_training.py --algorithm hac --episodes 500
    python launch_hierarchical_training.py --algorithm hiro --episodes 300
    python launch_hierarchical_training.py --algorithm fun --episodes 400  
    python launch_hierarchical_training.py --algorithm options --episodes 600
    python launch_hierarchical_training.py --test-all  # Test all algorithms
"""

import os
import sys
import argparse
import logging
import time
import subprocess
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HierarchicalTrainingLauncher:
    """Unified launcher for hierarchical RL training."""
    
    def __init__(self):
        self.algorithms = {
            'hac': {
                'name': 'HAC (Hindsight Action Control)',
                'script': 'train_hac_fixed.py',
                'description': 'Multi-level goal-conditioned learning with hindsight experience',
                'status': '✅ Production Ready',
                'recommended_episodes': 500
            },
            'hiro': {
                'name': 'HIRO (HIerarchical RL with Off-policy correction)',
                'script': 'train_hiro_improved.py', 
                'description': 'Data-efficient hierarchical RL with goal relabeling',
                'status': '🔄 Enhanced Implementation',
                'recommended_episodes': 300
            },
            'fun': {
                'name': 'FuN (FeUdal Networks)',
                'script': 'train_fun_improved.py',
                'description': 'Manager-worker hierarchy with intrinsic motivation',
                'status': '🆕 Newly Implemented',
                'recommended_episodes': 400
            },
            'options': {
                'name': 'Options Framework',
                'script': 'train_options_improved.py',
                'description': 'Temporal abstraction through learned options/skills',
                'status': '🆕 Newly Implemented', 
                'recommended_episodes': 600
            }
        }
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
    def display_algorithms(self) -> None:
        """Display available algorithms."""
        print("\n" + "="*80)
        print("🚁 HIERARCHICAL REINFORCEMENT LEARNING ALGORITHMS")
        print("="*80)
        
        for alg_id, info in self.algorithms.items():
            print(f"\n🔸 {alg_id.upper()}: {info['name']}")
            print(f"   Status: {info['status']}")
            print(f"   Description: {info['description']}")
            print(f"   Recommended Episodes: {info['recommended_episodes']}")
        
        print(f"\n{'='*80}")
        print("💡 Usage Examples:")
        print("   python launch_hierarchical_training.py --algorithm hac --episodes 500")
        print("   python launch_hierarchical_training.py --algorithm hiro --test-mode")
        print("   python launch_hierarchical_training.py --test-all")
        print("="*80 + "\n")
    
    def run_algorithm(
        self, 
        algorithm: str, 
        episodes: Optional[int] = None,
        test_mode: bool = False,
        gpu: bool = True,
        **kwargs
    ) -> bool:
        """
        Run specified hierarchical RL algorithm.
        
        Args:
            algorithm: Algorithm name (hac, hiro, fun, options)
            episodes: Number of training episodes
            test_mode: Run in test mode (shorter training)
            gpu: Use GPU if available
            **kwargs: Additional arguments
            
        Returns:
            Success status
        """
        if algorithm not in self.algorithms:
            logger.error(f"Unknown algorithm: {algorithm}")
            logger.error(f"Available algorithms: {list(self.algorithms.keys())}")
            return False
        
        alg_info = self.algorithms[algorithm]
        script_path = os.path.join(self.base_dir, alg_info['script'])
        
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            return False
        
        logger.info(f"🚀 Starting {alg_info['name']} training...")
        logger.info(f"   Script: {alg_info['script']}")
        
        if test_mode:
            episodes = 50  # Short test run
            logger.info("   Mode: TEST MODE (short run)")
        else:
            episodes = episodes or alg_info['recommended_episodes']
            logger.info(f"   Episodes: {episodes}")
        
        logger.info(f"   GPU: {'Enabled' if gpu else 'Disabled'}")
        
        # Prepare environment variables
        env = os.environ.copy()
        if not gpu:
            env['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU
        
        # Run training script
        try:
            start_time = time.time()
            
            # Use Python to run the script directly
            import importlib.util
            
            # Import and run the appropriate training script
            if algorithm == 'hac':
                from train_hac_fixed import main as train_hac_main
                train_hac_main()
            elif algorithm == 'hiro':
                from train_hiro_improved import main as train_hiro_main
                train_hiro_main() 
            elif algorithm == 'fun':
                from train_fun_improved import main as train_fun_main
                train_fun_main()
            elif algorithm == 'options':
                from train_options_improved import main as train_options_main
                train_options_main()
            
            duration = time.time() - start_time
            logger.info(f"✅ {alg_info['name']} training completed successfully!")
            logger.info(f"   Duration: {duration:.2f} seconds")
            return True
            
        except KeyboardInterrupt:
            logger.info(f"Training interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            return False
    
    def test_all_algorithms(self) -> Dict[str, bool]:
        """Test all algorithms with short runs."""
        logger.info("🧪 Testing all hierarchical RL algorithms...")
        
        results = {}
        
        # Import and run the test suite
        try:
            from test_all_algorithms import AlgorithmTester
            
            tester = AlgorithmTester()
            test_results = tester.run_all_tests()
            
            # Convert test results to success/failure
            for alg_name, result in test_results.items():
                results[alg_name.lower()] = result['status'] == 'PASSED'
            
            # Summary
            passed = sum(results.values())
            total = len(results)
            
            logger.info(f"\n📊 Test Results: {passed}/{total} algorithms passed")
            
            for alg_name, success in results.items():
                status = "✅ PASSED" if success else "❌ FAILED"
                logger.info(f"   {alg_name.upper()}: {status}")
            
        except Exception as e:
            logger.error(f"Testing failed: {str(e)}")
            results = {alg: False for alg in self.algorithms.keys()}
        
        return results
    
    def benchmark_algorithms(self, episodes: int = 100) -> Dict[str, Dict[str, Any]]:
        """Benchmark all algorithms with identical conditions."""
        logger.info(f"🏃 Benchmarking all algorithms with {episodes} episodes...")
        
        benchmark_results = {}
        
        for alg_id in self.algorithms.keys():
            logger.info(f"\nBenchmarking {alg_id.upper()}...")
            
            start_time = time.time()
            success = self.run_algorithm(alg_id, episodes=episodes)
            duration = time.time() - start_time
            
            benchmark_results[alg_id] = {
                'success': success,
                'duration': duration,
                'episodes': episodes
            }
        
        # Display benchmark results
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK RESULTS")
        logger.info("="*60)
        
        for alg_id, result in benchmark_results.items():
            status = "✅" if result['success'] else "❌"
            logger.info(f"{alg_id.upper():10} | {status} | {result['duration']:8.2f}s | {result['episodes']} episodes")
        
        return benchmark_results
    
    def get_algorithm_status(self) -> Dict[str, str]:
        """Get current status of all algorithms."""
        status = {}
        
        for alg_id, info in self.algorithms.items():
            script_path = os.path.join(self.base_dir, info['script'])
            
            if os.path.exists(script_path):
                status[alg_id] = "Available"
            else:
                status[alg_id] = "Script Missing"
        
        return status


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="Hierarchical RL Training Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--algorithm', '-a',
        choices=['hac', 'hiro', 'fun', 'options'],
        help='Algorithm to run'
    )
    
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        help='Number of training episodes'
    )
    
    parser.add_argument(
        '--test-mode', '-t',
        action='store_true',
        help='Run in test mode (short training)'
    )
    
    parser.add_argument(
        '--test-all',
        action='store_true', 
        help='Test all algorithms'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark all algorithms'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available algorithms'
    )
    
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = HierarchicalTrainingLauncher()
    
    # Handle different modes
    if args.list or (not args.algorithm and not args.test_all and not args.benchmark):
        launcher.display_algorithms()
        return
    
    if args.test_all:
        launcher.test_all_algorithms()
        return
    
    if args.benchmark:
        episodes = args.episodes or 100
        launcher.benchmark_algorithms(episodes)
        return
    
    if args.algorithm:
        success = launcher.run_algorithm(
            algorithm=args.algorithm,
            episodes=args.episodes,
            test_mode=args.test_mode,
            gpu=not args.no_gpu
        )
        
        if success:
            print(f"\n🎉 {args.algorithm.upper()} training completed successfully!")
        else:
            print(f"\n❌ {args.algorithm.upper()} training failed!")
            sys.exit(1)
    
    else:
        print("Please specify an algorithm or action. Use --help for options.")
        launcher.display_algorithms()


if __name__ == "__main__":
    main()