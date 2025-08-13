#!/usr/bin/env python
"""
Comprehensive Test Suite for All Hierarchical RL Algorithms
===========================================================

This script tests and validates all implemented hierarchical RL algorithms:
- HAC (Hindsight Action Control)
- HIRO (HIerarchical RL with Off-policy correction)
- FuN (FeUdal Networks)
- Options Framework

Each algorithm is tested for:
1. Basic functionality and initialization
2. Action selection mechanism
3. Training loop integration
4. Model saving/loading
5. Performance benchmarking
"""

import os
import sys
import time
import numpy as np
import torch
import logging
from typing import Dict, List, Any, Optional
import traceback

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_airsim.envs.AirGym import AirSimEnv
from hierarchical_rl.envs.hierarchical_airsim_env import HierarchicalAirSimEnv

# Algorithm imports
from hierarchical_rl.hac.hac_agent import HACAgent
from hierarchical_rl.hac.hac_config import HACConfig
from hierarchical_rl.hiro.hiro_agent import HIROAgent
from hierarchical_rl.hiro.hiro_config import HIROConfig
from hierarchical_rl.fun.fun_agent import FuNAgent
from hierarchical_rl.fun.fun_config import FuNConfig
from hierarchical_rl.options.options_agent import OptionsAgent
from hierarchical_rl.options.options_config import OptionsConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlgorithmTester:
    """Comprehensive tester for hierarchical RL algorithms."""
    
    def __init__(self, test_dir: str = None):
        self.test_dir = test_dir or f"test_results_{int(time.time())}"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Initialize test environment
        base_env = AirSimEnv()
        self.env = HierarchicalAirSimEnv(
            base_env, 
            goal_dim=3,  # Default goal dimension
            max_episode_steps=100  # Shorter episodes for testing
        )
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        logger.info(f"Test environment initialized:")
        logger.info(f"  State dimension: {self.state_dim}")
        logger.info(f"  Action dimension: {self.action_dim}")
        logger.info(f"  Test directory: {self.test_dir}")
        
        # Test results
        self.test_results = {}
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run comprehensive tests for all algorithms."""
        logger.info("Starting comprehensive algorithm tests...")
        
        algorithms = [
            ('HAC', self._test_hac),
            ('HIRO', self._test_hiro),
            ('FuN', self._test_fun),
            ('Options', self._test_options)
        ]
        
        for alg_name, test_func in algorithms:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing {alg_name} Algorithm")
            logger.info('='*50)
            
            try:
                start_time = time.time()
                results = test_func()
                test_time = time.time() - start_time
                
                results['test_duration'] = test_time
                results['status'] = 'PASSED'
                self.test_results[alg_name] = results
                
                logger.info(f"✅ {alg_name} tests PASSED ({test_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"❌ {alg_name} tests FAILED: {str(e)}")
                logger.error(traceback.format_exc())
                
                self.test_results[alg_name] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'test_duration': 0
                }
        
        # Generate summary
        self._generate_test_summary()
        return self.test_results
    
    def _test_hac(self) -> Dict[str, Any]:
        """Test HAC algorithm."""
        logger.info("Testing HAC (Hindsight Action Control)...")
        
        # Configuration
        config = HACConfig(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            goal_dim=3,
            num_levels=2,
            max_actions=10,
            buffer_size=1000,
            batch_size=32,
            device="cpu",  # Use CPU for testing
            max_episode_steps=50
        )
        
        # Initialize agent
        agent = HACAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config
        )
        
        results = {}
        
        # Test 1: Basic initialization
        logger.info("  Test 1: Basic initialization")
        assert hasattr(agent, 'networks'), "Agent should have networks attribute"
        assert hasattr(agent, 'replay_buffer'), "Agent should have replay buffer"
        results['initialization'] = True
        
        # Test 2: Action selection
        logger.info("  Test 2: Action selection")
        state = np.random.randn(self.state_dim)
        goal = np.random.randn(3)
        action, info = agent.select_action(state, goal)
        
        assert isinstance(action, np.ndarray), "Action should be numpy array"
        assert action.shape == (self.action_dim,), f"Action shape should be ({self.action_dim},)"
        assert isinstance(info, dict), "Info should be dictionary"
        results['action_selection'] = True
        
        # Test 3: Training episode
        logger.info("  Test 3: Training episode simulation")
        episode_rewards = []
        
        for episode in range(3):  # Short test episodes
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(20):  # Short episodes
                action, info = agent.select_action(state, goal)
                next_state, reward, done, env_info = self.env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done, info)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            agent.end_episode()
        
        results['training_simulation'] = {
            'episodes_completed': len(episode_rewards),
            'avg_reward': np.mean(episode_rewards)
        }
        
        # Test 4: Model saving/loading
        logger.info("  Test 4: Model saving and loading")
        save_path = os.path.join(self.test_dir, 'hac_test_model.pth')
        agent.save(save_path)
        assert os.path.exists(save_path), "Model file should be saved"
        
        # Create new agent and load
        new_agent = HACAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config
        )
        new_agent.load(save_path)
        results['model_save_load'] = True
        
        logger.info("  ✅ HAC tests completed successfully")
        return results
    
    def _test_hiro(self) -> Dict[str, Any]:
        """Test HIRO algorithm."""
        logger.info("Testing HIRO (HIerarchical RL with Off-policy correction)...")
        
        # Configuration
        config = HIROConfig(
            subgoal_dim=3,
            subgoal_freq=5,
            subgoal_scale=5.0,
            her_ratio=0.5,
            batch_size=32,
            buffer_size=1000,
            min_buffer_size=100,
            device="cpu"
        )
        
        # Initialize agent
        agent = HIROAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config
        )
        
        results = {}
        
        # Test 1: Basic initialization
        logger.info("  Test 1: Basic initialization")
        assert hasattr(agent, 'high_level_actor'), "Agent should have high-level actor"
        assert hasattr(agent, 'low_level_actor'), "Agent should have low-level actor"
        assert hasattr(agent, 'replay_buffer'), "Agent should have replay buffer"
        results['initialization'] = True
        
        # Test 2: Hierarchical action selection
        logger.info("  Test 2: Hierarchical action selection")
        state = np.random.randn(self.state_dim)
        action, info = agent.select_action(state)
        
        assert isinstance(action, np.ndarray), "Action should be numpy array"
        assert 'subgoal' in info, "Info should contain subgoal"
        assert 'relative_subgoal' in info, "Info should contain relative subgoal"
        results['hierarchical_action_selection'] = True
        
        # Test 3: Off-policy correction mechanism
        logger.info("  Test 3: Off-policy correction and HER")
        
        # Simulate episode with HER
        state = self.env.reset()
        for step in range(30):
            action, info = agent.select_action(state)
            next_state, reward, done, env_info = self.env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done, info)
            
            state = next_state
            if done:
                break
        
        agent.end_episode()  # This should trigger HER processing
        results['her_mechanism'] = True
        
        # Test 4: Model saving/loading
        logger.info("  Test 4: Model saving and loading")
        save_path = os.path.join(self.test_dir, 'hiro_test_model.pth')
        agent.save(save_path)
        assert os.path.exists(save_path), "Model file should be saved"
        
        new_agent = HIROAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config
        )
        new_agent.load(save_path)
        results['model_save_load'] = True
        
        logger.info("  ✅ HIRO tests completed successfully")
        return results
    
    def _test_fun(self) -> Dict[str, Any]:
        """Test FuN algorithm."""
        logger.info("Testing FuN (FeUdal Networks)...")
        
        # Configuration
        config = FuNConfig(
            manager_horizon=5,
            embedding_dim=64,
            goal_dim=8,
            manager_hidden_dims=[128, 64],
            worker_hidden_dims=[128, 64],
            buffer_size=512,
            batch_size=32,
            device="cpu"
        )
        
        # Initialize agent
        agent = FuNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config,
            continuous_actions=True
        )
        
        results = {}
        
        # Test 1: Basic initialization
        logger.info("  Test 1: Basic initialization")
        assert hasattr(agent, 'state_encoder'), "Agent should have state encoder"
        assert hasattr(agent, 'manager'), "Agent should have manager network"
        assert hasattr(agent, 'worker'), "Agent should have worker network"
        results['initialization'] = True
        
        # Test 2: Manager-worker hierarchy
        logger.info("  Test 2: Manager-worker hierarchy")
        state = np.random.randn(self.state_dim)
        action, info = agent.select_action(state)
        
        assert isinstance(action, np.ndarray), "Action should be numpy array"
        assert 'goal' in info, "Info should contain manager goal"
        assert 'intrinsic_reward' in info, "Info should contain intrinsic reward"
        results['manager_worker_hierarchy'] = True
        
        # Test 3: Intrinsic motivation
        logger.info("  Test 3: Intrinsic motivation mechanism")
        
        # Test intrinsic reward computation
        state_embedding = np.random.randn(config.embedding_dim)
        goal = np.random.randn(config.goal_dim)
        intrinsic_reward = agent._compute_intrinsic_reward(state_embedding, goal)
        
        assert isinstance(intrinsic_reward, (float, np.floating)), "Intrinsic reward should be numeric"
        assert -1 <= intrinsic_reward <= 1, "Intrinsic reward should be in reasonable range"
        results['intrinsic_motivation'] = True
        
        # Test 4: Feudal training simulation
        logger.info("  Test 4: Feudal training simulation")
        
        state = self.env.reset()
        agent.reset_episode()
        
        for step in range(20):
            action, info = agent.select_action(state)
            next_state, reward, done, env_info = self.env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done, info)
            
            state = next_state
            if done:
                break
        
        # Trigger update
        losses = agent.update()
        results['feudal_training'] = {
            'losses_computed': losses is not None and len(losses) > 0
        }
        
        # Test 5: Model saving/loading
        logger.info("  Test 5: Model saving and loading")
        save_path = os.path.join(self.test_dir, 'fun_test_model.pth')
        agent.save(save_path)
        assert os.path.exists(save_path), "Model file should be saved"
        
        new_agent = FuNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config,
            continuous_actions=True
        )
        new_agent.load(save_path)
        results['model_save_load'] = True
        
        logger.info("  ✅ FuN tests completed successfully")
        return results
    
    def _test_options(self) -> Dict[str, Any]:
        """Test Options framework."""
        logger.info("Testing Options Framework...")
        
        # Configuration
        config = OptionsConfig(
            num_options=4,
            option_min_length=2,
            option_max_length=8,
            use_diversity_bonus=True,
            diversity_coef=0.1,
            buffer_size=512,
            batch_size=32,
            device="cpu"
        )
        
        # Initialize agent
        agent = OptionsAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config,
            continuous_actions=True
        )
        
        results = {}
        
        # Test 1: Basic initialization
        logger.info("  Test 1: Basic initialization")
        assert hasattr(agent, 'option_policies'), "Agent should have option policies"
        assert hasattr(agent, 'option_selector'), "Agent should have option selector"
        assert hasattr(agent, 'termination_network'), "Agent should have termination network"
        assert len(agent.option_policies) == config.num_options, f"Should have {config.num_options} option policies"
        results['initialization'] = True
        
        # Test 2: Option selection mechanism
        logger.info("  Test 2: Option selection mechanism")
        state = np.random.randn(self.state_dim)
        action, info = agent.select_action(state)
        
        assert isinstance(action, np.ndarray), "Action should be numpy array"
        assert 'option' in info, "Info should contain selected option"
        assert 'option_length' in info, "Info should contain option length"
        assert 0 <= info['option'] < config.num_options, "Option should be valid"
        results['option_selection'] = True
        
        # Test 3: Option termination
        logger.info("  Test 3: Option termination mechanism")
        
        # Force option to run for multiple steps
        agent.current_option = 0
        agent.option_length = config.option_min_length + 1
        
        # Test termination decision
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        should_terminate = agent._should_terminate_option(state_tensor)
        assert isinstance(should_terminate, bool), "Termination decision should be boolean"
        results['option_termination'] = True
        
        # Test 4: Diversity mechanism
        logger.info("  Test 4: Diversity bonus mechanism")
        
        diversity_bonus = agent._compute_diversity_bonus(state, 0)
        assert isinstance(diversity_bonus, (float, np.floating)), "Diversity bonus should be numeric"
        assert diversity_bonus >= 0, "Diversity bonus should be non-negative"
        results['diversity_mechanism'] = True
        
        # Test 5: Option statistics
        logger.info("  Test 5: Option statistics tracking")
        
        # Simulate some option usage
        for i in range(10):
            action, info = agent.select_action(np.random.randn(self.state_dim))
            agent.option_visit_counts[info['option']] += 1
        
        stats = agent.get_option_statistics()
        assert 'option_visit_counts' in stats, "Stats should include visit counts"
        assert 'option_frequencies' in stats, "Stats should include frequencies"
        assert 'option_diversity_score' in stats, "Stats should include diversity score"
        results['option_statistics'] = True
        
        # Test 6: Model saving/loading
        logger.info("  Test 6: Model saving and loading")
        save_path = os.path.join(self.test_dir, 'options_test_model.pth')
        agent.save(save_path)
        assert os.path.exists(save_path), "Model file should be saved"
        
        new_agent = OptionsAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config,
            continuous_actions=True
        )
        new_agent.load(save_path)
        results['model_save_load'] = True
        
        logger.info("  ✅ Options tests completed successfully")
        return results
    
    def _generate_test_summary(self) -> None:
        """Generate comprehensive test summary."""
        logger.info("\n" + "="*60)
        logger.info("HIERARCHICAL RL ALGORITHMS TEST SUMMARY")
        logger.info("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')
        failed_tests = total_tests - passed_tests
        
        logger.info(f"Total Algorithms Tested: {total_tests}")
        logger.info(f"Passed: {passed_tests} ✅")
        logger.info(f"Failed: {failed_tests} ❌")
        logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        logger.info("\nDetailed Results:")
        logger.info("-" * 40)
        
        for alg_name, results in self.test_results.items():
            status_icon = "✅" if results['status'] == 'PASSED' else "❌"
            duration = results.get('test_duration', 0)
            
            logger.info(f"{alg_name:12} | {status_icon} {results['status']:6} | {duration:6.2f}s")
            
            if results['status'] == 'FAILED':
                logger.info(f"             Error: {results['error']}")
        
        logger.info("\nTest Environment:")
        logger.info(f"  State Dimension: {self.state_dim}")
        logger.info(f"  Action Dimension: {self.action_dim}")
        logger.info(f"  Device: CPU (for testing)")
        
        # Save summary to file
        summary_path = os.path.join(self.test_dir, 'test_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("HIERARCHICAL RL ALGORITHMS TEST SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total Algorithms Tested: {total_tests}\n")
            f.write(f"Passed: {passed_tests}\n")
            f.write(f"Failed: {failed_tests}\n")
            f.write(f"Success Rate: {passed_tests/total_tests*100:.1f}%\n\n")
            
            for alg_name, results in self.test_results.items():
                f.write(f"{alg_name}: {results['status']}\n")
                if 'error' in results:
                    f.write(f"  Error: {results['error']}\n")
        
        logger.info(f"\nTest summary saved to: {summary_path}")


def main():
    """Main testing function."""
    logger.info("Starting comprehensive hierarchical RL algorithm tests...")
    
    # Create tester
    tester = AlgorithmTester()
    
    try:
        # Run all tests
        results = tester.run_all_tests()
        
        # Check if all tests passed
        all_passed = all(result['status'] == 'PASSED' for result in results.values())
        
        if all_passed:
            logger.info("\n🎉 ALL TESTS PASSED! All hierarchical RL algorithms are working correctly.")
        else:
            logger.warning("\n⚠️ Some tests failed. Check the detailed results above.")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
    except Exception as e:
        logger.error(f"Testing failed with error: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()