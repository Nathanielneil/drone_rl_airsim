#!/usr/bin/env python
"""
Comprehensive testing script for hierarchical reinforcement learning components.
Tests each module independently to ensure proper functionality.
"""

import os
import sys
import numpy as np
import torch
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test basic imports of all HRL modules."""
    print("=" * 60)
    print("Testing Basic Imports...")
    print("=" * 60)
    
    try:
        # Test base components
        from hierarchical_rl.common.base_hierarchical_agent import BaseHierarchicalAgent, HierarchicalNetwork
        print("✓ BaseHierarchicalAgent imported successfully")
        
        from hierarchical_rl.common.hierarchical_replay_buffer import HierarchicalReplayBuffer
        print("✓ HierarchicalReplayBuffer imported successfully")
        
        # Test HAC
        from hierarchical_rl.hac import HACAgent, HACConfig
        print("✓ HAC components imported successfully")
        
        # Test FuN
        from hierarchical_rl.fun import FuNAgent, FuNConfig
        print("✓ FuN components imported successfully")
        
        # Test HIRO
        from hierarchical_rl.hiro import HIROAgent, HIROConfig
        print("✓ HIRO components imported successfully")
        
        # Test Options
        from hierarchical_rl.options import OptionsAgent, OptionsConfig
        print("✓ Options components imported successfully")
        
        # Test environments
        from hierarchical_rl.envs.hierarchical_airsim_env import HierarchicalAirSimEnv
        print("✓ HierarchicalAirSimEnv imported successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_config_classes():
    """Test configuration classes."""
    print("\n" + "=" * 60)
    print("Testing Configuration Classes...")
    print("=" * 60)
    
    try:
        from hierarchical_rl.hac import HACConfig
        from hierarchical_rl.fun import FuNConfig
        from hierarchical_rl.hiro import HIROConfig
        from hierarchical_rl.options import OptionsConfig
        
        # Test HAC config
        hac_config = HACConfig()
        print(f"✓ HAC Config created: num_levels={hac_config.num_levels}")
        
        # Test FuN config
        fun_config = FuNConfig()
        print(f"✓ FuN Config created: manager_horizon={fun_config.manager_horizon}")
        
        # Test HIRO config
        hiro_config = HIROConfig()
        print(f"✓ HIRO Config created: subgoal_freq={hiro_config.subgoal_freq}")
        
        # Test Options config
        options_config = OptionsConfig()
        print(f"✓ Options Config created: num_options={options_config.num_options}")
        
        print("\n✅ All configurations created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_base_components():
    """Test base hierarchical components."""
    print("\n" + "=" * 60)
    print("Testing Base Components...")
    print("=" * 60)
    
    try:
        from hierarchical_rl.common.base_hierarchical_agent import HierarchicalNetwork
        from hierarchical_rl.common.hierarchical_replay_buffer import HierarchicalReplayBuffer
        
        # Test HierarchicalNetwork
        network = HierarchicalNetwork(
            input_dim=10,
            output_dim=5,
            hidden_dims=[32, 32]
        )
        
        test_input = torch.randn(2, 10)
        output = network(test_input)
        assert output.shape == (2, 5), f"Expected shape (2, 5), got {output.shape}"
        print("✓ HierarchicalNetwork forward pass successful")
        
        # Test HierarchicalReplayBuffer
        buffer = HierarchicalReplayBuffer(
            capacity=1000,
            state_dim=10,
            action_dim=4,
            goal_dim=3
        )
        
        # Store some dummy transitions
        for i in range(10):
            state = np.random.randn(10)
            action = np.random.randn(4)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            goal = np.random.randn(3)
            
            buffer.store_low_level(state, action, reward, next_state, goal, False)
        
        print(f"✓ HierarchicalReplayBuffer stored {len(buffer)} transitions")
        
        # Test sampling
        if buffer.can_sample(5):
            batch = buffer.sample_low_level(5)
            print(f"✓ Successfully sampled batch with keys: {list(batch.keys())}")
        
        print("\n✅ Base components test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Base components test failed: {e}")
        traceback.print_exc()
        return False

def test_agent_creation():
    """Test creating instances of all HRL agents."""
    print("\n" + "=" * 60)
    print("Testing Agent Creation...")
    print("=" * 60)
    
    state_dim = 100  # Visual features + inform vector
    action_dim = 2   # Continuous control
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    agents_created = 0
    
    try:
        # Test HAC Agent
        from hierarchical_rl.hac import HACAgent, HACConfig
        hac_config = HACConfig(device=device, num_episodes=10)
        hac_agent = HACAgent(state_dim=state_dim, action_dim=action_dim, goal_dim=3, config=hac_config)
        print(f"✓ HAC Agent created successfully on {device}")
        agents_created += 1
        
        # Test action selection
        state = np.random.randn(state_dim)
        goal = np.random.randn(3)
        action, info = hac_agent.select_action(state, goal)
        print(f"  - Action shape: {action.shape}, Info keys: {list(info.keys())}")
        
    except Exception as e:
        print(f"❌ HAC Agent creation failed: {e}")
        traceback.print_exc()
    
    try:
        # Test FuN Agent
        from hierarchical_rl.fun import FuNAgent, FuNConfig
        fun_config = FuNConfig(device=device, num_episodes=10)
        fun_agent = FuNAgent(state_dim=state_dim, action_dim=action_dim, config=fun_config)
        print(f"✓ FuN Agent created successfully on {device}")
        agents_created += 1
        
        # Test action selection
        state = np.random.randn(state_dim)
        action, info = fun_agent.select_action(state)
        print(f"  - Action shape: {action.shape}, Info keys: {list(info.keys())}")
        
    except Exception as e:
        print(f"❌ FuN Agent creation failed: {e}")
        traceback.print_exc()
    
    try:
        # Test HIRO Agent
        from hierarchical_rl.hiro import HIROAgent, HIROConfig
        hiro_config = HIROConfig(device=device, num_episodes=10)
        hiro_agent = HIROAgent(state_dim=state_dim, action_dim=action_dim, config=hiro_config)
        print(f"✓ HIRO Agent created successfully on {device}")
        agents_created += 1
        
        # Test action selection
        state = np.random.randn(state_dim)
        action, info = hiro_agent.select_action(state)
        print(f"  - Action shape: {action.shape}, Info keys: {list(info.keys())}")
        
    except Exception as e:
        print(f"❌ HIRO Agent creation failed: {e}")
        traceback.print_exc()
    
    try:
        # Test Options Agent
        from hierarchical_rl.options import OptionsAgent, OptionsConfig
        options_config = OptionsConfig(device=device, num_episodes=10)
        options_agent = OptionsAgent(state_dim=state_dim, action_dim=action_dim, config=options_config)
        print(f"✓ Options Agent created successfully on {device}")
        agents_created += 1
        
        # Test action selection
        state = np.random.randn(state_dim)
        action, info = options_agent.select_action(state)
        print(f"  - Action shape: {action.shape}, Info keys: {list(info.keys())}")
        
    except Exception as e:
        print(f"❌ Options Agent creation failed: {e}")
        traceback.print_exc()
    
    print(f"\n✅ Successfully created {agents_created}/4 agents!")
    return agents_created == 4

def test_mock_environment():
    """Test with a mock environment to simulate training loop."""
    print("\n" + "=" * 60)
    print("Testing Mock Environment Integration...")
    print("=" * 60)
    
    try:
        # Create a simple mock environment
        class MockEnv:
            def __init__(self):
                self.state_dim = 109  # 100 visual + 9 inform
                self.action_dim = 2
                self.step_count = 0
                self.max_steps = 50
            
            def reset(self):
                self.step_count = 0
                obs = {
                    'observation': np.random.randn(4, 112, 112),  # Stack of depth images
                    'inform_vector': np.random.randn(9),
                    'desired_goal': np.random.randn(3),
                    'achieved_goal': np.random.randn(3)
                }
                info = {'goal': np.random.randn(3)}
                return obs, info
            
            def step(self, action):
                self.step_count += 1
                obs = {
                    'observation': np.random.randn(4, 112, 112),
                    'inform_vector': np.random.randn(9),
                    'desired_goal': np.random.randn(3),
                    'achieved_goal': np.random.randn(3)
                }
                reward = np.random.randn()
                terminated = self.step_count >= self.max_steps
                truncated = False
                info = {
                    'goal_achieved': np.random.random() < 0.1,
                    'goal_distance': np.random.uniform(0, 5)
                }
                return obs, reward, terminated, truncated, info
            
            def get_goal(self):
                return np.random.randn(3)
        
        # Test with Options Agent (simplest to test)
        from hierarchical_rl.options import OptionsAgent, OptionsConfig
        
        config = OptionsConfig(device="cpu", num_episodes=5, buffer_size=100)
        agent = OptionsAgent(state_dim=109, action_dim=2, config=config, continuous_actions=True)
        env = MockEnv()
        
        print("✓ Created mock environment and Options agent")
        
        # Run a short episode
        obs, info = env.reset()
        agent.reset_episode()
        
        episode_reward = 0
        step = 0
        done = False
        
        while not done and step < 20:
            # Process observation (simplified)
            visual_features = obs['observation'].flatten()[:100]
            visual_features = np.pad(visual_features, (0, max(0, 100 - len(visual_features))))
            state = np.concatenate([visual_features, obs['inform_vector']])
            
            # Select action
            action, action_info = agent.select_action(state)
            
            # Execute action
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            combined_info = {**action_info, **step_info}
            agent.store_transition(state, action, reward, next_obs, done, combined_info)
            
            episode_reward += reward
            step += 1
            obs = next_obs
        
        print(f"✓ Completed mock episode: {step} steps, reward: {episode_reward:.2f}")
        
        # Test agent update
        if step > 10:  # Only update if we have some experience
            losses = agent.update()
            print(f"✓ Agent update completed, losses: {list(losses.keys()) if losses else 'No losses'}")
        
        print("\n✅ Mock environment integration test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Mock environment test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Starting Hierarchical RL Component Tests...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device available: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    test_results = []
    
    # Run tests
    test_results.append(("Import Test", test_imports()))
    test_results.append(("Config Test", test_config_classes()))
    test_results.append(("Base Components Test", test_base_components()))
    test_results.append(("Agent Creation Test", test_agent_creation()))
    test_results.append(("Mock Environment Test", test_mock_environment()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("\n🎉 All tests passed! Hierarchical RL components are working correctly.")
    else:
        print(f"\n⚠️  {len(test_results) - passed} tests failed. Please check the errors above.")
    
    return passed == len(test_results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)