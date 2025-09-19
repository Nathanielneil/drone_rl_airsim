#!/usr/bin/env python
"""
Fixed testing script for hierarchical reinforcement learning components.
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
        
        from hierarchical_rl.common.goal_generation import GoalGenerator
        print("✓ GoalGenerator imported successfully")
        
        from hierarchical_rl.common.intrinsic_motivation import IntrinsicMotivation
        print("✓ IntrinsicMotivation imported successfully")
        
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
        
        from hierarchical_rl.envs.goal_conditioned_wrapper import GoalConditionedWrapper
        print("✓ GoalConditionedWrapper imported successfully")
        
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
        hac_config = HACConfig(device='cpu')
        print(f"✓ HAC Config created: num_levels={hac_config.num_levels}")
        
        # Test FuN config
        fun_config = FuNConfig(device='cpu')
        print(f"✓ FuN Config created: manager_horizon={fun_config.manager_horizon}")
        
        # Test HIRO config
        hiro_config = HIROConfig(device='cpu')
        print(f"✓ HIRO Config created: subgoal_freq={hiro_config.subgoal_freq}")
        
        # Test Options config
        options_config = OptionsConfig(device='cpu')
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
        hac_config = HACConfig(device=device)
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
        fun_config = FuNConfig(device=device)
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
        hiro_config = HIROConfig(device=device)
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
        options_config = OptionsConfig(device=device)
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

def test_simple_training_loop():
    """Test a simple training loop with one agent."""
    print("\n" + "=" * 60)
    print("Testing Simple Training Loop...")
    print("=" * 60)
    
    try:
        # Use Options agent (simplest)
        from hierarchical_rl.options import OptionsAgent, OptionsConfig
        
        config = OptionsConfig(device="cpu", buffer_size=100)
        agent = OptionsAgent(state_dim=50, action_dim=2, config=config, continuous_actions=True)
        
        print("✓ Created Options agent for training test")
        
        # Simulate simple training loop
        for episode in range(3):
            agent.reset_episode()
            episode_reward = 0
            
            for step in range(10):
                # Random state
                state = np.random.randn(50)
                
                # Select action
                action, info = agent.select_action(state)
                
                # Simulate environment step
                next_state = np.random.randn(50)
                reward = np.random.randn()
                done = step >= 9
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done, info)
                
                episode_reward += reward
            
            # Try to update agent
            if episode > 0:  # Skip first episode
                losses = agent.update()
                print(f"✓ Episode {episode}: reward={episode_reward:.2f}, losses={list(losses.keys()) if losses else 'None'}")
        
        print("\n✅ Simple training loop test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Training loop test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Starting Fixed Hierarchical RL Component Tests...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device available: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    test_results = []
    
    # Run tests
    test_results.append(("Import Test", test_imports()))
    test_results.append(("Config Test", test_config_classes()))
    test_results.append(("Base Components Test", test_base_components()))
    test_results.append(("Agent Creation Test", test_agent_creation()))
    test_results.append(("Simple Training Test", test_simple_training_loop()))
    
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
        print("\nAll tests passed! Hierarchical RL components are working correctly.")
    else:
        print(f"\n{len(test_results) - passed} tests failed. Please check the errors above.")
    
    return passed == len(test_results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)