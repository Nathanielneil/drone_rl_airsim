#!/usr/bin/env python
"""
基础功能测试 - 验证所有算法的基本导入和初始化
不依赖仿真环境，仅测试算法逻辑
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

def test_hac_basic():
    """测试HAC基本功能"""
    try:
        from hierarchical_rl.hac.hac_agent import HACAgent
        from hierarchical_rl.hac.hac_config import HACConfig
        
        config = HACConfig(
            num_levels=2,
            goal_dim=3,
            max_actions=10,
            batch_size=16,
            device="cpu"
        )
        
        agent = HACAgent(state_dim=109, action_dim=2, goal_dim=3, config=config)
        
        # 测试动作选择
        import numpy as np
        state = np.random.randn(109)
        goal = np.random.randn(3)
        action, info = agent.select_action(state, goal)
        
        print(f"✅ HAC: action shape {action.shape}, info keys: {list(info.keys())}")
        return True
    except Exception as e:
        print(f"❌ HAC failed: {e}")
        return False

def test_hiro_basic():
    """测试HIRO基本功能"""
    try:
        from hierarchical_rl.hiro.hiro_agent import HIROAgent
        from hierarchical_rl.hiro.hiro_config import HIROConfig
        
        config = HIROConfig(device="cpu")
        agent = HIROAgent(state_dim=109, action_dim=2, config=config)
        
        import numpy as np
        state = np.random.randn(109)
        action, info = agent.select_action(state)
        
        print(f"✅ HIRO: action shape {action.shape}, has subgoal: {'subgoal' in info}")
        return True
    except Exception as e:
        print(f"❌ HIRO failed: {e}")
        return False

def test_fun_basic():
    """测试FuN基本功能"""
    try:
        from hierarchical_rl.fun.fun_agent import FuNAgent
        from hierarchical_rl.fun.fun_config import FuNConfig
        
        config = FuNConfig(device="cpu")
        agent = FuNAgent(state_dim=109, action_dim=2, config=config)
        
        import numpy as np
        state = np.random.randn(109)
        action, info = agent.select_action(state)
        
        print(f"✅ FuN: action shape {action.shape}, has goal: {'goal' in info}")
        return True
    except Exception as e:
        print(f"❌ FuN failed: {e}")
        return False

def test_options_basic():
    """测试Options基本功能"""
    try:
        from hierarchical_rl.options.options_agent import OptionsAgent
        from hierarchical_rl.options.options_config import OptionsConfig
        
        config = OptionsConfig(device="cpu")
        agent = OptionsAgent(state_dim=109, action_dim=2, config=config)
        
        import numpy as np
        state = np.random.randn(109)
        action, info = agent.select_action(state)
        
        print(f"✅ Options: action shape {action.shape}, selected option: {info.get('option', -1)}")
        return True
    except Exception as e:
        print(f"❌ Options failed: {e}")
        return False

def main():
    """运行所有基础测试"""
    print("="*60)
    print("分层强化学习算法基础功能测试")
    print("="*60)
    
    tests = [
        ("HAC", test_hac_basic),
        ("HIRO", test_hiro_basic), 
        ("FuN", test_fun_basic),
        ("Options", test_options_basic)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n测试 {name}...")
        success = test_func()
        results.append((name, success))
    
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        icon = "✅" if success else "❌"
        print(f"{name:12} | {icon} {status}")
    
    print(f"\n总体结果: {passed}/{total} 算法通过基础测试")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)