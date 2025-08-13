#!/usr/bin/env python
"""
简化的分层算法测试脚本
专注于测试算法核心功能，不修改基础模块
"""

import os
import sys
import numpy as np
import torch
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleHierarchicalTester:
    """简化的分层算法测试器"""
    
    def __init__(self):
        self.state_dim = 109  # 固定状态维度
        self.action_dim = 2   # 固定动作维度
        self.test_results = {}
        
        logger.info(f"测试环境初始化:")
        logger.info(f"  状态维度: {self.state_dim}")
        logger.info(f"  动作维度: {self.action_dim}")
    
    def test_hac_algorithm(self) -> Dict[str, Any]:
        """测试HAC算法"""
        logger.info("测试HAC算法...")
        
        try:
            from hierarchical_rl.hac.hac_agent import HACAgent
            from hierarchical_rl.hac.hac_config import HACConfig
            
            # 配置
            config = HACConfig(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                goal_dim=3,
                num_levels=2,
                max_actions=10,
                buffer_size=1000,
                batch_size=32,
                device="cpu"
            )
            
            # 初始化agent
            agent = HACAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                config=config
            )
            
            # 测试动作选择
            state = np.random.randn(self.state_dim)
            goal = np.random.randn(3)
            action, info = agent.select_action(state, goal)
            
            # 验证输出
            assert isinstance(action, np.ndarray), "动作应为numpy数组"
            assert action.shape == (self.action_dim,), f"动作维度应为{self.action_dim}"
            assert isinstance(info, dict), "信息应为字典"
            
            return {
                'status': 'PASSED',
                'action_shape': action.shape,
                'info_keys': list(info.keys())
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e)
            }
    
    def test_hiro_algorithm(self) -> Dict[str, Any]:
        """测试HIRO算法"""
        logger.info("测试HIRO算法...")
        
        try:
            from hierarchical_rl.hiro.hiro_agent import HIROAgent
            from hierarchical_rl.hiro.hiro_config import HIROConfig
            
            # 配置
            config = HIROConfig(
                subgoal_dim=3,
                subgoal_freq=5,
                batch_size=32,
                buffer_size=1000,
                device="cpu"
            )
            
            # 初始化agent
            agent = HIROAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                config=config
            )
            
            # 测试动作选择
            state = np.random.randn(self.state_dim)
            action, info = agent.select_action(state)
            
            # 验证输出
            assert isinstance(action, np.ndarray), "动作应为numpy数组"
            assert 'subgoal' in info, "信息应包含subgoal"
            
            return {
                'status': 'PASSED',
                'action_shape': action.shape,
                'has_subgoal': 'subgoal' in info
            }
            
        except Exception as e:
            return {
                'status': 'FAILED', 
                'error': str(e)
            }
    
    def test_fun_algorithm(self) -> Dict[str, Any]:
        """测试FuN算法"""
        logger.info("测试FuN算法...")
        
        try:
            from hierarchical_rl.fun.fun_agent import FuNAgent
            from hierarchical_rl.fun.fun_config import FuNConfig
            
            # 配置
            config = FuNConfig(
                embedding_dim=64,
                goal_dim=8,
                manager_horizon=5,
                device="cpu"
            )
            
            # 初始化agent
            agent = FuNAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                config=config
            )
            
            # 测试动作选择
            state = np.random.randn(self.state_dim)
            action, info = agent.select_action(state)
            
            # 验证输出
            assert isinstance(action, np.ndarray), "动作应为numpy数组"
            assert 'goal' in info, "信息应包含goal"
            
            return {
                'status': 'PASSED',
                'action_shape': action.shape,
                'has_goal': 'goal' in info
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e)
            }
    
    def test_options_algorithm(self) -> Dict[str, Any]:
        """测试Options算法"""
        logger.info("测试Options算法...")
        
        try:
            from hierarchical_rl.options.options_agent import OptionsAgent
            from hierarchical_rl.options.options_config import OptionsConfig
            
            # 配置
            config = OptionsConfig(
                num_options=4,
                option_min_length=2,
                option_max_length=8,
                device="cpu"
            )
            
            # 初始化agent
            agent = OptionsAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                config=config
            )
            
            # 测试动作选择
            state = np.random.randn(self.state_dim)
            action, info = agent.select_action(state)
            
            # 验证输出
            assert isinstance(action, np.ndarray), "动作应为numpy数组"
            assert 'option' in info, "信息应包含option"
            
            return {
                'status': 'PASSED',
                'action_shape': action.shape,
                'selected_option': info.get('option', -1)
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """运行所有算法测试"""
        logger.info("开始运行分层算法测试...")
        
        algorithms = [
            ('HAC', self.test_hac_algorithm),
            ('HIRO', self.test_hiro_algorithm), 
            ('FuN', self.test_fun_algorithm),
            ('Options', self.test_options_algorithm)
        ]
        
        results = {}
        
        for alg_name, test_func in algorithms:
            logger.info(f"\n{'='*40}")
            logger.info(f"测试 {alg_name} 算法")
            logger.info('='*40)
            
            try:
                result = test_func()
                results[alg_name] = result
                
                if result['status'] == 'PASSED':
                    logger.info(f"✅ {alg_name} 测试通过")
                else:
                    logger.error(f"❌ {alg_name} 测试失败: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ {alg_name} 测试异常: {str(e)}")
                results[alg_name] = {
                    'status': 'FAILED',
                    'error': f"测试异常: {str(e)}"
                }
        
        # 生成测试总结
        self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict[str, Dict[str, Any]]) -> None:
        """生成测试总结"""
        logger.info("\n" + "="*60)
        logger.info("分层强化学习算法测试总结")
        logger.info("="*60)
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r['status'] == 'PASSED')
        failed_tests = total_tests - passed_tests
        
        logger.info(f"总算法数量: {total_tests}")
        logger.info(f"通过: {passed_tests}")
        logger.info(f"失败: {failed_tests}")
        logger.info(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        logger.info("\n详细结果:")
        logger.info("-" * 40)
        
        for alg_name, result in results.items():
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            logger.info(f"{alg_name:12} | {status_icon} {result['status']}")
            
            if result['status'] == 'FAILED':
                logger.info(f"             错误: {result.get('error', 'Unknown')}")


def main():
    """主测试函数"""
    logger.info("开始分层强化学习算法测试...")
    
    # 创建测试器
    tester = SimpleHierarchicalTester()
    
    try:
        # 运行所有测试
        results = tester.run_all_tests()
        
        # 检查是否所有测试都通过
        all_passed = all(result['status'] == 'PASSED' for result in results.values())
        
        if all_passed:
            logger.info("\n🎉 所有测试都通过了！分层强化学习算法工作正常。")
            return True
        else:
            logger.warning("\n⚠️ 部分测试失败，请检查上面的详细结果。")
            return False
        
    except Exception as e:
        logger.error(f"测试失败，错误: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)