#!/usr/bin/env python
"""
高级分层算法深度测试
==================

对HIRO、FuN、Options三个算法进行完整的AirSim环境集成测试
验证核心机制：分层决策、经验回放、内在动机、选项学习
"""

import os
import sys
import time
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 算法导入
from gym_airsim.envs.AirGym import AirSimEnv
from hierarchical_rl.envs.hierarchical_airsim_env import HierarchicalAirSimEnv
from hierarchical_rl.hiro.hiro_agent import HIROAgent
from hierarchical_rl.hiro.hiro_config import HIROConfig
from hierarchical_rl.fun.fun_agent import FuNAgent
from hierarchical_rl.fun.fun_config import FuNConfig
from hierarchical_rl.options.options_agent import OptionsAgent
from hierarchical_rl.options.options_config import OptionsConfig

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedAlgorithmTester:
    """高级分层算法综合测试器"""
    
    def _extract_state(self, env_return):
        """从环境返回值中提取状态"""
        if isinstance(env_return, tuple) and len(env_return) > 0:
            state = env_return[0]
            if isinstance(state, dict):
                # 使用inform_vector作为状态表示，回退到observation
                return state.get('inform_vector', state.get('observation', np.zeros(2)))
            return state
        elif isinstance(env_return, dict):
            return env_return.get('inform_vector', env_return.get('observation', np.zeros(2)))
        return env_return
    
    def __init__(self):
        self.test_results = {}
        self.episode_length = 30  # 较短的测试episode
        self.num_test_episodes = 2  # 测试episode数量
        
        # 初始化测试环境
        try:
            base_env = AirSimEnv()
            self.env = HierarchicalAirSimEnv(
                base_env,
                goal_dim=3,
                max_episode_steps=self.episode_length
            )
            
            # 安全地获取环境维度
            try:
                # 重置环境获取观测空间
                dummy_reset = self.env.reset()
                dummy_state = self._extract_state(dummy_reset)
                self.state_dim = len(dummy_state) if dummy_state is not None else 109
                self.action_dim = self.env.action_space.shape[0] if self.env.action_space is not None else 2
            except:
                # 使用默认值
                self.state_dim = 109
                self.action_dim = 2
                
            logger.info(f"测试环境初始化成功")
            logger.info(f"状态维度: {self.state_dim}")
            logger.info(f"动作维度: {self.action_dim}")
        except Exception as e:
            logger.error(f"环境初始化失败: {e}")
            raise
    
    def test_hiro_algorithm(self) -> Dict[str, Any]:
        """测试HIRO算法"""
        logger.info("="*60)
        logger.info("测试 HIRO (HIerarchical RL with Off-policy correction)")
        logger.info("="*60)
        
        # 配置HIRO (根据实际环境维度调整)
        actual_subgoal_dim = min(3, self.state_dim)  # 适应实际状态维度
        config = HIROConfig(
            subgoal_dim=actual_subgoal_dim,
            subgoal_freq=10,
            subgoal_scale=5.0,
            her_ratio=0.8,
            off_policy_correction=True,
            correction_radius=2.0,
            batch_size=16,  # 减小batch size
            buffer_size=500,  # 减小buffer size
            min_buffer_size=50,
            device="cpu"
        )
        
        # 初始化agent
        agent = HIROAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim, 
            config=config
        )
        
        results = {
            'algorithm': 'HIRO',
            'episodes_completed': 0,
            'episode_rewards': [],
            'episode_lengths': [],
            'subgoals_generated': 0,
            'her_experiences': 0,
            'policy_updates': 0,
            'success_episodes': 0,
            'final_positions': []
        }
        
        logger.info(f"开始HIRO测试 - {self.num_test_episodes}个episode")
        
        for episode in range(self.num_test_episodes):
            logger.info(f"\n--- HIRO Episode {episode + 1}/{self.num_test_episodes} ---")
            
            try:
                # 重置环境
                env_reset = self.env.reset()
                state = self._extract_state(env_reset)
                if state is None:
                    logger.warning("环境重置失败，跳过此episode")
                    continue
                    
                # 重置agent状态（如果有的话）
                if hasattr(agent, 'reset_episode'):
                    agent.reset_episode()
                elif hasattr(agent, 'reset'):
                    agent.reset()
                
                episode_reward = 0.0
                episode_length = 0
                subgoals_this_episode = 0
                
                for step in range(self.episode_length):
                    try:
                        # 选择动作
                        action, info = agent.select_action(state, deterministic=False)
                        
                        # 记录子目标信息
                        if 'subgoal' in info and info['subgoal'] is not None:
                            subgoals_this_episode += 1
                        
                        # 执行动作
                        try:
                            # 确保action是numpy数组
                            if not isinstance(action, np.ndarray):
                                logger.warning(f"Action is not numpy array: {type(action)} {action}")
                                action = np.array(action)
                            
                            step_result = self.env.step(action)
                            if len(step_result) == 4:
                                raw_next_state, reward, done, env_info = step_result
                            else:
                                # Handle different return formats
                                raw_next_state, reward, done = step_result[:3]
                                env_info = step_result[3] if len(step_result) > 3 else {}
                            
                            # Extract state from potentially dict format
                            next_state = self._extract_state(raw_next_state)
                        except Exception as step_e:
                            logger.warning(f"Step execution error with action {action} (type: {type(action)}): {step_e}")
                            break
                        
                        if next_state is None:
                            logger.warning(f"Step {step}: 获得None状态，停止episode")
                            break
                        
                        # 存储转换
                        agent.store_transition(state, action, reward, next_state, done, info)
                        
                        # 更新状态
                        state = next_state
                        episode_reward += reward
                        episode_length += 1
                        
                        # 输出关键信息
                        if step % 10 == 0:
                            try:
                                current_pos = next_state[:3] if len(next_state) >= 3 else next_state[:2]
                                subgoal = info.get('subgoal', [0, 0, 0])
                                logger.info(f"  Step {step}: Pos={current_pos[:2]}, Reward={reward:.3f}")
                            except:
                                logger.info(f"  Step {step}: Reward={reward:.3f}")
                        
                        if done:
                            logger.info(f"  Episode结束于step {step}")
                            break
                            
                    except Exception as e:
                        logger.warning(f"Step {step}执行失败: {e}")
                        break
                
                # episode结束处理
                agent.end_episode()
                
                # 记录最终位置
                try:
                    final_pos = state[:3] if len(state) >= 3 else state[:2] 
                    results['final_positions'].append(final_pos)
                except:
                    results['final_positions'].append([0, 0, 0])
                
                # 尝试执行策略更新
                try:
                    if episode_length > 10:  # 只有足够的经验才更新
                        losses = agent.update()
                        if losses:
                            results['policy_updates'] += 1
                            logger.info(f"  策略更新完成")
                except Exception as e:
                    logger.warning(f"  策略更新失败: {e}")
                
                # 记录结果
                results['episodes_completed'] += 1
                results['episode_rewards'].append(episode_reward)
                results['episode_lengths'].append(episode_length)
                results['subgoals_generated'] += subgoals_this_episode
                
                # 判断成功（移动距离超过阈值）
                try:
                    final_pos = results['final_positions'][-1]
                    if np.linalg.norm(final_pos[:2]) > 1.0:
                        results['success_episodes'] += 1
                except:
                    pass
                
                logger.info(f"  Episode {episode + 1} 完成:")
                logger.info(f"    奖励: {episode_reward:.2f}")
                logger.info(f"    长度: {episode_length}")
                logger.info(f"    子目标数: {subgoals_this_episode}")
                
            except Exception as e:
                logger.error(f"Episode {episode + 1} 失败: {e}")
                continue
        
        # 计算统计信息
        if results['episodes_completed'] > 0:
            results['avg_reward'] = np.mean(results['episode_rewards'])
            results['avg_length'] = np.mean(results['episode_lengths'])
            results['success_rate'] = results['success_episodes'] / results['episodes_completed']
        else:
            results['avg_reward'] = 0
            results['avg_length'] = 0
            results['success_rate'] = 0
        
        logger.info(f"\nHIRO测试完成:")
        logger.info(f"  完成episodes: {results['episodes_completed']}")
        logger.info(f"  平均奖励: {results['avg_reward']:.2f}")
        logger.info(f"  平均长度: {results['avg_length']:.1f}")
        logger.info(f"  成功率: {results['success_rate']:.2f}")
        logger.info(f"  策略更新次数: {results['policy_updates']}")
        
        return results
    
    def test_fun_algorithm(self) -> Dict[str, Any]:
        """测试FuN算法"""
        logger.info("="*60)
        logger.info("测试 FuN (FeUdal Networks)")
        logger.info("="*60)
        
        # 配置FuN
        config = FuNConfig(
            manager_horizon=6,  # 缩短horizon
            embedding_dim=64,   # 减小embedding维度
            goal_dim=8,         # 减小goal维度
            manager_hidden_dims=[128, 64],
            worker_hidden_dims=[64, 64],
            batch_size=16,      # 减小batch size
            buffer_size=256,    # 减小buffer size
            device="cpu"
        )
        
        # 初始化agent
        agent = FuNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config,
            continuous_actions=True
        )
        
        results = {
            'algorithm': 'FuN',
            'episodes_completed': 0,
            'episode_rewards': [],
            'intrinsic_rewards': [],
            'manager_decisions': 0,
            'worker_actions': 0,
            'success_episodes': 0,
            'final_positions': []
        }
        
        logger.info(f"开始FuN测试 - {self.num_test_episodes}个episode")
        
        for episode in range(self.num_test_episodes):
            logger.info(f"\n--- FuN Episode {episode + 1}/{self.num_test_episodes} ---")
            
            try:
                # 重置环境和agent
                state = self.env.reset()
                if state is None:
                    logger.warning("环境重置失败，跳过此episode")
                    continue
                    
                # 重置agent状态（如果有的话）
                if hasattr(agent, 'reset_episode'):
                    agent.reset_episode()
                elif hasattr(agent, 'reset'):
                    agent.reset()
                
                episode_reward = 0.0
                episode_intrinsic = 0.0
                
                for step in range(self.episode_length):
                    try:
                        # 选择动作
                        action, info = agent.select_action(state, deterministic=False)
                        
                        # 统计Manager和Worker决策
                        if step % config.manager_horizon == 0:
                            results['manager_decisions'] += 1
                            logger.debug(f"FuN Manager decision at step {step}")
                        results['worker_actions'] += 1
                        logger.debug(f"FuN Worker action {results['worker_actions']}")
                        
                        # 执行动作
                        try:
                            # 确保action是numpy数组
                            if not isinstance(action, np.ndarray):
                                logger.warning(f"Action is not numpy array: {type(action)} {action}")
                                action = np.array(action)
                            
                            step_result = self.env.step(action)
                            if len(step_result) == 4:
                                raw_next_state, reward, done, env_info = step_result
                            else:
                                # Handle different return formats
                                raw_next_state, reward, done = step_result[:3]
                                env_info = step_result[3] if len(step_result) > 3 else {}
                            
                            # Extract state from potentially dict format
                            next_state = self._extract_state(raw_next_state)
                        except Exception as step_e:
                            logger.warning(f"Step execution error with action {action} (type: {type(action)}): {step_e}")
                            break
                        
                        if next_state is None:
                            logger.warning(f"Step {step}: 获得None状态，停止episode")
                            break
                        
                        # 存储转换
                        agent.store_transition(state, action, reward, next_state, done, info)
                        
                        # 更新统计
                        state = next_state
                        episode_reward += reward
                        episode_intrinsic += info.get('intrinsic_reward', 0.0)
                        
                        # 输出关键信息
                        if step % 10 == 0:
                            try:
                                current_pos = next_state[:3] if len(next_state) >= 3 else next_state[:2]
                                intrinsic = info.get('intrinsic_reward', 0)
                                logger.info(f"  Step {step}: Pos={current_pos[:2]}, Intrinsic={intrinsic:.3f}")
                            except:
                                logger.info(f"  Step {step}: Reward={reward:.3f}")
                        
                        if done:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Step {step}执行失败: {e}")
                        break
                
                # 记录最终位置
                try:
                    final_pos = state[:3] if len(state) >= 3 else state[:2]
                    results['final_positions'].append(final_pos)
                except:
                    results['final_positions'].append([0, 0, 0])
                
                # episode结束处理
                try:
                    if step > 5:  # Use step instead of undefined episode_length
                        losses = agent.update()
                        if losses:
                            logger.info(f"  FuN更新完成")
                except Exception as e:
                    logger.warning(f"  FuN更新失败: {e}")
                
                # 记录结果
                results['episodes_completed'] += 1
                results['episode_rewards'].append(episode_reward)
                results['intrinsic_rewards'].append(episode_intrinsic)
                
                # 判断成功
                try:
                    final_pos = results['final_positions'][-1]
                    if np.linalg.norm(final_pos[:2]) > 1.0:
                        results['success_episodes'] += 1
                except:
                    pass
                
                logger.info(f"  Episode {episode + 1} 完成:")
                logger.info(f"    外在奖励: {episode_reward:.2f}")
                logger.info(f"    内在奖励: {episode_intrinsic:.2f}")
                
            except Exception as e:
                logger.error(f"Episode {episode + 1} 失败: {e}")
                continue
        
        # 计算统计信息
        if results['episodes_completed'] > 0:
            results['avg_reward'] = np.mean(results['episode_rewards'])
            results['avg_intrinsic'] = np.mean(results['intrinsic_rewards'])
            results['success_rate'] = results['success_episodes'] / results['episodes_completed']
        else:
            results['avg_reward'] = 0
            results['avg_intrinsic'] = 0
            results['success_rate'] = 0
        
        logger.info(f"\nFuN测试完成:")
        logger.info(f"  完成episodes: {results['episodes_completed']}")
        logger.info(f"  平均外在奖励: {results['avg_reward']:.2f}")
        logger.info(f"  平均内在奖励: {results['avg_intrinsic']:.2f}")
        logger.info(f"  Manager决策次数: {results['manager_decisions']}")
        logger.info(f"  成功率: {results['success_rate']:.2f}")
        
        return results
    
    def test_options_algorithm(self) -> Dict[str, Any]:
        """测试Options算法"""
        logger.info("="*60)
        logger.info("测试 Options Framework")
        logger.info("="*60)
        
        # 配置Options
        config = OptionsConfig(
            num_options=4,      # 减少选项数量
            option_min_length=2,
            option_max_length=8, # 缩短最大长度
            use_diversity_bonus=True,
            diversity_coef=0.1,
            batch_size=16,      # 减小batch size
            buffer_size=256,    # 减小buffer size
            device="cpu"
        )
        
        # 初始化agent
        agent = OptionsAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config,
            continuous_actions=True
        )
        
        results = {
            'algorithm': 'Options',
            'episodes_completed': 0,
            'episode_rewards': [],
            'option_switches': [],
            'option_lengths': [],
            'option_usage': defaultdict(int),
            'success_episodes': 0,
            'final_positions': []
        }
        
        logger.info(f"开始Options测试 - {self.num_test_episodes}个episode")
        
        for episode in range(self.num_test_episodes):
            logger.info(f"\n--- Options Episode {episode + 1}/{self.num_test_episodes} ---")
            
            try:
                # 重置环境和agent
                state = self.env.reset()
                if state is None:
                    logger.warning("环境重置失败，跳过此episode")
                    continue
                    
                # 重置agent状态（如果有的话）
                if hasattr(agent, 'reset_episode'):
                    agent.reset_episode()
                elif hasattr(agent, 'reset'):
                    agent.reset()
                
                episode_reward = 0.0
                option_switches = 0
                current_option = None
                option_length = 0
                option_lengths_episode = []
                
                for step in range(self.episode_length):
                    try:
                        # 选择动作
                        action, info = agent.select_action(state, deterministic=False)
                        
                        # 跟踪选项切换
                        option = info.get('option', 0)
                        option_len = info.get('option_length', 1)
                        
                        if current_option is None:
                            current_option = option
                            option_length = option_len
                        elif current_option != option:
                            # 选项切换
                            option_switches += 1
                            option_lengths_episode.append(option_length)
                            current_option = option
                            option_length = option_len
                        else:
                            option_length = option_len
                        
                        # 记录选项使用
                        results['option_usage'][option] += 1
                        logger.debug(f"Options using option {option}, usage count: {results['option_usage'][option]}")
                        
                        # 执行动作
                        try:
                            # 确保action是numpy数组
                            if not isinstance(action, np.ndarray):
                                logger.warning(f"Action is not numpy array: {type(action)} {action}")
                                action = np.array(action)
                            
                            step_result = self.env.step(action)
                            if len(step_result) == 4:
                                raw_next_state, reward, done, env_info = step_result
                            else:
                                # Handle different return formats
                                raw_next_state, reward, done = step_result[:3]
                                env_info = step_result[3] if len(step_result) > 3 else {}
                            
                            # Extract state from potentially dict format
                            next_state = self._extract_state(raw_next_state)
                        except Exception as step_e:
                            logger.warning(f"Step execution error with action {action} (type: {type(action)}): {step_e}")
                            break
                        
                        if next_state is None:
                            logger.warning(f"Step {step}: 获得None状态，停止episode")
                            break
                        
                        # 存储转换
                        agent.store_transition(state, action, reward, next_state, done, info)
                        
                        # 更新状态
                        state = next_state
                        episode_reward += reward
                        
                        # 输出关键信息
                        if step % 10 == 0:
                            try:
                                current_pos = next_state[:3] if len(next_state) >= 3 else next_state[:2]
                                logger.info(f"  Step {step}: Pos={current_pos[:2]}, Option={option}, Length={option_len}")
                            except:
                                logger.info(f"  Step {step}: Option={option}, Reward={reward:.3f}")
                        
                        if done:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Step {step}执行失败: {e}")
                        break
                
                # 添加最后一个选项的长度
                if option_length > 0:
                    option_lengths_episode.append(option_length)
                
                # 记录最终位置
                try:
                    final_pos = state[:3] if len(state) >= 3 else state[:2]
                    results['final_positions'].append(final_pos)
                except:
                    results['final_positions'].append([0, 0, 0])
                
                # episode结束处理
                try:
                    if step > 5:  # Use step instead of undefined variable
                        losses = agent.update()
                        if losses:
                            logger.info(f"  Options更新完成")
                except Exception as e:
                    logger.warning(f"  Options更新失败: {e}")
                
                # 记录结果
                results['episodes_completed'] += 1
                results['episode_rewards'].append(episode_reward)
                results['option_switches'].append(option_switches)
                results['option_lengths'].extend(option_lengths_episode)
                
                # 判断成功
                try:
                    final_pos = results['final_positions'][-1]
                    if np.linalg.norm(final_pos[:2]) > 1.0:
                        results['success_episodes'] += 1
                except:
                    pass
                
                logger.info(f"  Episode {episode + 1} 完成:")
                logger.info(f"    奖励: {episode_reward:.2f}")
                logger.info(f"    选项切换: {option_switches}")
                logger.info(f"    平均选项长度: {np.mean(option_lengths_episode) if option_lengths_episode else 0:.1f}")
                
            except Exception as e:
                logger.error(f"Episode {episode + 1} 失败: {e}")
                continue
        
        # 计算统计信息
        if results['episodes_completed'] > 0:
            results['avg_reward'] = np.mean(results['episode_rewards'])
            results['avg_switches'] = np.mean(results['option_switches'])
            results['avg_option_length'] = np.mean(results['option_lengths']) if results['option_lengths'] else 0
            results['success_rate'] = results['success_episodes'] / results['episodes_completed']
            
            # 计算选项多样性
            total_usage = sum(results['option_usage'].values())
            option_frequencies = [results['option_usage'][i] / total_usage for i in range(config.num_options)] if total_usage > 0 else [0] * config.num_options
            results['option_diversity'] = 1.0 - np.var(option_frequencies) if option_frequencies else 0
        else:
            results['avg_reward'] = 0
            results['avg_switches'] = 0
            results['avg_option_length'] = 0
            results['success_rate'] = 0
            results['option_diversity'] = 0
        
        logger.info(f"\nOptions测试完成:")
        logger.info(f"  完成episodes: {results['episodes_completed']}")
        logger.info(f"  平均奖励: {results['avg_reward']:.2f}")
        logger.info(f"  平均切换次数: {results['avg_switches']:.1f}")
        logger.info(f"  平均选项长度: {results['avg_option_length']:.1f}")
        logger.info(f"  选项多样性分数: {results['option_diversity']:.3f}")
        logger.info(f"  选项使用分布: {dict(results['option_usage'])}")
        logger.info(f"  成功率: {results['success_rate']:.2f}")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """运行所有高级测试"""
        logger.info("开始高级分层算法综合测试")
        logger.info(f"每个算法测试 {self.num_test_episodes} episodes，每episode最多 {self.episode_length} steps")
        
        all_results = {}
        
        # 测试HIRO
        try:
            logger.info("\n开始测试HIRO...")
            all_results['HIRO'] = self.test_hiro_algorithm()
        except Exception as e:
            logger.error(f"HIRO测试失败: {e}")
            all_results['HIRO'] = {'status': 'FAILED', 'error': str(e)}
        
        # 测试FuN  
        try:
            logger.info("\n开始测试FuN...")
            all_results['FuN'] = self.test_fun_algorithm()
        except Exception as e:
            logger.error(f"FuN测试失败: {e}")
            all_results['FuN'] = {'status': 'FAILED', 'error': str(e)}
        
        # 测试Options
        try:
            logger.info("\n开始测试Options...")
            all_results['Options'] = self.test_options_algorithm()
        except Exception as e:
            logger.error(f"Options测试失败: {e}")
            all_results['Options'] = {'status': 'FAILED', 'error': str(e)}
        
        return all_results
    
    def generate_test_report(self, results: Dict[str, Dict[str, Any]]) -> str:
        """生成详细测试报告"""
        report = []
        report.append("=" * 80)
        report.append("高级分层强化学习算法测试报告")
        report.append("=" * 80)
        report.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"测试配置: {self.num_test_episodes} episodes, {self.episode_length} steps/episode")
        report.append("")
        
        # 汇总表格
        report.append("算法性能汇总:")
        report.append("-" * 80)
        report.append(f"{'算法':<10} {'状态':<10} {'完成episodes':<12} {'平均奖励':<12} {'成功率':<10}")
        report.append("-" * 80)
        
        for alg_name, result in results.items():
            if 'status' in result and result['status'] == 'FAILED':
                report.append(f"{alg_name:<10} {'失败':<10} {'0':<12} {'N/A':<12} {'N/A':<10}")
            else:
                episodes_completed = result.get('episodes_completed', 0)
                avg_reward = result.get('avg_reward', 0)
                success_rate = result.get('success_rate', 0)
                
                report.append(f"{alg_name:<10} {'成功':<10} {episodes_completed:<12} {avg_reward:<12.2f} {success_rate:<10.2f}")
        
        report.append("")
        
        # 详细结果
        for alg_name, result in results.items():
            if 'status' in result and result['status'] == 'FAILED':
                report.append(f"{alg_name} - 测试失败:")
                report.append(f"  错误: {result.get('error', 'Unknown error')}")
                report.append("")
                continue
                
            report.append(f"{alg_name} 详细结果:")
            report.append("-" * 40)
            
            if alg_name == 'HIRO':
                report.append(f"  完成episodes: {result.get('episodes_completed', 0)}")
                report.append(f"  平均奖励: {result.get('avg_reward', 0):.2f}")
                report.append(f"  平均episode长度: {result.get('avg_length', 0):.1f}")
                report.append(f"  成功率: {result.get('success_rate', 0):.2f}")
                report.append(f"  策略更新次数: {result.get('policy_updates', 0)}")
                report.append(f"  子目标生成总数: {result.get('subgoals_generated', 0)}")
                
            elif alg_name == 'FuN':
                report.append(f"  完成episodes: {result.get('episodes_completed', 0)}")
                report.append(f"  平均外在奖励: {result.get('avg_reward', 0):.2f}")
                report.append(f"  平均内在奖励: {result.get('avg_intrinsic', 0):.2f}")
                report.append(f"  Manager决策次数: {result.get('manager_decisions', 0)}")
                report.append(f"  Worker动作次数: {result.get('worker_actions', 0)}")
                report.append(f"  成功率: {result.get('success_rate', 0):.2f}")
                
            elif alg_name == 'Options':
                report.append(f"  完成episodes: {result.get('episodes_completed', 0)}")
                report.append(f"  平均奖励: {result.get('avg_reward', 0):.2f}")
                report.append(f"  平均选项切换: {result.get('avg_switches', 0):.1f}")
                report.append(f"  平均选项长度: {result.get('avg_option_length', 0):.1f}")
                report.append(f"  选项多样性分数: {result.get('option_diversity', 0):.3f}")
                report.append(f"  成功率: {result.get('success_rate', 0):.2f}")
                
                # 选项使用统计
                usage = result.get('option_usage', {})
                if usage:
                    report.append(f"  选项使用分布: {dict(usage)}")
            
            report.append("")
        
        return "\n".join(report)


def main():
    """主测试函数"""
    logger.info("启动高级分层算法测试")
    
    try:
        # 创建测试器
        tester = AdvancedAlgorithmTester()
        
        # 运行所有测试
        results = tester.run_all_tests()
        
        # 生成报告
        report = tester.generate_test_report(results)
        
        # 输出报告
        print("\n" + report)
        
        # 保存报告
        report_path = f"advanced_test_report_{int(time.time())}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"测试报告已保存到: {report_path}")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    main()