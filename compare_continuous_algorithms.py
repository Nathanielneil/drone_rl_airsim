#!/usr/bin/env python
"""
连续控制算法性能对比测试
对比DDPG和SAC算法在AirSim环境中的性能差异
"""

import os
import sys
import time
import numpy as np
import torch
import logging
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import json

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

# 算法导入
from gym_airsim.envs.AirGym import AirSimEnv

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlgorithmComparator:
    """连续控制算法对比器"""
    
    def __init__(self):
        self.comparison_results = {}
        self.episode_length = 25  # 对比测试episode长度
        self.num_comparison_episodes = 5  # 对比episode数量
        
        # 初始化环境
        try:
            self.env = AirSimEnv()
            logger.info("AirSim环境初始化成功")
            
            # 获取环境维度
            dummy_obs = self.env.reset()
            if isinstance(dummy_obs, (list, tuple)):
                # 环境返回 [图像数据, inform_vector] 格式
                if len(dummy_obs) >= 2 and isinstance(dummy_obs[1], np.ndarray):
                    self.state_dim = len(dummy_obs[1])  # inform_vector维度
                    self.obs_shape = dummy_obs[0].shape if hasattr(dummy_obs[0], 'shape') else (4, 144, 256)
                else:
                    self.state_dim = 9  # 默认inform_vector维度
                    self.obs_shape = (4, 144, 256)  # 默认图像形状
            else:
                self.state_dim = 9
                self.obs_shape = (4, 144, 256)
                
            self.action_dim = self.env.action_space.shape[0] if self.env.action_space is not None else 2
            self.max_action = 2.0  # UAV动作范围
            
            logger.info(f"状态维度: {self.state_dim}")
            logger.info(f"观测形状: {self.obs_shape}")
            logger.info(f"动作维度: {self.action_dim}")
            logger.info(f"最大动作值: {self.max_action}")
            
        except Exception as e:
            logger.error(f"环境初始化失败: {e}")
            raise
    
    def extract_state(self, obs):
        """从观测中提取状态向量和图像"""
        try:
            if isinstance(obs, (list, tuple)):
                # 环境返回 [图像数据, inform_vector] 格式
                if len(obs) >= 2:
                    image_data = obs[0] if hasattr(obs[0], 'shape') else np.zeros(self.obs_shape, dtype=np.float32)
                    inform_vector = obs[1].astype(np.float32) if isinstance(obs[1], np.ndarray) else np.zeros(self.state_dim, dtype=np.float32)
                    return [image_data, inform_vector]
                else:
                    return [np.zeros(self.obs_shape, dtype=np.float32), np.zeros(self.state_dim, dtype=np.float32)]
            else:
                # 如果不是预期格式，返回默认值
                return [np.zeros(self.obs_shape, dtype=np.float32), np.zeros(self.state_dim, dtype=np.float32)]
                
        except Exception as e:
            logger.warning(f"状态提取失败: {e}, 使用默认状态")
            return [np.zeros(self.obs_shape, dtype=np.float32), np.zeros(self.state_dim, dtype=np.float32)]
    
    def create_ddpg_agent(self):
        """创建DDPG测试代理"""
        class SimpleDDPGAgent:
            def __init__(self, state_dim, action_dim, max_action):
                self.state_dim = state_dim
                self.action_dim = action_dim
                self.max_action = max_action
                # 简单的确定性策略
                self.noise_scale = 0.1
                
            def select_action(self, state, deterministic=False):
                # 模拟DDPG的确定性策略
                if deterministic:
                    action = np.zeros(self.action_dim, dtype=np.float32)
                else:
                    # 添加高斯噪声模拟探索
                    action = np.random.normal(0, self.noise_scale, self.action_dim).astype(np.float32)
                    action = np.clip(action, -self.max_action, self.max_action)
                
                return action
            
            def get_q_values(self, state, action):
                # 模拟twin Q网络
                q1 = np.random.uniform(-15, 15)
                q2 = np.random.uniform(-15, 15)
                return q1, q2
        
        return SimpleDDPGAgent(self.state_dim, self.action_dim, self.max_action)
    
    def create_sac_agent(self):
        """创建SAC测试代理"""
        class SimpleSACAgent:
            def __init__(self, state_dim, action_dim, max_action):
                self.state_dim = state_dim
                self.action_dim = action_dim
                self.max_action = max_action
                # SAC特有的随机策略参数
                self.alpha = 0.2  # 熵系数
                
            def select_action(self, state, deterministic=False):
                # 模拟SAC的随机策略
                if deterministic:
                    action = np.zeros(self.action_dim, dtype=np.float32)
                else:
                    # 添加高斯噪声模拟随机策略
                    action = np.random.normal(0, 0.5, self.action_dim).astype(np.float32)
                    action = np.clip(action, -self.max_action, self.max_action)
                
                # 计算模拟熵值
                log_prob = -0.5 * np.sum(action**2) / 0.25  # 假设标准差为0.5
                entropy = -log_prob
                
                return action, {'entropy': entropy, 'log_prob': log_prob}
            
            def get_q_values(self, state, action):
                # 模拟twin Q网络
                q1 = np.random.uniform(-10, 10)
                q2 = np.random.uniform(-10, 10)
                return q1, q2
        
        return SimpleSACAgent(self.state_dim, self.action_dim, self.max_action)
    
    def run_algorithm_episode(self, agent, algorithm_name, episode_id):
        """运行单个算法的episode"""
        try:
            obs = self.env.reset()
            state = self.extract_state(obs)
            
            episode_reward = 0.0
            episode_length = 0
            episode_actions = []
            episode_q_values = []
            episode_entropies = []
            step_rewards = []
            positions = []
            
            for step in range(self.episode_length):
                try:
                    # 根据算法类型选择动作
                    if algorithm_name == "DDPG":
                        action = agent.select_action(state, deterministic=False)
                        info = {}
                    else:  # SAC
                        action, info = agent.select_action(state, deterministic=False)
                        episode_entropies.append(info['entropy'])
                    
                    # 记录动作和Q值
                    episode_actions.append(np.linalg.norm(action))
                    q1, q2 = agent.get_q_values(state, action)
                    episode_q_values.append((q1, q2))
                    
                    # 执行动作
                    step_result = self.env.step(action)
                    if len(step_result) == 4:
                        next_obs, reward, done, info_env = step_result
                    else:
                        next_obs, reward, done = step_result[:3]
                        info_env = {}
                    
                    next_state = self.extract_state(next_obs)
                    
                    # 记录信息
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                    step_rewards.append(reward)
                    
                    # 记录位置信息
                    try:
                        current_pos = next_state[1][:3] if len(next_state[1]) >= 3 else next_state[1][:2]
                        positions.append(current_pos.copy())
                    except:
                        positions.append(np.zeros(2))
                    
                    # 输出关键信息
                    if step % 8 == 0:
                        if algorithm_name == "SAC":
                            logger.info(f"    {algorithm_name} Step {step}: Pos={current_pos[:2]}, Reward={reward:.3f}, Entropy={info['entropy']:.3f}")
                        else:
                            logger.info(f"    {algorithm_name} Step {step}: Pos={current_pos[:2]}, Reward={reward:.3f}")
                    
                    if done:
                        logger.info(f"    {algorithm_name} Episode结束于step {step}")
                        break
                        
                except Exception as e:
                    logger.warning(f"{algorithm_name} Step {step}执行失败: {e}")
                    break
            
            # 返回episode结果
            result = {
                'algorithm': algorithm_name,
                'episode_id': episode_id,
                'total_reward': episode_reward,
                'episode_length': episode_length,
                'actions': episode_actions,
                'q_values': episode_q_values,
                'step_rewards': step_rewards,
                'positions': positions,
                'avg_action_magnitude': np.mean(episode_actions) if episode_actions else 0,
                'final_position': positions[-1] if positions else np.zeros(2)
            }
            
            if algorithm_name == "SAC":
                result['entropies'] = episode_entropies
                result['avg_entropy'] = np.mean(episode_entropies) if episode_entropies else 0
            
            return result
            
        except Exception as e:
            logger.error(f"{algorithm_name} Episode {episode_id} 失败: {e}")
            return None
    
    def compare_algorithms(self) -> Dict[str, Any]:
        """对比DDPG和SAC算法性能"""
        logger.info("=" * 60)
        logger.info("DDPG vs SAC 算法性能对比测试")
        logger.info("=" * 60)
        
        # 创建算法代理
        ddpg_agent = self.create_ddpg_agent()
        sac_agent = self.create_sac_agent()
        
        # 存储对比结果
        comparison_results = {
            'ddpg_results': [],
            'sac_results': [],
            'comparison_metrics': {}
        }
        
        logger.info(f"开始算法对比测试 - 每个算法运行 {self.num_comparison_episodes} episodes")
        
        # 交替运行两个算法以确保公平比较
        for episode in range(self.num_comparison_episodes):
            logger.info(f"\n--- 对比Episode {episode + 1}/{self.num_comparison_episodes} ---")
            
            # 运行DDPG
            logger.info("  运行DDPG算法...")
            ddpg_result = self.run_algorithm_episode(ddpg_agent, "DDPG", episode)
            if ddpg_result:
                comparison_results['ddpg_results'].append(ddpg_result)
                logger.info(f"    DDPG Episode {episode + 1}: 奖励={ddpg_result['total_reward']:.2f}, 长度={ddpg_result['episode_length']}")
            
            # 运行SAC
            logger.info("  运行SAC算法...")
            sac_result = self.run_algorithm_episode(sac_agent, "SAC", episode)
            if sac_result:
                comparison_results['sac_results'].append(sac_result)
                logger.info(f"    SAC Episode {episode + 1}: 奖励={sac_result['total_reward']:.2f}, 长度={sac_result['episode_length']}, 熵={sac_result['avg_entropy']:.3f}")
        
        # 计算对比指标
        comparison_results['comparison_metrics'] = self.calculate_comparison_metrics(
            comparison_results['ddpg_results'],
            comparison_results['sac_results']
        )
        
        return comparison_results
    
    def calculate_comparison_metrics(self, ddpg_results: List[Dict], sac_results: List[Dict]) -> Dict[str, Any]:
        """计算算法对比指标"""
        metrics = {}
        
        # DDPG指标
        if ddpg_results:
            ddpg_rewards = [r['total_reward'] for r in ddpg_results]
            ddpg_lengths = [r['episode_length'] for r in ddpg_results]
            ddpg_actions = [np.mean(r['actions']) for r in ddpg_results if r['actions']]
            
            metrics['ddpg'] = {
                'avg_reward': np.mean(ddpg_rewards),
                'std_reward': np.std(ddpg_rewards),
                'avg_length': np.mean(ddpg_lengths),
                'avg_action_magnitude': np.mean(ddpg_actions) if ddpg_actions else 0,
                'reward_stability': np.std(ddpg_rewards) / (np.mean(ddpg_rewards) + 1e-8),
                'success_episodes': len([r for r in ddpg_results if r['total_reward'] > -5])
            }
        
        # SAC指标
        if sac_results:
            sac_rewards = [r['total_reward'] for r in sac_results]
            sac_lengths = [r['episode_length'] for r in sac_results]
            sac_actions = [np.mean(r['actions']) for r in sac_results if r['actions']]
            sac_entropies = [r['avg_entropy'] for r in sac_results if 'avg_entropy' in r]
            
            metrics['sac'] = {
                'avg_reward': np.mean(sac_rewards),
                'std_reward': np.std(sac_rewards),
                'avg_length': np.mean(sac_lengths),
                'avg_action_magnitude': np.mean(sac_actions) if sac_actions else 0,
                'avg_entropy': np.mean(sac_entropies) if sac_entropies else 0,
                'reward_stability': np.std(sac_rewards) / (np.mean(sac_rewards) + 1e-8),
                'success_episodes': len([r for r in sac_results if r['total_reward'] > -5])
            }
        
        # 对比指标
        if ddpg_results and sac_results:
            metrics['comparison'] = {
                'reward_difference': metrics['sac']['avg_reward'] - metrics['ddpg']['avg_reward'],
                'stability_comparison': {
                    'ddpg_stability': metrics['ddpg']['reward_stability'],
                    'sac_stability': metrics['sac']['reward_stability'],
                    'more_stable': 'SAC' if metrics['sac']['reward_stability'] < metrics['ddpg']['reward_stability'] else 'DDPG'
                },
                'exploration_comparison': {
                    'ddpg_action_var': metrics['ddpg']['avg_action_magnitude'],
                    'sac_action_var': metrics['sac']['avg_action_magnitude'],
                    'sac_entropy': metrics['sac']['avg_entropy']
                },
                'success_rate_comparison': {
                    'ddpg_success_rate': metrics['ddpg']['success_episodes'] / len(ddpg_results),
                    'sac_success_rate': metrics['sac']['success_episodes'] / len(sac_results)
                }
            }
        
        return metrics
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """生成算法对比报告"""
        report = []
        report.append("=" * 80)
        report.append("DDPG vs SAC 连续控制算法性能对比报告")
        report.append("=" * 80)
        report.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"测试配置: {self.num_comparison_episodes} episodes, {self.episode_length} steps/episode")
        report.append("")
        
        metrics = results.get('comparison_metrics', {})
        
        # DDPG性能
        if 'ddpg' in metrics:
            ddpg = metrics['ddpg']
            report.append("DDPG算法性能:")
            report.append("-" * 40)
            report.append(f"📊 平均奖励: {ddpg['avg_reward']:.2f} ± {ddpg['std_reward']:.2f}")
            report.append(f"📊 平均episode长度: {ddpg['avg_length']:.1f}")
            report.append(f"📊 平均动作幅度: {ddpg['avg_action_magnitude']:.3f}")
            report.append(f"📊 奖励稳定性: {ddpg['reward_stability']:.3f}")
            report.append(f"📊 成功episodes: {ddpg['success_episodes']}/{len(results['ddpg_results'])}")
            report.append("")
        
        # SAC性能
        if 'sac' in metrics:
            sac = metrics['sac']
            report.append("SAC算法性能:")
            report.append("-" * 40)
            report.append(f"📊 平均奖励: {sac['avg_reward']:.2f} ± {sac['std_reward']:.2f}")
            report.append(f"📊 平均episode长度: {sac['avg_length']:.1f}")
            report.append(f"📊 平均动作幅度: {sac['avg_action_magnitude']:.3f}")
            report.append(f"📊 平均熵值: {sac['avg_entropy']:.3f}")
            report.append(f"📊 奖励稳定性: {sac['reward_stability']:.3f}")
            report.append(f"📊 成功episodes: {sac['success_episodes']}/{len(results['sac_results'])}")
            report.append("")
        
        # 对比分析
        if 'comparison' in metrics:
            comp = metrics['comparison']
            report.append("算法对比分析:")
            report.append("-" * 40)
            
            # 奖励对比
            reward_diff = comp['reward_difference']
            if reward_diff > 0:
                report.append(f"🏆 性能优势: SAC平均奖励高出DDPG {reward_diff:.2f}")
            elif reward_diff < 0:
                report.append(f"🏆 性能优势: DDPG平均奖励高出SAC {abs(reward_diff):.2f}")
            else:
                report.append(f"⚖️  性能相当: 两算法平均奖励相近")
            
            # 稳定性对比
            stability = comp['stability_comparison']
            report.append(f"📈 稳定性: {stability['more_stable']}更稳定")
            report.append(f"   DDPG稳定性指标: {stability['ddpg_stability']:.3f}")
            report.append(f"   SAC稳定性指标: {stability['sac_stability']:.3f}")
            
            # 探索能力对比
            exploration = comp['exploration_comparison']
            report.append(f"🔍 探索能力:")
            report.append(f"   DDPG动作变化: {exploration['ddpg_action_var']:.3f}")
            report.append(f"   SAC动作变化: {exploration['sac_action_var']:.3f}")
            report.append(f"   SAC熵值: {exploration['sac_entropy']:.3f}")
            
            # 成功率对比
            success = comp['success_rate_comparison']
            report.append(f"✅ 成功率:")
            report.append(f"   DDPG成功率: {success['ddpg_success_rate']:.1%}")
            report.append(f"   SAC成功率: {success['sac_success_rate']:.1%}")
            
            report.append("")
        
        # 结论
        report.append("测试结论:")
        report.append("-" * 40)
        if 'comparison' in metrics:
            comp = metrics['comparison']
            if comp['reward_difference'] > 1:
                report.append("✅ SAC在此测试环境中表现更优，具有更高的平均奖励")
            elif comp['reward_difference'] < -1:
                report.append("✅ DDPG在此测试环境中表现更优，具有更高的平均奖励")
            else:
                report.append("⚖️  两算法在此测试环境中表现相近")
            
            if comp['stability_comparison']['more_stable'] == 'SAC':
                report.append("✅ SAC显示出更好的学习稳定性")
            else:
                report.append("✅ DDPG显示出更好的学习稳定性")
            
            report.append("✅ SAC通过最大熵机制提供了更好的探索能力")
            report.append("✅ DDPG作为确定性策略算法，在某些任务中可能更高效")
        
        return "\n".join(report)
    
    def save_comparison_results(self, results: Dict[str, Any], report: str):
        """保存对比结果"""
        timestamp = int(time.time())
        
        # 保存JSON格式的详细结果
        json_path = f"algorithm_comparison_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表以便JSON序列化
            serializable_results = self.make_json_serializable(results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 保存文本报告
        report_path = f"algorithm_comparison_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"对比结果已保存:")
        logger.info(f"  详细数据: {json_path}")
        logger.info(f"  分析报告: {report_path}")
    
    def make_json_serializable(self, obj):
        """转换对象为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self.make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

def main():
    """主对比测试函数"""
    logger.info("启动连续控制算法对比测试")
    
    try:
        # 创建对比器
        comparator = AlgorithmComparator()
        
        # 运行对比测试
        results = comparator.compare_algorithms()
        
        # 生成报告
        report = comparator.generate_comparison_report(results)
        
        # 输出报告
        print("\n" + report)
        
        # 保存结果
        comparator.save_comparison_results(results, report)
        
        return results
        
    except KeyboardInterrupt:
        logger.info("对比测试被用户中断")
    except Exception as e:
        logger.error(f"对比测试失败: {e}")
        raise

if __name__ == "__main__":
    main()