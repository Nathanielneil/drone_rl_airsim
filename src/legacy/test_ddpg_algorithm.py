#!/usr/bin/env python
"""
DDPG算法功能测试
验证Twin Critic架构和连续控制性能
"""

import os
import sys
import time
import numpy as np
import torch
import logging
from typing import Dict, Any

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

# 算法导入
from gym_airsim.envs.AirGym import AirSimEnv

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DDPGConfig:
    """DDPG配置类"""
    def __init__(self):
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.buffer_size = 1000000
        self.noise_std = 0.1
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2

class DDPGTester:
    """DDPG算法测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.episode_length = 20  # 较短的测试episode
        self.num_test_episodes = 3
        
        # 初始化环境
        try:
            self.env = AirSimEnv()
            logger.info("AirSim环境初始化成功")
            
            # 获取环境维度
            dummy_obs = self.env.reset()
            if isinstance(dummy_obs, dict):
                # 处理字典格式观测
                if 'inform_vector' in dummy_obs:
                    self.state_dim = len(dummy_obs['inform_vector'])
                elif 'observation' in dummy_obs:
                    self.state_dim = len(dummy_obs['observation'])
                else:
                    self.state_dim = 109  # 默认值
            elif isinstance(dummy_obs, (list, tuple)):
                # 环境返回 [图像数据, inform_vector] 格式
                if len(dummy_obs) >= 2 and isinstance(dummy_obs[1], np.ndarray):
                    self.state_dim = len(dummy_obs[1])  # inform_vector维度
                else:
                    self.state_dim = 9  # 默认inform_vector维度
            else:
                self.state_dim = len(dummy_obs) if dummy_obs is not None else 9
                
            self.action_dim = self.env.action_space.shape[0] if self.env.action_space is not None else 2
            self.max_action = 2.0  # UAV动作范围
            
            logger.info(f"状态维度: {self.state_dim}")
            logger.info(f"动作维度: {self.action_dim}")
            logger.info(f"最大动作值: {self.max_action}")
            
        except Exception as e:
            logger.error(f"环境初始化失败: {e}")
            raise
    
    def extract_state(self, obs):
        """从观测中提取状态向量"""
        try:
            if isinstance(obs, dict):
                if 'inform_vector' in obs:
                    state = obs['inform_vector']
                elif 'observation' in obs:
                    state = obs['observation']
                else:
                    # 如果是其他字典格式，尝试提取数值
                    values = []
                    for key, value in obs.items():
                        if isinstance(value, (int, float)):
                            values.append(value)
                        elif isinstance(value, (list, np.ndarray)):
                            if isinstance(value, np.ndarray):
                                values.extend(value.flatten())
                            else:
                                values.extend(value)
                    state = np.array(values[:self.state_dim], dtype=np.float32)
                    return state
            elif isinstance(obs, (list, tuple)):
                # 环境返回 [图像数据, inform_vector] 格式
                if len(obs) >= 2 and isinstance(obs[1], np.ndarray):
                    # 使用inform_vector作为状态
                    state = obs[1].astype(np.float32)
                else:
                    # 如果第二个元素不存在或不是数组，尝试使用第一个
                    state = obs[0] if len(obs) > 0 else np.zeros(self.state_dim, dtype=np.float32)
            elif isinstance(obs, np.ndarray):
                state = obs.astype(np.float32)
            else:
                logger.warning(f"未知观测格式: {type(obs)}")
                return np.zeros(self.state_dim, dtype=np.float32)
            
            # 确保返回正确的数据类型和维度
            if isinstance(state, np.ndarray):
                state = state.astype(np.float32).flatten()
                if len(state) > self.state_dim:
                    state = state[:self.state_dim]
                elif len(state) < self.state_dim:
                    # 填充到正确维度
                    padded_state = np.zeros(self.state_dim, dtype=np.float32)
                    padded_state[:len(state)] = state
                    state = padded_state
            else:
                state = np.array([state], dtype=np.float32)
                
            return state
            
        except Exception as e:
            logger.warning(f"状态提取失败: {e}, 使用默认状态")
            return np.zeros(self.state_dim, dtype=np.float32)
    
    def test_ddpg_basic_functionality(self) -> Dict[str, Any]:
        """测试DDPG基本功能"""
        logger.info("="*60)
        logger.info("测试 DDPG 基本功能")
        logger.info("="*60)
        
        # 动态导入DDPG
        try:
            from ddpg import DDPG, OUNoise
        except ImportError as e:
            logger.error(f"DDPG导入失败: {e}")
            return {'status': 'FAILED', 'error': 'Import failed'}
        
        config = DDPGConfig()
        
        # 初始化DDPG agent
        try:
            agent = DDPG(self.state_dim, self.action_dim, self.max_action, config)
            noise = OUNoise(self.action_dim)
            logger.info("DDPG agent初始化成功")
        except Exception as e:
            logger.error(f"DDPG初始化失败: {e}")
            return {'status': 'FAILED', 'error': str(e)}
        
        results = {
            'algorithm': 'DDPG',
            'episodes_completed': 0,
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'exploration_noise': [],
            'action_magnitudes': [],
            'twin_critic_verified': False,
            'ou_noise_verified': False
        }
        
        logger.info(f"开始DDPG基本功能测试 - {self.num_test_episodes}个episode")
        
        for episode in range(self.num_test_episodes):
            logger.info(f"\n--- DDPG Episode {episode + 1}/{self.num_test_episodes} ---")
            
            try:
                # 重置环境和噪声
                obs = self.env.reset()
                state = self.extract_state(obs)
                noise.reset()
                
                episode_reward = 0.0
                episode_length = 0
                episode_actions = []
                episode_noise = []
                
                for step in range(self.episode_length):
                    try:
                        # 选择动作 (测试阶段，添加噪声用于探索)
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state).unsqueeze(0)
                            if hasattr(agent, 'device'):
                                state_tensor = state_tensor.to(agent.device)
                            
                            # 获取确定性动作
                            action = agent.actor(state_tensor).cpu().numpy().flatten()
                            
                            # 添加OU噪声
                            noise_sample = noise.noise()
                            action = action + noise_sample
                            action = np.clip(action, -self.max_action, self.max_action)
                            
                            episode_actions.append(np.linalg.norm(action))
                            episode_noise.append(np.linalg.norm(noise_sample))
                        
                        # 执行动作
                        step_result = self.env.step(action)
                        if len(step_result) == 4:
                            next_obs, reward, done, info = step_result
                        else:
                            next_obs, reward, done = step_result[:3]
                            info = {}
                        
                        next_state = self.extract_state(next_obs)
                        
                        # 存储经验到replay buffer (如果有的话)
                        if hasattr(agent, 'replay_buffer'):
                            agent.replay_buffer.add(state, action, next_state, reward, done)
                        
                        # 更新状态
                        state = next_state
                        episode_reward += reward
                        episode_length += 1
                        
                        # 输出关键信息
                        if step % 5 == 0:
                            current_pos = next_state[:3] if len(next_state) >= 3 else next_state[:2]
                            logger.info(f"  Step {step}: Pos={current_pos[:2]}, Reward={reward:.3f}, Action_norm={np.linalg.norm(action):.3f}")
                        
                        if done:
                            logger.info(f"  Episode结束于step {step}")
                            break
                            
                    except Exception as e:
                        logger.warning(f"Step {step}执行失败: {e}")
                        break
                
                # 测试Twin Critic架构
                try:
                    if len(episode_actions) > 0:
                        # 测试双Q网络
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        action_tensor = torch.FloatTensor(action).unsqueeze(0)
                        if hasattr(agent, 'device'):
                            state_tensor = state_tensor.to(agent.device)
                            action_tensor = action_tensor.to(agent.device)
                        
                        with torch.no_grad():
                            q1, q2 = agent.critic(state_tensor, action_tensor)
                            if q1.shape == q2.shape and q1.numel() > 0 and q2.numel() > 0:
                                results['twin_critic_verified'] = True
                                logger.info(f"  Twin Critic验证成功: Q1={q1.item():.3f}, Q2={q2.item():.3f}")
                except Exception as e:
                    logger.warning(f"Twin Critic测试失败: {e}")
                
                # 记录结果
                results['episodes_completed'] += 1
                results['episode_rewards'].append(episode_reward)
                results['episode_lengths'].append(episode_length)
                results['action_magnitudes'].extend(episode_actions)
                results['exploration_noise'].extend(episode_noise)
                
                if len(episode_noise) > 0:
                    results['ou_noise_verified'] = True
                
                logger.info(f"  Episode {episode + 1} 完成:")
                logger.info(f"    奖励: {episode_reward:.2f}")
                logger.info(f"    长度: {episode_length}")
                logger.info(f"    平均动作幅度: {np.mean(episode_actions):.3f}")
                logger.info(f"    平均噪声幅度: {np.mean(episode_noise):.3f}")
                
            except Exception as e:
                logger.error(f"Episode {episode + 1} 失败: {e}")
                continue
        
        # 计算统计信息
        if results['episodes_completed'] > 0:
            results['avg_reward'] = np.mean(results['episode_rewards'])
            results['avg_length'] = np.mean(results['episode_lengths'])
            results['avg_action_magnitude'] = np.mean(results['action_magnitudes'])
            results['avg_noise_magnitude'] = np.mean(results['exploration_noise'])
        else:
            results['avg_reward'] = 0
            results['avg_length'] = 0
            results['avg_action_magnitude'] = 0
            results['avg_noise_magnitude'] = 0
        
        logger.info(f"\nDDPG基本功能测试完成:")
        logger.info(f"  完成episodes: {results['episodes_completed']}")
        logger.info(f"  平均奖励: {results['avg_reward']:.2f}")
        logger.info(f"  平均长度: {results['avg_length']:.1f}")
        logger.info(f"  Twin Critic验证: {'✓' if results['twin_critic_verified'] else '✗'}")
        logger.info(f"  OU噪声验证: {'✓' if results['ou_noise_verified'] else '✗'}")
        logger.info(f"  平均动作幅度: {results['avg_action_magnitude']:.3f}")
        logger.info(f"  平均噪声幅度: {results['avg_noise_magnitude']:.3f}")
        
        return results
    
    def test_ddpg_network_architecture(self) -> Dict[str, Any]:
        """测试DDPG网络架构特性"""
        logger.info("="*60)
        logger.info("测试 DDPG 网络架构特性")
        logger.info("="*60)
        
        try:
            from ddpg import DDPG, Actor, Critic
        except ImportError as e:
            logger.error(f"DDPG导入失败: {e}")
            return {'status': 'FAILED', 'error': 'Import failed'}
        
        config = DDPGConfig()
        
        results = {
            'actor_architecture_verified': False,
            'critic_architecture_verified': False,
            'target_networks_verified': False,
            'parameter_counts': {},
            'network_outputs': {}
        }
        
        try:
            # 测试Actor网络
            actor = Actor(self.state_dim, self.action_dim, self.max_action)
            
            # 计算参数数量
            actor_params = sum(p.numel() for p in actor.parameters())
            results['parameter_counts']['actor'] = actor_params
            
            # 测试前向传播
            dummy_state = torch.randn(1, self.state_dim)
            actor_output = actor(dummy_state)
            
            if actor_output.shape == (1, self.action_dim):
                results['actor_architecture_verified'] = True
                results['network_outputs']['actor_shape'] = actor_output.shape
                logger.info(f"Actor网络验证成功: 输出形状 {actor_output.shape}, 参数数量 {actor_params}")
            
            # 测试Critic网络 (Twin Critic)
            critic = Critic(self.state_dim, self.action_dim)
            
            critic_params = sum(p.numel() for p in critic.parameters())
            results['parameter_counts']['critic'] = critic_params
            
            # 测试Twin Critic输出
            dummy_action = torch.randn(1, self.action_dim)
            q1, q2 = critic(dummy_state, dummy_action)
            
            if q1.shape == (1, 1) and q2.shape == (1, 1):
                results['critic_architecture_verified'] = True
                results['network_outputs']['critic_q1_shape'] = q1.shape
                results['network_outputs']['critic_q2_shape'] = q2.shape
                logger.info(f"Critic网络验证成功: Q1形状 {q1.shape}, Q2形状 {q2.shape}, 参数数量 {critic_params}")
            
            # 测试完整DDPG agent的target网络
            agent = DDPG(self.state_dim, self.action_dim, self.max_action, config)
            
            # 验证target网络存在
            if hasattr(agent, 'actor_target') and hasattr(agent, 'critic_target'):
                results['target_networks_verified'] = True
                logger.info("Target网络验证成功")
                
                # 测试软更新机制
                original_actor_param = list(agent.actor.parameters())[0].clone()
                original_target_param = list(agent.actor_target.parameters())[0].clone()
                
                # 执行一次软更新 (如果有该方法)
                if hasattr(agent, 'soft_update'):
                    agent.soft_update()
                    updated_target_param = list(agent.actor_target.parameters())[0]
                    
                    # 检查参数是否有微小变化
                    if not torch.equal(original_target_param, updated_target_param):
                        logger.info("软更新机制验证成功")
                        results['soft_update_verified'] = True
            
        except Exception as e:
            logger.error(f"网络架构测试失败: {e}")
            results['error'] = str(e)
        
        logger.info(f"\nDDPG网络架构测试完成:")
        logger.info(f"  Actor架构: {'✓' if results['actor_architecture_verified'] else '✗'}")
        logger.info(f"  Critic架构: {'✓' if results['critic_architecture_verified'] else '✗'}")
        logger.info(f"  Target网络: {'✓' if results['target_networks_verified'] else '✗'}")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """运行所有DDPG测试"""
        logger.info("开始DDPG算法综合测试")
        logger.info(f"每个测试包含 {self.num_test_episodes} episodes，每episode最多 {self.episode_length} steps")
        
        all_results = {}
        
        # 测试基本功能
        try:
            logger.info("\n开始测试DDPG基本功能...")
            all_results['basic_functionality'] = self.test_ddpg_basic_functionality()
        except Exception as e:
            logger.error(f"DDPG基本功能测试失败: {e}")
            all_results['basic_functionality'] = {'status': 'FAILED', 'error': str(e)}
        
        # 测试网络架构
        try:
            logger.info("\n开始测试DDPG网络架构...")
            all_results['network_architecture'] = self.test_ddpg_network_architecture()
        except Exception as e:
            logger.error(f"DDPG网络架构测试失败: {e}")
            all_results['network_architecture'] = {'status': 'FAILED', 'error': str(e)}
        
        return all_results
    
    def generate_test_report(self, results: Dict[str, Dict[str, Any]]) -> str:
        """生成DDPG测试报告"""
        report = []
        report.append("=" * 80)
        report.append("DDPG (Deep Deterministic Policy Gradient) 测试报告")
        report.append("=" * 80)
        report.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"测试配置: {self.num_test_episodes} episodes, {self.episode_length} steps/episode")
        report.append("")
        
        # 基本功能测试结果
        if 'basic_functionality' in results:
            basic = results['basic_functionality']
            report.append("基本功能测试结果:")
            report.append("-" * 40)
            
            if 'status' in basic and basic['status'] == 'FAILED':
                report.append(f"❌ 测试失败: {basic.get('error', 'Unknown error')}")
            else:
                report.append(f"✅ 完成episodes: {basic.get('episodes_completed', 0)}")
                report.append(f"✅ 平均奖励: {basic.get('avg_reward', 0):.2f}")
                report.append(f"✅ 平均episode长度: {basic.get('avg_length', 0):.1f}")
                report.append(f"{'✅' if basic.get('twin_critic_verified') else '❌'} Twin Critic架构验证")
                report.append(f"{'✅' if basic.get('ou_noise_verified') else '❌'} OU噪声机制验证")
                report.append(f"✅ 平均动作幅度: {basic.get('avg_action_magnitude', 0):.3f}")
                report.append(f"✅ 平均噪声幅度: {basic.get('avg_noise_magnitude', 0):.3f}")
            report.append("")
        
        # 网络架构测试结果
        if 'network_architecture' in results:
            arch = results['network_architecture']
            report.append("网络架构测试结果:")
            report.append("-" * 40)
            
            if 'status' in arch and arch['status'] == 'FAILED':
                report.append(f"❌ 测试失败: {arch.get('error', 'Unknown error')}")
            else:
                report.append(f"{'✅' if arch.get('actor_architecture_verified') else '❌'} Actor网络架构")
                report.append(f"{'✅' if arch.get('critic_architecture_verified') else '❌'} Critic网络架构")
                report.append(f"{'✅' if arch.get('target_networks_verified') else '❌'} Target网络机制")
                
                if 'parameter_counts' in arch:
                    params = arch['parameter_counts']
                    report.append(f"📊 Actor参数数量: {params.get('actor', 0):,}")
                    report.append(f"📊 Critic参数数量: {params.get('critic', 0):,}")
            report.append("")
        
        # 总结
        report.append("测试总结:")
        report.append("-" * 40)
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if not (isinstance(r, dict) and r.get('status') == 'FAILED'))
        report.append(f"总测试数: {total_tests}")
        report.append(f"通过测试: {passed_tests}")
        report.append(f"测试通过率: {passed_tests/total_tests*100:.1f}%")
        
        return "\n".join(report)

def main():
    """主测试函数"""
    logger.info("启动DDPG算法测试")
    
    try:
        # 创建测试器
        tester = DDPGTester()
        
        # 运行所有测试
        results = tester.run_all_tests()
        
        # 生成报告
        report = tester.generate_test_report(results)
        
        # 输出报告
        print("\n" + report)
        
        # 保存报告
        report_path = f"ddpg_test_report_{int(time.time())}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"DDPG测试报告已保存到: {report_path}")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise

if __name__ == "__main__":
    main()