# UAV强化学习算法集合

UAV_RL（Ubuntu20.04-UE4.27-AirSim1.8.1）的最小实现，包含经典主流RL训练无人机导航的核心功能。。

## 支持的算法

### 策略梯度算法
- **PPO (Proximal Policy Optimization)** - `train_ppo.py`
- **A3C (Asynchronous Advantage Actor-Critic)** - `a3c.py`
- **SAC (Soft Actor-Critic)** - `SAC.py` + `eval_SAC.py`

### 价值函数算法  
- **DQN (Deep Q-Network)** - `dqn.py`
- **Prioritized DQN** - `prioritized_dqn.py`
- **Rainbow DQN** - `rainbow.py`

### 演员-评论家算法
- **TD3 (Twin Delayed DDPG)** - `td3.py`
- **DDPG (Deep Deterministic Policy Gradient)** - `baselines/ddpg/`
- **A2C (Advantage Actor-Critic)** - `baselines/a2c/`

### 其他算法
- **TRPO (Trust Region Policy Optimization)** - `baselines/trpo_mpi/`
- **ACER (Actor-Critic with Experience Replay)** - `baselines/acer/`
- **ACKTR (Actor-Critic using Kronecker-factored Trust Region)** - `baselines/acktr/`
- **HER (Hindsight Experience Replay)** - `baselines/her/`
- **GAIL (Generative Adversarial Imitation Learning)** - `baselines/gail/`

## 文件结构
```
drone_rl/
├── train_ppo.py              # PPO训练脚本
├── dqn.py                    # DQN算法实现
├── prioritized_dqn.py        # Prioritized DQN算法实现
├── rainbow.py                # Rainbow DQN算法实现
├── SAC.py                    # SAC算法实现
├── eval_SAC.py               # SAC评估脚本
├── a3c.py                    # A3C算法实现
├── td3.py                    # TD3算法实现
├── config.py                 # 训练配置
├── algorithm/                # PPO算法核心实现
├── DQN/                      # DQN相关支持文件
├── Rainbow/                  # Rainbow DQN支持文件
├── gym_airsim/              # AirSim环境封装
├── utils/                   # 工具函数
├── settings_folder/         # 环境设置
├── baselines/               # OpenAI Baselines算法库
├── common/                  # 通用函数
└── environment_randomization/ # 环境随机化

```

## 使用方法

### 环境准备
1. 确保AirSim环境已启动
2. 修改 `settings_folder/machine_dependent_settings.py` 中的路径配置

### 训练不同算法

#### PPO (推荐入门)
```bash
python train_ppo.py
```

#### DQN系列
```bash
python dqn.py                 # 基础DQN
python prioritized_dqn.py     # 优先经验回放DQN  
python rainbow.py             # Rainbow DQN
```

#### SAC (连续控制)
```bash
python SAC.py                 # SAC训练
python eval_SAC.py            # SAC评估
```

#### A3C (异步训练)
```bash
python a3c.py
```

#### TD3 (DDPG改进版)
```bash
python td3.py
```

#### Baselines算法
```bash
cd baselines
python -m baselines.run --alg=a2c --env=AirGym
python -m baselines.run --alg=ddpg --env=AirGym
python -m baselines.run --alg=trpo_mpi --env=AirGym
```

## 算法选择指南

- **离散动作空间**: DQN, Rainbow DQN, PPO, A3C
- **连续动作空间**: SAC, TD3, DDPG, PPO
- **初学者推荐**: PPO (稳定且易调参)
- **样本效率优先**: SAC, TD3
- **分布式训练**: A3C
- **模仿学习**: GAIL + 专家数据

## 依赖
- torch >= 1.7.0
- tensorboardX
- opencv-python
- numpy
- gym
- airsim
- matplotlib (用于可视化)
- tqdm (用于进度条)
