<div align="center">

# 🚁 UAV Reinforcement Learning Suite

**A comprehensive collection of state-of-the-art reinforcement learning algorithms for unmanned aerial vehicle navigation and control**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)](https://pytorch.org/)
[![AirSim](https://img.shields.io/badge/AirSim-1.8.1-green.svg)](https://microsoft.github.io/AirSim/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04-orange.svg)](https://ubuntu.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Powered by Ubuntu 20.04 • Unreal Engine 4.27 • AirSim 1.8.1*

</div>

---

## 🎯 Project Overview

This repository implements a complete suite of reinforcement learning algorithms specifically designed for UAV autonomous navigation, obstacle avoidance, and target tracking in realistic 3D environments. From classic value-based methods to cutting-edge actor-critic algorithms, this collection provides researchers and developers with battle-tested implementations ready for real-world drone applications.

## ⚡ Algorithm Arsenal

### 🔥 Policy Gradient Methods
| Algorithm | Implementation | Key Features |
|-----------|---------------|--------------|
| **PPO** | `train_ppo.py` | Stable, sample-efficient, beginner-friendly |
| **A3C** | `a3c.py` | Asynchronous training, distributed learning |
| **SAC** | `SAC.py` | Maximum entropy, continuous control expert |

### 🎲 Value-Based Methods
| Algorithm | Implementation | Specialization |
|-----------|---------------|----------------|
| **DQN** | `dqn.py` | Foundation discrete control |
| **Prioritized DQN** | `prioritized_dqn.py` | Enhanced experience replay |
| **Rainbow DQN** | `rainbow.py` | State-of-the-art Q-learning |

### 🎭 Actor-Critic Hybrid
| Algorithm | Implementation | Best For |
|-----------|---------------|----------|
| **TD3** | `td3.py` | Continuous control, reduced overestimation |
| **DDPG** | `baselines/ddpg/` | Deterministic policy gradients |
| **A2C** | `baselines/a2c/` | Synchronized actor-critic |

### 🚀 Advanced Methods
- **TRPO** - Trust region optimization for policy updates
- **ACER** - Off-policy actor-critic with experience replay
- **ACKTR** - Natural gradient optimization
- **HER** - Goal-conditioned reinforcement learning
- **GAIL** - Imitation learning from expert demonstrations

## 🏗️ Architecture

```
drone_rl/
│
├── 🧠 Core Algorithms
│   ├── train_ppo.py              # PPO: The go-to algorithm
│   ├── SAC.py & eval_SAC.py      # SAC: Continuous control master
│   ├── dqn.py                    # DQN: Discrete action foundation
│   ├── prioritized_dqn.py        # Enhanced DQN with prioritized replay
│   ├── rainbow.py                # Rainbow: DQN's ultimate evolution
│   ├── a3c.py                    # A3C: Asynchronous advantage actor-critic
│   └── td3.py                    # TD3: Twin delayed DDPG
│
├── 🎛️ Algorithm Components
│   ├── algorithm/                # PPO implementation details
│   ├── DQN/                      # DQN supporting modules
│   ├── Rainbow/                  # Rainbow DQN components
│   └── utils/                    # Shared utilities and helpers
│
├── 🌍 Environment Integration
│   ├── gym_airsim/              # AirSim-Gym interface
│   ├── environment_randomization/ # Domain randomization
│   └── settings_folder/         # Environment configurations
│
├── 📊 Baselines & Benchmarks
│   └── baselines/               # OpenAI Baselines integration
│       ├── a2c/, ddpg/, her/    # Classic implementations
│       ├── gail/, trpo_mpi/     # Advanced methods
│       └── ppo1/, ppo2/         # PPO variants
│
└── 🔧 Infrastructure
    ├── common/                   # Shared functionality
    ├── config.py                 # Global configuration
    └── start_simulation.py       # Environment launcher
```

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure AirSim environment is running
# Ubuntu 20.04 + Unreal Engine 4.27 + AirSim 1.8.1
```

### Installation
```bash
git clone https://github.com/Nathanielneil/drone_rl_airsim.git
cd drone_rl_airsim
pip install -r requirements.txt
```

### Configuration
```bash
# Edit machine-specific paths
nano settings_folder/machine_dependent_settings.py
```

## 🎮 Training Commands

### 🔰 Beginner: Start with PPO
```bash
python train_ppo.py
# Stable, reliable, great for learning the ropes
```

### 🎯 Discrete Control: DQN Family
```bash
python dqn.py                 # Classic DQN
python prioritized_dqn.py     # With prioritized experience replay
python rainbow.py             # State-of-the-art DQN variant
```

### 🎛️ Continuous Control: SAC
```bash
python SAC.py                 # Training phase
python eval_SAC.py            # Evaluation phase
# Best for smooth, continuous drone movements
```

### ⚡ Advanced Methods
```bash
python a3c.py                 # Asynchronous training
python td3.py                 # Twin delayed DDPG
```

### 🏛️ OpenAI Baselines
```bash
cd baselines
python -m baselines.run --alg=a2c --env=AirGym
python -m baselines.run --alg=ddpg --env=AirGym
python -m baselines.run --alg=trpo_mpi --env=AirGym
```

## 📋 Algorithm Selection Guide

| Use Case | Recommended Algorithm | Rationale |
|----------|----------------------|-----------|
| **First-time users** | PPO | Stable, forgiving, well-documented |
| **Discrete actions** | Rainbow DQN | Most advanced Q-learning variant |
| **Continuous control** | SAC | Maximum entropy, robust performance |
| **Sample efficiency** | TD3, SAC | Improved sample complexity |
| **Distributed training** | A3C | Asynchronous parallel learning |
| **Imitation learning** | GAIL | Learn from expert demonstrations |
| **Goal-oriented tasks** | HER | Learns from failed attempts |

## 🛠️ Technical Requirements

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.7+ | Core runtime |
| **PyTorch** | 1.7+ | Deep learning framework |
| **AirSim** | 1.8.1 | Simulation environment |
| **OpenCV** | Latest | Computer vision |
| **NumPy** | Latest | Numerical computing |
| **Gym** | Latest | RL environment interface |
| **TensorboardX** | Latest | Training visualization |

## 📈 Performance Monitoring

All algorithms include built-in tensorboard logging:
```bash
tensorboard --logdir=runs/
```

## 🎯 Environment Features

- **Photorealistic 3D environments** powered by Unreal Engine 4.27
- **Physics-accurate drone dynamics** via AirSim
- **Customizable weather and lighting conditions**
- **Multiple drone models and sensor configurations**
- **Real-time obstacle generation and randomization**

## 🤝 Contributing

We welcome contributions! Whether it's new algorithms, performance improvements, or bug fixes, please feel free to submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built upon the excellent work of:
- Microsoft AirSim team for the simulation environment
- OpenAI for the Baselines implementations  
- PyTorch community for the deep learning framework

---

<div align="center">

**Ready to take your drone AI to the next level?**

[Get Started](#-quick-start) • [Choose Algorithm](#-algorithm-selection-guide) • [View Examples](examples/)

</div>