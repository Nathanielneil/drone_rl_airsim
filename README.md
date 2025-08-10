<div align="center">

# UAV Reinforcement Learning Suite

**A comprehensive collection of state-of-the-art reinforcement learning algorithms for unmanned aerial vehicle navigation and control**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)](https://pytorch.org/)
[![AirSim](https://img.shields.io/badge/AirSim-1.8.1-green.svg)](https://microsoft.github.io/AirSim/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04-orange.svg)](https://ubuntu.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Powered by Ubuntu 20.04 • Unreal Engine 4.27 • AirSim 1.8.1*<br>
*Email:guowei_ni@bit.edu.cn*
</div>

---

## Project Overview

This repository implements a complete suite of reinforcement learning algorithms specifically designed for UAV autonomous navigation, obstacle avoidance in realistic 3D environments. From classic value-based methods to cutting-edge actor-critic algorithms, this collection provides researchers and developers with battle-tested implementations ready for real-world drone applications.

## Algorithm Arsenal

### 🚀 Ready-to-Run Algorithms (✅ Tested & Verified)

| Algorithm | Implementation | Status | Control Mode | Key Features |
|-----------|---------------|---------|--------------|--------------|
| **PPO** | `train_ppo.py` | ✅ Production Ready | Discrete/Continuous | Stable, sample-efficient, beginner-friendly |
| **SAC** | `SAC.py` + `eval_SAC.py` | ✅ Production Ready | Continuous | Maximum entropy, robust exploration |
| **TD3** | `td3.py` | ✅ Production Ready | Continuous | Twin critics, delayed policy updates |
| **DQN** | `dqn.py` | ✅ Production Ready | Discrete | Classic deep Q-learning |
| **Rainbow DQN** | `rainbow.py` | ✅ Production Ready | Discrete | Multi-component DQN enhancement |
| **Prioritized DQN** | `prioritized_dqn.py` | ✅ Production Ready | Discrete | Experience replay prioritization |
| **A3C** | `a3c.py` | ✅ Newly Completed | Discrete | Asynchronous advantage actor-critic |
| **DDPG** | `ddpg.py` | ✅ Enhanced Twin-Critic | Continuous | Deterministic policy gradients |

### 🔧 Extended Algorithm Suite (Baselines)
| Algorithm | Implementation | Status |
|-----------|---------------|---------|
| **A2C** | `baselines/a2c/` | Available |
| **ACER** | `baselines/acer/` | Available |
| **ACKTR** | `baselines/acktr/` | Available |
| **DDPG (Original)** | `baselines/ddpg/` | Available |
| **HER** | `baselines/her/` | Available |
| **GAIL** | `baselines/gail/` | Available |
| **TRPO** | `baselines/trpo_mpi/` | Available |

### Advanced Methods
- **TRPO** - Trust region optimization for policy updates
- **ACER** - Off-policy actor-critic with experience replay
- **ACKTR** - Natural gradient optimization
- **HER** - Goal-conditioned reinforcement learning
- **GAIL** - Imitation learning from expert demonstrations

## Architecture

```
drone_rl/
│
├── Core Algorithms
│   ├── train_ppo.py              # PPO: The go-to algorithm
│   ├── SAC.py & eval_SAC.py      # SAC: Continuous control master
│   ├── dqn.py                    # DQN: Discrete action foundation
│   ├── prioritized_dqn.py        # Enhanced DQN with prioritized replay
│   ├── rainbow.py                # Rainbow: DQN's ultimate evolution
│   ├── a3c.py                    # A3C: Asynchronous advantage actor-critic
│   └── td3.py                    # TD3: Twin delayed DDPG
│
├── Algorithm Components
│   ├── algorithm/                # PPO implementation details
│   ├── DQN/                      # DQN supporting modules
│   ├── Rainbow/                  # Rainbow DQN components
│   └── utils/                    # Shared utilities and helpers
│
├── Environment Integration
│   ├── gym_airsim/              # AirSim-Gym interface
│   ├── environment_randomization/ # Domain randomization
│   └── settings_folder/         # Environment configurations
│
├── Baselines & Benchmarks
│   └── baselines/               # OpenAI Baselines integration
│       ├── a2c/, ddpg/, her/    # Classic implementations
│       ├── gail/, trpo_mpi/     # Advanced methods
│       └── ppo1/, ppo2/         # PPO variants
│
└── Infrastructure
    ├── common/                   # Shared functionality
    ├── config.py                 # Global configuration
    └── start_simulation.py       # Environment launcher
```

## Quick Start

### Option 1: Docker (Recommended)

#### Prerequisites
- Docker Engine 20.10+
- NVIDIA Docker runtime (for GPU support)
- Docker Compose 1.28+

#### Quick Start with Docker
```bash
git clone https://github.com/Nathanielneil/drone_rl_airsim.git
cd drone_rl_airsim

# Simple training with GPU
docker-compose up drone-rl-gpu

# Development environment
docker-compose up dev

# Jupyter Lab for experimentation
docker-compose up jupyter
```

#### Custom Docker Commands
```bash
# Build image
./docker/build.sh

# Run interactive development
./docker/run.sh --dev

# Start specific training
./docker/run.sh --training ppo
```

### Option 2: Local Installation

#### Prerequisites
```bash
# Ensure AirSim environment is running
# Ubuntu 20.04 + Unreal Engine 4.27 + AirSim 1.8.1
```

#### Installation
```bash
git clone https://github.com/Nathanielneil/drone_rl_airsim.git
cd drone_rl_airsim
pip install -r requirements.txt
```

#### Configuration
```bash
# Edit machine-specific paths
nano settings_folder/machine_dependent_settings.py
```

## Training Commands

### 🎯 Quick Start (Recommended)
```bash
# For beginners - most stable algorithm
python train_ppo.py

# For continuous control enthusiasts
python SAC.py                 # Training
python eval_SAC.py            # Evaluation
```

### 🔥 Discrete Control Algorithms
```bash
python dqn.py                 # Classic deep Q-learning
python prioritized_dqn.py     # Enhanced experience replay
python rainbow.py             # Multi-component DQN (state-of-the-art)
python a3c.py                 # Asynchronous actor-critic (newly completed)
```

### ⚡ Continuous Control Algorithms
```bash
# Note: These require control_mode="moveByVelocity" in settings.py
python SAC.py                 # Soft actor-critic (entropy-based)
python td3.py                 # Twin delayed DDPG (twin critics)
python ddpg.py                # Enhanced DDPG with twin critics
```

### 🛠️ Configuration
```bash
# Switch control modes in settings_folder/settings.py:
control_mode="Discrete"       # For DQN family, A3C, PPO
control_mode="moveByVelocity" # For SAC, TD3, DDPG
```

### 📊 Advanced Training (OpenAI Baselines)
```bash
cd baselines
python -m baselines.run --alg=a2c --env=AirGym
python -m baselines.run --alg=ddpg --env=AirGym
python -m baselines.run --alg=trpo_mpi --env=AirGym
```

## 🎯 Algorithm Selection Guide

| Use Case | Recommended Algorithm | Control Mode | Rationale |
|----------|----------------------|--------------|-----------|
| **🚀 First-time users** | PPO | Discrete | Stable, forgiving, well-documented |
| **🎮 Discrete actions** | Rainbow DQN | Discrete | State-of-the-art Q-learning with all improvements |
| **🕹️ Continuous control** | SAC | Continuous | Maximum entropy, robust exploration |
| **⚡ Fast convergence** | TD3 | Continuous | Twin critics reduce overestimation bias |
| **🔄 Stable deterministic** | DDPG (Enhanced) | Continuous | Twin-critic version for improved stability |
| **🌐 Distributed training** | A3C | Discrete | Asynchronous parallel learning |
| **🎯 Sample efficiency** | TD3, SAC | Continuous | Advanced off-policy methods |
| **🧠 Imitation learning** | GAIL | Both | Learn from expert demonstrations |
| **🎖️ Goal-oriented tasks** | HER | Both | Learns from failed attempts |

### 💡 **Quick Decision Tree**:
- **New to RL?** → Start with **PPO** (most forgiving)
- **Need discrete actions?** → Use **Rainbow DQN** (best Q-learning)
- **Want smooth control?** → Choose **SAC** (entropy-based) or **TD3** (deterministic)
- **Research cutting-edge?** → Try **TD3** or enhanced **DDPG**

## Technical Requirements

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.7+ | Core runtime |
| **PyTorch** | 1.7+ | Deep learning framework |
| **AirSim** | 1.8.1 | Simulation environment |
| **OpenCV** | Latest | Computer vision |
| **NumPy** | Latest | Numerical computing |
| **Gym** | Latest | RL environment interface |
| **TensorboardX** | Latest | Training visualization |

## Performance Monitoring

All algorithms include built-in tensorboard logging:
```bash
tensorboard --logdir=runs/
```

## 🌟 Latest Features & Improvements

### 🔧 **Algorithm Enhancements**
- **A3C Complete Implementation**: Full training loop, loss computation, and network architecture
- **DDPG Twin-Critic Version**: Enhanced stability with dual Q-networks (TD3-inspired)
- **Unified State Handling**: All algorithms now properly handle 9-dimensional inform_vector
- **Smart Action Processing**: Automatic handling of continuous vs discrete action spaces

### 🛡️ **Intelligent Collision Recovery System**
- **Progressive Penalty**: Escalating collision penalties (1st: -5, 2nd: -15, 3rd+: -30)
- **Auto-Recovery**: Automatic repositioning to safe zones after multiple collisions
- **Collision Counting**: Smart reset mechanism after collision-free intervals
- **Training Continuity**: Collisions don't terminate episodes, preserving training data

### 🎨 **Visual Enhancements**
- **Fluorescent Trail Effects**: Ultra-cool neon trails during training
- **Dynamic Color Changes**: Trail colors change every 20 episodes
- **Multiple Effects**: laser_red, ghost_white, electric_blue, toxic_green, and more

## Environment Features

- **Photorealistic 3D environments** powered by Unreal Engine 4.27
- **Physics-accurate drone dynamics** via AirSim
- **Dual control modes**: Discrete actions & continuous velocity control
- **Intelligent collision handling** with progressive recovery
- **9-dimensional state space**: position, velocity, orientation, and goal information
- **Real-time obstacle generation and randomization**

## Contributing

We welcome contributions! Whether it's new algorithms, performance improvements, or bug fixes, please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built upon the excellent work of:
- Microsoft AirSim team for the simulation environment
- OpenAI for the Baselines implementations  
- PyTorch community for the deep learning framework

---

<div align="center">

**Ready to take your drone AI to the next level?**

[Get Started](#quick-start) • [Choose Algorithm](#algorithm-selection-guide) • [View Examples](examples/)

</div>
