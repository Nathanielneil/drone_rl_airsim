<div align="center">

# Modern UAV Reinforcement Learning Suite

**Next-generation reinforcement learning algorithms optimized for Windows 10 + AirSim 1.8.1 + CUDA 12.1**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2+cu121-red.svg)](https://pytorch.org/)
[![AirSim](https://img.shields.io/badge/AirSim-1.8.1-green.svg)](https://microsoft.github.io/AirSim/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Windows](https://img.shields.io/badge/Windows-10+-0078D4.svg)](https://www.microsoft.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Optimized for Windows 10 • Unreal Engine 4.27.2 • AirSim 1.8.1 • CUDA 12.1*<br>
*Modern Architecture • Mixed Precision Training • GPU Optimization*<br>
*Email:guowei_ni@bit.edu.cn*
</div>

---

## Project Overview

This repository implements a **modern, GPU-optimized** suite of reinforcement learning algorithms specifically designed for UAV autonomous navigation. Fully optimized for **Windows 10 + AirSim 1.8.1 + CUDA 12.1**, featuring mixed precision training, asynchronous processing, and intelligent performance monitoring for maximum training efficiency.

### 🚀 **New Features (2025/09/20)**
- **CUDA 12.1 Optimization**: Native support for latest CUDA with TF32 acceleration
- **Mixed Precision Training**: Up to 2x faster training with AMP (Automatic Mixed Precision)
- **Modern Architecture**: Gymnasium interface, OmegaConf configuration, async processing
- **Windows Optimization**: Performance tuning, GPU monitoring, intelligent memory management
- **Real-time Monitoring**: Live GPU metrics, performance analytics, automated optimization

## Algorithm Arsenal

### 🔥 **Modern Optimized Algorithms**

| Algorithm | Implementation | CUDA 12.1 | Mixed Precision | Key Features |
|-----------|---------------|------------|-----------------|--------------|
| **Modern SAC** | `scripts/modern_train.py` | ✅ | ✅ | GPU-optimized, async processing, AirSim 1.8.1 |
| **Modern PPO** | `train_ppo.py` | ✅ | ✅ | Stable, sample-efficient, beginner-friendly |
| **Legacy SAC** | `SAC.py` + `eval_SAC.py` | ⚠️ | ❌ | Maximum entropy, robust exploration |
| **TD3** | `td3.py` | ⚠️ | ❌ | Twin critics, delayed policy updates |
| **DQN** | `dqn.py` | ⚠️ | ❌ | Classic deep Q-learning |
| **Rainbow DQN** | `rainbow.py` | ⚠️ | ❌ | Multi-component DQN enhancement |
| **A3C** | `a3c.py` | ⚠️ | ❌ | Asynchronous advantage actor-critic |
| **DDPG** | `ddpg.py` | ⚠️ | ❌ | Deterministic policy gradients |

> 💡 **Recommendation**: Use **Modern SAC** (`scripts/modern_train.py`) for best performance with Windows 10 + CUDA 12.1

### Hierarchical Reinforcement Learning (New!)

| Algorithm | Implementation | Type | Key Features |
|-----------|---------------|------|--------------|
| **HAC** | `hierarchical_rl/hac/` | Hierarchical | Goal-conditioned multi-level learning |
| **HIRO** | `hierarchical_rl/hiro/` | Hierarchical | Off-policy hierarchical learning |
| **FUN** | `hierarchical_rl/fun/` | Hierarchical | Feudal networks for temporal abstraction |
| **Options** | `hierarchical_rl/options/` | Hierarchical | Semi-Markov option-based learning |

### Extended Algorithm Suite (Baselines)
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
├── Hierarchical RL Suite
│   ├── hierarchical_rl/          # Complete HRL framework
│   │   ├── hac/                  # HAC: Hindsight Action Control
│   │   ├── hiro/                 # HIRO: Off-policy hierarchical RL
│   │   ├── fun/                  # FUN: Feudal networks
│   │   ├── options/              # Options framework
│   │   ├── envs/                 # Goal-conditioned environments
│   │   └── common/               # Shared HRL components
│   └── train_hac_fixed.py        # HAC training script (fixed version)
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

### 🚀 **Modern Installation (Windows 10 + CUDA 12.1)**

#### Prerequisites
```bash
# System Requirements
✅ Windows 10/11 (64-bit)
✅ CUDA 12.1 + cuDNN 8.9+
✅ Unreal Engine 4.27.2 + AirSim 1.8.1
✅ Python 3.9+ (Recommended: 3.11)
✅ 16GB+ RAM, 8GB+ VRAM (RTX 3080+ recommended)
```

#### 🔧 **Automated Installation**
```bash
git clone https://github.com/Nathanielneil/drone_rl_airsim.git
cd drone_rl_airsim

# Option 1: Full Windows + CUDA 12.1 setup (Recommended)
scripts/install_windows_cuda121.bat

# Option 2: Manual installation
pip install -r requirements_windows_cuda121.txt
```

#### 🔍 **Environment Verification**
```bash
# Check compatibility and performance
python scripts/check_cuda_compatibility.py

# Verify AirSim connection
python -c "from src.environments.airsim_env.modern_airsim_env import ModernAirSimEnv; print('✅ AirSim ready')"
```

## Training Commands

### 🏃‍♂️ **Modern Training (Optimized)**
```bash
# 🔥 MODERN SAC - Best performance with CUDA 12.1
python scripts/modern_train.py --algorithm sac --device cuda

# 🎯 MODERN SAC with custom settings
python scripts/modern_train.py \
    --algorithm sac \
    --total-timesteps 1000000 \
    --batch-size 512 \
    --learning-rate 3e-4 \
    --env-host 127.0.0.1 \
    --tensorboard-log experiments/logs

# 🚀 Quick performance test
python scripts/modern_train.py --total-timesteps 10000 --experiment-name quick_test
```

### 📊 **Performance Monitoring**
```bash
# Real-time training monitoring
tensorboard --logdir experiments/logs

# Performance analysis
python -c "from src.utils.performance.gpu_manager import get_performance_monitor; monitor = get_performance_monitor(); print(monitor.get_detailed_stats())"
```

### Discrete Control Algorithms
```bash
python dqn.py                 # Classic deep Q-learning
python prioritized_dqn.py     # Enhanced experience replay
python rainbow.py             # Multi-component DQN (state-of-the-art)
python a3c.py                 # Asynchronous actor-critic (newly completed)
```

### Continuous Control Algorithms
```bash
# Note: These require control_mode="moveByVelocity" in settings.py
python SAC.py                 # Soft actor-critic (entropy-based)
python td3.py                 # Twin delayed DDPG (twin critics)
python ddpg.py                # Enhanced DDPG with twin critics
```

### Hierarchical Reinforcement Learning
```bash
# Hierarchical algorithms for complex goal-oriented tasks
python train_hac_fixed.py     # HAC: Multi-level goal-conditioned learning (recommended)

# Alternative HRL algorithms (development versions)
cd hierarchical_rl
python train_hierarchical.py --algorithm hiro    # HIRO: Off-policy HRL
python train_hierarchical.py --algorithm fun     # FUN: Feudal networks
python train_hierarchical.py --algorithm options # Options framework
```

### Configuration
```bash
# Switch control modes in settings_folder/settings.py:
control_mode="Discrete"       # For DQN family, A3C, PPO
control_mode="moveByVelocity" # For SAC, TD3, DDPG
```

### Advanced Training (OpenAI Baselines)
```bash
cd baselines
python -m baselines.run --alg=a2c --env=AirGym
python -m baselines.run --alg=ddpg --env=AirGym
python -m baselines.run --alg=trpo_mpi --env=AirGym
```

## Algorithm Selection Guide

| Use Case | Recommended Algorithm | Control Mode | Rationale |
|----------|----------------------|--------------|-----------|
| **First-time users** | PPO | Discrete | Stable, forgiving, well-documented |
| **Discrete actions** | Rainbow DQN | Discrete | State-of-the-art Q-learning with all improvements |
| **Continuous control** | SAC | Continuous | Maximum entropy, robust exploration |
| **Fast convergence** | TD3 | Continuous | Twin critics reduce overestimation bias |
| **Stable deterministic** | DDPG (Enhanced) | Continuous | Twin-critic version for improved stability |
| **Distributed training** | A3C | Discrete | Asynchronous parallel learning |
| **Sample efficiency** | TD3, SAC | Continuous | Advanced off-policy methods |
| **Imitation learning** | GAIL | Both | Learn from expert demonstrations |
| **Goal-oriented tasks** | HER, HAC | Both | Learns from failed attempts |
| **Complex navigation** | HAC | Continuous | Multi-level hierarchical planning |
| **Long-horizon tasks** | HAC, HIRO | Continuous | Temporal abstraction and subgoals |

### Quick Decision Tree:
- **New to RL?** → Start with **PPO** (most forgiving)
- **Need discrete actions?** → Use **Rainbow DQN** (best Q-learning)
- **Want smooth control?** → Choose **SAC** (entropy-based) or **TD3** (deterministic)
- **Complex goal-oriented tasks?** → Try **HAC** (hierarchical learning)
- **Research cutting-edge?** → Try **TD3** or enhanced **DDPG**

## Technical Requirements

### 🔧 **Modern Stack (Optimized)**
| Component | Version | Purpose | CUDA 12.1 | Mixed Precision |
|-----------|---------|---------|------------|-----------------|
| **Python** | 3.9+ | Core runtime | ✅ | ✅ |
| **PyTorch** | 2.1.2+cu121 | GPU-optimized DL framework | ✅ | ✅ |
| **AirSim** | 1.8.1 | Modern simulation environment | ✅ | ✅ |
| **Gymnasium** | 0.29+ | Modern RL interface | ✅ | ✅ |
| **OmegaConf** | 2.3+ | Advanced configuration | ✅ | ✅ |
| **CUDA** | 12.1 | GPU acceleration | ✅ | ✅ |
| **TensorBoard** | Latest | Training visualization | ✅ | ✅ |

### 📋 **Hardware Recommendations**
| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **GPU** | GTX 1660 (6GB) | RTX 3080 (10GB) | RTX 4090 (24GB) |
| **RAM** | 16GB | 32GB | 64GB |
| **Storage** | 50GB SSD | 100GB NVMe | 500GB NVMe |
| **CPU** | 6-core | 8-core | 12+ core |

## Performance Monitoring

All algorithms include built-in tensorboard logging:
```bash
tensorboard --logdir=runs/
```

## 🚀 Latest Features & Modern Optimizations (2024)

### 🔥 **Major Update: Windows 10 + CUDA 12.1 Optimization**
- **Modern SAC Implementation**: Complete rewrite with CUDA 12.1 native support
- **Mixed Precision Training**: AMP support for 2x faster training with minimal memory
- **GPU Memory Management**: Intelligent memory allocation and automatic cleanup
- **Windows Performance Tuning**: High-priority processes, CPU affinity, power optimization
- **Real-time Performance Monitoring**: Live GPU metrics, memory usage, temperature tracking

### ⚡ **Algorithm Modernization**
- **AirSim 1.8.1 Integration**: Native compatibility with latest AirSim API
- **Gymnasium Interface**: Modern RL environment standards
- **Async Processing**: Non-blocking image processing and action execution
- **OmegaConf Configuration**: Advanced YAML-based configuration management
- **Hardware-Aware Optimization**: Automatic batch size and memory optimization

### 🛠 **Development Tools**
- **Automated Installation**: One-click Windows + CUDA 12.1 setup
- **Compatibility Checker**: Hardware detection and optimization recommendations
- **Performance Profiler**: Detailed training metrics and bottleneck analysis
- **Configuration Generator**: Hardware-specific config generation
- **Error Recovery**: Robust error handling and training continuation

### 📊 **Monitoring & Analytics**
- **TensorBoard Integration**: Real-time training visualization
- **GPU Performance Tracking**: Memory, utilization, temperature monitoring
- **Training Reports**: Comprehensive performance analysis
- **Automated Optimization**: Dynamic batch size and memory adjustment
- **Performance Alerts**: Real-time warnings for hardware issues

### 🎯 **Legacy Algorithm Support**
- **HAC Algorithm Fixed**: Resolved UAV hovering issue with proper action scaling
- **Complete HRL Framework**: HAC, HIRO, FUN, and Options algorithms
- **Intelligent Collision Recovery**: Progressive penalty system
- **Visual Enhancements**: Fluorescent trail effects and dynamic colors

## Environment Features

### 🌟 **Modern Environment (AirSim 1.8.1)**
- **GPU-Optimized Image Processing**: CUDA-accelerated computer vision pipeline
- **Async Operation Support**: Non-blocking environment interactions
- **Modern Gymnasium Interface**: Standard RL environment protocol
- **Windows-Native Integration**: Optimized for Windows 10/11 performance
- **Real-time Performance Monitoring**: Live FPS and latency tracking

### 🎮 **Enhanced Simulation**
- **Photorealistic 3D environments** powered by Unreal Engine 4.27.2
- **Physics-accurate drone dynamics** via AirSim 1.8.1
- **Dual control modes**: Discrete actions & continuous velocity control
- **Hierarchical goal-conditioned environments** for complex navigation tasks
- **Intelligent collision handling** with progressive recovery
- **9-dimensional state space**: position, velocity, orientation, and goal information
- **Real-time obstacle generation and randomization**
- **Multi-level reward structures** supporting hierarchical learning

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
