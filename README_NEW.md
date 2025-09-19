# 🚁 Drone RL AirSim

A comprehensive reinforcement learning framework for autonomous drone navigation using Microsoft AirSim.

## 🌟 Features

- **Multiple RL Algorithms**: SAC, PPO, DDPG, TD3, DQN, Rainbow DQN
- **Hierarchical RL Support**: FuN, HIRO, HAC, Options framework
- **Modular Architecture**: Clean, extensible codebase
- **Comprehensive Evaluation**: Built-in benchmarking and comparison tools
- **Rich Configuration**: YAML-based configuration system
- **Documentation**: Complete documentation and tutorials

## 🏗️ Project Structure

```
drone_rl_airsim/
├── src/                           # Source code
│   ├── algorithms/               # RL algorithm implementations
│   │   ├── value_based/         # DQN, Rainbow, etc.
│   │   ├── policy_based/        # PPO, A3C, TRPO
│   │   ├── actor_critic/        # SAC, DDPG, TD3
│   │   └── hierarchical/        # Hierarchical RL methods
│   ├── environments/            # Environment wrappers
│   │   ├── airsim_env/         # AirSim environment interface
│   │   ├── gym_airsim/         # Gym wrapper for AirSim
│   │   └── environment_randomization/  # Domain randomization
│   ├── core/                    # Core framework components
│   ├── utils/                   # Utility functions and helpers
│   └── legacy/                  # Legacy code (to be refactored)
├── config/                      # Configuration files
│   ├── algorithms/             # Algorithm-specific configs
│   ├── environments/           # Environment configs
│   └── default.yaml           # Default settings
├── experiments/                 # Experiment management
│   ├── scripts/               # Training, evaluation, comparison scripts
│   ├── configs/               # Experiment configurations
│   ├── results/               # Results and analysis
│   └── logs/                  # Training logs
├── scripts/                    # Convenience scripts
├── tests/                      # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── benchmarks/            # Performance benchmarks
├── docs/                       # Documentation
│   ├── user_guide/            # User guides and tutorials
│   ├── api_reference/         # API documentation
│   └── algorithms/            # Algorithm documentation
├── data/                       # Data directory
├── models/                     # Trained models
├── requirements/               # Dependency specifications
└── tools/                      # Development tools
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd drone_rl_airsim

# Install dependencies
pip install -r requirements/base.txt

# For training (includes additional dependencies)
pip install -r requirements/training.txt

# Test installation
python scripts/test_installation.py
```

### 2. Basic Usage

```bash
# Train SAC agent
python experiments/scripts/train.py --algorithm sac

# Train PPO agent with custom config
python experiments/scripts/train.py --algorithm ppo --config config/algorithms/ppo.yaml

# Evaluate trained model
python experiments/scripts/evaluate.py --model-path models/sac_model.zip --algorithm sac

# Compare multiple models
python experiments/scripts/compare.py --models model1.zip model2.zip --algorithms sac ppo
```

### 3. Quick Training Scripts

```bash
# Quick SAC training
python scripts/train_sac.py

# Quick PPO training  
python scripts/train_ppo.py
```

## 📊 Supported Algorithms

### Value-Based Methods
- **DQN**: Deep Q-Network
- **Rainbow DQN**: DQN with multiple improvements
- **Prioritized DQN**: Experience replay with prioritization

### Policy-Based Methods
- **PPO**: Proximal Policy Optimization
- **A3C**: Asynchronous Advantage Actor-Critic
- **TRPO**: Trust Region Policy Optimization

### Actor-Critic Methods
- **SAC**: Soft Actor-Critic
- **DDPG**: Deep Deterministic Policy Gradient
- **TD3**: Twin Delayed Deep Deterministic Policy Gradient

### Hierarchical Methods
- **FuN**: FeUdal Networks
- **HIRO**: HIerarchy and Robust Option
- **HAC**: Hierarchical Actor-Critic
- **Options**: Option-Critic Architecture

## ⚙️ Configuration

The framework uses YAML configuration files for easy customization:

```yaml
# config/algorithms/sac.yaml
algorithm: "sac"
sac:
  learning_rate: 0.0003
  buffer_size: 1000000
  batch_size: 256
  tau: 0.005
  gamma: 0.99
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test category
python -m pytest tests/unit/
python -m pytest tests/integration/

# Run with coverage
python -m pytest tests/ --cov=src
```

## 📚 Documentation

- [Installation Guide](docs/user_guide/installation.md)
- [Quick Start Tutorial](docs/user_guide/quickstart.md) 
- [Configuration Guide](docs/user_guide/configuration.md)
- [API Reference](docs/api_reference/)
- [Algorithm Documentation](docs/algorithms/)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Microsoft AirSim team for the simulation platform
- OpenAI and Stable Baselines communities
- Research papers and implementations that inspired this work

## 📞 Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/your-repo/issues)
- 💬 [Discussions](https://github.com/your-repo/discussions)