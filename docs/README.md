# Drone RL AirSim Documentation

Welcome to the Drone RL AirSim documentation! This project provides a comprehensive framework for training reinforcement learning agents for autonomous drone navigation using Microsoft AirSim.

## 📚 Documentation Structure

### User Guides
- [Installation Guide](user_guide/installation.md) - Step-by-step installation instructions
- [Quick Start](user_guide/quickstart.md) - Get up and running quickly
- [Configuration Guide](user_guide/configuration.md) - How to configure training and environments
- [Training Guide](user_guide/training.md) - Complete training workflow
- [Evaluation Guide](user_guide/evaluation.md) - Model evaluation and analysis

### API Reference
- [Core Components](api_reference/core.md) - Core framework components
- [Algorithms](api_reference/algorithms.md) - RL algorithm implementations
- [Environments](api_reference/environments.md) - Environment interfaces
- [Utilities](api_reference/utilities.md) - Utility functions and helpers

### Tutorials
- [Basic Training Tutorial](tutorials/basic_training.md) - Your first training session
- [Custom Environment Tutorial](tutorials/custom_environment.md) - Creating custom environments
- [Hyperparameter Tuning](tutorials/hyperparameter_tuning.md) - Optimizing training
- [Multi-Agent Training](tutorials/multi_agent.md) - Training multiple agents

### Algorithm Documentation
- [SAC (Soft Actor-Critic)](algorithms/sac.md) - Continuous control with SAC
- [PPO (Proximal Policy Optimization)](algorithms/ppo.md) - Policy gradient method
- [DDPG (Deep Deterministic Policy Gradient)](algorithms/ddpg.md) - Actor-critic for continuous control
- [TD3 (Twin Delayed Deep Deterministic)](algorithms/td3.md) - Improved DDPG
- [DQN (Deep Q-Network)](algorithms/dqn.md) - Value-based method for discrete actions
- [Rainbow DQN](algorithms/rainbow.md) - Enhanced DQN with multiple improvements

## 🚀 Quick Links

- **New to the project?** Start with the [Installation Guide](user_guide/installation.md)
- **Want to train your first model?** Follow the [Quick Start](user_guide/quickstart.md)
- **Looking for specific algorithm details?** Check the [Algorithm Documentation](algorithms/)
- **Need API details?** Browse the [API Reference](api_reference/)
- **Want hands-on examples?** Try the [Tutorials](tutorials/)

## 🏗️ Project Architecture

```
drone_rl_airsim/
├── src/                     # Source code
│   ├── algorithms/         # RL algorithm implementations
│   ├── environments/       # Environment wrappers and interfaces
│   ├── core/              # Core framework components
│   └── utils/             # Utility functions
├── config/                 # Configuration files
├── experiments/           # Experiment management
├── scripts/              # Convenience scripts
└── docs/                 # This documentation
```

## 📖 Getting Help

1. **Check the documentation** - Most questions are answered here
2. **Look at examples** - The tutorials provide working examples
3. **Check the issues** - Someone might have had the same problem
4. **Create an issue** - If you can't find an answer

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details on:
- Code style and standards
- How to submit pull requests
- Reporting bugs and requesting features
- Development setup

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft AirSim team for the excellent simulation platform
- OpenAI Baselines for reference implementations
- Stable Baselines3 for high-quality RL implementations
- The broader RL community for research and open source contributions