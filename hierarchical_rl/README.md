# Enhanced Hierarchical Reinforcement Learning Suite
## Complete Implementation of Advanced Hierarchical RL Algorithms

This directory contains a comprehensive suite of state-of-the-art hierarchical reinforcement learning algorithms, specifically enhanced and optimized for UAV navigation tasks in the AirSim environment.

---

## Quick Start

### Run All Algorithm Tests
```bash
cd hierarchical_rl
python launch_hierarchical_training.py --test-all
```

### Train Specific Algorithms
```bash
# HAC (Production Ready - Best Performance)
python launch_hierarchical_training.py --algorithm hac --episodes 500

# HIRO (Enhanced with Off-policy Correction)  
python launch_hierarchical_training.py --algorithm hiro --episodes 300

# FuN (Feudal Networks - Manager-Worker Hierarchy)
python launch_hierarchical_training.py --algorithm fun --episodes 400

# Options Framework (Skill Discovery)
python launch_hierarchical_training.py --algorithm options --episodes 600
```

### Benchmark All Algorithms
```bash
python launch_hierarchical_training.py --benchmark
```

---

## Algorithm Status & Performance

| Algorithm | Status | Implementation | Key Features | Recommended Use |
|-----------|---------|---------------|--------------|-----------------|
| **HAC** | Production Ready | `hac/` | Multi-level goal learning, HER | **Primary Choice** - Best results |
| **HIRO** | Enhanced | `hiro/` | Off-policy correction, Goal relabeling | Sample-efficient learning |
| **FuN** | Enhanced | `fun/` | Manager-worker feudal hierarchy | Complex temporal tasks |
| **Options** | Enhanced | `options/` | Skill discovery, Temporal abstraction | Diverse behavior learning |

---

## Architecture Overview

```
hierarchical_rl/
├── Core Algorithms
│   ├── hac/                    # HAC: Multi-level goal-conditioned RL
│   │   ├── hac_agent.py       # Main HAC implementation
│   │   ├── hac_config.py      # Configuration settings
│   │   └── train_hac_fixed.py # Fixed training script (PRODUCTION READY)
│   │
│   ├── hiro/                   # HIRO: Hierarchical RL with off-policy correction
│   │   ├── hiro_agent.py      # Enhanced HIRO implementation
│   │   ├── hiro_config.py     # HIRO configuration
│   │   └── train_hiro_improved.py # Enhanced training script
│   │
│   ├── fun/                    # FuN: Feudal Networks
│   │   ├── fun_agent.py       # Manager-worker architecture
│   │   ├── fun_config.py      # FuN configuration
│   │   └── train_fun_improved.py # Feudal training script
│   │
│   └── options/                # Options Framework
│       ├── options_agent.py   # Multi-option skill learning
│       ├── options_config.py  # Options configuration
│       └── train_options_improved.py # Skill discovery training
│
├── Shared Infrastructure
│   ├── common/                 # Shared components
│   │   ├── base_hierarchical_agent.py
│   │   ├── hierarchical_replay_buffer.py
│   │   ├── goal_generation.py
│   │   └── intrinsic_motivation.py
│   │
│   └── envs/                   # Environment wrappers
│       ├── hierarchical_airsim_env.py
│       └── goal_conditioned_wrapper.py
│
├── Testing & Validation
│   ├── test_all_algorithms.py  # Comprehensive test suite
│   ├── test_hrl_components.py  # Component testing
│   └── eval_hierarchical.py    # Performance evaluation
│
├── Training & Utilities
│   ├── launch_hierarchical_training.py # Unified launcher
│   ├── train_hierarchical.py   # Generic training framework
│   └── setup_airsim_config.py  # Environment setup
│
└── Documentation
    ├── README_UPDATED.md       # This file
    ├── DEBUGGING_LOG.md        # Debug history
    └── ALGORITHM_COMPARISON.md # Performance comparison
```

---

## Algorithm Deep Dive

### 1. HAC (Hindsight Action Control) - **PRODUCTION READY**

**Status:** Fully Working - Best Performance

**Key Improvements:**
- Fixed action space scaling ([-2.0, 2.0] instead of [-0.3, 0.3])
- Proper network weight initialization with tanh activation
- Enhanced collision recovery with progressive penalties
- Multi-level goal-conditioned learning with HER
- Successful 100-episode training completion

**Architecture:**
```
Level 2 (High): Goal → Subgoal Generation
Level 1 (Low):  Subgoal → Action Execution
```

**Training Results:**
- Episodes: 100 Completed
- Final Model: `results/hierarchical/hac_fixed/final_model.pth`
- Action Space: [-2.0, 2.0] for effective UAV movement
- Collision Recovery: Progressive penalty system

### 2. HIRO (HIerarchical RL with Off-policy correction) - **ENHANCED**

**Status:** Enhanced Implementation

**Key Features:**
- Two-level hierarchy with high-level and low-level policies
- Off-policy correction for improved sample efficiency
- Advanced Hindsight Experience Replay (HER)
- Goal relabeling with dense reward computation
- Twin DDPG critics for stability

**Enhancements:**
- Specialized replay buffer with HER integration
- Off-policy correction mechanisms
- Advanced goal relabeling strategies
- Comprehensive logging and monitoring

### 3. FuN (FeUdal Networks) - **ENHANCED**

**Status:** Enhanced Implementation

**Key Features:**
- Manager-worker feudal architecture
- Intrinsic motivation through cosine similarity
- Dilated LSTM for temporal abstraction
- Goal embedding and transition policies

**Architecture:**
```
Manager: State → Goal Generation (every c steps)
Worker:  State + Goal → Action Selection
Motivation: Cosine(State_Embedding, Goal)
```

**Enhancements:**
- Advanced state encoding with embedding networks
- Intrinsic reward computation via cosine similarity
- Feudal hierarchy with proper temporal abstraction
- PPO-based policy optimization for both levels

### 4. Options Framework - **ENHANCED**

**Status:** Enhanced Implementation  

**Key Features:**
- Multiple learned options/skills (6 default)
- Option termination conditions
- Diversity bonuses for skill discovery
- Semi-Markov decision process formulation

**Components:**
```
Option Policies: Individual skill behaviors
Option Selector: High-level option selection
Termination Network: When to switch options
Diversity Mechanism: Encourage exploration
```

**Enhancements:**
- Advanced option discovery with diversity bonuses
- Comprehensive option analysis and visualization
- t-SNE visualization of option states
- Option usage statistics and performance tracking

---

## Training Recommendations

### For Best Results (HAC - Production Ready):
```bash
python train_hac_fixed.py
# OR
python launch_hierarchical_training.py --algorithm hac --episodes 500
```

### For Experimental Research:
```bash
# Test all algorithms
python launch_hierarchical_training.py --test-all

# Compare performance
python launch_hierarchical_training.py --benchmark

# Individual algorithm testing
python launch_hierarchical_training.py --algorithm hiro --test-mode
python launch_hierarchical_training.py --algorithm fun --test-mode  
python launch_hierarchical_training.py --algorithm options --test-mode
```

---

## Performance Comparison

| Metric | HAC | HIRO | FuN | Options |
|--------|-----|------|-----|---------|
| **Convergence Speed** | 5/5 | 4/5 | 3/5 | 3/5 |
| **Sample Efficiency** | 4/5 | 5/5 | 3/5 | 2/5 |
| **Goal Achievement** | 5/5 | 4/5 | 3/5 | 3/5 |
| **Exploration Quality** | 4/5 | 3/5 | 4/5 | 5/5 |
| **Implementation Status** | Production | Enhanced | Enhanced | Enhanced |

---

## Configuration Guide

### Environment Setup
```python
# All algorithms use unified configuration
base_env = AirGymEnv()
env = HierarchicalAirSimEnv(
    base_env,
    goal_dim=3,  # 3D position goals
    max_episode_steps=200
)
```

### Key Parameters

#### HAC (Production):
```python
config = HACConfig(
    num_levels=2,
    max_actions=15,
    subgoal_bounds=[-8.0, 8.0],
    atomic_noise=0.5,  # Fixed from original 0.3
    subgoal_noise=0.2
)
```

#### HIRO (Enhanced):
```python
config = HIROConfig(
    subgoal_freq=10,
    her_ratio=0.8,
    off_policy_correction=True,
    correction_radius=2.0
)
```

#### FuN (Enhanced):
```python
config = FuNConfig(
    manager_horizon=8,
    embedding_dim=256,
    goal_dim=16,
    alpha=0.5  # Intrinsic reward coefficient
)
```

#### Options (Enhanced):
```python
config = OptionsConfig(
    num_options=6,
    use_diversity_bonus=True,
    diversity_coef=0.1,
    option_min_length=4,
    option_max_length=15
)
```

---

## Monitoring & Analysis

### TensorBoard Visualization
```bash
# View training progress
tensorboard --logdir=results/hierarchical/

# Algorithm-specific logs
tensorboard --logdir=results/hierarchical/hac_fixed/logs
tensorboard --logdir=results/hierarchical/hiro_improved/logs
tensorboard --logdir=results/hierarchical/fun_improved/logs  
tensorboard --logdir=results/hierarchical/options_improved/logs
```

### Key Metrics Tracked:
- **Episode Rewards**: Total and component rewards
- **Goal Achievement**: Success rates and distances
- **Hierarchical Metrics**: Subgoal changes, option usage
- **Training Losses**: Policy, value, and auxiliary losses
- **Exploration Quality**: Diversity scores and coverage

---

## Troubleshooting

### Common Issues & Solutions:

#### 1. HAC Training Issues:
```bash
# If UAV only moves vertically:
# FIXED: Action space scaled to [-2.0, 2.0]
# FIXED: Proper network initialization
# SOLUTION: Use train_hac_fixed.py
```

#### 2. HIRO Memory Issues:
```bash
# Reduce buffer size for limited memory:
config.buffer_size = 100000  # Instead of 1000000
config.batch_size = 64       # Instead of 256
```

#### 3. FuN Convergence Issues:
```bash
# Adjust intrinsic reward coefficient:
config.alpha = 0.3           # Reduce if too much exploration
config.manager_horizon = 5   # Shorter horizons for faster feedback
```

#### 4. Options Diversity Issues:
```bash
# Increase diversity encouragement:
config.diversity_coef = 0.2  # Higher diversity bonus
config.num_options = 4       # Fewer options for better specialization
```

---

## Contributing

### Adding New Algorithms:
1. Create algorithm directory: `hierarchical_rl/new_algorithm/`
2. Implement agent class inheriting from `BaseHierarchicalAgent`
3. Add configuration class
4. Create training script
5. Add tests to `test_all_algorithms.py`
6. Update launcher: `launch_hierarchical_training.py`

### Code Standards:
- Follow existing code structure and naming conventions
- Include comprehensive docstrings and type hints
- Add proper logging and error handling
- Write corresponding tests
- Update documentation

---

## References

1. **HAC**: "Learning Multi-Level Hierarchies with Hindsight" (Levy et al., 2018)
2. **HIRO**: "Data-Efficient Hierarchical Reinforcement Learning" (Nachum et al., 2018)  
3. **FuN**: "FeUdal Networks for Hierarchical Reinforcement Learning" (Vezhnevets et al., 2017)
4. **Options**: "Between MDPs and Semi-MDPs" (Sutton et al., 1999) + "The Option-Critic Architecture" (Bacon et al., 2017)

---

## Support

For issues, questions, or contributions:

1. **Check Existing Issues**: Review `DEBUGGING_LOG.md` for known solutions
2. **Run Tests**: Use `python test_all_algorithms.py` to identify problems
3. **Algorithm Status**: Check individual algorithm status in launcher
4. **Performance Issues**: Refer to troubleshooting section above

---

**Status**: HAC Production Ready | Others Enhanced & Tested
**Last Updated**: August 2025
**Maintainer**: Hierarchical RL Development Team