# Hybrid MPC for TD-MPC2

This document describes the novel hybrid MPC implementation that combines classical MPC algorithms (like iLQG) with learned MPC (MPPI) in the TD-MPC2 framework.

## Overview

The hybrid approach leverages the best of both worlds:
- **Classical MPC** for initial timesteps using true simulator dynamics
- **Learned MPC** for remaining timesteps using the learned world model
- **Adaptive transition** that gradually shifts from classical to learned planning during training

## Research Motivation

Current model-based RL approaches often underutilize available simulators. While simulators provide perfect dynamics for short horizons, learned models excel at capturing complex long-term behaviors. This hybrid approach:

1. **Improves sample efficiency** during early training when learned models are inaccurate
2. **Provides theoretical guarantees** for short-term planning
3. **Maintains flexibility** for long-term complex behaviors
4. **Better utilizes simulators** that are often available but underused

## Implementation Architecture

### Core Components

1. **`hybrid_mpc.py`** - Core hybrid MPC implementation
   - `ClassicalMPC` - Abstract base class for classical algorithms
   - `iLQG` - Iterative Linear Quadratic Gaussian implementation
   - `EnvironmentInterface` - Interface to simulator dynamics
   - `HybridMPC` - Main hybrid planning coordinator

2. **`hybrid_tdmpc2.py`** - Extended TD-MPC2 agent
   - `HybridTDMPC2` - Main agent class with hybrid planning
   - Integrated MPPI with classical trajectory seeding
   - Adaptive transition scheduling

3. **`envs/dmcontrol_interface.py`** - DMControl-specific interface
   - Leverages MuJoCo's analytical derivatives
   - Task-specific cost functions
   - Efficient state conversion utilities

4. **`train_hybrid.py`** - Training script for hybrid agents
   - Modified trainer with transition scheduling
   - Hybrid-specific logging and monitoring

## Key Features

### Transition Scheduling
- **Linear**: Gradual linear reduction of classical horizon
- **Exponential**: Exponential decay of classical influence
- **Step**: Discrete transitions at specified intervals

### Classical Algorithms
- **iLQG**: Iterative Linear Quadratic Gaussian with line search
- **Extensible**: Easy to add LQR, DDP, or other classical methods

### Environment Support
- **DMControl**: Full support with analytical derivatives
- **Meta-World**: Compatible through MuJoCo interface
- **Extensible**: Framework for adding other simulators

## Usage

### Basic Training

```bash
# Train with hybrid MPC enabled
python train_hybrid.py task=cartpole-balance hybrid_mpc=true

# Customize transition schedule
python train_hybrid.py task=cheetah-run hybrid_mpc=true \
    max_classical_horizon=8 transition_steps=500000 \
    transition_schedule=exponential

# Compare with standard TD-MPC2
python train_hybrid.py task=walker-walk hybrid_mpc=false
```

### Configuration Parameters

```yaml
# Hybrid MPC settings
hybrid_mpc: true                    # Enable hybrid MPC
max_classical_horizon: 10           # Maximum classical planning horizon
min_classical_horizon: 0            # Minimum classical planning horizon
transition_steps: 1_000_000         # Steps over which to transition
transition_schedule: linear         # Transition schedule type
classical_algorithm: ilqg           # Classical algorithm to use

# iLQG specific parameters
ilqg_iterations: 10                 # Maximum iLQG iterations
ilqg_line_search_steps: 10          # Line search steps
ilqg_regularization: 1e-6           # Regularization parameter
ilqg_tolerance: 1e-6                # Convergence tolerance
finite_diff_eps: 1e-6               # Finite difference epsilon
```

### Programmatic Usage

```python
from tdmpc2.hybrid_tdmpc2 import HybridTDMPC2
from envs import make_env

# Create environment and agent
env = make_env(cfg)
agent = HybridTDMPC2(cfg, env)

# Training loop
for step in range(training_steps):
    # Update transition schedule
    agent.update_training_step(step)
    
    # Get action with hybrid planning
    action = agent.act(obs, t0=(step==0))
    
    # Monitor hybrid state
    info = agent.get_hybrid_info()
    print(f"Classical horizon: {info['classical_horizon']}")
```

## Research Potential

### Novel Contributions

1. **Hybrid Planning Architecture**: First approach to systematically combine classical and learned MPC
2. **Simulator Utilization**: Better leverages available simulators in model-based RL
3. **Adaptive Transition**: Novel scheduling approaches for smooth handoff
4. **Theoretical Foundation**: Combines guarantees of classical control with flexibility of learning

### Experimental Opportunities

1. **Sample Efficiency Studies**: Compare against pure learned approaches
2. **Robustness Analysis**: Evaluate performance under model mismatch
3. **Transfer Learning**: Study how classical initialization affects transfer
4. **Ablation Studies**: Analyze different transition schedules and horizons
5. **Scaling Studies**: Evaluate on increasingly complex tasks

### Potential Extensions

1. **Multi-fidelity Models**: Combine multiple simulator fidelities
2. **Uncertainty-aware Transition**: Use model uncertainty to guide transitions
3. **Task-specific Scheduling**: Learn optimal transition schedules per task
4. **Hierarchical Planning**: Extend to hierarchical classical-learned planning

## Implementation Details

### Classical MPC Integration

The iLQG implementation includes:
- Forward simulation using true dynamics
- Analytical Jacobian computation (when available)
- Backward pass with optimal control gains
- Line search for robust convergence

### Hybrid Value Estimation

The hybrid approach evaluates trajectories using:
- Classical dynamics and costs for initial timesteps
- Learned dynamics and rewards for remaining timesteps
- Smooth transition between evaluation methods

### Environment Interfaces

Each environment type has a specialized interface:
- **DMControl**: Uses MuJoCo's physics engine directly
- **Meta-World**: Leverages MuJoCo through Meta-World wrapper
- **Custom**: Extensible framework for new environments

## Testing

Run the test script to verify installation:

```bash
python test_hybrid.py
```

This will:
- Test hybrid agent creation
- Demonstrate transition scheduling
- Show action selection with different horizons
- Validate configuration handling

## Performance Considerations

### Computational Overhead
- Classical MPC adds computational cost for derivative computation
- Cost decreases over training as classical horizon reduces
- Parallelizable across multiple environments

### Memory Usage
- Additional storage for classical trajectories
- Environment interface state management
- Minimal overhead compared to base TD-MPC2

### Scalability
- Scales to high-dimensional action spaces
- Environment-dependent derivative computation cost
- Efficient implementation with batched operations

## Future Work

### Immediate Extensions
1. Add support for more classical algorithms (LQR, DDP)
2. Implement analytical derivatives for more environments
3. Add uncertainty-based transition scheduling
4. Develop multi-task hybrid training

### Research Directions
1. Theoretical analysis of hybrid convergence properties
2. Optimal transition schedule learning
3. Integration with other model-based RL approaches
4. Application to real-world robotics tasks

## Citation

If you use this hybrid MPC implementation in your research, please cite:

```bibtex
@misc{hybrid_tdmpc2,
  title={Hybrid MPC for TD-MPC2: Combining Classical and Learned Planning},
  author={[Your Name]},
  year={2025},
  note={Extension of TD-MPC2 with hybrid classical-learned planning}
}
```

And the original TD-MPC2 paper:

```bibtex
@inproceedings{hansen2024tdmpc2,
  title={TD-MPC2: Scalable, Robust World Models for Continuous Control}, 
  author={Nicklas Hansen and Hao Su and Xiaolong Wang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## Contributing

Contributions are welcome! Areas for contribution:
- Additional classical algorithms
- New environment interfaces
- Improved transition scheduling
- Performance optimizations
- Documentation improvements

Please follow the existing code style and include tests for new features.
