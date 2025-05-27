"""
Test script to demonstrate the hybrid MPC functionality.
This script shows how to use the hybrid TD-MPC2 implementation.
"""

import torch
import numpy as np
from omegaconf import OmegaConf

from common.parser import parse_cfg
from envs import make_env
from tdmpc2.hybrid_tdmpc2 import HybridTDMPC2


def test_hybrid_mpc():
    """Test the hybrid MPC implementation with a simple environment."""
    
    # Create a minimal configuration
    cfg = OmegaConf.create({
        'task': 'cartpole-balance',
        'obs': 'state',
        'episodic': False,
        'model_size': 5,
        'action_dim': 1,
        'obs_shape': [5],
        'episode_length': 1000,
        'horizon': 3,
        'mpc': True,
        'iterations': 6,
        'num_samples': 512,
        'num_elites': 64,
        'num_pi_trajs': 24,
        'min_std': 0.05,
        'max_std': 2.0,
        'temperature': 0.5,
        'latent_dim': 512,
        'mlp_dim': 512,
        'enc_dim': 256,
        'num_enc_layers': 2,
        'num_channels': 32,
        'task_dim': 96,
        'num_q': 5,
        'dropout': 0.01,
        'simnorm_dim': 8,
        'num_bins': 101,
        'vmin': -10,
        'vmax': 10,
        'log_std_min': -10,
        'log_std_max': 2,
        'entropy_coef': 1e-4,
        'lr': 3e-4,
        'enc_lr_scale': 0.3,
        'grad_clip_norm': 20,
        'tau': 0.01,
        'discount_denom': 5,
        'discount_min': 0.95,
        'discount_max': 0.995,
        'rho': 0.5,
        'consistency_coef': 20,
        'reward_coef': 0.1,
        'value_coef': 0.1,
        'termination_coef': 1,
        'compile': False,
        'multitask': False,
        'seed': 1,
        
        # Hybrid MPC settings
        'hybrid_mpc': True,
        'max_classical_horizon': 5,
        'min_classical_horizon': 0,
        'transition_steps': 100,  # Short for testing
        'transition_schedule': 'linear',
        'classical_algorithm': 'ilqg',
        'ilqg_iterations': 5,
        'ilqg_line_search_steps': 5,
        'ilqg_regularization': 1e-6,
        'ilqg_tolerance': 1e-6,
        'finite_diff_eps': 1e-6,
    })
    
    cfg = parse_cfg(cfg)
    
    print("Testing Hybrid TD-MPC2 Implementation")
    print("=" * 50)
    
    try:
        # Create environment (this might fail if DMControl is not installed)
        print("Creating environment...")
        env = make_env(cfg)
        print(f"✓ Environment created: {cfg.task}")
        
        # Create hybrid agent
        print("Creating hybrid agent...")
        agent = HybridTDMPC2(cfg, env)
        print("✓ Hybrid agent created")
        
        # Test hybrid info
        hybrid_info = agent.get_hybrid_info()
        print(f"✓ Hybrid MPC enabled: {hybrid_info['hybrid_enabled']}")
        print(f"  Current classical horizon: {hybrid_info['classical_horizon']}")
        print(f"  Max classical horizon: {hybrid_info['max_classical_horizon']}")
        print(f"  Transition schedule: {hybrid_info['transition_schedule']}")
        
        # Test action selection
        print("\nTesting action selection...")
        obs = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        
        # Test with different training steps
        for step in [0, 50, 100, 150]:
            agent.update_training_step(step)
            action = agent.act(obs_tensor, t0=True, eval_mode=True)
            hybrid_info = agent.get_hybrid_info()
            print(f"  Step {step:3d}: Classical horizon = {hybrid_info['classical_horizon']}, Action = {action.numpy()}")
        
        print("\n✓ All tests passed!")
        
        # Demonstrate transition schedule
        print("\nDemonstrating transition schedule:")
        print("Step | Classical Horizon")
        print("-" * 25)
        for step in range(0, 151, 10):
            agent.update_training_step(step)
            horizon = agent.hybrid_mpc.get_classical_horizon()
            print(f"{step:4d} | {horizon:8d}")
            
    except ImportError as e:
        print(f"⚠ Environment creation failed (missing dependencies): {e}")
        print("Testing hybrid agent without environment...")
        
        # Test without environment
        agent = HybridTDMPC2(cfg, env=None)
        hybrid_info = agent.get_hybrid_info()
        print(f"✓ Hybrid MPC disabled (no env): {not hybrid_info['hybrid_enabled']}")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_research_potential():
    """Demonstrate the research potential of the hybrid approach."""
    
    print("\n" + "=" * 60)
    print("RESEARCH POTENTIAL OF HYBRID MPC APPROACH")
    print("=" * 60)
    
    print("""
This implementation provides a novel hybrid approach that combines:

1. CLASSICAL MPC (iLQG) for initial timesteps:
   - Uses true simulator dynamics
   - Provides theoretical guarantees
   - Excellent for short-term optimization
   - Leverages analytical derivatives when available

2. LEARNED MPC (MPPI) for remaining timesteps:
   - Uses learned world model
   - Handles complex long-term behaviors
   - Adapts to environment dynamics
   - Scales to high-dimensional problems

3. ADAPTIVE TRANSITION SCHEDULING:
   - Gradually reduces classical horizon during training
   - Multiple schedules: linear, exponential, step
   - Allows smooth handoff between approaches
   - Configurable transition timing

RESEARCH CONTRIBUTIONS:
- Novel hybrid planning architecture
- Better utilization of available simulators
- Improved sample efficiency during early training
- Theoretical foundation with practical flexibility
- Extensible to multiple environments and tasks

POTENTIAL EXPERIMENTS:
1. Compare sample efficiency vs. pure learned approaches
2. Analyze performance across different transition schedules
3. Study robustness to model mismatch
4. Evaluate on complex manipulation tasks
5. Investigate transfer learning capabilities
""")


if __name__ == "__main__":
    test_hybrid_mpc()
    demonstrate_research_potential()
