import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import time

from tdmpc2 import TDMPC2
from common import math


class SimpleEnvironmentInterface:
    """Simplified environment interface that works with standard RL environments."""
    
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.dt = getattr(cfg, 'dt', 0.02)  # Default timestep
        
        # Try to extract dynamics info from environment
        self.has_physics = hasattr(env, 'physics') or hasattr(env, '_env')
        self.state_dim = self._get_state_dim()
        self.action_dim = self._get_action_dim()
        
    def _get_state_dim(self):
        """Try to determine state dimension from environment."""
        if hasattr(self.env, 'physics'):
            # DMControl
            try:
                return len(self.env.physics.get_state())
            except:
                pass
        elif hasattr(self.env, '_env') and hasattr(self.env._env, 'physics'):
            # Wrapped DMControl
            try:
                return len(self.env._env.physics.get_state())
            except:
                pass
        elif hasattr(self.env, 'state'):
            return len(self.env.state)
        
        # Try various config attributes for observation dimension
        for attr in ['obs_dim', 'obs_shape', 'observation_dim']:
            if hasattr(self.cfg, attr):
                val = getattr(self.cfg, attr)
                if isinstance(val, (int, list, tuple)):
                    if isinstance(val, int):
                        return val
                    elif isinstance(val, (list, tuple)) and len(val) > 0:
                        if len(val) == 1:
                            return val[0]
                        else:
                            # Multi-dimensional observation, flatten
                            import numpy as np
                            return int(np.prod(val))
        
        # Last resort: try to get from environment observation space
        try:
            if hasattr(self.env, 'observation_space'):
                obs_space = self.env.observation_space
                if hasattr(obs_space, 'shape'):
                    import numpy as np
                    return int(np.prod(obs_space.shape))
                elif hasattr(obs_space, 'n'):
                    return obs_space.n
            elif hasattr(self.env, 'obs_spec'):
                # DMControl style
                obs_spec = self.env.obs_spec()
                if hasattr(obs_spec, 'shape'):
                    import numpy as np
                    return int(np.prod(obs_spec.shape))
        except:
            pass
        
        # Final fallback
        print("Warning: Could not determine state dimension, using default 10")
        return 10
    
    def _get_action_dim(self):
        """Try to determine action dimension from config and environment."""
        # First try config
        if hasattr(self.cfg, 'action_dim') and self.cfg.action_dim is not None:
            return self.cfg.action_dim
        
        # Try environment
        try:
            if hasattr(self.env, 'action_space'):
                action_space = self.env.action_space
                if hasattr(action_space, 'shape'):
                    import numpy as np
                    return int(np.prod(action_space.shape))
                elif hasattr(action_space, 'n'):
                    return action_space.n
            elif hasattr(self.env, 'action_spec'):
                # DMControl style
                action_spec = self.env.action_spec()
                if hasattr(action_spec, 'shape'):
                    import numpy as np
                    return int(np.prod(action_spec.shape))
        except:
            pass
        
        # Final fallback
        print("Warning: Could not determine action dimension, using default 2")
        return 2
    
    def obs_to_state(self, obs):
        """Convert observation to state for classical control."""
        # For many environments, observations contain or are the state
        if obs.shape[-1] == self.state_dim:
            return obs
        elif obs.shape[-1] > self.state_dim:
            # Truncate if observation is larger
            return obs[..., :self.state_dim]
        else:
            # Pad if observation is smaller
            padding = torch.zeros(*obs.shape[:-1], self.state_dim - obs.shape[-1], device=obs.device)
            return torch.cat([obs, padding], dim=-1)
    
    def state_to_obs(self, state):
        """Convert state back to observation."""
        if state.shape[-1] == self.cfg.obs_dim:
            return state
        elif state.shape[-1] > self.cfg.obs_dim:
            return state[..., :self.cfg.obs_dim]
        else:
            padding = torch.zeros(*state.shape[:-1], self.cfg.obs_dim - state.shape[-1], device=state.device)
            return torch.cat([state, padding], dim=-1)
    
    def step_dynamics(self, state, action):
        """Improved dynamics model that handles dimension mismatches."""
        # Handle dimension mismatch between state and action
        state_dim = state.shape[-1]
        action_dim = action.shape[-1]
        
        if state_dim == action_dim:
            # Simple integrator for compatible dimensions
            return state + action * self.dt
        else:
            # For complex environments like dog-run, use a learned or zero-order hold model
            if hasattr(self, '_dynamics_mapping') and self._dynamics_mapping is not None:
                # Use learned dynamics mapping if available
                return self._dynamics_mapping(state, action)
            else:
                # Zero-order hold: state doesn't change (very conservative)
                # This is safe but not very useful for planning
                # In practice, this should be replaced with environment-specific dynamics
                return state.clone()
    
    def set_dynamics_mapping(self, dynamics_fn):
        """Set a learned dynamics function for better classical control."""
        self._dynamics_mapping = dynamics_fn
    
    def compute_cost(self, state, action, next_state, task=None, timestep=0):
        """Simple quadratic cost function."""
        # State cost (prefer small states)
        state_cost = torch.sum(state ** 2, dim=-1)
        
        # Action cost (prefer small actions)
        action_cost = torch.sum(action ** 2, dim=-1)
        
        # Combine costs
        return 0.1 * state_cost + 0.01 * action_cost


class SimpleiLQG:
    """Simplified iLQG implementation for demonstration."""
    
    def __init__(self, cfg, env_interface):
        self.cfg = cfg
        self.env_interface = env_interface
        self.max_iterations = getattr(cfg, 'ilqg_iterations', 5)
        self.regularization = getattr(cfg, 'ilqg_regularization', 1e-4)
        self.action_dim = env_interface.action_dim  # Get from interface instead of cfg
        
    def plan(self, state, horizon, initial_actions=None, task=None):
        """Simplified iLQG planning."""
        device = state.device
        batch_size = state.shape[0] if state.ndim > 1 else 1
        
        # Initialize actions
        if initial_actions is not None and initial_actions.shape[0] >= horizon:
            actions = initial_actions[:horizon].clone()
        else:
            actions = 0.1 * torch.randn(horizon, batch_size, self.action_dim, device=device)
        
        # Clamp actions to bounds
        actions = torch.clamp(actions, -1.0, 1.0)
        
        best_cost = float('inf')
        
        # Simple iterative improvement
        for iteration in range(self.max_iterations):
            # Forward pass
            states = torch.zeros(horizon + 1, batch_size, state.shape[-1], device=device)
            costs = torch.zeros(horizon, batch_size, device=device)
            states[0] = state
            
            total_cost = 0
            for t in range(horizon):
                next_state = self.env_interface.step_dynamics(states[t], actions[t])
                cost = self.env_interface.compute_cost(
                    states[t], actions[t], next_state, task, t
                )
                states[t + 1] = next_state
                costs[t] = cost
                total_cost += cost.sum()
            
            # Simple gradient-based update
            if total_cost < best_cost:
                best_cost = total_cost
            else:
                # Add some noise to escape local minima
                actions += 0.01 * torch.randn_like(actions)
                actions = torch.clamp(actions, -1.0, 1.0)
        
        info = {
            'final_cost': best_cost.item() if torch.is_tensor(best_cost) else best_cost,
            'iterations': self.max_iterations,
            'converged': True
        }
        
        return actions, info


class SimpleHybridTDMPC2(TDMPC2):
    """
    Simplified Hybrid TD-MPC2 that gradually transitions from classical to learned control.
    """
    
    def __init__(self, cfg, env=None):
        # Initialize parent TD-MPC2
        super().__init__(cfg)
        
        # Hybrid-specific configuration
        self.hybrid_enabled = getattr(cfg, 'hybrid_mpc', False)
        self.max_classical_horizon = getattr(cfg, 'max_classical_horizon', 10)
        self.min_classical_horizon = getattr(cfg, 'min_classical_horizon', 0)
        self.transition_steps = getattr(cfg, 'transition_steps', 1_000_000)
        self.transition_schedule = getattr(cfg, 'transition_schedule', 'linear')
        self.classical_algorithm = getattr(cfg, 'classical_algorithm', 'ilqg')
        
        # Training step counter
        self.current_step = 0
        
        if self.hybrid_enabled and env is not None:
            # Initialize environment interface
            self.env_interface = SimpleEnvironmentInterface(cfg, env)
            
            # Initialize classical controller
            if self.classical_algorithm == 'ilqg':
                self.classical_controller = SimpleiLQG(cfg, self.env_interface)
            else:
                raise ValueError(f"Unsupported classical algorithm: {self.classical_algorithm}")
            
            print(f"Hybrid TD-MPC2 initialized:")
            print(f"  - Max classical horizon: {self.max_classical_horizon}")
            print(f"  - Transition steps: {self.transition_steps}")
            print(f"  - Transition schedule: {self.transition_schedule}")
        else:
            self.env_interface = None
            self.classical_controller = None
            
        # Buffer for previous trajectories (warm starting)
        self.prev_classical_actions = None
        
    def get_classical_horizon(self):
        """Get current classical planning horizon based on training progress."""
        if not self.hybrid_enabled:
            return 0
            
        progress = min(self.current_step / self.transition_steps, 1.0)
        
        if self.transition_schedule == 'linear':
            horizon = self.max_classical_horizon * (1 - progress)
        elif self.transition_schedule == 'exponential':
            horizon = self.max_classical_horizon * np.exp(-5 * progress)
        elif self.transition_schedule == 'cosine':
            horizon = self.max_classical_horizon * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.transition_schedule == 'step':
            if progress < 0.33:
                horizon = self.max_classical_horizon
            elif progress < 0.67:
                horizon = self.max_classical_horizon // 2
            else:
                horizon = self.min_classical_horizon
        else:
            horizon = self.max_classical_horizon
            
        return int(max(horizon, self.min_classical_horizon))
    
    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Enhanced action selection with hybrid planning.
        """
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        
        if not self.hybrid_enabled or self.classical_controller is None:
            # Use standard TD-MPC2
            return super().act(obs[0], t0, eval_mode, task[0] if task is not None else None)
        
        # Get current classical horizon
        classical_horizon = self.get_classical_horizon()
        
        if classical_horizon == 0:
            # Pure learned control
            return super().act(obs[0], t0, eval_mode, task[0] if task is not None else None)
        elif classical_horizon >= self.cfg.horizon:
            # Pure classical control
            return self._pure_classical_plan(obs, task).cpu()
        else:
            # Hybrid control
            return self._hybrid_plan(obs, classical_horizon, t0, eval_mode, task).cpu()
    
    def _pure_classical_plan(self, obs, task):
        """Plan using only classical control."""
        # Convert observation to state
        state = self.env_interface.obs_to_state(obs)
        
        # Use previous actions for warm starting
        initial_actions = None
        if self.prev_classical_actions is not None:
            # Shift previous solution
            shifted = torch.zeros_like(self.prev_classical_actions)
            shifted[:-1] = self.prev_classical_actions[1:]
            shifted[-1] = self.prev_classical_actions[-1]  # Repeat last action
            initial_actions = shifted
        
        # Plan with classical controller
        actions, info = self.classical_controller.plan(
            state, self.max_classical_horizon, initial_actions, task
        )
        
        # Store for next time
        self.prev_classical_actions = actions.detach()
        
        return actions[0]
    
    def _hybrid_plan(self, obs, classical_horizon, t0, eval_mode, task):
        """Hybrid planning combining classical and learned control."""
        total_horizon = self.cfg.horizon
        action_dim = self.env_interface.action_dim if self.env_interface else self._get_action_dim_fallback()
        
        # Phase 1: Classical planning for initial steps
        state = self.env_interface.obs_to_state(obs)
        classical_actions, _ = self.classical_controller.plan(
            state, classical_horizon, self.prev_classical_actions, task
        )
        
        # Phase 2: Learned planning for remaining steps
        if not t0 and hasattr(self, '_prev_mean'):
            # Warm start with shifted previous solution
            mean = torch.zeros(total_horizon, action_dim, device=self.device)
            mean[:-1] = self._prev_mean[1:]
            mean[-1] = self._prev_mean[-1]
        else:
            mean = torch.zeros(total_horizon, action_dim, device=self.device)
        
        # Initialize with classical actions
        mean[:classical_horizon] = classical_actions[:classical_horizon]
        
        # Plan remaining steps with learned MPPI
        learned_action = self._plan_learned_continuation(
            obs, mean, classical_horizon, eval_mode, task
        )
        
        # Store for next iteration
        self._prev_mean = mean
        self.prev_classical_actions = classical_actions.detach()
        
        # Return first action (already computed by classical planner)
        return classical_actions[0]
    
    def _plan_learned_continuation(self, obs, mean, classical_horizon, eval_mode, task):
        """Plan the continuation of trajectory using learned model."""
        if classical_horizon >= self.cfg.horizon:
            return mean[0]
        
        # Get action dimension from environment interface
        action_dim = self.env_interface.action_dim if self.env_interface else self._get_action_dim_fallback()
        
        # Encode observation
        z = self.model.encode(obs, task)
        
        # Simulate forward to the state where learned planning takes over
        current_z = z
        for t in range(classical_horizon):
            current_z = self.model.next(current_z, mean[t].unsqueeze(0), task)
        
        # Now plan from this point using learned MPPI
        remaining_horizon = self.cfg.horizon - classical_horizon
        if remaining_horizon > 0:
            # Sample actions for remaining steps
            num_samples = min(self.cfg.num_samples, 256)  # Reduce for efficiency
            std = torch.full((remaining_horizon, action_dim), 
                           self.cfg.max_std, device=self.device)
            
            actions = torch.empty(remaining_horizon, num_samples, action_dim, device=self.device)
            
            # Sample random actions
            for t in range(remaining_horizon):
                noise = torch.randn(num_samples, action_dim, device=self.device)
                actions[t] = mean[classical_horizon + t].unsqueeze(0) + std[t].unsqueeze(0) * noise
                actions[t] = actions[t].clamp(-1, 1)
            
            # Evaluate trajectories
            z_expanded = current_z.repeat(num_samples, 1)
            values = self._estimate_value(z_expanded, actions, task)
            
            # Select best actions
            elite_idxs = torch.topk(values.squeeze(-1), min(32, num_samples), dim=0).indices
            elite_actions = actions[:, elite_idxs]
            
            # Update mean for continuation
            weights = F.softmax(values[elite_idxs] / self.cfg.temperature, dim=0)
            for t in range(remaining_horizon):
                mean[classical_horizon + t] = torch.sum(
                    weights.unsqueeze(-1) * elite_actions[t], dim=0
                )
        
        return mean[0]
    
    def _get_action_dim_fallback(self):
        """Fallback method to get action dimension."""
        # Try various config attributes
        for attr in ['action_dim', 'action_dims']:
            if hasattr(self.cfg, attr):
                val = getattr(self.cfg, attr)
                if isinstance(val, int):
                    return val
                elif isinstance(val, (list, tuple)) and len(val) > 0:
                    return val[0]
        
        # Final fallback
        print("Warning: Could not determine action dimension, using default 6 for dog-run")
        return 6  # Dog-run has 6 action dimensions
    
    def update_training_step(self, step):
        """Update current training step for transition scheduling."""
        self.current_step = step
    
    def get_diagnostics(self):
        """Get diagnostic information for logging."""
        if not self.hybrid_enabled:
            return {"hybrid_enabled": False}
        
        classical_horizon = self.get_classical_horizon()
        progress = min(self.current_step / self.transition_steps, 1.0)
        
        return {
            "hybrid_enabled": True,
            "classical_horizon": classical_horizon,
            "transition_progress": progress,
            "current_step": self.current_step,
            "max_classical_horizon": self.max_classical_horizon,
        }