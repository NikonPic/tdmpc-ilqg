import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict
from tdmpc2.tdmpc2 import TDMPC2
from tdmpc2.hybrid_mpc import HybridMPC, MPCConfig, EnvironmentInterface


@dataclass
class HybridTDMPC2Config(MPCConfig):
    """Extended configuration for Hybrid TD-MPC2."""
    # Hybrid-specific parameters
    hybrid_mpc: bool = True
    hybrid_value_estimation: str = 'mixed'  # 'mixed', 'learned', 'classical'
    classical_weight_schedule: str = 'adaptive'  # 'fixed', 'linear', 'adaptive'
    state_conversion_method: str = 'identity'  # 'identity', 'encoder', 'learned'
    
    # Blending parameters
    blend_actions: bool = True
    blend_horizon: int = 3  # Number of steps to blend over
    blend_temperature: float = 0.1
    
    # Classical MPC trust region
    classical_trust_weight: float = 0.3
    classical_trust_decay: float = 0.99
    
    # Performance tracking
    track_performance: bool = True
    performance_window: int = 100
    
    # Warm-starting
    warm_start_classical: bool = True
    warm_start_learned: bool = True
    
    # Multi-task specific
    task_specific_horizons: Dict[int, int] = field(default_factory=dict)


class ImprovedEnvironmentInterface(EnvironmentInterface):
    """Enhanced environment interface with better state/observation handling."""
    
    def __init__(self, cfg: HybridTDMPC2Config, env, world_model: WorldModel):
        super().__init__(cfg, env)
        self.world_model = world_model
        self._state_dim = None
        self._obs_dim = None
        
    @property
    def state_dim(self):
        """Get state dimension from environment."""
        if self._state_dim is None:
            if hasattr(self.env, 'state_dim'):
                self._state_dim = self.env.state_dim
            elif hasattr(self.env, 'observation_spec'):
                # DMControl style
                spec = self.env.observation_spec()
                self._state_dim = sum([np.prod(v.shape) for v in spec.values()])
            else:
                # Fallback to observation dimension
                self._state_dim = self.cfg.obs_dim
        return self._state_dim
    
    @property
    def obs_dim(self):
        """Get observation dimension."""
        if self._obs_dim is None:
            self._obs_dim = self.cfg.obs_dim
        return self._obs_dim
    
    def obs_to_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Convert observation to state for classical control."""
        if self.cfg.state_conversion_method == 'identity':
            # Direct mapping (assumes obs contains full state)
            return obs
        elif self.cfg.state_conversion_method == 'encoder':
            # Use world model encoder then project to state space
            with torch.no_grad():
                z = self.world_model.encode(obs)
                # Project latent to state dimension
                if hasattr(self.world_model, 'latent_to_state'):
                    return self.world_model.latent_to_state(z)
                else:
                    # Simple linear projection
                    if not hasattr(self, '_latent_to_state_proj'):
                        self._latent_to_state_proj = torch.nn.Linear(
                            z.shape[-1], self.state_dim
                        ).to(z.device)
                    return self._latent_to_state_proj(z)
        elif self.cfg.state_conversion_method == 'learned':
            # Use a learned mapping
            if hasattr(self.world_model, 'obs_to_state'):
                return self.world_model.obs_to_state(obs)
            else:
                return obs  # Fallback
        else:
            return obs
    
    def state_to_obs(self, state: torch.Tensor) -> torch.Tensor:
        """Convert state back to observation format."""
        if self.cfg.state_conversion_method == 'identity':
            return state
        elif hasattr(self.world_model, 'state_to_obs'):
            return self.world_model.state_to_obs(state)
        else:
            # Pad or truncate as needed
            if state.shape[-1] < self.obs_dim:
                padding = torch.zeros(*state.shape[:-1], self.obs_dim - state.shape[-1], device=state.device)
                return torch.cat([state, padding], dim=-1)
            else:
                return state[..., :self.obs_dim]
    
    def compute_reward_from_cost(self, cost: torch.Tensor) -> torch.Tensor:
        """Convert cost to reward for value estimation."""
        # Simple negation with optional scaling
        return -cost * self.cfg.get('cost_to_reward_scale', 1.0)


class HybridTDMPC2(TDMPC2):
    """
    Enhanced Hybrid TD-MPC2 agent with improved integration between classical and learned control.
    """

    def __init__(self, cfg: HybridTDMPC2Config, env=None):
        # Initialize parent TD-MPC2
        super().__init__(cfg)
        
        self.cfg = cfg  # Use hybrid config
        self.env = env
        self.hybrid_mpc_enabled = cfg.hybrid_mpc
        
        if self.hybrid_mpc_enabled and env is not None:
            # Create enhanced environment interface
            self.env_interface = ImprovedEnvironmentInterface(cfg, env, self.model)
            
            # Initialize hybrid MPC components
            self.hybrid_mpc = HybridMPC(cfg, env, self.model)
            self.hybrid_mpc.env_interface = self.env_interface
            
            # Performance tracking
            self.performance_tracker = PerformanceTracker(cfg) if cfg.track_performance else None
            
            # Trajectory buffers for warm-starting
            self.classical_trajectory_buffer = TrajectoryBuffer(
                cfg.horizon, cfg.action_dim, device=self.device
            )
            self.learned_trajectory_buffer = TrajectoryBuffer(
                cfg.horizon, cfg.action_dim, device=self.device
            )
            
            print(f"Hybrid TD-MPC2 initialized with:")
            print(f"  - Max classical horizon: {cfg.max_classical_horizon}")
            print(f"  - Transition schedule: {cfg.transition_schedule}")
            print(f"  - Value estimation: {cfg.hybrid_value_estimation}")
            print(f"  - State conversion: {cfg.state_conversion_method}")
        else:
            self.hybrid_mpc = None
            self.env_interface = None
            print("Using standard TD-MPC2 planning")

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """Enhanced action selection with hybrid planning."""
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
            
        if self.hybrid_mpc_enabled and self.hybrid_mpc is not None:
            if self.cfg.mpc:
                # Use enhanced hybrid planning
                action, info = self._enhanced_hybrid_plan(
                    obs, t0=t0, eval_mode=eval_mode, task=task, return_info=True
                )
                
                # Track performance if enabled
                if self.performance_tracker and not eval_mode:
                    self.performance_tracker.update(info)
                
                return action.cpu()
            else:
                # Fallback to policy
                z = self.model.encode(obs, task)
                action, info = self.model.pi(z, task)
                if eval_mode:
                    action = info["mean"]
                return action[0].cpu()
        else:
            # Use standard TD-MPC2
            return super().act(obs[0], t0, eval_mode, task[0] if task is not None else None)

    @torch.no_grad()
    def _enhanced_hybrid_plan(self, obs, t0=False, eval_mode=False, task=None, return_info=False):
        """
        Enhanced hybrid planning with better integration between classical and learned control.
        """
        # Get current planning horizons
        classical_horizon = self._get_adaptive_classical_horizon(task)
        total_horizon = self.cfg.horizon
        
        info = {
            'classical_horizon': classical_horizon,
            'total_horizon': total_horizon,
            'planning_mode': 'hybrid'
        }
        
        # Encode observation for learned planning
        z = self.model.encode(obs, task)
        
        # Get state for classical planning
        state = self.env_interface.obs_to_state(obs)
        
        if classical_horizon == 0:
            # Pure learned MPC
            info['planning_mode'] = 'learned'
            action = self._plan(obs, t0, eval_mode, task)
            
        elif classical_horizon >= total_horizon:
            # Pure classical MPC
            info['planning_mode'] = 'classical'
            action = self._pure_classical_plan(state, obs, z, classical_horizon, task, info)
            
        else:
            # True hybrid planning
            action = self._integrated_hybrid_plan(
                state, obs, z, classical_horizon, t0, eval_mode, task, info
            )
        
        if return_info:
            return action, info
        return action

    def _get_adaptive_classical_horizon(self, task=None):
        """Get classical horizon with task-specific and performance-based adaptation."""
        # Task-specific horizon if available
        if task is not None and task.item() in self.cfg.task_specific_horizons:
            base_horizon = self.cfg.task_specific_horizons[task.item()]
        else:
            base_horizon = self.hybrid_mpc.get_classical_horizon()
        
        # Performance-based adaptation
        if self.performance_tracker and self.cfg.classical_weight_schedule == 'adaptive':
            performance_factor = self.performance_tracker.get_performance_factor()
            adapted_horizon = int(base_horizon * performance_factor)
            return max(self.cfg.min_classical_horizon, 
                      min(adapted_horizon, self.cfg.max_classical_horizon))
        
        return base_horizon

    @torch.no_grad()
    def _pure_classical_plan(self, state, obs, z, horizon, task, info):
        """Pure classical planning with learned value function terminal cost."""
        # Get warm-start from buffer
        initial_actions = None
        if self.cfg.warm_start_classical:
            initial_actions = self.classical_trajectory_buffer.get_shifted_trajectory()
        
        # Plan with classical MPC
        actions, plan_info = self.hybrid_mpc.classical_mpc.plan(
            state, horizon, initial_actions, task
        )
        
        # Store trajectory
        self.classical_trajectory_buffer.update(actions)
        
        # Augment with learned value estimate for terminal state
        if self.cfg.hybrid_value_estimation in ['mixed', 'learned']:
            # Simulate to terminal state
            terminal_state = state
            for t in range(min(horizon, actions.shape[0])):
                terminal_state = self.env_interface.step_dynamics(terminal_state, actions[t])
            
            # Convert to observation and get value estimate
            terminal_obs = self.env_interface.state_to_obs(terminal_state)
            terminal_z = self.model.encode(terminal_obs, task)
            terminal_action, _ = self.model.pi(terminal_z, task)
            terminal_value = self.model.Q(terminal_z, terminal_action, task, return_type='avg')
            
            info['terminal_value'] = terminal_value.item()
        
        info.update(plan_info)
        return actions[0]

    @torch.no_grad()
    def _integrated_hybrid_plan(self, state, obs, z, classical_horizon, t0, eval_mode, task, info):
        """
        Fully integrated hybrid planning that seamlessly combines classical and learned control.
        """
        # Initialize trajectory collections
        trajectories = self._initialize_trajectory_collection(
            state, obs, z, classical_horizon, task
        )
        
        # Run integrated MPPI with hybrid dynamics
        mean = self._get_initial_mean(classical_horizon, trajectories['classical'])
        std = self._get_adaptive_std(classical_horizon)
        
        # Main MPPI loop with hybrid dynamics
        for iteration in range(self.cfg.iterations):
            # Sample actions with trust region around classical solution
            actions = self._sample_hybrid_actions(
                mean, std, trajectories, classical_horizon, iteration
            )
            
            # Evaluate trajectories with hybrid dynamics
            values = self._evaluate_hybrid_trajectories(
                state, z, actions, classical_horizon, task
            )
            
            # Update distribution with classical trust region
            mean, std = self._update_hybrid_distribution(
                actions, values, mean, classical_horizon, iteration
            )
        
        # Select final action with optional blending
        action = self._select_hybrid_action(
            mean[0], std[0], trajectories['classical'][0], 
            classical_horizon, eval_mode
        )
        
        # Update trajectory buffers
        self._update_trajectory_buffers(mean, trajectories['classical'])
        
        # Store mean for next iteration
        if hasattr(self, '_prev_mean'):
            self._prev_mean.copy_(mean)
        
        return action.clamp(-1, 1)

    def _initialize_trajectory_collection(self, state, obs, z, classical_horizon, task):
        """Initialize trajectory collection with classical, learned, and policy rollouts."""
        trajectories = {}
        
        # Classical trajectory
        initial_actions = None
        if self.cfg.warm_start_classical:
            initial_actions = self.classical_trajectory_buffer.get_shifted_trajectory()
            
        classical_actions, _ = self.hybrid_mpc.classical_mpc.plan(
            state, self.cfg.horizon, initial_actions, task
        )
        trajectories['classical'] = classical_actions
        
        # Policy trajectory
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(
                self.cfg.horizon, self.cfg.num_pi_trajs, 
                self.cfg.action_dim, device=self.device
            )
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon):
                pi_actions[t], _ = self.model.pi(_z, task)
                _z = self.model.next(_z, pi_actions[t], task)
            trajectories['policy'] = pi_actions
        
        # Previous learned trajectory
        if self.cfg.warm_start_learned and hasattr(self, '_prev_mean'):
            trajectories['previous'] = self._prev_mean.clone()
        
        return trajectories

    def _sample_hybrid_actions(self, mean, std, trajectories, classical_horizon, iteration):
        """Sample actions with trust region around classical solution."""
        num_samples = self.cfg.num_samples
        actions = torch.empty(
            self.cfg.horizon, num_samples, self.cfg.action_dim, 
            device=self.device
        )
        
        # Deterministic samples (classical, policy, etc.)
        sample_idx = 0
        
        # Add classical trajectory
        actions[:, sample_idx] = trajectories['classical']
        sample_idx += 1
        
        # Add policy trajectories
        if 'policy' in trajectories and self.cfg.num_pi_trajs > 0:
            num_pi = min(self.cfg.num_pi_trajs, num_samples - sample_idx)
            actions[:, sample_idx:sample_idx+num_pi] = trajectories['policy'][:, :num_pi]
            sample_idx += num_pi
        
        # Sample remaining actions with trust region
        num_random = num_samples - sample_idx
        if num_random > 0:
            # Use different sampling strategies for classical and learned horizons
            for t in range(self.cfg.horizon):
                if t < classical_horizon:
                    # Tighter distribution around classical solution
                    trust_weight = self.cfg.classical_trust_weight * \
                                 (self.cfg.classical_trust_decay ** iteration)
                    sample_mean = trust_weight * trajectories['classical'][t] + \
                                (1 - trust_weight) * mean[t]
                    sample_std = std[t] * (1 - trust_weight)
                else:
                    # Standard sampling for learned horizon
                    sample_mean = mean[t]
                    sample_std = std[t]
                
                noise = torch.randn(num_random, self.cfg.action_dim, device=self.device)
                actions[t, sample_idx:] = (sample_mean + sample_std * noise).clamp(-1, 1)
        
        return actions

    def _evaluate_hybrid_trajectories(self, initial_state, initial_z, actions, 
                                    classical_horizon, task):
        """Evaluate trajectories using hybrid dynamics and value estimation."""
        num_samples = actions.shape[1]
        device = actions.device
        
        # Initialize values
        values = torch.zeros(num_samples, device=device)
        discount = 1.0
        
        # State for classical dynamics
        states = initial_state.repeat(num_samples, 1)
        # Latent state for learned dynamics
        zs = initial_z.repeat(num_samples, 1)
        
        # Track which dynamics to use
        use_classical = torch.ones(num_samples, dtype=torch.bool, device=device)
        
        for t in range(self.cfg.horizon):
            if t < classical_horizon:
                # Classical dynamics phase
                if self.cfg.hybrid_value_estimation in ['mixed', 'classical']:
                    # Use classical dynamics
                    next_states = self.env_interface.step_dynamics(states, actions[t])
                    costs = self.env_interface.compute_cost(
                        states, actions[t], next_states, task, timestep=t
                    )
                    rewards = self.env_interface.compute_reward_from_cost(costs)
                    states = next_states
                    
                    # Update latent states for consistency
                    if t == classical_horizon - 1:
                        # Convert to observation for learned phase
                        obs = self.env_interface.state_to_obs(states)
                        zs = self.model.encode(obs, task)
                else:
                    # Use learned dynamics even in classical horizon
                    rewards = math.two_hot_inv(
                        self.model.reward(zs, actions[t], task), self.cfg
                    )
                    zs = self.model.next(zs, actions[t], task)
            else:
                # Learned dynamics phase
                rewards = math.two_hot_inv(
                    self.model.reward(zs, actions[t], task), self.cfg
                )
                zs = self.model.next(zs, actions[t], task)
            
            # Accumulate discounted rewards
            values += discount * rewards
            discount *= self.discount if isinstance(self.discount, float) else self.discount.mean()
            
            # Handle termination if episodic
            if self.cfg.episodic and t >= classical_horizon:
                termination = (self.model.termination(zs, task) > 0.5).float()
                discount *= (1 - termination).squeeze()
        
        # Add terminal value
        with torch.no_grad():
            terminal_actions, _ = self.model.pi(zs, task)
            terminal_values = self.model.Q(zs, terminal_actions, task, return_type='avg')
            values += discount * terminal_values.squeeze()
        
        return values

    def _update_hybrid_distribution(self, actions, values, prev_mean, 
                                  classical_horizon, iteration):
        """Update action distribution with classical bias."""
        # Compute elite actions
        num_elites = self.cfg.num_elites
        elite_idxs = torch.topk(values, num_elites, dim=0).indices
        elite_actions = actions[:, elite_idxs]
        elite_values = values[elite_idxs]
        
        # Temperature-scaled weights
        max_value = elite_values.max()
        weights = F.softmax(
            (elite_values - max_value) / self.cfg.temperature, dim=0
        )
        
        # Weighted mean with classical bias
        new_mean = torch.sum(
            weights.unsqueeze(0).unsqueeze(-1) * elite_actions, dim=1
        )
        
        # Adaptive blending based on iteration and horizon
        for t in range(self.cfg.horizon):
            if t < classical_horizon:
                # Stronger classical influence for early timesteps
                blend_factor = 0.8 * (1 - iteration / self.cfg.iterations)
                new_mean[t] = blend_factor * prev_mean[t] + (1 - blend_factor) * new_mean[t]
        
        # Compute standard deviation
        std = torch.sqrt(torch.sum(
            weights.unsqueeze(0).unsqueeze(-1) * (elite_actions - new_mean.unsqueeze(1))**2, 
            dim=1
        ) + 1e-6)
        
        # Clamp standard deviation
        std = std.clamp(self.cfg.min_std, self.cfg.max_std)
        
        # Apply action mask if multi-task
        if self.cfg.multitask:
            task_mask = self.model._action_masks[task[0]]
            new_mean = new_mean * task_mask
            std = std * task_mask
        
        return new_mean, std

    def _get_initial_mean(self, classical_horizon, classical_actions):
        """Get initial mean with proper warm-starting."""
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        
        # Initialize with classical actions for classical horizon
        mean[:classical_horizon] = classical_actions[:classical_horizon]
        
        # Warm-start remaining with previous solution
        if hasattr(self, '_prev_mean'):
            remaining = self.cfg.horizon - classical_horizon
            if remaining > 0:
                mean[classical_horizon:] = self._prev_mean[classical_horizon:classical_horizon+remaining]
        
        return mean

    def _get_adaptive_std(self, classical_horizon):
        """Get adaptive standard deviation based on planning phase."""
        std = torch.full(
            (self.cfg.horizon, self.cfg.action_dim), 
            self.cfg.max_std, device=self.device
        )
        
        # Lower variance for classical horizon
        std[:classical_horizon] *= 0.5
        
        # Gradual increase for transition region
        if self.cfg.blend_horizon > 0:
            blend_start = classical_horizon
            blend_end = min(classical_horizon + self.cfg.blend_horizon, self.cfg.horizon)
            for t in range(blend_start, blend_end):
                alpha = (t - blend_start) / self.cfg.blend_horizon
                std[t] *= (0.5 + 0.5 * alpha)
        
        return std

    def _select_hybrid_action(self, mean_action, std_action, classical_action, 
                            classical_horizon, eval_mode):
        """Select final action with optional blending."""
        if eval_mode:
            # Deterministic action
            if classical_horizon > 0 and self.cfg.blend_actions:
                # Blend with classical action
                blend_weight = min(1.0, classical_horizon / self.cfg.max_classical_horizon)
                return blend_weight * classical_action + (1 - blend_weight) * mean_action
            else:
                return mean_action
        else:
            # Stochastic action
            base_action = mean_action + std_action * torch.randn_like(std_action)
            
            if classical_horizon > 0 and self.cfg.blend_actions:
                # Blend with classical action
                blend_weight = min(1.0, classical_horizon / self.cfg.max_classical_horizon)
                return blend_weight * classical_action + (1 - blend_weight) * base_action
            else:
                return base_action

    def _update_trajectory_buffers(self, learned_trajectory, classical_trajectory):
        """Update trajectory buffers for warm-starting."""
        self.learned_trajectory_buffer.update(learned_trajectory)
        self.classical_trajectory_buffer.update(classical_trajectory)

    def update_training_step(self, step):
        """Update training step for all components."""
        if self.hybrid_mpc is not None:
            self.hybrid_mpc.update_step(step)

    def get_diagnostics(self):
        """Get diagnostic information for logging."""
        diagnostics = {
            "hybrid_enabled": self.hybrid_mpc_enabled,
        }
        
        if self.hybrid_mpc is not None:
            diagnostics.update({
                "classical_horizon": self.hybrid_mpc.get_classical_horizon(),
                "transition_progress": min(1.0, self.hybrid_mpc.current_step / self.cfg.transition_steps),
            })
        
        if self.performance_tracker is not None:
            diagnostics.update(self.performance_tracker.get_stats())
        
        return diagnostics


class TrajectoryBuffer:
    """Buffer for storing and warm-starting trajectories."""
    
    def __init__(self, horizon, action_dim, device='cuda'):
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = device
        self.trajectory = None
        
    def update(self, trajectory):
        """Update stored trajectory."""
        self.trajectory = trajectory.detach().clone()
        
    def get_shifted_trajectory(self):
        """Get time-shifted trajectory for warm-starting."""
        if self.trajectory is None:
            return None
            
        # Shift trajectory forward in time
        shifted = torch.zeros_like(self.trajectory)
        shifted[:-1] = self.trajectory[1:]
        # Repeat last action or use zero
        shifted[-1] = self.trajectory[-1]
        
        return shifted


class PerformanceTracker:
    """Track performance metrics for adaptive horizon selection."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.window_size = cfg.performance_window
        self.metrics_buffer = []
        
    def update(self, info):
        """Update performance metrics."""
        metrics = {
            'planning_mode': info.get('planning_mode', 'unknown'),
            'cost': info.get('final_cost', float('inf')),
            'classical_horizon': info.get('classical_horizon', 0),
            'planning_time': info.get('planning_time', 0),
        }
        
        self.metrics_buffer.append(metrics)
        if len(self.metrics_buffer) > self.window_size:
            self.metrics_buffer.pop(0)
    
    def get_performance_factor(self):
        """Get performance factor for adaptive horizon selection."""
        if len(self.metrics_buffer) < 10:
            return 1.0  # Not enough data
        
        # Separate metrics by planning mode
        classical_costs = [m['cost'] for m in self.metrics_buffer if m['planning_mode'] == 'classical']
        learned_costs = [m['cost'] for m in self.metrics_buffer if m['planning_mode'] == 'learned']
        hybrid_costs = [m['cost'] for m in self.metrics_buffer if m['planning_mode'] == 'hybrid']
        
        # Compute average costs
        avg_classical = np.mean(classical_costs) if classical_costs else float('inf')
        avg_learned = np.mean(learned_costs) if learned_costs else float('inf')
        avg_hybrid = np.mean(hybrid_costs) if hybrid_costs else float('inf')
        
        # Determine best mode
        best_cost = min(avg_classical, avg_learned, avg_hybrid)
        
        if best_cost == avg_learned:
            # Learned is best, reduce classical horizon
            return 0.5
        elif best_cost == avg_classical:
            # Classical is best, increase classical horizon
            return 1.5
        else:
            # Hybrid is best, maintain current balance
            return 1.0
    
    def get_stats(self):
        """Get performance statistics."""
        if not self.metrics_buffer:
            return {}
        
        costs = [m['cost'] for m in self.metrics_buffer]
        planning_times = [m['planning_time'] for m in self.metrics_buffer]
        
        return {
            'avg_cost': np.mean(costs),
            'min_cost': np.min(costs),
            'max_cost': np.max(costs),
            'avg_planning_time': np.mean(planning_times),
            'buffer_size': len(self.metrics_buffer),
        }


# Environment-specific interface implementations
class DMControlInterface(ImprovedEnvironmentInterface):
    """DMControl-specific environment interface."""
    
    def __init__(self, cfg, env, world_model):
        super().__init__(cfg, env, world_model)
        self._setup_dmcontrol_specifics()
        
    def _setup_dmcontrol_specifics(self):
        """Setup DMControl-specific parameters."""
        # Get physics and spec
        self.physics = self.env.physics if hasattr(self.env, 'physics') else None
        self.task_spec = self.env.task if hasattr(self.env, 'task') else None
        
    def step_dynamics(self, state, action):
        """Step dynamics using DMControl physics."""
        if self.physics is not None:
            # Set state
            with self.physics.reset_context():
                self.physics.set_state(state.cpu().numpy())
            
            # Apply action
            self.physics.set_control(action.cpu().numpy())
            
            # Step physics
            self.physics.step()
            
            # Get next state
            next_state = torch.tensor(
                self.physics.get_state(), 
                device=state.device, 
                dtype=state.dtype
            )
            return next_state
        else:
            # Fallback to learned dynamics
            return super().step_dynamics(state, action)
    
    def compute_cost(self, state, action, next_state, task=None, timestep=0):
        """Compute cost using DMControl reward function."""
        if self.task_spec is not None and hasattr(self.task_spec, 'get_reward'):
            # Use environment's reward function
            reward = self.task_spec.get_reward(self.physics)
            return -reward  # Convert reward to cost
        else:
            return super().compute_cost(state, action, next_state, task, timestep)