import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

from common import math
from common.world_model import WorldModel


@dataclass
class MPCConfig:
    """Configuration for MPC algorithms."""
    # iLQG parameters
    ilqg_iterations: int = 10
    ilqg_line_search_steps: int = 10
    ilqg_regularization: float = 1e-6
    ilqg_tolerance: float = 1e-6
    ilqg_alpha_init: float = 1.0
    ilqg_alpha_decay: float = 0.5
    
    # Finite difference parameters
    finite_diff_eps: float = 1e-4
    parallel_finite_diff: bool = True
    
    # Constraint parameters
    action_bounds: Tuple[float, float] = (-1.0, 1.0)
    state_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    
    # Hybrid MPC parameters
    max_classical_horizon: int = 10
    min_classical_horizon: int = 0
    transition_steps: int = 1_000_000
    transition_schedule: str = 'cosine'  # 'linear', 'exponential', 'step', 'cosine'
    
    # Performance parameters
    cache_jacobians: bool = True
    warm_start: bool = True
    parallel_rollouts: bool = True
    
    # Cost function parameters
    state_cost_weight: float = 1.0
    action_cost_weight: float = 0.1
    terminal_cost_weight: float = 10.0
    
    # General parameters
    horizon: int = 15
    action_dim: int = 2
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class ClassicalMPC(ABC):
    """Abstract base class for classical MPC algorithms."""
    
    @abstractmethod
    def plan(self, state: torch.Tensor, horizon: int, 
             initial_actions: Optional[torch.Tensor] = None,
             task: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Plan actions using classical MPC algorithm.
        
        Returns:
            actions: Planned action sequence
            info: Dictionary containing planning information
        """
        pass


class iLQG(ClassicalMPC):
    """
    Improved Iterative Linear Quadratic Gaussian (iLQG) implementation.
    """
    
    def __init__(self, cfg: MPCConfig, env_interface):
        self.cfg = cfg
        self.env_interface = env_interface
        self.jacobian_cache = {} if cfg.cache_jacobians else None
        self.prev_actions = None  # For warm starting
        
    def plan(self, state: torch.Tensor, horizon: int, 
             initial_actions: Optional[torch.Tensor] = None,
             task: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Plan actions using iLQG algorithm with improvements.
        """
        start_time = time.time()
        info = {'iterations': 0, 'converged': False, 'final_cost': float('inf')}
        
        batch_size = state.shape[0] if state.ndim > 1 else 1
        state_dim = state.shape[-1]
        action_dim = self.cfg.action_dim
        device = state.device
        
        # Initialize or warm-start action sequence
        if initial_actions is not None:
            actions = initial_actions.to(device)
        elif self.cfg.warm_start and self.prev_actions is not None and self.prev_actions.shape[0] >= horizon:
            # Shift previous solution and pad with zeros
            actions = torch.cat([
                self.prev_actions[1:horizon],
                torch.zeros(1, batch_size, action_dim, device=device)
            ], dim=0)
        else:
            # Initialize with small random actions
            actions = 0.01 * torch.randn(horizon, batch_size, action_dim, device=device)
        
        # Clamp initial actions to bounds
        actions = self._clamp_actions(actions)
        
        # Forward pass: simulate trajectory
        states, costs = self._forward_pass(state, actions, task)
        best_cost = torch.sum(costs)
        
        # iLQG iterations
        for iteration in range(self.cfg.ilqg_iterations):
            # Linearize dynamics and quadratize cost
            A, B, Q, q, R, r = self._linearize_dynamics_and_cost(states, actions, task)
            
            # Add regularization for numerical stability
            reg = self.cfg.ilqg_regularization * (1.0 + iteration)
            
            # Backward pass to compute gains
            K, k, expected_improvement = self._backward_pass(A, B, Q, q, R, r, reg)
            
            # Forward pass with adaptive line search
            new_actions, new_states, new_costs, alpha = self._adaptive_line_search(
                state, actions, states, K, k, expected_improvement, task
            )
            
            # Check convergence
            cost_improvement = best_cost - torch.sum(new_costs)
            relative_improvement = cost_improvement / (best_cost + 1e-10)
            
            if relative_improvement < self.cfg.ilqg_tolerance:
                info['converged'] = True
                break
            
            # Update best solution
            if torch.sum(new_costs) < best_cost:
                actions = new_actions
                states = new_states
                costs = new_costs
                best_cost = torch.sum(new_costs)
            
            info['iterations'] = iteration + 1
        
        # Store for warm starting
        if self.cfg.warm_start:
            self.prev_actions = actions.detach()
        
        # Clear cache if needed
        if self.cfg.cache_jacobians:
            self.jacobian_cache.clear()
        
        info['final_cost'] = best_cost.item()
        info['planning_time'] = time.time() - start_time
        
        return actions, info
    
    def _forward_pass(self, initial_state: torch.Tensor, actions: torch.Tensor, 
                     task: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward simulation with parallel rollouts if enabled."""
        horizon = actions.shape[0]
        batch_size = initial_state.shape[0] if initial_state.ndim > 1 else 1
        state_dim = initial_state.shape[-1]
        device = initial_state.device
        
        states = torch.zeros(horizon + 1, batch_size, state_dim, device=device)
        costs = torch.zeros(horizon, batch_size, device=device)
        
        states[0] = initial_state
        
        if self.cfg.parallel_rollouts and batch_size > 1:
            # Vectorized forward pass
            for t in range(horizon):
                next_state = self.env_interface.step_dynamics(states[t], actions[t])
                cost = self.env_interface.compute_cost(
                    states[t], actions[t], next_state, task, timestep=t
                )
                
                # Apply state constraints if specified
                if self.cfg.state_bounds is not None:
                    next_state = self._clamp_states(next_state)
                
                states[t + 1] = next_state
                costs[t] = cost
        else:
            # Sequential forward pass
            for t in range(horizon):
                next_state = self.env_interface.step_dynamics(states[t], actions[t])
                cost = self.env_interface.compute_cost(
                    states[t], actions[t], next_state, task, timestep=t
                )
                
                if self.cfg.state_bounds is not None:
                    next_state = self._clamp_states(next_state)
                
                states[t + 1] = next_state
                costs[t] = cost
        
        # Add terminal cost
        terminal_cost = self.env_interface.compute_terminal_cost(states[-1], task)
        costs[-1] += self.cfg.terminal_cost_weight * terminal_cost
        
        return states, costs
    
    def _linearize_dynamics_and_cost(self, states: torch.Tensor, actions: torch.Tensor,
                                   task: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """Improved linearization with caching and parallel computation."""
        horizon = actions.shape[0]
        batch_size = states.shape[1]
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        device = states.device
        
        # Initialize matrices
        A = torch.zeros(horizon, batch_size, state_dim, state_dim, device=device)
        B = torch.zeros(horizon, batch_size, state_dim, action_dim, device=device)
        Q = torch.zeros(horizon + 1, batch_size, state_dim, state_dim, device=device)
        q = torch.zeros(horizon + 1, batch_size, state_dim, device=device)
        R = torch.zeros(horizon, batch_size, action_dim, action_dim, device=device)
        r = torch.zeros(horizon, batch_size, action_dim, device=device)
        
        # Compute in parallel if possible
        if self.cfg.parallel_finite_diff:
            # Batch compute all Jacobians at once
            A_batch, B_batch = self.env_interface.compute_dynamics_jacobians_batch(
                states[:-1], actions
            )
            A[:] = A_batch
            B[:] = B_batch
            
            # Batch compute cost Hessians
            Q_batch, q_batch, R_batch, r_batch = self.env_interface.compute_cost_hessians_batch(
                states[:-1], actions, states[1:], task
            )
            Q[:-1] = Q_batch
            q[:-1] = q_batch
            R[:] = R_batch
            r[:] = r_batch
        else:
            # Sequential computation
            for t in range(horizon):
                # Check cache first
                cache_key = (states[t].detach(), actions[t].detach()) if self.cfg.cache_jacobians else None
                
                if cache_key and cache_key in self.jacobian_cache:
                    A[t], B[t] = self.jacobian_cache[cache_key]
                else:
                    A[t], B[t] = self.env_interface.compute_dynamics_jacobians(
                        states[t], actions[t]
                    )
                    if cache_key:
                        self.jacobian_cache[cache_key] = (A[t].detach(), B[t].detach())
                
                # Compute cost Hessians
                Q[t], q[t], R[t], r[t] = self.env_interface.compute_cost_hessians(
                    states[t], actions[t], states[t + 1], task, timestep=t
                )
        
        # Terminal cost Hessian
        Q[-1], q[-1] = self.env_interface.compute_terminal_cost_hessian(states[-1], task)
        Q[-1] *= self.cfg.terminal_cost_weight
        q[-1] *= self.cfg.terminal_cost_weight
        
        return A, B, Q, q, R, r
    
    def _backward_pass(self, A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, 
                      q: torch.Tensor, R: torch.Tensor, r: torch.Tensor,
                      regularization: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Improved backward pass with better numerical stability."""
        horizon = A.shape[0]
        batch_size = A.shape[1]
        state_dim = A.shape[-1]
        action_dim = B.shape[-1]
        device = A.device
        
        # Initialize value function with terminal cost
        V = Q[-1].clone()
        v = q[-1].clone()
        
        # Initialize gain matrices
        K = torch.zeros(horizon, batch_size, action_dim, state_dim, device=device)
        k = torch.zeros(horizon, batch_size, action_dim, device=device)
        
        # Expected cost improvement
        expected_improvement = torch.zeros(batch_size, device=device)
        
        # Backward recursion
        for t in reversed(range(horizon)):
            # Q-function approximation
            Q_xx = Q[t] + A[t].transpose(-2, -1) @ V @ A[t]
            Q_uu = R[t] + B[t].transpose(-2, -1) @ V @ B[t]
            Q_ux = B[t].transpose(-2, -1) @ V @ A[t]
            Q_x = q[t] + A[t].transpose(-2, -1) @ v
            Q_u = r[t] + B[t].transpose(-2, -1) @ v
            
            # Symmetrize for numerical stability
            Q_xx = 0.5 * (Q_xx + Q_xx.transpose(-2, -1))
            Q_uu = 0.5 * (Q_uu + Q_uu.transpose(-2, -1))
            
            # Add regularization
            Q_uu_reg = Q_uu + regularization * torch.eye(action_dim, device=device)
            
            # Compute gains using Cholesky decomposition for better stability
            try:
                L = torch.linalg.cholesky(Q_uu_reg)
                K[t] = -torch.cholesky_solve(Q_ux.unsqueeze(-1), L).squeeze(-1).transpose(-2, -1)
                k[t] = -torch.cholesky_solve(Q_u.unsqueeze(-1), L).squeeze(-1)
            except:
                # Fallback to SVD for singular matrices
                U, S, Vh = torch.linalg.svd(Q_uu_reg)
                S_inv = torch.where(S > 1e-6, 1.0 / S, torch.zeros_like(S))
                Q_uu_inv = Vh.transpose(-2, -1) @ torch.diag_embed(S_inv) @ U.transpose(-2, -1)
                K[t] = -Q_uu_inv @ Q_ux
                k[t] = -Q_uu_inv @ Q_u.squeeze(-1) if Q_u.ndim > 2 else -Q_uu_inv @ Q_u
            
            # Update value function
            V = Q_xx + K[t].transpose(-2, -1) @ Q_uu @ K[t] + \
                K[t].transpose(-2, -1) @ Q_ux + Q_ux.transpose(-2, -1) @ K[t]
            v = Q_x + K[t].transpose(-2, -1) @ Q_uu @ k[t].unsqueeze(-1) + \
                K[t].transpose(-2, -1) @ Q_u.unsqueeze(-1) + \
                Q_ux.transpose(-2, -1) @ k[t].unsqueeze(-1)
            v = v.squeeze(-1)
            
            # Compute expected improvement
            expected_improvement += 0.5 * (k[t].unsqueeze(-2) @ Q_uu @ k[t].unsqueeze(-1)).squeeze()
        
        return K, k, expected_improvement
    
    def _adaptive_line_search(self, initial_state: torch.Tensor, actions: torch.Tensor,
                            states: torch.Tensor, K: torch.Tensor, k: torch.Tensor,
                            expected_improvement: torch.Tensor,
                            task: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """Adaptive line search with backtracking."""
        best_actions = actions.clone()
        best_states, best_costs = self._forward_pass(initial_state, best_actions, task)
        best_total_cost = torch.sum(best_costs)
        
        # Adaptive line search parameters
        alpha = self.cfg.ilqg_alpha_init
        alpha_min = 1e-8
        
        for _ in range(self.cfg.ilqg_line_search_steps):
            # Apply control update
            new_actions = torch.zeros_like(actions)
            state = initial_state.clone()
            
            for t in range(actions.shape[0]):
                # Compute state error
                state_error = state - states[t]
                
                # Control update with line search parameter
                du = alpha * k[t] + K[t] @ state_error.unsqueeze(-1)
                du = du.squeeze(-1) if du.ndim > 2 else du
                
                new_actions[t] = actions[t] + du
                new_actions[t] = self._clamp_actions(new_actions[t])
                
                # Simulate forward
                state = self.env_interface.step_dynamics(state, new_actions[t])
                if self.cfg.state_bounds is not None:
                    state = self._clamp_states(state)
            
            # Evaluate new trajectory
            new_states, new_costs = self._forward_pass(initial_state, new_actions, task)
            total_cost = torch.sum(new_costs)
            
            # Check Armijo condition
            actual_improvement = best_total_cost - total_cost
            expected = alpha * expected_improvement.sum()
            
            if actual_improvement > 0.1 * expected:  # Armijo condition satisfied
                return new_actions, new_states, new_costs, alpha
            
            # Backtrack
            alpha *= self.cfg.ilqg_alpha_decay
            if alpha < alpha_min:
                break
        
        # Return original if line search failed
        return best_actions, best_states, best_costs, 0.0
    
    def _clamp_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Clamp actions to bounds."""
        return torch.clamp(actions, self.cfg.action_bounds[0], self.cfg.action_bounds[1])
    
    def _clamp_states(self, states: torch.Tensor) -> torch.Tensor:
        """Clamp states to bounds if specified."""
        if self.cfg.state_bounds is not None:
            lower, upper = self.cfg.state_bounds
            return torch.clamp(states, lower, upper)
        return states


class EnvironmentInterface:
    """Improved interface to access true simulator dynamics and derivatives."""
    
    def __init__(self, cfg: MPCConfig, env):
        self.cfg = cfg
        self.env = env
        self.dynamics_model = None  # Can be set to use a learned dynamics model
        
    def set_dynamics_model(self, model):
        """Set a learned dynamics model for hybrid approaches."""
        self.dynamics_model = model
    
    def step_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Step the true simulator dynamics or learned model."""
        if hasattr(self.env, 'step_batch'):
            # Batched environment step
            return self.env.step_batch(state, action)
        elif hasattr(self.env, 'dynamics'):
            # Direct dynamics function
            return self.env.dynamics(state, action)
        elif self.dynamics_model is not None:
            # Use learned dynamics model
            with torch.no_grad():
                return self.dynamics_model(state, action)
        else:
            # Fallback to finite differences
            return self._finite_diff_dynamics(state, action)
    
    def compute_cost(self, state: torch.Tensor, action: torch.Tensor, 
                    next_state: torch.Tensor, task: Optional[torch.Tensor] = None,
                    timestep: int = 0) -> torch.Tensor:
        """Compute cost with task and time awareness."""
        if hasattr(self.env, 'cost_function'):
            return self.env.cost_function(state, action, next_state, task)
        
        # Default: quadratic cost with task-specific weights
        if task is not None and hasattr(self, 'task_weights'):
            state_weight = self.task_weights[task, 0]
            action_weight = self.task_weights[task, 1]
        else:
            state_weight = self.cfg.state_cost_weight
            action_weight = self.cfg.action_cost_weight
        
        # State cost (distance to goal if available)
        if hasattr(self.env, 'goal'):
            state_error = state - self.env.goal
            state_cost = state_weight * torch.sum(state_error ** 2, dim=-1)
        else:
            state_cost = state_weight * torch.sum(state ** 2, dim=-1)
        
        # Action cost with optional smoothness term
        action_cost = action_weight * torch.sum(action ** 2, dim=-1)
        
        # Time-varying cost (e.g., increasing urgency)
        time_factor = 1.0 + 0.01 * timestep
        
        return time_factor * (state_cost + action_cost)
    
    def compute_terminal_cost(self, state: torch.Tensor, 
                            task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute terminal cost."""
        if hasattr(self.env, 'terminal_cost'):
            return self.env.terminal_cost(state, task)
        
        # Default: distance to goal
        if hasattr(self.env, 'goal'):
            state_error = state - self.env.goal
            return torch.sum(state_error ** 2, dim=-1)
        else:
            return torch.sum(state ** 2, dim=-1)
    
    def compute_dynamics_jacobians(self, state: torch.Tensor, 
                                 action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Jacobians using automatic differentiation when possible."""
        if hasattr(self.env, 'dynamics') and torch.is_grad_enabled():
            # Use automatic differentiation
            return self._autodiff_jacobians(state, action)
        else:
            # Use finite differences
            return self._finite_diff_jacobians(state, action)
    
    def compute_dynamics_jacobians_batch(self, states: torch.Tensor, 
                                       actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch computation of Jacobians for efficiency."""
        batch_size, horizon = states.shape[0], states.shape[1]
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        device = states.device
        
        A = torch.zeros(horizon, batch_size, state_dim, state_dim, device=device)
        B = torch.zeros(horizon, batch_size, state_dim, action_dim, device=device)
        
        # Parallel computation across time steps
        for t in range(horizon):
            A[t], B[t] = self.compute_dynamics_jacobians(states[:, t], actions[:, t])
        
        return A, B
    
    def compute_cost_hessians(self, state: torch.Tensor, action: torch.Tensor,
                            next_state: torch.Tensor, task: Optional[torch.Tensor] = None,
                            timestep: int = 0) -> Tuple[torch.Tensor, ...]:
        """Compute Hessians of cost function."""
        if hasattr(self.env, 'cost_hessians'):
            return self.env.cost_hessians(state, action, next_state, task)
        
        batch_size = state.shape[0] if state.ndim > 1 else 1
        state_dim = state.shape[-1]
        action_dim = action.shape[-1]
        device = state.device
        
        # Default quadratic cost Hessians
        state_weight = self.cfg.state_cost_weight * (1.0 + 0.01 * timestep)
        action_weight = self.cfg.action_cost_weight
        
        Q = 2 * state_weight * torch.eye(state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        q = 2 * state_weight * state
        R = 2 * action_weight * torch.eye(action_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        r = 2 * action_weight * action
        
        return Q, q, R, r
    
    def compute_cost_hessians_batch(self, states: torch.Tensor, actions: torch.Tensor,
                                  next_states: torch.Tensor, 
                                  task: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """Batch computation of cost Hessians."""
        horizon, batch_size = states.shape[0], states.shape[1]
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        device = states.device
        
        Q = torch.zeros(horizon, batch_size, state_dim, state_dim, device=device)
        q = torch.zeros(horizon, batch_size, state_dim, device=device)
        R = torch.zeros(horizon, batch_size, action_dim, action_dim, device=device)
        r = torch.zeros(horizon, batch_size, action_dim, device=device)
        
        for t in range(horizon):
            Q[t], q[t], R[t], r[t] = self.compute_cost_hessians(
                states[t], actions[t], next_states[t], task, timestep=t
            )
        
        return Q, q, R, r
    
    def compute_terminal_cost_hessian(self, state: torch.Tensor,
                                    task: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Hessian of terminal cost."""
        if hasattr(self.env, 'terminal_cost_hessian'):
            return self.env.terminal_cost_hessian(state, task)
        
        batch_size = state.shape[0] if state.ndim > 1 else 1
        state_dim = state.shape[-1]
        device = state.device
        
        # Default: quadratic terminal cost
        Q = 2 * torch.eye(state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        if hasattr(self.env, 'goal'):
            q = 2 * (state - self.env.goal)
        else:
            q = 2 * state
        
        return Q, q
    
    def _autodiff_jacobians(self, state: torch.Tensor, 
                          action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Jacobians using automatic differentiation."""
        state.requires_grad_(True)
        action.requires_grad_(True)
        
        # Compute dynamics
        next_state = self.env.dynamics(state, action)
        
        # Compute Jacobians
        batch_size = state.shape[0] if state.ndim > 1 else 1
        state_dim = state.shape[-1]
        action_dim = action.shape[-1]
        
        A = torch.zeros(batch_size, state_dim, state_dim, device=state.device)
        B = torch.zeros(batch_size, state_dim, action_dim, device=state.device)
        
        for i in range(state_dim):
            grad_outputs = torch.zeros_like(next_state)
            grad_outputs[..., i] = 1.0
            
            grads = torch.autograd.grad(
                outputs=next_state,
                inputs=(state, action),
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False
            )
            
            A[..., i, :] = grads[0]
            B[..., i, :] = grads[1]
        
        state.requires_grad_(False)
        action.requires_grad_(False)
        
        return A, B
    
    def _finite_diff_jacobians(self, state: torch.Tensor, 
                             action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Improved finite difference Jacobian computation."""
        eps = self.cfg.finite_diff_eps
        state_dim = state.shape[-1]
        action_dim = action.shape[-1]
        batch_size = state.shape[0] if state.ndim > 1 else 1
        device = state.device
        
        # Pre-allocate for efficiency
        A = torch.zeros(batch_size, state_dim, state_dim, device=device)
        B = torch.zeros(batch_size, state_dim, action_dim, device=device)
        
        # Central differences for state Jacobian
        for i in range(state_dim):
            state_plus = state.clone()
            state_minus = state.clone()
            state_plus[..., i] += eps
            state_minus[..., i] -= eps
            
            f_plus = self.step_dynamics(state_plus, action)
            f_minus = self.step_dynamics(state_minus, action)
            
            A[..., :, i] = (f_plus - f_minus) / (2 * eps)
        
        # Central differences for action Jacobian
        for i in range(action_dim):
            action_plus = action.clone()
            action_minus = action.clone()
            action_plus[..., i] += eps
            action_minus[..., i] -= eps
            
            # Ensure actions stay within bounds
            action_plus = torch.clamp(action_plus, self.cfg.action_bounds[0], self.cfg.action_bounds[1])
            action_minus = torch.clamp(action_minus, self.cfg.action_bounds[0], self.cfg.action_bounds[1])
            
            f_plus = self.step_dynamics(state, action_plus)
            f_minus = self.step_dynamics(state, action_minus)
            
            B[..., :, i] = (f_plus - f_minus) / (2 * eps)
        
        return A, B
    
    def _finite_diff_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Fallback dynamics (should be replaced with actual environment dynamics)."""
        # This should be implemented based on your specific environment
        # For now, using a simple linear dynamics model as placeholder
        dt = 0.1  # timestep
        
        # Example: simple integrator dynamics
        # x' = x + v * dt
        # v' = v + a * dt
        # where action represents acceleration
        
        state_dim = state.shape[-1]
        if state_dim >= 2:
            # Assume first half is position, second half is velocity
            pos_dim = state_dim // 2
            pos = state[..., :pos_dim]
            vel = state[..., pos_dim:]
            
            # Update position and velocity
            new_pos = pos + vel * dt
            new_vel = vel + action[..., :pos_dim] * dt
            
            # Combine
            next_state = torch.cat([new_pos, new_vel], dim=-1)
        else:
            # Simple integrator
            next_state = state + action * dt
        
        return next_state


class HybridMPC:
    """
    Enhanced Hybrid MPC with better integration between classical and learned control.
    """
    
    def __init__(self, cfg: MPCConfig, env, world_model: WorldModel):
        self.cfg = cfg
        self.env = env
        self.world_model = world_model
        self.env_interface = EnvironmentInterface(cfg, env)
        self.classical_mpc = iLQG(cfg, self.env_interface)
        
        # For smooth transition
        self.current_step = 0
        self.performance_history = []
        self.use_adaptive_transition = cfg.get('adaptive_transition', False)
        
        # Initialize learned dynamics interface
        if hasattr(world_model, 'dynamics'):
            self.env_interface.set_dynamics_model(world_model.dynamics)
    
    def get_classical_horizon(self) -> int:
        """Get current classical horizon with improved scheduling."""
        if self.use_adaptive_transition:
            # Adaptive transition based on performance
            return self._adaptive_horizon()
        
        # Fixed schedule
        if self.cfg.transition_schedule == 'linear':
            progress = min(self.current_step / self.cfg.transition_steps, 1.0)
            horizon = self.cfg.max_classical_horizon * (1 - progress)
        elif self.cfg.transition_schedule == 'exponential':
            decay_rate = -np.log(0.01) / self.cfg.transition_steps  # Decay to 1% at end
            horizon = self.cfg.max_classical_horizon * np.exp(-decay_rate * self.current_step)
        elif self.cfg.transition_schedule == 'cosine':
            progress = min(self.current_step / self.cfg.transition_steps, 1.0)
            horizon = self.cfg.max_classical_horizon * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.cfg.transition_schedule == 'step':
            if self.current_step < self.cfg.transition_steps // 3:
                horizon = self.cfg.max_classical_horizon
            elif self.current_step < 2 * self.cfg.transition_steps // 3:
                horizon = self.cfg.max_classical_horizon // 2
            else:
                horizon = self.cfg.min_classical_horizon
        else:
            horizon = self.cfg.max_classical_horizon
        
        return int(max(horizon, self.cfg.min_classical_horizon))
    
    def _adaptive_horizon(self) -> int:
        """Adaptively choose horizon based on performance metrics."""
        if len(self.performance_history) < 100:
            return self.cfg.max_classical_horizon
        
        # Compare recent performance of classical vs learned
        recent_perf = self.performance_history[-100:]
        classical_perf = np.mean([p['classical_cost'] for p in recent_perf if 'classical_cost' in p])
        learned_perf = np.mean([p['learned_cost'] for p in recent_perf if 'learned_cost' in p])
        
        # Adjust horizon based on relative performance
        if learned_perf < classical_perf * 0.9:  # Learned is significantly better
            return max(self.get_classical_horizon() - 1, self.cfg.min_classical_horizon)
        elif classical_perf < learned_perf * 0.9:  # Classical is significantly better
            return min(self.get_classical_horizon() + 1, self.cfg.max_classical_horizon)
        else:
            return self.get_classical_horizon()
    
    def plan(self, obs: torch.Tensor, t0: bool = False, eval_mode: bool = False, 
             task: Optional[torch.Tensor] = None,
             return_info: bool = False) -> torch.Tensor:
        """
        Enhanced hybrid planning with smooth transitions and better integration.
        """
        classical_horizon = self.get_classical_horizon()
        total_horizon = self.cfg.horizon
        
        planning_info = {
            'classical_horizon': classical_horizon,
            'total_horizon': total_horizon,
            'mode': 'hybrid'
        }
        
        if classical_horizon == 0:
            # Pure learned MPC
            planning_info['mode'] = 'learned'
            action = self._learned_mpc_plan(obs, t0, eval_mode, task)
        elif classical_horizon >= total_horizon:
            # Pure classical MPC
            planning_info['mode'] = 'classical'
            action = self._classical_mpc_plan(obs, classical_horizon, task, planning_info)
        else:
            # Hybrid approach with blending
            action = self._hybrid_plan_with_blending(
                obs, classical_horizon, t0, eval_mode, task, planning_info
            )
        
        # Update performance tracking
        if not eval_mode and 'cost' in planning_info:
            self.performance_history.append({
                'step': self.current_step,
                'cost': planning_info['cost'],
                'mode': planning_info['mode']
            })
        
        if return_info:
            return action, planning_info
        return action
    
    def _classical_mpc_plan(self, obs: torch.Tensor, horizon: int, 
                           task: Optional[torch.Tensor] = None,
                           info: Optional[Dict] = None) -> torch.Tensor:
        """Plan using classical MPC with improved interface."""
        # Convert observation to state
        if hasattr(self.world_model, 'obs_to_state'):
            state = self.world_model.obs_to_state(obs)
        else:
            state = obs
        
        # Get warm-start from previous solution if available
        initial_actions = None
        if hasattr(self, 'prev_trajectory') and self.prev_trajectory is not None:
            initial_actions = self.prev_trajectory[1:horizon+1]
        
        actions, plan_info = self.classical_mpc.plan(state, horizon, initial_actions, task)
        
        # Store trajectory for warm-starting
        self.prev_trajectory = actions
        
        # Update info
        if info is not None:
            info.update(plan_info)
            info['classical_cost'] = plan_info['final_cost']
        
        return actions[0]
    
    def _learned_mpc_plan(self, obs: torch.Tensor, t0: bool = False, 
                         eval_mode: bool = False, 
                         task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Plan using learned MPC (TD-MPC2 style)."""
        # Encode observation
        z = self.world_model.encode(obs, task)
        
        if t0 or not hasattr(self, 'prev_mean'):
            # First timestep or no previous solution
            self.prev_mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=obs.device)
        
        # MPPI-style planning
        action = self._mppi(z, task, eval_mode)
        
        return action
    
    def _mppi(self, z: torch.Tensor, task: Optional[torch.Tensor] = None,
              eval_mode: bool = False) -> torch.Tensor:
        """Model Predictive Path Integral planning."""
        horizon = self.cfg.horizon
        n_samples = self.cfg.get('n_samples', 512)
        device = z.device
        
        # Sample action sequences
        noise = torch.randn(n_samples, horizon, self.cfg.action_dim, device=device)
        actions = self.prev_mean.unsqueeze(0) + self.cfg.get('noise_std', 1.0) * noise
        actions = torch.clamp(actions, self.cfg.action_bounds[0], self.cfg.action_bounds[1])
        
        # Rollout trajectories
        costs = torch.zeros(n_samples, device=device)
        state = z.unsqueeze(0).repeat(n_samples, 1)
        
        for t in range(horizon):
            # Predict next state
            next_state, reward = self.world_model.dynamics(state, actions[:, t], task)
            costs -= reward  # Negative reward is cost
            state = next_state
        
        # Compute weights
        weights = F.softmax(-costs / self.cfg.get('temperature', 1.0), dim=0)
        
        # Weighted average
        self.prev_mean = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * actions, dim=0)
        
        if eval_mode:
            return self.prev_mean[0]
        else:
            # Add exploration noise
            return self.prev_mean[0] + 0.1 * torch.randn_like(self.prev_mean[0])
    
    def _hybrid_plan_with_blending(self, obs: torch.Tensor, classical_horizon: int,
                                 t0: bool, eval_mode: bool, 
                                 task: Optional[torch.Tensor] = None,
                                 info: Optional[Dict] = None) -> torch.Tensor:
        """Enhanced hybrid planning with trajectory blending."""
        total_horizon = self.cfg.horizon
        device = obs.device
        
        # Phase 1: Get classical trajectory
        state = self.world_model.obs_to_state(obs) if hasattr(self.world_model, 'obs_to_state') else obs
        classical_actions, classical_info = self.classical_mpc.plan(
            state, classical_horizon, task=task
        )
        
        # Phase 2: Get learned trajectory from current state
        learned_action = self._learned_mpc_plan(obs, t0, eval_mode, task)
        
        # Phase 3: Blend trajectories for smooth transition
        if hasattr(self, 'prev_learned_trajectory'):
            # Use blending weight based on horizon position
            blend_weight = 1.0 - (classical_horizon / self.cfg.max_classical_horizon)
            blend_weight = np.clip(blend_weight, 0.0, 1.0)
            
            # Exponential blending for smoother transition
            blended_action = (1 - blend_weight) * classical_actions[0] + blend_weight * learned_action
        else:
            # No previous learned trajectory, use classical
            blended_action = classical_actions[0]
        
        # Store for next iteration
        self.prev_learned_trajectory = learned_action
        
        # Update info
        if info is not None:
            info.update(classical_info)
            info['blend_weight'] = blend_weight if 'blend_weight' in locals() else 0.0
        
        return blended_action
    
    def update_step(self, step: int):
        """Update the current training step."""
        self.current_step = step
    
    def reset(self):
        """Reset planning state for new episode."""
        if hasattr(self, 'classical_mpc') and hasattr(self.classical_mpc, 'prev_actions'):
            self.classical_mpc.prev_actions = None
        if hasattr(self, 'prev_mean'):
            delattr(self, 'prev_mean')
        if hasattr(self, 'prev_trajectory'):
            delattr(self, 'prev_trajectory')
        if hasattr(self, 'prev_learned_trajectory'):
            delattr(self, 'prev_learned_trajectory')