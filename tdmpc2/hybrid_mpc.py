import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod

from common import math
from common.world_model import WorldModel


class ClassicalMPC(ABC):
    """Abstract base class for classical MPC algorithms."""
    
    @abstractmethod
    def plan(self, state: torch.Tensor, horizon: int, task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Plan actions using classical MPC algorithm."""
        pass


class iLQG(ClassicalMPC):
    """
    Iterative Linear Quadratic Gaussian (iLQG) implementation for classical MPC.
    Uses the true simulator dynamics for the first few timesteps.
    """
    
    def __init__(self, cfg, env_interface):
        self.cfg = cfg
        self.env_interface = env_interface
        self.max_iterations = cfg.get('ilqg_iterations', 10)
        self.line_search_steps = cfg.get('ilqg_line_search_steps', 10)
        self.regularization = cfg.get('ilqg_regularization', 1e-6)
        self.tolerance = cfg.get('ilqg_tolerance', 1e-6)
        
    def plan(self, state: torch.Tensor, horizon: int, task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Plan actions using iLQG algorithm.
        
        Args:
            state: Current state
            horizon: Planning horizon for classical MPC
            task: Task identifier (for multi-task settings)
            
        Returns:
            Planned actions for the classical horizon
        """
        batch_size = state.shape[0] if state.ndim > 1 else 1
        state_dim = state.shape[-1]
        action_dim = self.cfg.action_dim
        
        # Initialize action sequence
        actions = torch.zeros(horizon, batch_size, action_dim, device=state.device)
        
        # Forward pass: simulate trajectory
        states, costs = self._forward_pass(state, actions, task)
        
        # Backward pass: compute optimal control
        for iteration in range(self.max_iterations):
            # Linearize dynamics and quadratize cost
            A, B, Q, q, R, r = self._linearize_dynamics_and_cost(states, actions, task)
            
            # Backward pass to compute gains
            K, k, cost_to_go = self._backward_pass(A, B, Q, q, R, r)
            
            # Forward pass with line search
            new_actions, new_states, new_costs = self._forward_pass_with_line_search(
                state, actions, K, k, task
            )
            
            # Check convergence
            cost_improvement = torch.sum(costs) - torch.sum(new_costs)
            if cost_improvement < self.tolerance:
                break
                
            actions = new_actions
            states = new_states
            costs = new_costs
            
        return actions
    
    def _forward_pass(self, initial_state: torch.Tensor, actions: torch.Tensor, 
                     task: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward simulation using true dynamics."""
        horizon = actions.shape[0]
        batch_size = initial_state.shape[0] if initial_state.ndim > 1 else 1
        state_dim = initial_state.shape[-1]
        
        states = torch.zeros(horizon + 1, batch_size, state_dim, device=initial_state.device)
        costs = torch.zeros(horizon, batch_size, device=initial_state.device)
        
        states[0] = initial_state
        
        for t in range(horizon):
            # Use true simulator dynamics
            next_state = self.env_interface.step_dynamics(states[t], actions[t])
            cost = self.env_interface.compute_cost(states[t], actions[t], next_state)
            
            states[t + 1] = next_state
            costs[t] = cost
            
        return states, costs
    
    def _linearize_dynamics_and_cost(self, states: torch.Tensor, actions: torch.Tensor,
                                   task: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """Linearize dynamics and quadratize cost around current trajectory."""
        horizon = actions.shape[0]
        batch_size = states.shape[1]
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        
        # Initialize matrices
        A = torch.zeros(horizon, batch_size, state_dim, state_dim, device=states.device)
        B = torch.zeros(horizon, batch_size, state_dim, action_dim, device=states.device)
        Q = torch.zeros(horizon, batch_size, state_dim, state_dim, device=states.device)
        q = torch.zeros(horizon, batch_size, state_dim, device=states.device)
        R = torch.zeros(horizon, batch_size, action_dim, action_dim, device=states.device)
        r = torch.zeros(horizon, batch_size, action_dim, device=states.device)
        
        for t in range(horizon):
            # Compute dynamics Jacobians
            A[t], B[t] = self.env_interface.compute_dynamics_jacobians(states[t], actions[t])
            
            # Compute cost Hessians
            Q[t], q[t], R[t], r[t] = self.env_interface.compute_cost_hessians(
                states[t], actions[t], states[t + 1]
            )
            
        return A, B, Q, q, R, r
    
    def _backward_pass(self, A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, 
                      q: torch.Tensor, R: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Backward pass to compute optimal gains."""
        horizon = A.shape[0]
        batch_size = A.shape[1]
        state_dim = A.shape[-1]
        action_dim = B.shape[-1]
        
        # Initialize value function
        V = Q[-1]  # Terminal cost
        v = q[-1]
        
        # Initialize gain matrices
        K = torch.zeros(horizon, batch_size, action_dim, state_dim, device=A.device)
        k = torch.zeros(horizon, batch_size, action_dim, device=A.device)
        cost_to_go = torch.zeros(horizon + 1, batch_size, device=A.device)
        
        cost_to_go[-1] = 0  # Terminal cost-to-go
        
        # Backward recursion
        for t in reversed(range(horizon)):
            # Q-function approximation
            Q_xx = Q[t] + A[t].transpose(-2, -1) @ V @ A[t]
            Q_uu = R[t] + B[t].transpose(-2, -1) @ V @ B[t]
            Q_ux = B[t].transpose(-2, -1) @ V @ A[t]
            Q_x = q[t] + A[t].transpose(-2, -1) @ v
            Q_u = r[t] + B[t].transpose(-2, -1) @ v
            
            # Add regularization
            Q_uu_reg = Q_uu + self.regularization * torch.eye(action_dim, device=Q_uu.device)
            
            # Compute gains
            try:
                Q_uu_inv = torch.inverse(Q_uu_reg)
                K[t] = -Q_uu_inv @ Q_ux
                k[t] = -Q_uu_inv @ Q_u
            except:
                # Fallback to pseudo-inverse if singular
                K[t] = -torch.pinverse(Q_uu_reg) @ Q_ux
                k[t] = -torch.pinverse(Q_uu_reg) @ Q_u
            
            # Update value function
            V = Q_xx + K[t].transpose(-2, -1) @ Q_uu @ K[t] + K[t].transpose(-2, -1) @ Q_ux + Q_ux.transpose(-2, -1) @ K[t]
            v = Q_x + K[t].transpose(-2, -1) @ Q_uu @ k[t] + K[t].transpose(-2, -1) @ Q_u + Q_ux.transpose(-2, -1) @ k[t]
            
            cost_to_go[t] = cost_to_go[t + 1]  # Placeholder for actual cost-to-go computation
            
        return K, k, cost_to_go
    
    def _forward_pass_with_line_search(self, initial_state: torch.Tensor, actions: torch.Tensor,
                                     K: torch.Tensor, k: torch.Tensor, 
                                     task: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """Forward pass with line search to find optimal step size."""
        best_actions = actions.clone()
        best_states, best_costs = self._forward_pass(initial_state, best_actions, task)
        best_total_cost = torch.sum(best_costs)
        
        # Line search
        for alpha in torch.logspace(0, -3, self.line_search_steps):
            # Apply control update
            new_actions = actions.clone()
            state = initial_state
            
            for t in range(actions.shape[0]):
                # Compute control update
                if t == 0:
                    state_error = torch.zeros_like(state)
                else:
                    state_error = state - best_states[t]
                
                control_update = alpha * k[t] + K[t] @ state_error
                new_actions[t] = actions[t] + control_update
                new_actions[t] = torch.clamp(new_actions[t], -1, 1)  # Action bounds
                
                # Simulate forward
                state = self.env_interface.step_dynamics(state, new_actions[t])
            
            # Evaluate new trajectory
            new_states, new_costs = self._forward_pass(initial_state, new_actions, task)
            total_cost = torch.sum(new_costs)
            
            if total_cost < best_total_cost:
                best_actions = new_actions
                best_states = new_states
                best_costs = new_costs
                best_total_cost = total_cost
                break
                
        return best_actions, best_states, best_costs


class EnvironmentInterface:
    """Interface to access true simulator dynamics and derivatives."""
    
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.finite_diff_eps = cfg.get('finite_diff_eps', 1e-6)
        
    def step_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Step the true simulator dynamics."""
        # This needs to be implemented for each environment type
        # For now, we'll use finite differences as a fallback
        return self._finite_diff_dynamics(state, action)
    
    def compute_cost(self, state: torch.Tensor, action: torch.Tensor, 
                    next_state: torch.Tensor) -> torch.Tensor:
        """Compute cost for the current state-action pair."""
        # Default quadratic cost - should be customized per environment
        state_cost = torch.sum(state ** 2, dim=-1)
        action_cost = 0.1 * torch.sum(action ** 2, dim=-1)
        return state_cost + action_cost
    
    def compute_dynamics_jacobians(self, state: torch.Tensor, 
                                 action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Jacobians of dynamics with respect to state and action."""
        return self._finite_diff_jacobians(state, action)
    
    def compute_cost_hessians(self, state: torch.Tensor, action: torch.Tensor,
                            next_state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute Hessians of cost function."""
        batch_size = state.shape[0] if state.ndim > 1 else 1
        state_dim = state.shape[-1]
        action_dim = action.shape[-1]
        
        # Default quadratic cost Hessians
        Q = 2 * torch.eye(state_dim, device=state.device).unsqueeze(0).repeat(batch_size, 1, 1)
        q = 2 * state
        R = 0.2 * torch.eye(action_dim, device=action.device).unsqueeze(0).repeat(batch_size, 1, 1)
        r = 0.2 * action
        
        return Q, q, R, r
    
    def _finite_diff_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Finite difference approximation of dynamics (fallback)."""
        # This is a placeholder - real implementation would call simulator
        # For now, assume simple linear dynamics
        A = torch.eye(state.shape[-1], device=state.device) * 0.9
        B = torch.randn(state.shape[-1], action.shape[-1], device=state.device) * 0.1
        return state @ A.T + action @ B.T
    
    def _finite_diff_jacobians(self, state: torch.Tensor, 
                             action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Jacobians using finite differences."""
        state_dim = state.shape[-1]
        action_dim = action.shape[-1]
        batch_size = state.shape[0] if state.ndim > 1 else 1
        
        # Initialize Jacobians
        A = torch.zeros(batch_size, state_dim, state_dim, device=state.device)
        B = torch.zeros(batch_size, state_dim, action_dim, device=state.device)
        
        # Finite differences for state Jacobian
        for i in range(state_dim):
            state_plus = state.clone()
            state_minus = state.clone()
            state_plus[..., i] += self.finite_diff_eps
            state_minus[..., i] -= self.finite_diff_eps
            
            f_plus = self.step_dynamics(state_plus, action)
            f_minus = self.step_dynamics(state_minus, action)
            
            A[..., :, i] = (f_plus - f_minus) / (2 * self.finite_diff_eps)
        
        # Finite differences for action Jacobian
        for i in range(action_dim):
            action_plus = action.clone()
            action_minus = action.clone()
            action_plus[..., i] += self.finite_diff_eps
            action_minus[..., i] -= self.finite_diff_eps
            
            f_plus = self.step_dynamics(state, action_plus)
            f_minus = self.step_dynamics(state, action_minus)
            
            B[..., :, i] = (f_plus - f_minus) / (2 * self.finite_diff_eps)
        
        return A, B


class HybridMPC:
    """
    Hybrid MPC that combines classical MPC (iLQG) for initial timesteps
    with learned MPC (MPPI) for remaining timesteps.
    """
    
    def __init__(self, cfg, env, world_model: WorldModel):
        self.cfg = cfg
        self.world_model = world_model
        self.env_interface = EnvironmentInterface(cfg, env)
        self.classical_mpc = iLQG(cfg, self.env_interface)
        
        # Transition scheduling parameters
        self.max_classical_horizon = cfg.get('max_classical_horizon', 10)
        self.min_classical_horizon = cfg.get('min_classical_horizon', 0)
        self.transition_steps = cfg.get('transition_steps', 1_000_000)
        self.transition_schedule = cfg.get('transition_schedule', 'linear')
        self.current_step = 0
        
    def get_classical_horizon(self) -> int:
        """Get current classical horizon based on training progress."""
        if self.transition_schedule == 'linear':
            progress = min(self.current_step / self.transition_steps, 1.0)
            horizon = self.max_classical_horizon * (1 - progress) + self.min_classical_horizon * progress
            return int(horizon)
        elif self.transition_schedule == 'exponential':
            decay_rate = 0.001
            horizon = self.max_classical_horizon * np.exp(-decay_rate * self.current_step)
            return max(int(horizon), self.min_classical_horizon)
        elif self.transition_schedule == 'step':
            if self.current_step < self.transition_steps // 2:
                return self.max_classical_horizon
            elif self.current_step < self.transition_steps:
                return self.max_classical_horizon // 2
            else:
                return self.min_classical_horizon
        else:
            return self.max_classical_horizon
    
    def plan(self, obs: torch.Tensor, t0: bool = False, eval_mode: bool = False, 
             task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Hybrid planning: classical MPC for first N steps, learned MPC for remainder.
        
        Args:
            obs: Current observation
            t0: Whether this is the first timestep
            eval_mode: Whether in evaluation mode
            task: Task identifier
            
        Returns:
            Action to execute
        """
        classical_horizon = self.get_classical_horizon()
        total_horizon = self.cfg.horizon
        
        if classical_horizon == 0:
            # Pure learned MPC
            return self._learned_mpc_plan(obs, t0, eval_mode, task)
        elif classical_horizon >= total_horizon:
            # Pure classical MPC
            return self._classical_mpc_plan(obs, classical_horizon, task)
        else:
            # Hybrid approach
            return self._hybrid_plan(obs, classical_horizon, t0, eval_mode, task)
    
    def _classical_mpc_plan(self, obs: torch.Tensor, horizon: int, 
                           task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Plan using only classical MPC."""
        # Convert observation to state (this might need environment-specific handling)
        state = obs  # Assuming state-based observations for now
        
        actions = self.classical_mpc.plan(state, horizon, task)
        return actions[0]  # Return first action
    
    def _learned_mpc_plan(self, obs: torch.Tensor, t0: bool = False, 
                         eval_mode: bool = False, task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Plan using only learned MPC (original MPPI)."""
        # This would call the original TD-MPC2 planning method
        # For now, we'll implement a simplified version
        z = self.world_model.encode(obs, task)
        action, _ = self.world_model.pi(z, task)
        if eval_mode:
            return action.mean(dim=0)
        return action[0]
    
    def _hybrid_plan(self, obs: torch.Tensor, classical_horizon: int, t0: bool = False,
                    eval_mode: bool = False, task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Hybrid planning combining classical and learned MPC."""
        # Phase 1: Classical MPC for first classical_horizon steps
        state = obs  # Convert obs to state if needed
        classical_actions = self.classical_mpc.plan(state, classical_horizon, task)
        
        # Phase 2: Simulate forward using classical actions to get future state
        future_state = state
        for t in range(classical_horizon):
            future_state = self.env_interface.step_dynamics(future_state, classical_actions[t])
        
        # Phase 3: Use learned MPC from future state for remaining horizon
        remaining_horizon = self.cfg.horizon - classical_horizon
        if remaining_horizon > 0:
            # Convert future state back to observation format
            future_obs = future_state  # This might need environment-specific conversion
            
            # Plan remaining steps with learned model
            learned_action = self._learned_mpc_plan(future_obs, False, eval_mode, task)
            
            # For now, we return the first classical action
            # In a full implementation, you might want to consider the full trajectory
        
        return classical_actions[0]
    
    def update_step(self, step: int):
        """Update the current training step for transition scheduling."""
        self.current_step = step
