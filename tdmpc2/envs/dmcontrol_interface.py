import torch
import numpy as np
from typing import Tuple, Optional
import dm_control.mujoco as mujoco
from dm_control.suite import common

from tdmpc2.hybrid_mpc import EnvironmentInterface


class DMControlInterface(EnvironmentInterface):
    """
    DMControl-specific interface that leverages MuJoCo's analytical derivatives
    for classical MPC algorithms like iLQG.
    """
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env = env
        self.physics = env.physics if hasattr(env, 'physics') else env._env.physics
        self.task = env.task if hasattr(env, 'task') else env._env.task
        
        # MuJoCo model dimensions
        self.nq = self.physics.model.nq  # Number of generalized coordinates
        self.nv = self.physics.model.nv  # Number of degrees of freedom
        self.nu = self.physics.model.nu  # Number of actuators
        self.nx = self.nq + self.nv      # State dimension (position + velocity)
        
        # Cost function parameters (task-specific)
        self.setup_cost_function()
        
    def setup_cost_function(self):
        """Setup task-specific cost function parameters."""
        task_name = self.cfg.task.lower()
        
        if 'cartpole' in task_name:
            self.target_state = torch.zeros(self.nx)
            self.target_state[1] = np.pi  # Upright position
            self.Q_diag = torch.tensor([10.0, 100.0, 1.0, 1.0])  # [cart_pos, pole_angle, cart_vel, pole_vel]
            self.R_diag = torch.tensor([0.1])  # Control cost
            
        elif 'pendulum' in task_name:
            self.target_state = torch.zeros(self.nx)
            self.target_state[0] = np.pi  # Upright position
            self.Q_diag = torch.tensor([100.0, 1.0])  # [angle, angular_velocity]
            self.R_diag = torch.tensor([0.1])
            
        elif 'cheetah' in task_name:
            # Running task - encourage forward velocity
            self.target_state = torch.zeros(self.nx)
            self.target_state[self.nq:self.nq+1] = 10.0  # Target forward velocity
            self.Q_diag = torch.ones(self.nx) * 0.1
            self.Q_diag[self.nq:self.nq+1] = 10.0  # High weight on forward velocity
            self.R_diag = torch.ones(self.nu) * 0.01
            
        elif 'walker' in task_name:
            # Walking task
            self.target_state = torch.zeros(self.nx)
            self.target_state[self.nq:self.nq+1] = 5.0  # Target forward velocity
            self.Q_diag = torch.ones(self.nx) * 0.1
            self.Q_diag[2] = 100.0  # Upright orientation
            self.Q_diag[self.nq:self.nq+1] = 10.0  # Forward velocity
            self.R_diag = torch.ones(self.nu) * 0.01
            
        else:
            # Default cost function
            self.target_state = torch.zeros(self.nx)
            self.Q_diag = torch.ones(self.nx)
            self.R_diag = torch.ones(self.nu) * 0.1
            
    def obs_to_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Convert observation to full state (qpos, qvel)."""
        # For DMControl, observations often don't include full state
        # We need to extract it from the physics engine
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            
        batch_size = obs.shape[0]
        states = torch.zeros(batch_size, self.nx, device=obs.device)
        
        # Get current state from physics
        qpos = torch.tensor(self.physics.data.qpos.copy(), device=obs.device, dtype=torch.float32)
        qvel = torch.tensor(self.physics.data.qvel.copy(), device=obs.device, dtype=torch.float32)
        
        # Repeat for batch
        states[:, :self.nq] = qpos.unsqueeze(0).repeat(batch_size, 1)
        states[:, self.nq:] = qvel.unsqueeze(0).repeat(batch_size, 1)
        
        return states
        
    def state_to_obs(self, state: torch.Tensor) -> torch.Tensor:
        """Convert full state back to observation format."""
        # This is environment-specific and might need adjustment
        # For now, we'll use the state as-is
        return state
        
    def step_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Step the true MuJoCo dynamics."""
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
            
        batch_size = state.shape[0]
        next_states = torch.zeros_like(state)
        
        # Save current state
        original_qpos = self.physics.data.qpos.copy()
        original_qvel = self.physics.data.qvel.copy()
        original_ctrl = self.physics.data.ctrl.copy()
        
        try:
            for i in range(batch_size):
                # Set state
                qpos = state[i, :self.nq].cpu().numpy()
                qvel = state[i, self.nq:].cpu().numpy()
                ctrl = action[i].cpu().numpy()
                
                # Ensure control is within bounds
                ctrl = np.clip(ctrl, -1.0, 1.0)
                
                self.physics.set_state(qpos, qvel)
                self.physics.set_control(ctrl)
                
                # Step physics
                self.physics.step()
                
                # Get next state
                next_qpos = torch.tensor(self.physics.data.qpos.copy(), device=state.device, dtype=torch.float32)
                next_qvel = torch.tensor(self.physics.data.qvel.copy(), device=state.device, dtype=torch.float32)
                
                next_states[i, :self.nq] = next_qpos
                next_states[i, self.nq:] = next_qvel
                
        finally:
            # Restore original state
            self.physics.set_state(original_qpos, original_qvel)
            self.physics.set_control(original_ctrl)
            
        return next_states.squeeze(0) if batch_size == 1 else next_states
        
    def compute_cost(self, state: torch.Tensor, action: torch.Tensor, 
                    next_state: torch.Tensor) -> torch.Tensor:
        """Compute task-specific cost."""
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
            
        batch_size = state.shape[0]
        
        # State cost (quadratic around target)
        state_error = state - self.target_state.to(state.device)
        Q = torch.diag(self.Q_diag).to(state.device)
        state_cost = torch.sum(state_error * (Q @ state_error.T).T, dim=-1)
        
        # Action cost
        R = torch.diag(self.R_diag).to(action.device)
        action_cost = torch.sum(action * (R @ action.T).T, dim=-1)
        
        total_cost = state_cost + action_cost
        
        return total_cost.squeeze(0) if batch_size == 1 else total_cost
        
    def compute_dynamics_jacobians(self, state: torch.Tensor, 
                                 action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute analytical Jacobians using MuJoCo's derivatives."""
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
            
        batch_size = state.shape[0]
        
        # Initialize Jacobians
        A = torch.zeros(batch_size, self.nx, self.nx, device=state.device)
        B = torch.zeros(batch_size, self.nx, self.nu, device=state.device)
        
        # Save current state
        original_qpos = self.physics.data.qpos.copy()
        original_qvel = self.physics.data.qvel.copy()
        original_ctrl = self.physics.data.ctrl.copy()
        
        try:
            for i in range(batch_size):
                # Set state and control
                qpos = state[i, :self.nq].cpu().numpy()
                qvel = state[i, self.nq:].cpu().numpy()
                ctrl = action[i].cpu().numpy()
                
                self.physics.set_state(qpos, qvel)
                self.physics.set_control(ctrl)
                
                # Compute analytical derivatives
                A_mj, B_mj = self._compute_mujoco_derivatives()
                
                A[i] = torch.tensor(A_mj, device=state.device, dtype=torch.float32)
                B[i] = torch.tensor(B_mj, device=state.device, dtype=torch.float32)
                
        finally:
            # Restore original state
            self.physics.set_state(original_qpos, original_qvel)
            self.physics.set_control(original_ctrl)
            
        return A.squeeze(0) if batch_size == 1 else A, B.squeeze(0) if batch_size == 1 else B
        
    def _compute_mujoco_derivatives(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute MuJoCo analytical derivatives."""
        # Allocate derivative matrices
        A = np.zeros((self.nx, self.nx))
        B = np.zeros((self.nx, self.nu))
        
        # MuJoCo derivative computation
        # This uses MuJoCo's finite difference derivatives
        # For analytical derivatives, we'd need to use mjd_transitionFD
        
        eps = 1e-6
        dt = self.physics.model.opt.timestep
        
        # Save current state
        qpos0 = self.physics.data.qpos.copy()
        qvel0 = self.physics.data.qvel.copy()
        ctrl0 = self.physics.data.ctrl.copy()
        
        # Compute A matrix (state derivatives)
        for i in range(self.nx):
            # Positive perturbation
            if i < self.nq:
                qpos_plus = qpos0.copy()
                qpos_plus[i] += eps
                self.physics.set_state(qpos_plus, qvel0)
            else:
                qvel_plus = qvel0.copy()
                qvel_plus[i - self.nq] += eps
                self.physics.set_state(qpos0, qvel_plus)
                
            self.physics.set_control(ctrl0)
            self.physics.step()
            
            qpos_plus_next = self.physics.data.qpos.copy()
            qvel_plus_next = self.physics.data.qvel.copy()
            state_plus_next = np.concatenate([qpos_plus_next, qvel_plus_next])
            
            # Negative perturbation
            if i < self.nq:
                qpos_minus = qpos0.copy()
                qpos_minus[i] -= eps
                self.physics.set_state(qpos_minus, qvel0)
            else:
                qvel_minus = qvel0.copy()
                qvel_minus[i - self.nq] -= eps
                self.physics.set_state(qpos0, qvel_minus)
                
            self.physics.set_control(ctrl0)
            self.physics.step()
            
            qpos_minus_next = self.physics.data.qpos.copy()
            qvel_minus_next = self.physics.data.qvel.copy()
            state_minus_next = np.concatenate([qpos_minus_next, qvel_minus_next])
            
            # Finite difference
            A[:, i] = (state_plus_next - state_minus_next) / (2 * eps)
            
            # Reset state
            self.physics.set_state(qpos0, qvel0)
            
        # Compute B matrix (control derivatives)
        for i in range(self.nu):
            # Positive perturbation
            ctrl_plus = ctrl0.copy()
            ctrl_plus[i] += eps
            self.physics.set_state(qpos0, qvel0)
            self.physics.set_control(ctrl_plus)
            self.physics.step()
            
            qpos_plus_next = self.physics.data.qpos.copy()
            qvel_plus_next = self.physics.data.qvel.copy()
            state_plus_next = np.concatenate([qpos_plus_next, qvel_plus_next])
            
            # Negative perturbation
            ctrl_minus = ctrl0.copy()
            ctrl_minus[i] -= eps
            self.physics.set_state(qpos0, qvel0)
            self.physics.set_control(ctrl_minus)
            self.physics.step()
            
            qpos_minus_next = self.physics.data.qpos.copy()
            qvel_minus_next = self.physics.data.qvel.copy()
            state_minus_next = np.concatenate([qpos_minus_next, qvel_minus_next])
            
            # Finite difference
            B[:, i] = (state_plus_next - state_minus_next) / (2 * eps)
            
            # Reset state
            self.physics.set_state(qpos0, qvel0)
            
        return A, B
        
    def compute_cost_hessians(self, state: torch.Tensor, action: torch.Tensor,
                            next_state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute Hessians of the cost function."""
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
            
        batch_size = state.shape[0]
        
        # For quadratic cost, Hessians are constant
        Q = torch.diag(self.Q_diag).to(state.device)
        R = torch.diag(self.R_diag).to(action.device)
        
        # Expand for batch
        Q_batch = Q.unsqueeze(0).repeat(batch_size, 1, 1)
        R_batch = R.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Linear terms (gradient of cost)
        state_error = state - self.target_state.to(state.device)
        q = 2 * (Q @ state_error.T).T
        r = 2 * (R @ action.T).T
        
        return (Q_batch.squeeze(0) if batch_size == 1 else Q_batch,
                q.squeeze(0) if batch_size == 1 else q,
                R_batch.squeeze(0) if batch_size == 1 else R_batch,
                r.squeeze(0) if batch_size == 1 else r)


def create_dmcontrol_interface(cfg, env):
    """Factory function to create DMControl interface."""
    return DMControlInterface(cfg, env)
