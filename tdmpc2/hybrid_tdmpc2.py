import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict
from tdmpc2.tdmpc2 import TDMPC2
from tdmpc2.hybrid_mpc import HybridMPC
from tdmpc2.envs.dmcontrol_interface import create_dmcontrol_interface


class HybridTDMPC2(TDMPC2):
    """
    Hybrid TD-MPC2 agent that combines classical MPC (iLQG) for initial timesteps
    with learned MPC (MPPI) for remaining timesteps.
    
    This extends the original TDMPC2 with hybrid planning capabilities.
    """

    def __init__(self, cfg, env=None):
        super().__init__(cfg)
        self.env = env
        self.hybrid_mpc_enabled = cfg.get('hybrid_mpc', False)
        
        if self.hybrid_mpc_enabled and env is not None:
            # Initialize hybrid MPC
            self.hybrid_mpc = self._create_hybrid_mpc(cfg, env)
            print(f"Hybrid MPC enabled with max classical horizon: {cfg.get('max_classical_horizon', 10)}")
        else:
            self.hybrid_mpc = None
            print("Using standard TD-MPC2 planning")
            
    def _create_hybrid_mpc(self, cfg, env):
        """Create hybrid MPC instance with appropriate environment interface."""
        # Determine environment type and create appropriate interface
        env_name = cfg.task.lower()
        
        if any(domain in env_name for domain in ['cartpole', 'pendulum', 'cheetah', 'walker', 'dog', 'fish', 'humanoid']):
            # DMControl environment
            env_interface = create_dmcontrol_interface(cfg, env)
            return HybridMPC(cfg, env, self.model)
        else:
            # For other environments, use the base implementation
            return HybridMPC(cfg, env, self.model)

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Select an action using hybrid planning when enabled, otherwise use standard TD-MPC2.
        
        Args:
            obs (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (int): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
        """
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
            
        if self.hybrid_mpc_enabled and self.hybrid_mpc is not None:
            # Use hybrid planning
            if self.cfg.mpc:
                return self.hybrid_mpc.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
            else:
                # Fallback to policy if MPC is disabled
                z = self.model.encode(obs, task)
                action, info = self.model.pi(z, task)
                if eval_mode:
                    action = info["mean"]
                return action[0].cpu()
        else:
            # Use standard TD-MPC2 planning
            return super().act(obs, t0, eval_mode, task)

    @torch.no_grad()
    def _hybrid_plan(self, obs, t0=False, eval_mode=False, task=None):
        """
        Hybrid planning that combines classical and learned MPC.
        This method integrates the original MPPI with classical MPC.
        """
        if self.hybrid_mpc is None:
            # Fallback to original planning
            return self._plan(obs, t0, eval_mode, task)
            
        # Get current classical horizon
        classical_horizon = self.hybrid_mpc.get_classical_horizon()
        total_horizon = self.cfg.horizon
        
        if classical_horizon == 0:
            # Pure learned MPC (original TD-MPC2)
            return self._plan(obs, t0, eval_mode, task)
        elif classical_horizon >= total_horizon:
            # Pure classical MPC
            return self.hybrid_mpc._classical_mpc_plan(obs, classical_horizon, task)
        else:
            # True hybrid approach
            return self._integrated_hybrid_plan(obs, classical_horizon, t0, eval_mode, task)

    @torch.no_grad()
    def _integrated_hybrid_plan(self, obs, classical_horizon, t0=False, eval_mode=False, task=None):
        """
        Integrated hybrid planning that considers both classical and learned trajectories
        in the MPPI optimization.
        """
        # Sample policy trajectories (original TD-MPC2 approach)
        z = self.model.encode(obs, task)
        
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon-1):
                pi_actions[t], _ = self.model.pi(_z, task)
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1], _ = self.model.pi(_z, task)

        # Get classical MPC trajectory
        state = self.hybrid_mpc.env_interface.obs_to_state(obs)
        classical_actions = self.hybrid_mpc.classical_mpc.plan(state, classical_horizon, task)
        
        # Pad classical actions to full horizon if needed
        if classical_horizon < self.cfg.horizon:
            # Use learned policy for remaining steps
            future_state = state
            for t in range(classical_horizon):
                future_state = self.hybrid_mpc.env_interface.step_dynamics(future_state, classical_actions[t])
            
            # Convert back to observation and encode
            future_obs = self.hybrid_mpc.env_interface.state_to_obs(future_state)
            future_z = self.model.encode(future_obs, task)
            
            # Generate remaining actions with learned policy
            remaining_actions = torch.empty(self.cfg.horizon - classical_horizon, 1, self.cfg.action_dim, device=self.device)
            _z = future_z
            for t in range(self.cfg.horizon - classical_horizon):
                action, _ = self.model.pi(_z, task)
                remaining_actions[t] = action[:1]  # Take first sample
                _z = self.model.next(_z, action[:1], task)
            
            # Combine classical and learned actions
            full_classical_actions = torch.cat([
                classical_actions.unsqueeze(1),  # Add batch dimension
                remaining_actions
            ], dim=0)
        else:
            full_classical_actions = classical_actions.unsqueeze(1)

        # Initialize state and parameters for MPPI
        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
        
        # Initialize mean with classical trajectory
        mean[:classical_horizon] = full_classical_actions[:classical_horizon, 0]
        
        if not t0:
            # Use previous mean for remaining steps
            remaining_steps = self.cfg.horizon - classical_horizon
            if remaining_steps > 0 and hasattr(self, '_prev_mean'):
                mean[classical_horizon:] = self._prev_mean[1:1+remaining_steps]

        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        
        # Include policy trajectories
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions
            
        # Include classical trajectory as one of the samples
        actions[:, -1] = full_classical_actions.squeeze(1)

        # Iterate MPPI with hybrid initialization
        for iteration in range(self.cfg.iterations):
            # Sample actions
            num_random_samples = self.cfg.num_samples - self.cfg.num_pi_trajs - 1  # -1 for classical trajectory
            if num_random_samples > 0:
                r = torch.randn(self.cfg.horizon, num_random_samples, self.cfg.action_dim, device=std.device)
                actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
                actions_sample = actions_sample.clamp(-1, 1)
                actions[:, self.cfg.num_pi_trajs:-1] = actions_sample
                
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions with hybrid evaluation
            value = self._hybrid_estimate_value(z, actions, task, classical_horizon).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters with bias towards classical solution
            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score = score / score.sum(0)
            
            # Weighted update with classical bias
            classical_weight = max(0.1, 1.0 - iteration / self.cfg.iterations)  # Decrease classical influence over iterations
            mean_update = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            mean = classical_weight * mean + (1 - classical_weight) * mean_update
            
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action
        rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, std = actions[0], std[0]
        
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
            
        self._prev_mean.copy_(mean)
        return a.clamp(-1, 1)

    @torch.no_grad()
    def _hybrid_estimate_value(self, z, actions, task, classical_horizon):
        """
        Estimate value with hybrid dynamics: classical for first steps, learned for remainder.
        """
        G, discount = 0, 1
        termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
        
        # Convert latent state to full state for classical simulation
        # This is a simplification - in practice, you'd need proper state conversion
        obs = z  # Placeholder - would need proper conversion
        
        for t in range(self.cfg.horizon):
            if t < classical_horizon:
                # Use classical dynamics and cost
                if hasattr(self.hybrid_mpc, 'env_interface'):
                    # Convert to state space for classical evaluation
                    state = obs  # Simplified - would need proper conversion
                    next_state = self.hybrid_mpc.env_interface.step_dynamics(state, actions[t])
                    reward = -self.hybrid_mpc.env_interface.compute_cost(state, actions[t], next_state)
                    obs = next_state  # Update for next iteration
                else:
                    # Fallback to learned dynamics
                    reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
                    z = self.model.next(z, actions[t], task)
            else:
                # Use learned dynamics
                reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
                z = self.model.next(z, actions[t], task)
                
            G = G + discount * (1-termination) * reward
            discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
            discount = discount * discount_update
            
            if self.cfg.episodic:
                termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
                
        # Final value estimate
        action, _ = self.model.pi(z, task)
        return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

    def update_training_step(self, step):
        """Update the training step for hybrid MPC transition scheduling."""
        if self.hybrid_mpc is not None:
            self.hybrid_mpc.update_step(step)

    def get_hybrid_info(self):
        """Get information about the current hybrid MPC state."""
        if self.hybrid_mpc is None:
            return {"hybrid_enabled": False}
            
        return {
            "hybrid_enabled": True,
            "current_step": self.hybrid_mpc.current_step,
            "classical_horizon": self.hybrid_mpc.get_classical_horizon(),
            "max_classical_horizon": self.hybrid_mpc.max_classical_horizon,
            "transition_schedule": self.hybrid_mpc.transition_schedule,
        }

    def save(self, fp):
        """Save state dict including hybrid MPC state."""
        state_dict = {"model": self.model.state_dict()}
        
        if self.hybrid_mpc is not None:
            state_dict["hybrid_mpc"] = {
                "current_step": self.hybrid_mpc.current_step,
                "max_classical_horizon": self.hybrid_mpc.max_classical_horizon,
                "min_classical_horizon": self.hybrid_mpc.min_classical_horizon,
                "transition_steps": self.hybrid_mpc.transition_steps,
                "transition_schedule": self.hybrid_mpc.transition_schedule,
            }
            
        torch.save(state_dict, fp)

    def load(self, fp):
        """Load state dict including hybrid MPC state."""
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
            
        # Load model
        model_state = state_dict["model"] if "model" in state_dict else state_dict
        model_state = api_model_conversion(self.model.state_dict(), model_state)
        self.model.load_state_dict(model_state)
        
        # Load hybrid MPC state if available
        if "hybrid_mpc" in state_dict and self.hybrid_mpc is not None:
            hybrid_state = state_dict["hybrid_mpc"]
            self.hybrid_mpc.current_step = hybrid_state.get("current_step", 0)
            self.hybrid_mpc.max_classical_horizon = hybrid_state.get("max_classical_horizon", 10)
            self.hybrid_mpc.min_classical_horizon = hybrid_state.get("min_classical_horizon", 0)
            self.hybrid_mpc.transition_steps = hybrid_state.get("transition_steps", 1_000_000)
            self.hybrid_mpc.transition_schedule = hybrid_state.get("transition_schedule", "linear")
            
        return

    # Override the plan property to use hybrid planning
    @property
    def plan(self):
        if self.hybrid_mpc_enabled and self.hybrid_mpc is not None:
            _plan_val = getattr(self, "_hybrid_plan_val", None)
            if _plan_val is not None:
                return _plan_val
            if self.cfg.compile:
                plan = torch.compile(self._hybrid_plan, mode="reduce-overhead")
            else:
                plan = self._hybrid_plan
            self._hybrid_plan_val = plan
            return self._hybrid_plan_val
        else:
            # Use original planning
            return super().plan
