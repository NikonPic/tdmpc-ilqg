import torch
import torch.nn.functional as F
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict


@dataclass
class ILQGConfig:
    """Configuration for ILQG integration"""
    enabled: bool = True
    max_iterations: int = 5  # Reduziert von typischen 10-20
    convergence_threshold: float = 1e-2  # Lockerer als 1e-6
    line_search_steps: int = 5
    regularization: float = 1e-4
    horizon: int = 15  # Kann länger als MPPI horizon sein
    num_workers: int = 2  # Thread-Pool Größe
    trigger_uncertainty_threshold: float = 0.1
    trigger_frequency: int = 3  # Alle N Steps
    warm_start_from_tdmpc2: bool = True
    max_time_budget_ms: float = 5.0  # Zeitlimit pro ILQG call


class FastILQG:
    """
    Performance-optimierte ILQG Implementation für Integration mit TDMPC2.
    Verwendet shared TOLD model und TDMPC2 warm-starts.
    """
    
    def __init__(self, world_model: WorldModel, config: ILQGConfig, device: torch.device):
        self.model = world_model
        self.config = config
        self.device = device
        self.horizon = config.horizon
        
        # Caching für Linearisierungen (Performance!)
        self.jacobian_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _compute_jacobians(self, z: torch.Tensor, u: torch.Tensor, task: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Berechnet Jacobians der Dynamik bzgl. state und action.
        Mit Caching für Performance.
        """
        # Cache-Key für identical states/actions
        cache_key = (z.data_ptr(), u.data_ptr())
        if cache_key in self.jacobian_cache:
            self.cache_hits += 1
            return self.jacobian_cache[cache_key]
        
        self.cache_misses += 1
        
        # Enable gradients für Jacobian-Berechnung
        z_grad = z.detach().requires_grad_(True)
        u_grad = u.detach().requires_grad_(True)
        
        # Forward pass durch dynamics
        next_z = self.model.next(z_grad, u_grad, task)
        
        # Batch-Jacobian Berechnung (performanter als loops)
        A = torch.autograd.grad(
            outputs=next_z, inputs=z_grad,
            grad_outputs=torch.eye(next_z.shape[-1], device=self.device).repeat(z.shape[0], 1, 1),
            create_graph=False, retain_graph=True, only_inputs=True
        )[0]
        
        B = torch.autograd.grad(
            outputs=next_z, inputs=u_grad,
            grad_outputs=torch.eye(next_z.shape[-1], device=self.device).repeat(z.shape[0], 1, 1),
            create_graph=False, retain_graph=False, only_inputs=True
        )[0]
        
        # Cache result (aber limitiert für Memory)
        if len(self.jacobian_cache) < 1000:  # Cache size limit
            self.jacobian_cache[cache_key] = (A.detach(), B.detach())
        
        return A, B
    
    def _cost_and_derivatives(self, z: torch.Tensor, u: torch.Tensor, task: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Berechnet Cost und dessen Ableitungen (für LQR backward pass).
        """
        # Reward als negative cost
        reward = math.two_hot_inv(self.model.reward(z, u, task), self.model.cfg if hasattr(self.model, 'cfg') else type('cfg', (), {'bins': 255})())
        cost = -reward
        
        # Erste Ableitungen
        z_grad = z.requires_grad_(True)
        u_grad = u.requires_grad_(True)
        
        cost_z = torch.autograd.grad(cost.sum(), z_grad, create_graph=True, retain_graph=True)[0]
        cost_u = torch.autograd.grad(cost.sum(), u_grad, create_graph=True, retain_graph=True)[0]
        
        # Zweite Ableitungen (Hessians)
        cost_zz = torch.autograd.grad(cost_z.sum(), z_grad, create_graph=False, retain_graph=True)[0]
        cost_uu = torch.autograd.grad(cost_u.sum(), u_grad, create_graph=False, retain_graph=False)[0]
        
        return cost, cost_z, cost_u, cost_zz, cost_uu
    
    def _backward_pass(self, trajectory_z: torch.Tensor, trajectory_u: torch.Tensor, 
                      terminal_value_fn: Optional[torch.Tensor] = None, task: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ILQG Backward Pass - berechnet Feedback gains und feedforward terms.
        """
        horizon = trajectory_z.shape[0] - 1
        k = torch.zeros(horizon, trajectory_u.shape[-1], device=self.device)  # feedforward
        K = torch.zeros(horizon, trajectory_u.shape[-1], trajectory_z.shape[-1], device=self.device)  # feedback
        
        # Terminal Value Function (von TDMPC2)
        if terminal_value_fn is not None:
            V = terminal_value_fn
            V_z = torch.autograd.grad(V.sum(), trajectory_z[-1], create_graph=True)[0]
            V_zz = torch.autograd.grad(V_z.sum(), trajectory_z[-1], create_graph=False)[0]
        else:
            # Fallback: Zero terminal cost
            V = torch.zeros(trajectory_z.shape[0], device=self.device)
            V_z = torch.zeros_like(trajectory_z[-1])
            V_zz = torch.zeros(trajectory_z.shape[-1], trajectory_z.shape[-1], device=self.device)
        
        # Rückwärts durch Zeit
        for t in reversed(range(horizon)):
            z_t, u_t = trajectory_z[t], trajectory_u[t]
            
            # Dynamics Jacobians
            A, B = self._compute_jacobians(z_t, u_t, task)
            
            # Cost derivatives
            cost, cost_z, cost_u, cost_zz, cost_uu = self._cost_and_derivatives(z_t, u_t, task)
            
            # Q-function approximation (LQR)
            Q_z = cost_z + A.T @ V_z
            Q_u = cost_u + B.T @ V_z
            Q_zz = cost_zz + A.T @ V_zz @ A
            Q_uu = cost_uu + B.T @ V_zz @ B + self.config.regularization * torch.eye(B.shape[-1], device=self.device)
            Q_uz = B.T @ V_zz @ A
            
            # Solve for gains (regularized inverse)
            try:
                Q_uu_inv = torch.linalg.inv(Q_uu)
                k[t] = -Q_uu_inv @ Q_u
                K[t] = -Q_uu_inv @ Q_uz
            except torch.linalg.LinAlgError:
                # Fallback bei singular matrix
                k[t] = torch.zeros_like(Q_u)
                K[t] = torch.zeros_like(Q_uz)
            
            # Value function update
            V_z = Q_z + K[t].T @ Q_uu @ k[t] + K[t].T @ Q_u + Q_uz.T @ k[t]
            V_zz = Q_zz + K[t].T @ Q_uu @ K[t] + K[t].T @ Q_uz + Q_uz.T @ K[t]
        
        return k, K
    
    def _forward_pass(self, trajectory_z: torch.Tensor, trajectory_u: torch.Tensor,
                     k: torch.Tensor, K: torch.Tensor, alpha: float = 1.0,
                     task: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        ILQG Forward Pass mit Line Search.
        """
        horizon = len(k)
        new_trajectory_z = torch.zeros_like(trajectory_z)
        new_trajectory_u = torch.zeros_like(trajectory_u)
        new_trajectory_z[0] = trajectory_z[0]  # Gleicher Startzustand
        
        total_cost = 0.0
        
        for t in range(horizon):
            z_t = new_trajectory_z[t]
            
            # Control update mit Line Search
            delta_u = k[t] + K[t] @ (z_t - trajectory_z[t])
            new_trajectory_u[t] = trajectory_u[t] + alpha * delta_u
            new_trajectory_u[t] = torch.clamp(new_trajectory_u[t], -1, 1)  # Action bounds
            
            # Forward dynamics
            new_trajectory_z[t+1] = self.model.next(z_t, new_trajectory_u[t], task)
            
            # Accumulate cost
            reward = math.two_hot_inv(self.model.reward(z_t, new_trajectory_u[t], task), 
                                    self.model.cfg if hasattr(self.model, 'cfg') else type('cfg', (), {'bins': 255})())
            total_cost += -reward.sum().item()
        
        return new_trajectory_z, new_trajectory_u, total_cost
    
    def optimize(self, initial_state: torch.Tensor, warm_start_actions: Optional[torch.Tensor] = None,
                terminal_value_fn: Optional[torch.Tensor] = None, task: Optional[torch.Tensor] = None) -> Dict:
        """
        Hauptfunktion: ILQG Optimierung mit Time Budget.
        """
        start_time = time.time()
        
        # Initialisierung
        if warm_start_actions is not None:
            actions = warm_start_actions[:self.horizon].clone()
        else:
            actions = torch.zeros(self.horizon, initial_state.shape[-1], device=self.device)
        
        # Forward rollout für initiale Trajektorie
        states = torch.zeros(self.horizon + 1, initial_state.shape[-1], device=self.device)
        states[0] = initial_state
        
        for t in range(self.horizon):
            states[t+1] = self.model.next(states[t], actions[t], task)
        
        best_cost = float('inf')
        converged = False
        
        # ILQG Iterationen mit Time Budget
        for iteration in range(self.config.max_iterations):
            # Check time budget
            if (time.time() - start_time) * 1000 > self.config.max_time_budget_ms:
                break
                
            # Backward pass
            k, K = self._backward_pass(states, actions, terminal_value_fn, task)
            
            # Forward pass mit Line Search
            best_alpha = 1.0
            best_new_cost = best_cost
            
            for alpha in [1.0, 0.5, 0.25, 0.1]:  # Simple line search
                new_states, new_actions, cost = self._forward_pass(states, actions, k, K, alpha, task)
                
                if cost < best_new_cost:
                    best_alpha = alpha
                    best_new_cost = cost
                    best_states = new_states
                    best_actions = new_actions
            
            # Update wenn Verbesserung
            if best_new_cost < best_cost:
                cost_improvement = best_cost - best_new_cost
                states = best_states
                actions = best_actions
                best_cost = best_new_cost
                
                # Konvergenz Check
                if cost_improvement < self.config.convergence_threshold:
                    converged = True
                    break
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            'states': states,
            'actions': actions,
            'cost': best_cost,
            'converged': converged,
            'iterations': iteration + 1,
            'execution_time_ms': execution_time,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }


class TDMPC2WithILQG(torch.nn.Module):
    """
    TDMPC2 mit ILQG Integration. Erweitert die originale TDMPC2 Klasse
    um asynchrone ILQG-Optimierung ohne Performance-Einbußen.
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        # Original TDMPC2 Initialisierung
        self.cfg = cfg
        self.device = torch.device('cuda:0')
        self.model = WorldModel(cfg).to(self.device)
        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._termination.parameters() if self.cfg.episodic else []},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []
             }
        ], lr=self.cfg.lr, capturable=True)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20)
        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
        ) if self.cfg.multitask else self._get_discount(cfg.episode_length)
        self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
        
        # ILQG Integration
        self.ilqg_config = ILQGConfig(
            enabled=getattr(cfg, 'use_ilqg', True),
            max_iterations=getattr(cfg, 'ilqg_max_iterations', 5),
            horizon=getattr(cfg, 'ilqg_horizon', 15),
            num_workers=getattr(cfg, 'ilqg_workers', 2)
        )
        
        if self.ilqg_config.enabled:
            self.ilqg_solver = FastILQG(self.model, self.ilqg_config, self.device)
            self.ilqg_executor = ThreadPoolExecutor(max_workers=self.ilqg_config.num_workers, thread_name_prefix="ILQG")
            self.ilqg_futures = {}  # Track running ILQG optimizations
            self.ilqg_results_buffer = deque(maxlen=100)  # Buffer für high-quality Trajektorien
            self.step_counter = 0
            self.ilqg_stats = {
                'total_runs': 0,
                'successful_runs': 0,
                'avg_execution_time': 0.0,
                'cache_hit_rate': 0.0
            }
        
        # Compile wenn gewünscht
        if cfg.compile:
            print('Compiling update function with torch.compile...')
            self._update = torch.compile(self._update, mode="reduce-overhead")

    def _should_trigger_ilqg(self, z: torch.Tensor, task: Optional[torch.Tensor] = None) -> bool:
        """
        Entscheidet intelligently, ob ILQG für diesen State ausgeführt werden soll.
        """
        if not self.ilqg_config.enabled:
            return False
            
        # Frequency-based triggering
        if self.step_counter % self.ilqg_config.trigger_frequency != 0:
            return False
            
        # Value uncertainty triggering (heuristisch)
        with torch.no_grad():
            action_samples = torch.randn(10, self.cfg.action_dim, device=self.device) * 0.1
            q_values = []
            for a in action_samples:
                q_val = self.model.Q(z, a.unsqueeze(0), task, return_type='avg')
                q_values.append(q_val)
            
            q_values = torch.stack(q_values)
            uncertainty = q_values.std().item()
            
            return uncertainty > self.ilqg_config.trigger_uncertainty_threshold

    def _collect_ilqg_results(self):
        """
        Sammelt fertige ILQG Resultate (non-blocking).
        """
        if not hasattr(self, 'ilqg_futures'):
            return
            
        completed_futures = []
        for future_id, future in self.ilqg_futures.items():
            if future.done():
                try:
                    result = future.result()
                    if result['converged']:
                        self.ilqg_results_buffer.append(result)
                        self.ilqg_stats['successful_runs'] += 1
                        
                        # Update stats
                        self.ilqg_stats['avg_execution_time'] = (
                            0.9 * self.ilqg_stats['avg_execution_time'] + 
                            0.1 * result['execution_time_ms']
                        )
                        self.ilqg_stats['cache_hit_rate'] = result.get('cache_hit_rate', 0.0)
                        
                    self.ilqg_stats['total_runs'] += 1
                    completed_futures.append(future_id)
                except Exception as e:
                    print(f"ILQG future failed: {e}")
                    completed_futures.append(future_id)
        
        # Cleanup completed futures
        for future_id in completed_futures:
            del self.ilqg_futures[future_id]

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Erweiterte act-Funktion mit ILQG Integration.
        Behält vollständige Kompatibilität zur Original-API.
        """
        self.step_counter += 1
        
        # Sammle fertige ILQG Resultate (non-blocking)
        if self.ilqg_config.enabled:
            self._collect_ilqg_results()
        
        # Original TDMPC2 action selection
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
            
        if self.cfg.mpc:
            action = self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
        else:
            z = self.model.encode(obs, task)
            action, info = self.model.pi(z, task)
            if eval_mode:
                action = info["mean"]
            action = action[0].cpu()
        
        # Asynchrone ILQG Optimierung triggern
        if self.ilqg_config.enabled and not eval_mode:
            z = self.model.encode(obs, task)
            if self._should_trigger_ilqg(z, task):
                self._submit_ilqg_optimization(z, task)
        
        return action
    
    def _submit_ilqg_optimization(self, z: torch.Tensor, task: Optional[torch.Tensor]):
        """
        Startet asynchrone ILQG Optimierung (non-blocking).
        """
        # Check if we have capacity
        if len(self.ilqg_futures) >= self.ilqg_config.num_workers:
            return
            
        # Warm start von TDMPC2 policy
        warm_start_actions = None
        if self.ilqg_config.warm_start_from_tdmpc2:
            with torch.no_grad():
                warm_start_actions = torch.zeros(self.ilqg_config.horizon, self.cfg.action_dim, device=self.device)
                z_temp = z.clone()
                for t in range(min(self.ilqg_config.horizon, 10)):  # Limit für Performance
                    action, _ = self.model.pi(z_temp, task)
                    warm_start_actions[t] = action[0]
                    z_temp = self.model.next(z_temp, action, task)
        
        # Terminal value function von TDMPC2
        terminal_value = None
        if self.ilqg_config.horizon > self.cfg.horizon:
            with torch.no_grad():
                z_terminal = z.clone()
                # Rollout to terminal state
                for t in range(self.cfg.horizon):
                    action, _ = self.model.pi(z_terminal, task)
                    z_terminal = self.model.next(z_terminal, action, task)
                terminal_action, _ = self.model.pi(z_terminal, task)
                terminal_value = self.model.Q(z_terminal, terminal_action, task, return_type='avg')
        
        # Submit asynchron
        future = self.ilqg_executor.submit(
            self.ilqg_solver.optimize,
            z.detach().clone(),
            warm_start_actions.detach().clone() if warm_start_actions is not None else None,
            terminal_value.detach().clone() if terminal_value is not None else None,
            task.detach().clone() if task is not None else None
        )
        
        self.ilqg_futures[id(future)] = future

    def get_ilqg_enhanced_buffer_data(self):
        """
        Gibt high-quality Trajektorien aus ILQG für Experience Replay zurück.
        """
        if not self.ilqg_config.enabled or len(self.ilqg_results_buffer) == 0:
            return None
            
        # Beste Trajektorie basierend auf Cost
        best_result = min(self.ilqg_results_buffer, key=lambda x: x['cost'])
        return {
            'states': best_result['states'],
            'actions': best_result['actions'],
            'priority_weight': 2.0,  # Höhere Priorität für ILQG Trajektorien
            'source': 'ilqg'
        }
    
    def get_ilqg_stats(self) -> Dict:
        """
        Gibt ILQG Performance-Statistiken zurück.
        """
        if not self.ilqg_config.enabled:
            return {}
            
        return {
            'ilqg_enabled': True,
            'ilqg_runs_total': self.ilqg_stats['total_runs'],
            'ilqg_success_rate': self.ilqg_stats['successful_runs'] / max(1, self.ilqg_stats['total_runs']),
            'ilqg_avg_time_ms': self.ilqg_stats['avg_execution_time'],
            'ilqg_cache_hit_rate': self.ilqg_stats['cache_hit_rate'],
            'ilqg_buffer_size': len(self.ilqg_results_buffer),
            'ilqg_active_optimizations': len(self.ilqg_futures)
        }

    # Alle anderen Methoden aus der originalen TDMPC2 Klasse bleiben unverändert
    def _get_discount(self, episode_length):
        frac = episode_length/self.cfg.discount_denom
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
        state_dict = state_dict["model"] if "model" in state_dict else state_dict
        state_dict = api_model_conversion(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict)

    @property
    def plan(self):
        _plan_val = getattr(self, "_plan_val", None)
        if _plan_val is not None:
            return _plan_val
        if self.cfg.compile:
            plan = torch.compile(self._plan, mode="reduce-overhead")
        else:
            plan = self._plan
        self._plan_val = plan
        return self._plan_val

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        G, discount = 0, 1
        termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G = G + discount * (1-termination) * reward
            discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
            discount = discount * discount_update
            if self.cfg.episodic:
                termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
        action, _ = self.model.pi(z, task)
        return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

    @torch.no_grad()
    def _plan(self, obs, t0=False, eval_mode=False, task=None):
        # Original TDMPC2 _plan method bleibt identisch
        z = self.model.encode(obs, task)
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon-1):
                pi_actions[t], _ = self.model.pi(_z, task)
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1], _ = self.model.pi(_z, task)

        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions

        for _ in range(self.cfg.iterations):
            r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.cfg.num_pi_trajs:] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            value = self._estimate_value(z, actions, task).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, std = actions[0], std[0]
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
        self._prev_mean.copy_(mean)
        return a.clamp(-1, 1)

    def update_pi(self, zs, task):
        action, info = self.model.pi(zs, task)
        qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)

        info = TensorDict({
            "pi_loss": pi_loss,
            "pi_grad_norm": pi_grad_norm,
            "pi_entropy": info["entropy"],
            "pi_scaled_entropy": info["scaled_entropy"],
            "pi_scale": self.scale.value,
        })
        return info

    @torch.no_grad()
    def _td_target(self, next_z, reward, terminated, task):
        action, _ = self.model.pi(next_z, task)
        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

    def _update(self, obs, action, reward, terminated, task=None):
        # Original _update method bleibt unverändert
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, terminated, task)

        self.model.train()

        zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
            z = self.model.next(z, _action, task)
            consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
            zs[t+1] = z

        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)
        if self.cfg.episodic:
            termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

        reward_loss, value_loss = 0, 0
        for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
            reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
            for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
                value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon
        if self.cfg.episodic:
            termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
        else:
            termination_loss = 0.
        value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.termination_coef * termination_loss +
            self.cfg.value_coef * value_loss
        )

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        pi_info = self.update_pi(zs.detach(), task)
        self.model.soft_update_target_Q()

        self.model.eval()
        info = TensorDict({
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "termination_loss": termination_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
        })
        if self.cfg.episodic:
            info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
        info.update(pi_info)
        
        # Füge ILQG stats hinzu
        info.update(self.get_ilqg_stats())
        
        return info.detach().mean()

    def update(self, buffer):
        obs, action, reward, terminated, task = buffer.sample()
        kwargs = {}
        if task is not None:
            kwargs["task"] = task
        torch.compiler.cudagraph_mark_step_begin()
        return self._update(obs, action, reward, terminated, **kwargs)

    def __del__(self):
        """Cleanup beim Zerstören der Instanz"""
        if hasattr(self, 'ilqg_executor'):
            self.ilqg_executor.shutdown(wait=False)


# Verwendungsbeispiel:
# cfg.use_ilqg = True  # Aktiviert ILQG Integration
# cfg.ilqg_max_iterations = 5  # Schnelle ILQG für Performance
# cfg.ilqg_horizon = 15  # Längerer Horizont als MPPI
# cfg.ilqg_workers = 2  # Thread pool size
# 
# agent = TDMPC2WithILQG(cfg)
# 
# # Normale Nutzung - API bleibt identisch!
# action = agent.act(observation)
# loss_info = agent.update(buffer)
# 
# # ILQG Statistics abrufen
# stats = agent.get_ilqg_stats()
# print(f"ILQG success rate: {stats.get('ilqg_success_rate', 0):.2%}")