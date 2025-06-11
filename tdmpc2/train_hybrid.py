import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from simple_hybrid_tdmpc2 import SimpleHybridTDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def convert_tensors_to_scalars(metrics_dict):
    """Convert tensor values to scalars for logging."""
    if not isinstance(metrics_dict, dict):
        return metrics_dict
    
    converted = {}
    for key, value in metrics_dict.items():
        if hasattr(value, 'item') and callable(getattr(value, 'item')):
            try:
                converted[key] = value.item()
            except:
                converted[key] = float(value) if hasattr(value, '__float__') else value
        else:
            converted[key] = value
    return converted


class HybridOnlineTrainer(OnlineTrainer):
    """
    Enhanced online trainer that supports hybrid MPC training.
    """
    
    def __init__(self, cfg, env, agent, buffer, logger):
        super().__init__(cfg, env, agent, buffer, logger)
        self.hybrid_agent = agent if isinstance(agent, SimpleHybridTDMPC2) else None
        
        # Track hybrid-specific metrics
        self.hybrid_metrics = {}
        
    def train(self):
        """Train a Hybrid TD-MPC2 agent."""
        train_metrics, done, eval_next = {}, True, False
        
        while self._step <= self.cfg.steps:
            # Update hybrid MPC transition schedule
            if self.hybrid_agent is not None:
                self.hybrid_agent.update_training_step(self._step)
                
            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    
                    # Log hybrid diagnostics during evaluation
                    if self.hybrid_agent is not None:
                        hybrid_info = self.hybrid_agent.get_diagnostics()
                        eval_metrics.update(hybrid_info)
                        
                        # Print progress for hybrid MPC
                        if self._step % self.cfg.get('log_freq', 10000) == 0:
                            classical_horizon = hybrid_info.get('classical_horizon', 0)
                            progress = hybrid_info.get('transition_progress', 0.0)
                            print(f"Step {self._step}: Classical horizon = {classical_horizon}, "
                                  f"Transition progress = {progress:.3f}")
                    
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(convert_tensors_to_scalars(eval_metrics), 'eval')
                    eval_next = False

                if self._step > 0:
                    # Check for episode termination in episodic environments
                    if hasattr(self, '_tds') and len(self._tds) > 1:
                        # Get episode info
                        episode_reward = sum([td.get('reward', 0) for td in self._tds[1:]])
                        episode_length = len(self._tds)
                        
                        # Check if environment provides termination info
                        episode_terminated = False
                        episode_success = False
                        if hasattr(self, '_last_info') and self._last_info is not None:
                            episode_terminated = self._last_info.get('terminated', False)
                            episode_success = self._last_info.get('success', False)
                        
                        # Handle episodic mode
                        if episode_terminated and not self.cfg.episodic:
                            print(f"Warning: Termination detected at step {self._step} but episodic=false. "
                                  f"Consider setting episodic=true in config.")
                        
                        train_metrics.update({
                            'episode_reward': episode_reward,
                            'episode_success': episode_success,
                            'episode_length': episode_length,
                            'episode_terminated': episode_terminated
                        })
                        
                        # Add hybrid diagnostics to training metrics
                        if self.hybrid_agent is not None:
                            hybrid_info = self.hybrid_agent.get_diagnostics()
                            if hybrid_info.get("hybrid_enabled", False):
                                train_metrics.update({
                                    'classical_horizon': hybrid_info.get("classical_horizon", 0),
                                    'transition_progress': hybrid_info.get("transition_progress", 0.0),
                                })
                    
                    train_metrics.update(self.common_metrics())
                    self.logger.log(convert_tensors_to_scalars(train_metrics), 'train')
                    
                    # Add episode to buffer if we have trajectory data
                    if hasattr(self, '_tds') and len(self._tds) > 1:
                        try:
                            episode_data = torch.cat(self._tds)
                            self._ep_idx = self.buffer.add(episode_data)
                        except Exception as e:
                            print(f"Warning: Failed to add episode to buffer: {e}")

                # Reset for new episode
                obs = self.env.reset()
                self._tds = [self.to_td(obs)]
                self._last_info = None

            # Collect experience
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds)==1)
            else:
                action = self.env.rand_act()
                
            # Step environment
            try:
                obs, reward, done, info = self.env.step(action)
                self._last_info = info
                
                # Create trajectory data
                td_data = self.to_td(obs, action, reward, info.get('terminated', done))
                self._tds.append(td_data)
                
            except Exception as e:
                print(f"Error during environment step: {e}")
                # Reset environment on error
                obs = self.env.reset()
                self._tds = [self.to_td(obs)]
                done = True
                continue

            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print('Pretraining agent on seed data...')
                else:
                    num_updates = 1
                    
                # Perform agent updates
                for _ in range(num_updates):
                    try:
                        _train_metrics = self.agent.update(self.buffer)
                        train_metrics.update(_train_metrics)
                    except Exception as e:
                        print(f"Error during agent update: {e}")
                        break

            self._step += 1

        # Finish training
        self.logger.finish(self.agent)
        
        # Print final hybrid statistics
        if self.hybrid_agent is not None:
            final_diagnostics = self.hybrid_agent.get_diagnostics()
            print("\nFinal Hybrid Training Statistics:")
            for key, value in final_diagnostics.items():
                print(f"  {key}: {value}")


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
    """
    Script for training hybrid TD-MPC2 agents.

    Most relevant args:
        `task`: task name (or mt30/mt80 for multi-task training)
        `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
        `steps`: number of training/environment steps (default: 10M)
        `seed`: random seed (default: 1)
        `hybrid_mpc`: enable hybrid MPC (default: false)
        `max_classical_horizon`: maximum classical horizon (default: 10)
        `transition_steps`: steps over which to transition (default: 1M)

    See config.yaml for a full list of args.

    Example usage:
    ```
        $ python train_hybrid.py task=cartpole-balance hybrid_mpc=true max_classical_horizon=5
        $ python train_hybrid.py task=cheetah-run hybrid_mpc=true transition_schedule=exponential
        $ python train_hybrid.py task=walker-walk hybrid_mpc=true transition_steps=500000
    ```
    """
    assert torch.cuda.is_available(), "CUDA is required for training"
    assert cfg.steps > 0, 'Must train for at least 1 step.'
    
    # Parse and validate configuration
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
    
    # Validate hybrid MPC configuration
    if getattr(cfg, 'hybrid_mpc', False):
        # Set default values for hybrid parameters if not specified
        if not hasattr(cfg, 'max_classical_horizon'):
            cfg.max_classical_horizon = 10
        if not hasattr(cfg, 'min_classical_horizon'):
            cfg.min_classical_horizon = 0
        if not hasattr(cfg, 'transition_steps'):
            cfg.transition_steps = min(1_000_000, cfg.steps // 2)
        if not hasattr(cfg, 'transition_schedule'):
            cfg.transition_schedule = 'linear'
        if not hasattr(cfg, 'classical_algorithm'):
            cfg.classical_algorithm = 'ilqg'
        if not hasattr(cfg, 'ilqg_iterations'):
            cfg.ilqg_iterations = 5
        if not hasattr(cfg, 'ilqg_regularization'):
            cfg.ilqg_regularization = 1e-4
            
        # Validate parameters
        assert cfg.max_classical_horizon >= cfg.min_classical_horizon, \
            "max_classical_horizon must be >= min_classical_horizon"
        assert cfg.transition_steps > 0, "transition_steps must be positive"
        assert cfg.transition_schedule in ['linear', 'exponential', 'cosine', 'step'], \
            "transition_schedule must be one of: linear, exponential, cosine, step"
        assert cfg.classical_algorithm in ['ilqg'], \
            "classical_algorithm must be 'ilqg'"
    
    # Create environment
    try:
        env = make_env(cfg)
        print(f"Created environment: {cfg.task}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        raise
    
    # Create hybrid agent
    try:
        agent = SimpleHybridTDMPC2(cfg, env)
        print(f"Created agent: {'Hybrid' if cfg.get('hybrid_mpc', False) else 'Standard'} TD-MPC2")
    except Exception as e:
        print(f"Error creating agent: {e}")
        raise
    
    # Print configuration
    print(colored('Training Configuration:', 'green', attrs=['bold']))
    print(f"  Task: {cfg.task}")
    print(f"  Model size: {cfg.model_size}")
    print(f"  Training steps: {cfg.steps:,}")
    print(f"  Seed: {cfg.seed}")
    
    if cfg.get('hybrid_mpc', False):
        print(colored('Hybrid MPC Configuration:', 'cyan', attrs=['bold']))
        print(f"  Max classical horizon: {cfg.max_classical_horizon}")
        print(f"  Min classical horizon: {cfg.min_classical_horizon}")
        print(f"  Transition steps: {cfg.transition_steps:,}")
        print(f"  Transition schedule: {cfg.transition_schedule}")
        print(f"  Classical algorithm: {cfg.classical_algorithm}")
        print(f"  iLQG iterations: {cfg.ilqg_iterations}")
    else:
        print(colored('Using standard TD-MPC2 (hybrid MPC disabled)', 'yellow'))
    
    # Select trainer
    trainer_cls = OfflineTrainer if cfg.multitask else HybridOnlineTrainer
    
    try:
        trainer = trainer_cls(
            cfg=cfg,
            env=env,
            agent=agent,
            buffer=Buffer(cfg),
            logger=Logger(cfg),
        )
        print(f"Created trainer: {trainer_cls.__name__}")
    except Exception as e:
        print(f"Error creating trainer: {e}")
        raise
    
    # Start training
    print(colored('\nStarting training...', 'green', attrs=['bold']))
    try:
        trainer.train()
        print(colored('\nTraining completed successfully!', 'green', attrs=['bold']))
    except KeyboardInterrupt:
        print(colored('\nTraining interrupted by user', 'yellow'))
    except Exception as e:
        print(colored(f'\nTraining failed with error: {e}', 'red'))
        raise


if __name__ == '__main__':
    train()