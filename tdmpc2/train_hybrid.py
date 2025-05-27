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
from tdmpc2.hybrid_tdmpc2 import HybridTDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


class HybridOnlineTrainer(OnlineTrainer):
    """
    Modified online trainer that supports hybrid MPC training.
    """
    
    def __init__(self, cfg, env, agent, buffer, logger):
        super().__init__(cfg, env, agent, buffer, logger)
        self.hybrid_agent = agent if isinstance(agent, HybridTDMPC2) else None
        
    def train(self):
        """Training loop with hybrid MPC support."""
        train_metrics, time_metrics = {}, {}
        ep_idx, ep_len, ep_return = 0, 0, 0
        obs = self.env.reset()
        
        for step in range(int(self.cfg.seed_steps + self.cfg.steps)):
            # Update hybrid MPC transition schedule
            if self.hybrid_agent is not None:
                self.hybrid_agent.update_training_step(step)
                
            # Collect experience
            if step < self.cfg.seed_steps:
                action = torch.randn(self.cfg.action_dim)
            else:
                action = self.agent.act(obs, t0=(ep_len == 0))
                
            obs, reward, done, info = self.env.step(action)
            ep_len += 1
            ep_return += reward
            self.buffer.add(obs, action, reward, done)
            
            if done or ep_len >= self.cfg.episode_length:
                # Log hybrid MPC info
                if self.hybrid_agent is not None:
                    hybrid_info = self.hybrid_agent.get_hybrid_info()
                    if hybrid_info["hybrid_enabled"]:
                        self.logger.log({
                            'classical_horizon': hybrid_info["classical_horizon"],
                            'training_step': step,
                        }, step, ty='train')
                        
                        # Print progress
                        if step % 10000 == 0:
                            print(f"Step {step}: Classical horizon = {hybrid_info['classical_horizon']}")
                
                # Episode finished
                self.logger.log({'episode_return': ep_return, 'episode_length': ep_len}, step, ty='train')
                obs = self.env.reset()
                ep_idx += 1
                ep_len, ep_return = 0, 0
                
            # Training
            if step >= self.cfg.seed_steps:
                if step == self.cfg.seed_steps:
                    num_updates = int(self.cfg.seed_steps)
                    print('Pretraining agent on seed data...')
                else:
                    num_updates = 1
                    
                for _ in range(num_updates):
                    train_metrics.update(self.agent.update(self.buffer))
                    
            # Evaluation
            if step > 0 and step % self.cfg.eval_freq == 0:
                eval_metrics = self.eval()
                
                # Log hybrid info during evaluation
                if self.hybrid_agent is not None:
                    hybrid_info = self.hybrid_agent.get_hybrid_info()
                    eval_metrics.update(hybrid_info)
                    
                self.logger.log(eval_metrics, step, ty='eval')
                
                # Save agent
                if self.cfg.save_agent:
                    self.agent.save(self.logger.log_dir / f'agent_{step}.pt')
                    
            # Logging
            if step > 0 and step % 1000 == 0:
                self.logger.log(train_metrics, step, ty='train')
                train_metrics = {}


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
    assert torch.cuda.is_available()
    assert cfg.steps > 0, 'Must train for at least 1 step.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
    
    # Create environment
    env = make_env(cfg)
    
    # Create hybrid agent
    agent = HybridTDMPC2(cfg, env)
    
    # Print hybrid MPC configuration
    if cfg.get('hybrid_mpc', False):
        print(colored('Hybrid MPC Configuration:', 'green', attrs=['bold']))
        print(f"  Max classical horizon: {cfg.get('max_classical_horizon', 10)}")
        print(f"  Min classical horizon: {cfg.get('min_classical_horizon', 0)}")
        print(f"  Transition steps: {cfg.get('transition_steps', 1_000_000)}")
        print(f"  Transition schedule: {cfg.get('transition_schedule', 'linear')}")
        print(f"  Classical algorithm: {cfg.get('classical_algorithm', 'ilqg')}")
    else:
        print(colored('Using standard TD-MPC2 (hybrid MPC disabled)', 'yellow'))
    
    # Select trainer
    trainer_cls = OfflineTrainer if cfg.multitask else HybridOnlineTrainer
    trainer = trainer_cls(
        cfg=cfg,
        env=env,
        agent=agent,
        buffer=Buffer(cfg),
        logger=Logger(cfg),
    )
    
    trainer.train()
    print('\nTraining completed successfully')


if __name__ == '__main__':
    train()
