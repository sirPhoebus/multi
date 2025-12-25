import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, List, Optional

from marl_scientist.core import ExperimentConfig, ExperimentResult
from marl_scientist.env.lab_env import LabEnvironment

class ResearcherEnv(gym.Env):
    """
    A Gym environment that wraps the Lab.
    The 'Agent' in this env is the Researcher.
    Action: Selecting hyperparameters for the next experiment.
    Observation: History of past performance.
    Reward: The score obtained by the proposed experiment.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, lab: LabEnvironment, history_len: int = 5):
        super(ResearcherEnv, self).__init__()
        self.lab = lab
        self.history_len = history_len
        self.history: List[ExperimentResult] = []
        
        # Action Space: MultiDiscrete
        # 0: Algorithm (0=PPO, 1=A2C, 2=DQN)
        # 1: Learning Rate (Log bins: 0..10 => 1e-5 to 1e-2)
        # 2: Gamma (0=0.9, 1=0.95, 2=0.98, 3=0.99, 4=0.995, 5=0.999)
        # 3: Net Arch (0=Small, 1=Medium, 2=Large)
        self.action_space = spaces.MultiDiscrete([3, 11, 6, 3])
        
        # Observation Space: Flattened history
        # For each past step: [Algo_ID, LR_Val, Gamma_Val, Reward]
        # Shape: (history_len * 4, )
        self.observation_space = spaces.Box(
            low=-1.0, high=1000.0, shape=(self.history_len * 4,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.history = []
        return self._get_obs(), {}

    def step(self, action):
        # 1. Decode Action to Config
        config = self._action_to_config(action)
        
        # 2. Run in Lab
        # Note: Lab expects a list of configs (Multi-Agent), we pass one.
        # We assume single-agent training for now.
        results, rewards = self.lab.step([config])
        result = results[0]
        reward = rewards[0]
        
        # 3. Update State
        self.history.append(result)
        if len(self.history) > self.history_len:
            self.history.pop(0)
            
        # 4. Return
        # We treat each step as a new 'experiment'.
        # Episode ends after N steps (handled by TimeLimit wrapper usually) or if we want to reset periodically.
        done = False
        truncated = False
        info = {
            "config": config,
            "raw_reward": result.final_mean_reward
        }
        
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        obs = np.zeros(self.history_len * 4, dtype=np.float32)
        
        for i, res in enumerate(self.history):
            # Encode Algo
            algo_map = {"PPO": 0, "A2C": 1, "DQN": 2}
            algo_idx = algo_map.get(res.config.algorithm, 0)
            
            # Encode Params
            lr = res.config.hyperparameters.get("learning_rate", 0.0)
            gamma = res.config.hyperparameters.get("gamma", 0.0)
            
            # Encode Reward
            rew = res.final_mean_reward
            
            idx = i * 4
            obs[idx] = float(algo_idx)
            obs[idx+1] = float(lr) * 1000.0 # Scale up
            obs[idx+2] = float(gamma)
            obs[idx+3] = float(rew) / 500.0 # Normalize roughly
            
        return obs

    def _action_to_config(self, action) -> ExperimentConfig:
        algo_idx, lr_idx, gamma_idx, net_idx = action
        
        # Algorithms
        algos = ["PPO", "A2C", "DQN"]
        chosen_algo = algos[algo_idx]
        
        # Hyperparameters decoding
        # LR: Log scale from 1e-5 to 1e-2
        lr_log = np.linspace(-5, -2, 11)
        lr_val = 10 ** lr_log[lr_idx]
        
        # Gamma
        gammas = [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
        gamma_val = gammas[gamma_idx]
        
        # Net Arch (Simplified)
        # None passed to SB3 defaults, but we can adding mapping later
        # 0: [64,64], 1: [128,128], 2: [256,256]
        # Currently SB3 runner needs update to handle this.
        
        return ExperimentConfig(
            algorithm=chosen_algo,
            hyperparameters={
                "learning_rate": float(lr_val),
                "gamma": float(gamma_val),
                # Add defaults for other required params
                 "ent_coef": 0.01 
            }
        )
