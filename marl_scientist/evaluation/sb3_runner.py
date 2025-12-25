from typing import Dict, Any, Tuple
import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

from marl_scientist.core import ExperimentConfig, ExperimentResult

class SB3ExperimentRunner:
    """Runs a single RL experiment using Stable Baselines3."""
    
    def __init__(self, benchmark_env_id: str = "CartPole-v1"):
        self.env_id = benchmark_env_id
        
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Trains an agent according to config and returns the result.
        """
        try:
            # 1. Create Environment
            env = gym.make(self.env_id)
            
            # 2. Instantiate Algorithm
            algo_class = self._get_algo_class(config.algorithm)
            
            # Extract hyperparameters
            hp = config.hyperparameters.copy()
            
            # Parse Policy Kwargs (Architecture & Activation)
            policy_kwargs = {}
            if "net_arch" in hp:
                policy_kwargs["net_arch"] = hp.pop("net_arch")
                
            if "activation_fn" in hp:
                act_fn_name = hp.pop("activation_fn")
                if act_fn_name == "ReLU": policy_kwargs["activation_fn"] = nn.ReLU
                elif act_fn_name == "Tanh": policy_kwargs["activation_fn"] = nn.Tanh
                elif act_fn_name == "ELU": policy_kwargs["activation_fn"] = nn.ELU
                elif act_fn_name == "LeakyReLU": policy_kwargs["activation_fn"] = nn.LeakyReLU
            
            # Common SB3 Params that might need filtering per algo
            # We construct a dictionary of valid args
            
            algo_kwargs = {}
            
            # Universal Params (most algos support these)
            for key in ["learning_rate", "gamma", "seed", "device", "verbose"]:
                if key in hp: algo_kwargs[key] = hp.pop(key)
            
            # On-Policy Params (PPO, A2C)
            if config.algorithm in ["PPO", "A2C"]:
                for key in ["gae_lambda", "ent_coef", "vf_coef", "max_grad_norm", "n_steps"]:
                    if key in hp: algo_kwargs[key] = hp.pop(key)
                    
            # Off-Policy Params (DQN, SAC)
            if config.algorithm in ["DQN", "SAC"]:
                for key in ["buffer_size", "learning_starts", "batch_size", "tau", "train_freq", "gradient_steps"]:
                    if key in hp: algo_kwargs[key] = hp.pop(key)

            # Specific Params
            if config.algorithm == "PPO":
                for key in ["n_epochs", "clip_range", "target_kl"]:
                    if key in hp: algo_kwargs[key] = hp.pop(key)
                    
            elif config.algorithm == "DQN":
                for key in ["target_update_interval", "exploration_fraction", "exploration_final_eps"]:
                    if key in hp: algo_kwargs[key] = hp.pop(key)
            
            elif config.algorithm == "SAC":
                # Guard: SAC only supports Box (Continuous) action spaces
                import gymnasium
                if not isinstance(env.action_space, gymnasium.spaces.Box):
                    raise ValueError(f"SAC does not support action space {env.action_space}. Use PPO/DQN/A2C for Discrete environments.")
                if "ent_coef" in hp: algo_kwargs["ent_coef"] = hp.pop("ent_coef")
                
            # 3. Instantiate Model
            model = algo_class(
                policy="MlpPolicy",
                env=env,
                verbose=0,
                policy_kwargs=policy_kwargs if policy_kwargs else None,
                **algo_kwargs
            )
            
            # 4. Train
            # Use a simple callback to populate training curve every 1000 steps
            from stable_baselines3.common.callbacks import BaseCallback
            
            class CurveCallback(BaseCallback):
                def __init__(self, eval_env, eval_freq=1000):
                    super().__init__(verbose=0)
                    self.eval_env = eval_env
                    self.eval_freq = eval_freq
                    self.curve = []
                def _on_step(self) -> bool:
                    # eval_freq needs to be relative to num_timesteps
                    if self.num_timesteps % self.eval_freq == 0:
                        m_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=5)
                        self.curve.append(float(m_reward))
                    return True
            
            eval_env = gym.make(self.env_id)
            curve_callback = CurveCallback(eval_env, eval_freq=1000)
            
            total_timesteps = 10000
            model.learn(total_timesteps=total_timesteps, callback=curve_callback)
            
            # 5. Evaluate Final
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
            
            training_curve = curve_callback.curve
            
            # Cleanup! CRITICAL to prevent resource leaks
            eval_env.close()
            env.close()
            
            metrics = {
                "std_reward": float(std_reward),
                "stability": 1.0 / (float(std_reward) + 1e-6),
            }
            
            return ExperimentResult(
                config=config,
                final_mean_reward=mean_reward,
                training_curve=training_curve, 
                metrics=metrics
            )
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            # Handle crash (e.g., unstable params)
            print(f"Experiment failed ({config.algorithm}): {e}\n{tb}")
            
            # Attempt cleanup if envs exist
            try:
                if 'eval_env' in locals(): eval_env.close()
                if 'env' in locals(): env.close()
            except:
                pass
                
            return ExperimentResult(
                config=config,
                final_mean_reward=-500.0, # Penalize crash
                training_curve=[],
                metrics={"error": 1.0}
            )
            
    def _get_algo_class(self, algo_name: str):
        if algo_name == "PPO": return PPO
        if algo_name == "A2C": return A2C
        if algo_name == "DQN": return DQN
        if algo_name == "SAC": return SAC
        raise ValueError(f"Unknown algorithm: {algo_name}")
