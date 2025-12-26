
from typing import Dict, Any, Tuple
import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import time

from marl_scientist.core import ExperimentConfig, ExperimentResult

class SB3ExperimentRunner:
    """Runs a single RL experiment using Stable Baselines3."""
    
    def __init__(self, benchmark_env_id: str = "CartPole-v1"):
        self.env_id = benchmark_env_id
        
    def run(self, config: ExperimentConfig, visual: bool = False, agent_id: str = "Unknown") -> ExperimentResult:
        """
        Trains an agent according to config and returns the result.
        """
        try:
            # 1. Create Environment
            from stable_baselines3.common.monitor import Monitor
            try:
                env = gym.make(self.env_id)
                env = Monitor(env) 
                
                eval_render_mode = "rgb_array" if visual else None
                eval_env = gym.make(self.env_id, render_mode=eval_render_mode)
            except Exception as e:
                return ExperimentResult(
                    config=config,
                    final_mean_reward=-500.0,
                    training_curve=[],
                    metrics={"env_init_error": 1.0},
                    info={"error": str(e)}
                )
            
            # 2. Instantiate Algorithm
            algo_class = self._get_algo_class(config.algorithm)
            
            # Extract hyperparameters
            hp = config.hyperparameters.copy()
            
            # [FIX] Batch Size / Buffer / n_steps Validation
            # If batch_size is larger than rollout buffer (n_steps * n_envs), it fails.
            if "batch_size" in hp:
                 if config.algorithm in ["PPO", "A2C"]:
                     # PPO uses n_steps * n_envs as buffer size
                     n_steps = hp.get("n_steps", 2048) # Default PPO n_steps
                     buffer_size = n_steps # Assuming n_envs=1
                     
                     if hp["batch_size"] > buffer_size:
                         # Case 1: Batch too big -> reduce batch
                         hp["batch_size"] = buffer_size
                     else:
                         # Case 2: Batch must be a factor of n_steps (buffer_size)
                         # If not, we adjust n_steps to be a multiple
                         bs = hp["batch_size"]
                         remainder = n_steps % bs
                         if remainder != 0:
                             # Round up n_steps to nearest multiple of bs
                             # This ensures the buffer is large enough for the batch
                             new_n_steps = ((n_steps // bs) + 1) * bs
                             hp["n_steps"] = new_n_steps
                             # print(f"[SB3Runner] Adjusted n_steps {n_steps} -> {new_n_steps} for batch_size {bs}")
                         
                 elif config.algorithm in ["DQN", "SAC", "TD3"]:
                     # Off-policy uses explicit buffer_size
                     pass

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
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[{agent_id}] Training {config.algorithm} on {config.env_id} | Device: {device} | Batch: {hp.get('batch_size', 'N/A')} | Arch: {hp.get('net_arch', 'default')}")
            
            for key in ["learning_rate", "gamma", "seed", "verbose"]:
                if key in hp: algo_kwargs[key] = hp.pop(key)
            
            algo_kwargs["device"] = device
            
            # On-Policy Params (PPO, A2C)
            if config.algorithm in ["PPO", "A2C"]:
                for key in ["gae_lambda", "ent_coef", "vf_coef", "max_grad_norm", "n_steps", "batch_size"]:
                    if key in hp: algo_kwargs[key] = hp.pop(key)
                    
            # Off-Policy Params (DQN, SAC, TD3)
            if config.algorithm in ["DQN", "SAC", "TD3"]:
                for key in ["buffer_size", "learning_starts", "batch_size", "tau", "train_freq", "gradient_steps"]:
                    if key in hp: algo_kwargs[key] = hp.pop(key)

            # Specific Params
            if config.algorithm == "PPO":
                for key in ["n_epochs", "clip_range", "target_kl"]:
                    if key in hp: algo_kwargs[key] = hp.pop(key)
                    
            elif config.algorithm == "DQN":
                for key in ["target_update_interval", "exploration_fraction", "exploration_final_eps"]:
                    if key in hp: algo_kwargs[key] = hp.pop(key)
                
                # Guard: DQN only supports Discrete action spaces
                if not isinstance(env.action_space, gym.spaces.Discrete):
                     raise ValueError(f"DQN does not support action space {env.action_space}. Use PPO/A2C/SAC/TD3 for Continuous environments.")
            
            elif config.algorithm == "SAC":
                # Guard: SAC only supports Box (Continuous) action spaces
                if not isinstance(env.action_space, gym.spaces.Box):
                    raise ValueError(f"SAC does not support action space {env.action_space}. Use PPO/DQN/A2C for Discrete environments.")
                
                if "ent_coef" in hp: algo_kwargs["ent_coef"] = hp.pop("ent_coef")
            
            elif config.algorithm == "TD3":
                # TD3 Needs Action Noise
                if isinstance(env.action_space, gym.spaces.Box):
                    n_actions = env.action_space.shape[0]
                    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                    algo_kwargs["action_noise"] = action_noise
                else:
                    raise ValueError("TD3 requires a continuous action space (Box).")

            # [NEW] Architectural Search: Load Custom Policy if provided
            policy_type = "MlpPolicy"
            if "policy_code" in hp:
                custom_policy = self._load_custom_policy(hp.pop("policy_code"), hp)
                if custom_policy:
                    policy_type = custom_policy

            # 3. Instantiate Model
            model = algo_class(
                policy=policy_type,
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
                    self.eval_env = Monitor(eval_env) # Fix: Wrap eval env too
                    self.eval_freq = eval_freq
                    self.curve = []
                def _on_step(self) -> bool:
                    # eval_freq needs to be relative to num_timesteps
                    if self.num_timesteps % self.eval_freq == 0:
                        try:
                            m_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=5, warn=False)
                            self.curve.append(float(m_reward))
                        except Exception:
                            # Sometimes eval fails on Discrete vs Box mismatches if env not reset
                            pass 
                    return True

            class StagnationCallback(BaseCallback):
                def __init__(self, threshold=1.0, window=50000, check_freq=10000):
                    super().__init__(verbose=0)
                    self.threshold = threshold
                    self.window = window
                    self.check_freq = check_freq
                    self.history = [] # (timestep, mean_reward)
                    self.stopped_early = False
                
                def _on_step(self) -> bool:
                    if self.num_timesteps % self.check_freq == 0:
                        if len(self.model.ep_info_buffer) > 0:
                            current_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                            self.history.append((self.num_timesteps, current_reward))
                            
                            if self.num_timesteps >= self.window:
                                start_ts = self.num_timesteps - self.window
                                past_rewards = [r for ts, r in self.history if ts <= start_ts]
                                if past_rewards:
                                    improvement = current_reward - past_rewards[-1]
                                    if improvement < self.threshold:
                                        print(f"[SB3Runner] EARLY STOP: Timestep {self.num_timesteps}, Improvement {improvement:.2f} < {self.threshold}")
                                        self.stopped_early = True
                                        return False
                    return True
            
            eval_env = gym.make(self.env_id, render_mode="rgb_array" if visual else None)
            curve_callback = CurveCallback(eval_env, eval_freq=1000)
            
            # [PHASE 4] Early Stopping
            stagnation_thresh = config.hyperparameters.get("stagnation_threshold", 1.0)
            stagnation_window = config.hyperparameters.get("stagnation_window", 50000)
            stagnation_callback = StagnationCallback(threshold=stagnation_thresh, window=stagnation_window)
            
            from stable_baselines3.common.callbacks import CallbackList
            callbacks = CallbackList([curve_callback, stagnation_callback])
            
            total_timesteps = hp.get("total_timesteps", 10000)
            model.learn(total_timesteps=total_timesteps, callback=callbacks)
            
            # 5. Evaluate Final (Best-of-N)
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
            
            # [PHASE 4] Stagnation Penalty
            if stagnation_callback.stopped_early:
                mean_reward -= 10.0 # Small penalty for stagnation
                print(f"[{agent_id}] Applying stagnation penalty (-10.0). Final Adj Reward: {mean_reward:.2f}")
            
            training_curve = curve_callback.curve
            
            # Cleanup! CRITICAL to prevent resource leaks
            eval_env.close()
            env.close()
            
            metrics = {
                "std_reward": float(std_reward),
                "stability": 1.0 / (float(std_reward) + 1e-6),
            }
            
            # Optional Visual Snapshot
            visual_path = None
            if visual:
                try:
                    import os
                    os.makedirs("saves/snapshots", exist_ok=True)
                    frame = eval_env.render()
                    if frame is not None:
                        from PIL import Image
                        img = Image.fromarray(frame)
                        visual_path = f"saves/snapshots/{agent_id}_{config.env_id}_{int(time.time())}.png"
                        img.save(visual_path)
                except Exception as ve:
                    print(f"Failed to capture snapshot: {ve}")

            return ExperimentResult(
                config=config,
                final_mean_reward=mean_reward,
                training_curve=training_curve, 
                metrics=metrics,
                visual_snapshot=visual_path
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
        if algo_name == "TD3": return TD3
        raise ValueError(f"Unknown algorithm: {algo_name}")

    def _load_custom_policy(self, code: str, hp: Dict[str, Any]):
        """Dynamically loads a custom SB3 policy class from code."""
        from stable_baselines3.common.policies import BasePolicy
        import torch
        import torch.nn as nn
        
        local_scope = {}
        try:
            exec(code, {"torch": torch, "nn": nn, "BasePolicy": BasePolicy}, local_scope)
            for name, obj in local_scope.items():
                if isinstance(obj, type) and issubclass(obj, BasePolicy) and obj is not BasePolicy:
                    return obj
            print("[SB3Runner] No subclass of BasePolicy found in custom code.")
            return None
        except Exception as e:
            print(f"[SB3Runner] Error loading custom policy: {e}")
            return None

