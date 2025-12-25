import math
import time
from typing import List, Dict, Any, Tuple, Optional
from marl_scientist.core import ExperimentConfig

class SymbolicValidator:
    """
    Ensures experiment configurations follow deterministic safety and sanity rules.
    """
    def __init__(self, time_budget_per_experiment: float = 300.0):
        self.time_budget = time_budget_per_experiment # Max seconds allocated
        
        # Sane ranges for common RL hyperparameters
        self.hp_constraints = {
            "learning_rate": (1e-7, 0.1),
            "gamma": (0.5, 0.99999),
            "ent_coef": (0.0, 0.5),
            "batch_size": [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
            "n_steps": (1, 10000),
            "clip_range": (0.01, 0.5),
            "tau": (0.0001, 0.1)
        }

    def validate(self, config: ExperimentConfig) -> Tuple[bool, List[str]]:
        """
        Validates the configuration against symbolic rules.
        Returns (is_valid, list_of_violations).
        """
        violations = []
        
        hps = config.hyperparameters
        
        # 1. Check HP Ranges
        for key, value in hps.items():
            if key in self.hp_constraints:
                constraint = self.hp_constraints[key]
                if isinstance(constraint, tuple):
                    low, high = constraint
                    if not (low <= float(value) <= high):
                        violations.append(f"Hyperparameter '{key}' value {value} out of range [{low}, {high}].")
                elif isinstance(constraint, list):
                    if value not in constraint:
                        violations.append(f"Hyperparameter '{key}' value {value} must be one of {constraint}.")

        # 2. Constraint: Batch size must be power of 2 (SB3 requirement for some algos)
        if "batch_size" in hps:
            try:
                bs = int(hps["batch_size"])
                if not (bs > 0 and (bs & (bs - 1) == 0)):
                     if bs not in self.hp_constraints["batch_size"]:
                        violations.append(f"Batch size {bs} is not a power of 2.")
            except (ValueError, TypeError):
                violations.append(f"Invalid batch_size type: {type(hps['batch_size'])}")

        # 3. Environment Specific Rules
        if config.env_id == "CartPole-v1" and hps.get("learning_rate", 0) > 0.01:
             violations.append("Learning rate too high for CartPole instability.")

        # 4. Resource Estimation
        valid_res, res_msg = self.check_resource_budget(config)
        if not valid_res:
            violations.append(res_msg)

        return len(violations) == 0, violations

    def check_resource_budget(self, config: ExperimentConfig) -> Tuple[bool, str]:
        """
        Estimates the wall-clock time and rejects if it exceeds the budget.
        """
        steps = config.hyperparameters.get("total_timesteps", 50000)
        env_id = config.env_id
        
        # Estimated steps per second (very rough heuristic)
        # CartPole: ~2000 steps/sec
        # LunarLander: ~500 steps/sec
        # Hopper/Walker: ~100 steps/sec
        sps_map = {
            "CartPole-v1": 2000,
            "Pendulum-v1": 1500,
            "Acrobot-v1": 1500,
            "LunarLander-v3": 500,
            "MountainCarContinuous-v0": 800,
            "Hopper-v4": 200,
            "Walker2d-v4": 150,
            "HalfCheetah-v4": 100
        }
        
        sps = sps_map.get(env_id, 300)
        estimated_time = steps / sps
        
        # Bias for algorithm complexity
        if config.algorithm == "SAC":
            estimated_time *= 1.5
        elif config.algorithm == "PPO":
            estimated_time *= 1.2
            
        if estimated_time > self.time_budget:
            return False, f"Estimated time {estimated_time:.1f}s exceeds budget {self.time_budget}s (Steps: {steps})."
            
        return True, ""
