from marl_scientist.evaluation.sb3_runner import SB3ExperimentRunner
from marl_scientist.core import ExperimentConfig
import gymnasium as gym
import numpy as np

def test_overrides():
    print("Test 1: Check TD3 instantiation")
    runner = SB3ExperimentRunner(benchmark_env_id="MountainCarContinuous-v0")
    
    cfg = ExperimentConfig(
        algorithm="TD3",
        env_id="MountainCarContinuous-v0",
        hyperparameters={
            "total_timesteps": 1000,
            "learning_starts": 100
        }
    )
    
    # We set visual=False to avoid render bugs
    try:
        res = runner.run(cfg, visual=False, agent_id="TestBot")
        if res.metrics.get("error"):
            print("FAILURE:", res.metrics["error"])
        else:
            print("SUCCESS: TD3 ran for 1000 steps.")
            print(f"Reward: {res.final_mean_reward}")
            
    except Exception as e:
        print(f"CRASH: {e}")

if __name__ == "__main__":
    test_overrides()
