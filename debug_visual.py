
import gymnasium as gym
from marl_scientist.evaluation.sb3_runner import SB3ExperimentRunner
from marl_scientist.core import ExperimentConfig
import torch

def test():
    config = ExperimentConfig(
        algorithm="PPO",
        hyperparameters={"learning_rate": 0.0003, "total_timesteps": 1000},
        env_id="CartPole-v1"
    )
    runner = SB3ExperimentRunner(benchmark_env_id="CartPole-v1")
    print("Running experiment with visual=True...")
    result = runner.run(config, visual=True, agent_id="TestAgent")
    print(f"Result: {result.final_mean_reward}")
    print(f"Snapshot path: {result.visual_snapshot}")

if __name__ == "__main__":
    test()
