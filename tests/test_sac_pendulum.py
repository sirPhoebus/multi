import unittest
import gymnasium as gym
from marl_scientist.core import ExperimentConfig
from marl_scientist.evaluation.sb3_runner import SB3ExperimentRunner

class TestSACPendulum(unittest.TestCase):
    def test_sac_execution(self):
        """Test that SAC runs correctly on a continuous environment (Pendulum-v1)."""
        config = ExperimentConfig(
            algorithm="SAC",
            env_id="Pendulum-v1",
            hyperparameters={
                "learning_rate": 1e-3,
                "gamma": 0.99,
                "buffer_size": 1000, # Small for testing
                "learning_starts": 100,
                "batch_size": 64
            }
        )
        runner = SB3ExperimentRunner(benchmark_env_id="Pendulum-v1")
        result = runner.run(config)
        
        self.assertIsNotNone(result)
        # Pendulum rewards are negative (max is 0). We just want it not to crash and collect curve
        self.assertGreater(len(result.training_curve), 0)
        print(f"[Pass] SAC on Pendulum Reward: {result.final_mean_reward}")

if __name__ == "__main__":
    unittest.main()
