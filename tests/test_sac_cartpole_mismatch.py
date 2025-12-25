import unittest
import gymnasium as gym
from marl_scientist.core import ExperimentConfig
from marl_scientist.evaluation.sb3_runner import SB3ExperimentRunner

class TestSACMismatch(unittest.TestCase):
    def test_sac_on_discrete_penalty(self):
        """Confirm that SAC on Discrete env results in -500 reward penalty."""
        config = ExperimentConfig(
            algorithm="SAC",
            env_id="CartPole-v1", # DISCRETE
            hyperparameters={
                "learning_rate": 1e-3,
                "gamma": 0.99
            }
        )
        runner = SB3ExperimentRunner(benchmark_env_id="CartPole-v1")
        result = runner.run(config)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.final_mean_reward, -500.0)
        print(f"[Pass] Confirmed: SAC on Discrete env correctly triggers -500 penalty.")

if __name__ == "__main__":
    unittest.main()
