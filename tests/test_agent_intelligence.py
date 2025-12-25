import unittest
from marl_scientist.core import Observation, ExperimentConfig
from marl_scientist.agents.researcher import ResearcherAgent

class TestAgentIntelligence(unittest.TestCase):
    def test_algo_selection_discrete(self):
        """Verify agent picks PPO/A2C/DQN for Discrete envs, not SAC."""
        agent = ResearcherAgent(agent_id="TestAgent")
        obs = Observation(
            experiment_history=[],
            performance_trends={},
            novelty_landscape={},
            knowledge_summary="",
            env_metadata={
                "CartPole-v1": {"is_discrete": True, "is_continuous": False}
            }
        )
        # Force a random config
        for _ in range(20):
            config = agent._random_config(obs)
            self.assertEqual(config.env_id, "CartPole-v1")
            self.assertIn(config.algorithm, ["PPO", "A2C", "DQN"])
            self.assertNotEqual(config.algorithm, "SAC")

    def test_algo_selection_continuous(self):
        """Verify agent picks SAC/PPO/A2C for Continuous envs, not DQN."""
        agent = ResearcherAgent(agent_id="TestAgent")
        obs = Observation(
            experiment_history=[],
            performance_trends={},
            novelty_landscape={},
            knowledge_summary="",
            env_metadata={
                "Pendulum-v1": {"is_discrete": False, "is_continuous": True}
            }
        )
        # Force a random config
        for _ in range(20):
            config = agent._random_config(obs)
            self.assertEqual(config.env_id, "Pendulum-v1")
            self.assertIn(config.algorithm, ["SAC", "PPO", "A2C"])
            self.assertNotEqual(config.algorithm, "DQN")

if __name__ == "__main__":
    unittest.main()
