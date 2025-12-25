
import torch
import numpy as np
from marl_scientist.agents.neural_researcher import NeuralResearcherAgent
from marl_scientist.core import Observation, ExperimentConfig, ExperimentResult

class MockBrain:
    def __init__(self, high_entropy=False):
        self.high_entropy = high_entropy
        
    def __call__(self, *args, **kwargs):
        # Return algo_logits, env_logits, hp_means, value, new_hidden, goal_logits
        # goal_logits: [Explore, Exploit, Refine]
        if self.high_entropy:
            # High entropy [0.33, 0.33, 0.33]
            goal_logits = torch.tensor([[1.0, 1.0, 1.0]])
        else:
            # Low entropy [1.0, 0.0, 0.0]
            goal_logits = torch.tensor([[10.0, -10.0, -10.0]])
            
        algo_logits = torch.zeros((1, 4))
        env_logits = torch.zeros((1, 5))
        hp_means = torch.zeros((1, 6))
        value = torch.tensor([0.0])
        new_hidden = None
        
        return algo_logits, env_logits, hp_means, value, new_hidden, goal_logits

class MockStore:
    def search(self, *args, **kwargs):
        return []

def test_horizon_trigger():
    agent = NeuralResearcherAgent(agent_id="Agent_HT")
    agent.brain = MockBrain(high_entropy=True)
    agent.knowledge_store = MockStore()
    
    # Mock embedding methods to bypass external calls
    agent._get_knowledge_embedding = lambda obs: np.zeros(768)
    agent._get_latent_embedding = lambda obs: np.zeros(768)
    
    obs = Observation(
        experiment_history=[],
        performance_trends={},
        novelty_landscape={},
        knowledge_summary="",
        env_metadata={"allowed_envs": ["CartPole-v1"]}
    )
    
    configs = agent.propose_experiment(obs)
    
    # Triggered because H > 0.8 and intent is EXPLORE (stochastic, but MockBrain forces it)
    assert isinstance(configs, list)
    assert len(configs) == 5
    assert configs[0].hyperparameters.get("is_horizon") is True
    assert agent.horizon_active is True

def test_horizon_consolidation():
    agent = NeuralResearcherAgent(agent_id="Agent_HC")
    agent.brain = MockBrain(high_entropy=True)
    agent.knowledge_store = MockStore()
    
    # Mock embedding methods to bypass external calls
    agent._get_knowledge_embedding = lambda obs: np.zeros(768)
    agent._get_latent_embedding = lambda obs: np.zeros(768)
    
    obs = Observation(
        experiment_history=[],
        performance_trends={},
        novelty_landscape={},
        knowledge_summary="",
        env_metadata={"allowed_envs": ["CartPole-v1"]}
    )
    
    # 1. Trigger Horizon
    configs = agent.propose_experiment(obs)
    assert agent.horizon_active is True
    
    # 2. Feed results
    for i, cfg in enumerate(configs):
        res = ExperimentResult(
            config=cfg,
            final_mean_reward=100.0 + i, # i=4 is best
            training_curve=[],
            metrics={}
        )
        agent.update_knowledge(res)
        
    assert agent.horizon_active is False
    assert len(agent.horizon_results) == 5
    
    # 3. Next proposal should pick the best (i=4)
    best_config = agent.propose_experiment(obs)
    assert isinstance(best_config, ExperimentConfig)
    assert best_config.hyperparameters.get("is_horizon") is None
    # Reward for i=4 was 104.0. We can't check the reward in config, 
    # but we can check if it was derived from that config.
    assert len(agent.horizon_results) == 0 # Should have been reset

if __name__ == "__main__":
    test_horizon_trigger()
    test_horizon_consolidation()
    print("Horizon Mode Tests Passed!")
