import torch
from marl_scientist.agents.brain import MetaBrain, BrainEncoder
from marl_scientist.agents.trainer import MetaPPOTrainer
from marl_scientist.agents.neural_researcher import NeuralResearcherAgent
from marl_scientist.core import Observation, ExperimentConfig, ExperimentResult

def test_instantiation():
    print("Testing Brain...")
    brain = MetaBrain()
    print("Brain OK.")
    
    print("Testing Encoder...")
    encoder = BrainEncoder()
    # Create a mock observation
    obs = Observation(
        experiment_history=[],
        performance_trends={"mean_reward_all": 0.0},
        novelty_landscape={"explored_ratio": 0.0},
        knowledge_summary="Test"
    )
    inputs = encoder.encode_observation(obs, "test_agent")
    print("Encoder OK.")
    
    print("Testing Trainer...")
    trainer = MetaPPOTrainer(brain)
    print("Trainer OK.")
    
    print("Testing Neural Agent...")
    agent = NeuralResearcherAgent("test_agent")
    print("Agent OK.")

if __name__ == "__main__":
    try:
        test_instantiation()
        print("\nAll components instantiated successfully!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nTest FAILED: {e}")
