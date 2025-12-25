import torch
import numpy as np
from marl_scientist.agents.brain import MetaBrain, BrainEncoder
from marl_scientist.core import Observation, ExperimentResult, ExperimentConfig

def test_brain_latent_split():
    print("Testing MetaBrain with Latent Split...")
    
    latent_dim = 768
    brain = MetaBrain(latent_dim=latent_dim)
    encoder = BrainEncoder(history_len=5)
    
    # Create a mock observation
    obs = Observation(
        experiment_history=[],
        env_metadata={"allowed_envs": ["CartPole-v1"]},
        performance_trends={},
        novelty_landscape={},
        knowledge_summary="No research found."
    )
    
    # 1. Test encoding with and without latent vec
    print("Encoding observation...")
    inputs = encoder.encode_observation(obs, "Agent_Test", knowledge_vec=None, latent_vec=None)
    assert "latent" in inputs
    assert inputs["latent"].shape == (1, latent_dim)
    
    latent_vec = np.random.rand(latent_dim).astype(np.float32)
    inputs_with_latent = encoder.encode_observation(obs, "Agent_Test", knowledge_vec=None, latent_vec=latent_vec)
    assert torch.allclose(inputs_with_latent["latent"], torch.from_numpy(latent_vec).unsqueeze(0))

    # 2. Test forward pass
    print("Running forward pass...")
    history = inputs["history"]
    trends = inputs["trends"]
    knowledge = inputs["knowledge"]
    latent_context = inputs["latent"]
    
    algo_logits, env_logits, hp_means, value, new_hidden, goal_logits = brain(
        history, trends, knowledge, latent_context=latent_context
    )
    
    print(f"Forward pass successful. Algo logits shape: {algo_logits.shape}")
    assert algo_logits.shape == (1, 4)
    assert hp_means.shape == (1, 6)

    print("\nMetaBrain Latent Split Test Passed!")

if __name__ == "__main__":
    test_brain_latent_split()
