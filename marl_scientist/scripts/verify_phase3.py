import torch
import numpy as np
from marl_scientist.agents.memory import TrajectoryMemory
from marl_scientist.core import ExperimentConfig, ExperimentResult
from marl_scientist.agents.brain import MetaBrain, BrainEncoder

def verify_phase3():
    print("=== Phase 3 Verification: Trajectory Embeddings ===")
    
    # 1. Setup Memory
    # Use a temporary path for verification
    test_path = "test_trajectories.pkl"
    if os.path.exists(test_path): os.remove(test_path)
    
    memory = TrajectoryMemory(persistence_path=test_path)
    
    # 2. Add Dummy Trajectories
    print("Adding dummy trajectories...")
    env_ids = ["CartPole-v1", "LunarLander-v3"]
    for env in env_ids:
        for lr in [1e-3, 1e-4]:
            config = ExperimentConfig(env_id=env, algorithm="PPO", hyperparameters={"learning_rate": lr})
            result = ExperimentResult(
                config=config, 
                final_mean_reward=100.0,
                training_curve=[10.0, 50.0, 100.0],
                metrics={"sample_efficiency": 0.5}
            ) # Dummy reward
            # Manually add scores to satisfy summarize_result
            result.performance_score = 1.0
            result.stability_score = 1.0
            result.efficiency_score = 1.0
            memory.add(result)
            
    print(f"Store size: {len(memory.store.summaries)}")
    
    # 3. Test Retrieval
    print("\nTesting Retrieval...")
    query_config = ExperimentConfig(env_id="CartPole-v1", algorithm="PPO", hyperparameters={})
    context_vec = memory.retrieve_context(query_config, k=2)
    
    print(f"Context Vector Shape: {context_vec.shape}")
    print(f"Context Vector Norm: {np.linalg.norm(context_vec):.4f}")
    
    assert context_vec.shape == (768,), f"Expected (768,), got {context_vec.shape}"
    assert np.linalg.norm(context_vec) > 0, "Context vector should be non-zero"
    
    # 4. Test Brain Integration
    print("\nTesting Brain Integration...")
    brain = MetaBrain()
    encoder = BrainEncoder()
    
    # Dummy observation
    from marl_scientist.core import Observation
    obs = Observation(
        experiment_history=[],
        performance_trends={},
        novelty_landscape={},
        knowledge_summary=""
    )
    obs.env_metadata["allowed_envs"] = ["CartPole-v1"]
    
    inputs = encoder.encode_observation(obs, "TestAgent", latent_vec=context_vec)
    
    # Forward pass
    with torch.no_grad():
        algo_logits, env_logits, hp_means, value, hidden, goal_logits = brain(
            inputs["history"],
            inputs["trends"],
            inputs["knowledge"],
            inputs["latent"]
        )
        
    print("Brain Forward Pass: SUCCESS")
    print(f"Latent Projection Sample: {hidden[0, :5].numpy()}")
    
    # Cleanup
    if os.path.exists(test_path): os.remove(test_path)
    print("\n=== Phase 3 Verification: PASSED ===")

if __name__ == "__main__":
    import os
    try:
        verify_phase3()
    except Exception as e:
        print(f"Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
