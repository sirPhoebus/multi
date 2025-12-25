import asyncio
import numpy as np
import os
import shutil
from marl_scientist.knowledge.trajectory_store import TrajectoryStore
from marl_scientist.agents.memory import TrajectoryMemory
from marl_scientist.core import ExperimentConfig, ExperimentResult
from marl_scientist.agents.neural_researcher import NeuralResearcherAgent

async def test_d3_phase3():
    print("--- [Test] D3 Engine Phase 3: Active/Latent Split ---")
    
    test_db = "test_trajectories.pkl"
    if os.path.exists(test_db):
        os.remove(test_db)
        
    # 1. Initialize Memory
    memory = TrajectoryMemory(persistence_path=test_db)
    
    # 2. Add some "latant" knowledge (past experiments)
    print("[Test] Adding sample trajectories to Latent Store...")
    
    # Success on CartPole
    res1 = ExperimentResult(
        config=ExperimentConfig(algorithm="PPO", env_id="CartPole-v1", hyperparameters={"learning_rate": 0.001}),
        final_mean_reward=500.0,
        training_curve=[10, 50, 200, 500],
        metrics={},
        performance_score=1.0,
        stability_score=1.0,
        efficiency_score=1.0
    )
    await memory.add(res1)
    
    # Failure on CartPole
    res2 = ExperimentResult(
        config=ExperimentConfig(algorithm="DQN", env_id="CartPole-v1", hyperparameters={"learning_rate": 0.1}),
        final_mean_reward=10.0,
        training_curve=[5, 10, 10],
        metrics={},
        performance_score=0.1,
        stability_score=0.1,
        efficiency_score=0.1
    )
    await memory.add(res2)
    
    print(f"[Test] Stored {len(memory.store.summaries)} trajectories.")
    
    # 3. Retrieve Context
    print("[Test] Retrieving Latent Context for CartPole-v1...")
    query_config = ExperimentConfig(algorithm="PPO", env_id="CartPole-v1", hyperparameters={})
    latent_vec = await memory.retrieve_context(query_config, k=2)
    
    print(f"[Test] Latent Vector Shape: {latent_vec.shape}")
    assert latent_vec.shape == (768,), f"Latent vector shape mismatch: {latent_vec.shape}"
    assert not np.all(latent_vec == 0), "Latent vector should not be all zeros"
    
    # 4. Integrate with Agent
    print("[Test] Initializing Agent with Latent Memory...")
    
    class MockKB:
        async def search(self, query, k=1):
            return [{"embedding": np.zeros(768, dtype=np.float32)}]
            
    agent = NeuralResearcherAgent("TestAgent", trajectory_memory=memory, knowledge_store=MockKB())
    
    # Mock Observation
    from marl_scientist.core import Observation
    obs = Observation(
        experiment_history=[res1], # Active Memory (recent)
        performance_trends={"mean_reward_all": 250},
        novelty_landscape={"explored_ratio": 0.2},
        knowledge_summary="Sample knowledge summary for testing",
        env_metadata={}
    )
    
    print("[Test] Proposing experiment (should use both Active and Latent memory)...")
    config, train_data = await agent.propose_experiment(obs)
    
    print(f"[Test] Proposed: {config.algorithm} on {config.env_id}")
    assert config is not None
    
    # 5. Check failures retrieval
    print("[Test] Checking failure retrieval...")
    failures = await memory.get_failures(query_config, k=1)
    print(f"[Test] Similar failures found: {failures}")
    assert len(failures) > 0, "Should have found at least one failure trajectory"
    
    print("\n[Test] D3 Engine Phase 3: PASSED.")
    
    # Cleanup
    if os.path.exists(test_db):
        os.remove(test_db)
    if os.path.exists("kb_cache.pkl"): # Created by LLMClient
        os.remove("kb_cache.pkl")

if __name__ == "__main__":
    asyncio.run(test_d3_phase3())
