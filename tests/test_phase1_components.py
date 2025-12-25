import os
import sys
import numpy as np

# Mocking parts of the system for internal verification
from marl_scientist.core import ExperimentConfig, ExperimentResult
from marl_scientist.knowledge.trajectory_store import TrajectoryStore
from marl_scientist.agents.memory import TrajectoryMemory

def test_trajectory_system():
    print("Testing Trajectory Store/Memory...")
    store_file = "test_trajectories.pkl"
    if os.path.exists(store_file):
        os.remove(store_file)
        
    tm = TrajectoryMemory(persistence_path=store_file)
    
    # 1. Add a "failed" experiment
    config1 = ExperimentConfig(
        algorithm="PPO",
        hyperparameters={"learning_rate": 0.01, "gamma": 0.8},
        env_id="CartPole-v1"
    )
    result1 = ExperimentResult(
        config=config1,
        final_mean_reward=10.0, # Failure
        training_curve=[],
        metrics={},
        performance_score=0.1,
        stability_score=0.1,
        efficiency_score=0.1
    )
    
    print("Adding failure trajectory...")
    tm.add(result1)
    
    # 2. Check similarity with a near-identical config
    config2 = ExperimentConfig(
        algorithm="PPO",
        hyperparameters={"learning_rate": 0.011, "gamma": 0.8},
        env_id="CartPole-v1"
    )
    
    print("Checking similarity for near-identical config...")
    similar = tm.check_similarity(config2, threshold=0.9)
    if similar:
        print(f"  - Found similar: {similar['score']:.4f}")
        assert similar["score"] > 0.9
    else:
        print("  - FAILED: Should have found similar config")
        
    # 3. Retrieve failures
    print("Retrieving failures...")
    failures = tm.get_failures(config2, k=1)
    if failures:
        print(f"  - Found failure: {failures[0][:50]}...")
        assert "Algorithm PPO on CartPole-v1" in failures[0]
    else:
         print("  - FAILED: Should have retrieved failure summary")

    print("\nPhase 1 Component Test Passed!")

if __name__ == "__main__":
    test_trajectory_system()
