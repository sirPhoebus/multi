import asyncio
import numpy as np
import time
from marl_scientist.evaluation.sb3_runner import SB3ExperimentRunner
from marl_scientist.core import ExperimentConfig

async def test_gpu_load():
    print("--- [Test] GPU Load & Scalability ---")
    
    # 1. High-load Config
    config = ExperimentConfig(
        algorithm="SAC", 
        env_id="MountainCarContinuous-v0", 
        hyperparameters={
            "learning_rate": 0.0003,
            "batch_size": 1024, # Large batch
            "net_arch": [256, 256], # Complex model
            "total_timesteps": 50000 
        }
    )
    
    runner = SB3ExperimentRunner(benchmark_env_id="MountainCarContinuous-v0")
    
    print("[Test] Starting high-load run on GPU...")
    start_t = time.time()
    result = runner.run(config, visual=False, agent_id="GPU_Test_Agent")
    end_t = time.time()
    
    print(f"[Test] Completed in {end_t - start_t:.1f}s")
    print(f"[Test] Final Reward: {result.final_mean_reward}")
    
    if result.final_mean_reward != -500.0:
        print("[Test] SUCCESS: Experiment completed without crash.")
    else:
        print("[Test] FAILED: Experiment crashed.")

if __name__ == "__main__":
    asyncio.run(test_gpu_load())
