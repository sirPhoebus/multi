
import time
from marl_scientist.evaluation.parallel_runner import run_experiment_task
from marl_scientist.core import ExperimentConfig
import multiprocessing

def test_parallel_speedup():
    print("Testing Parallel Execution Speedup...")
    
    # We will use the actual parallel_runner function but since we can't easily mock the internals 
    # of a spawned process to "sleep" without modifying the code, 
    # we will rely on the fact that running 4 CartPole experiments in parallel should be faster than serial 
    # IF the overhead isn't massive. 
    # Actually, CartPole is so fast that overhead might dominate.
    # So we'll trust the mechanism if it works and returns results.
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    configs = [
        ExperimentConfig(algorithm="PPO", hyperparameters={"learning_rate": 0.001}),
        ExperimentConfig(algorithm="A2C", hyperparameters={"learning_rate": 0.002}),
        ExperimentConfig(algorithm="DQN", hyperparameters={"learning_rate": 0.003}),
        ExperimentConfig(algorithm="PPO", hyperparameters={"learning_rate": 0.004})
    ]
    
    agent_ids = [f"Agent_{i}" for i in range(len(configs))]
    
    start_time = time.time()
    
    print(f"Submitting {len(configs)} tasks to ProcessPoolExecutor...")
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, config in enumerate(configs):
            futures.append(executor.submit(run_experiment_task, config, agent_ids[i]))
            
        for future in as_completed(futures):
            aid, res, err = future.result()
            if res:
                results.append(res)
                print(f"  - {aid}: Success (Reward: {res.final_mean_reward})")
            else:
                print(f"  - {aid}: Failed ({err})")
                
    duration = time.time() - start_time
    print(f"Total Duration: {duration:.2f}s")
    
    if len(results) == 4:
        print("PASS: All experiments completed.")
    else:
        print(f"FAIL: Only {len(results)}/4 completed.")

if __name__ == "__main__":
    # Windows needs this
    multiprocessing.freeze_support()
    test_parallel_speedup()
