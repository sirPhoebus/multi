
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marl_scientist.env.lab_env import LabEnvironment
from marl_scientist.core import ExperimentConfig

def main():
    print("Initializing Multi-Domain Lab...")
    lab = LabEnvironment(max_workers=2)
    
    print(f"Allowed Envs: {lab.allowed_envs}")
    
    # Check if StringReverse is present
    if "StringReverse" not in lab.allowed_envs:
        print("ERROR: StringReverse not found in allowed envs!")
        print(f"Domains: {[d.__class__.__name__ for d in lab.domains]}")
        lab.close()
        return

    # 1. Correct Solution
    config_good = ExperimentConfig(
        algorithm="PythonScript",
        hyperparameters={"code": "def solution(x): return x[::-1]"},
        env_id="StringReverse",
        domain="coding"
    )
    
    # 2. Bad Solution
    config_bad = ExperimentConfig(
        algorithm="PythonScript",
        hyperparameters={"code": "def solution(x): return x"},
        env_id="StringReverse",
        domain="coding"
    )
    
    # 3. RL Solution (to ensure regression)
    config_rl = ExperimentConfig(
        algorithm="PPO",
        hyperparameters={
            "learning_rate": 3e-4, 
            "n_steps": 128, 
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "net_arch": [64, 64],
            "activation_fn": "Tanh"
        },
        env_id="CartPole-v1",
        domain="rl"
    )

    print("\nSubmitting experiments...")
    lab.submit_experiment("agent_coder_good", config_good)
    lab.submit_experiment("agent_coder_bad", config_bad)
    lab.submit_experiment("agent_rl", config_rl)
    
    # Poll for results
    results = {}
    start = time.time()
    while len(results) < 3 and time.time() - start < 30:
        res, rewards = lab.poll_results()
        if res:
            for aid, r in res.items():
                print(f"Got result for {aid}: Reward={r.final_mean_reward:.1f}, Perf={r.performance_score:.2f}")
                results[aid] = r
        time.sleep(1)
        
    lab.close()
    
    # Verification
    if "agent_coder_good" in results:
        g = results["agent_coder_good"]
        if g.performance_score >= 0.99:
            print("[PASS] Good Code Scored 1.0")
        else:
            print(f"[FAIL] Good Code Scored {g.performance_score}")
    else:
        print("[FAIL] Good Code Timed Out")

    if "agent_coder_bad" in results:
        b = results["agent_coder_bad"]
        if b.performance_score < 0.5: # Identity reverses palindromes correctly but fails others
            print("[PASS] Bad Code Scored low")
        else:
            print(f"[FAIL] Bad Code Scored {b.performance_score}")
            
    if "agent_rl" in results:
        print(f"[PASS] RL Agent Finished")
    else:
        print("[FAIL] RL Agent Timed Out")

if __name__ == "__main__":
    main()
