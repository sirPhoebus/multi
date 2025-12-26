
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marl_scientist.env.lab_env import LabEnvironment
from marl_scientist.core import ExperimentConfig

def main():
    print("Initializing Multi-Domain Lab (with Vision)...")
    lab = LabEnvironment(max_workers=1)
    
    print(f"Allowed Envs: {lab.allowed_envs[:5]} ...")
    
    # Check if CIFAR is present
    vision_task = "CIFAR100-Task-0"
    if vision_task not in lab.allowed_envs:
        print(f"ERROR: {vision_task} not found in allowed envs!")
        print(f"Domains: {[d.__class__.__name__ for d in lab.domains]}")
        lab.close()
        return

    print("Submitting Vision Experiment...")
    
    config_vis = ExperimentConfig(
        algorithm="DCC",
        hyperparameters={
            "fast_lr": 1e-3, 
            "epochs": 1, # Fast test
            "input_dim": 3072,
            "hidden_dim": 128 # Smaller for test
        },
        env_id=vision_task,
        domain="vision"
    )

    lab.submit_experiment("agent_vision_1", config_vis)
    
    # Poll for results
    results = {}
    start = time.time()
    while len(results) < 1 and time.time() - start < 45:
        try:
            res, rewards = lab.poll_results()
            if res:
                for aid, r in res.items():
                    print(f"\nGot result for {aid}: Reward={r.final_mean_reward:.1f}, Perf={r.performance_score:.2f}")
                    if "model_path" in r.metrics:
                        print(f"Model saved to: {r.metrics['model_path']}")
                    if "error" in r.info:
                        print(f"Error Info: {r.info['error']}")
                    results[aid] = r
            time.sleep(1)
        except Exception as e:
            print(f"Polling loop error: {e}")
            break
        
    lab.close()
    
    if "agent_vision_1" in results:
        r = results["agent_vision_1"]
        if r.performance_score >= 0.0:
            print("[PASS] Vision Experiment Finished Successfully (even if accuracy is low/random).")
        else:
             print("[FAIL] Vision Experiment Failed logic.")
    else:
        print("[FAIL] Vision Agent Timed Out")

if __name__ == "__main__":
    main()
