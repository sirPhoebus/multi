
import sys
import os
import random
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marl_scientist.agents.researcher import ResearcherAgent
from marl_scientist.core import Observation, ExperimentConfig
from marl_scientist.env.lab_env import LabEnvironment

def main():
    print("Testing Neural Architecture Search (Self-Design)...")
    
    # Needs LLM server running, else falls back to DCC
    agent = ResearcherAgent("agent_arch_designer")
    agent.competence["vision"] = 0.9 # Force high competence to trigger design mode
    
    # Mock Observation
    obs = Observation(
        experiment_history=[],
        performance_trends={},
        novelty_landscape={},
        knowledge_summary="",
        env_metadata={"CIFAR100-Task-0": {"domain": "vision", "input_dim": 3072}}
    )
    
    # Force generate a proposal
    # We loop until we get a NAS one (since there is randomness)
    print("Agent thinking...")
    nas_config = None
    for i in range(10):
        # We manually call _select_target_domain internal logic simulation
        # The agent.propose_experiment calls _random_config or similar
        # Let's call the generator directly if possible, or just force propose
        # But propose_experiment is complex.
        # Let's use internal method directly for test
        try:
            nas_config = agent._generate_architecture_proposal("CIFAR100-Task-0", obs)
            if nas_config.algorithm == "NAS":
                break
        except Exception as e:
            print(f"Error: {e}")
            
    if not nas_config:
        print("Failed to generate NAS config.")
        return

    print(f"Generated Algorithm: {nas_config.algorithm}")
    code = nas_config.hyperparameters.get("model_code", "")
    print(f"Code Length: {len(code)} chars")
    if len(code) > 0:
        print("--- Code Snippet ---")
        print(code[:200] + "...")
        print("--------------------")
    
    # Now try to run it in Lab
    print("\nRunning in Lab...")
    lab = LabEnvironment(max_workers=1)
    lab.submit_experiment("arch_designer", nas_config)
    
    # Poll
    start = time.time()
    result_found = False
    r = None
    while time.time() - start < 60:
        res, rewards = lab.poll_results()
        if "arch_designer" in res:
            r = res["arch_designer"]
            if r:
                result_found = True
                print(f"Result: Reward={r.final_mean_reward:.1f}, Perf={r.performance_score:.2f}")
                if "error" in r.metrics:
                    print(f"Error Metric: {r.metrics['error']}")
                if "error" in r.info:
                    print(f"Error Info: {r.info['error']}")
            else:
                print("Result is None (Failed)")
            break
        time.sleep(1)
        
    lab.close()
    
    with open("test_nas_result.txt", "w") as f:
        if nas_config:
            f.write(f"Algorithm: {nas_config.algorithm}\n")
            f.write(f"Code Length: {len(code)}\n")
        else:
             f.write("Failed to generate NAS config.\n")

        if result_found:
             f.write(f"Reward: {r.final_mean_reward}\n")
             f.write(f"Performance: {r.performance_score}\n")
             f.write(f"Metrics: {r.metrics}\n")
             f.write(f"Info: {r.info}\n")
             f.write(f"CWD: {os.getcwd()}\n")
        else:
             f.write("No result returned from Lab.\n")

if __name__ == "__main__":
    main()
