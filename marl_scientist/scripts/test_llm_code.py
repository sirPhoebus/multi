
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marl_scientist.agents.researcher import ResearcherAgent
from marl_scientist.core import Observation
from marl_scientist.env.domains import CodingDomain

def main():
    print("Testing LLM Code Generation...")
    
    agent = ResearcherAgent("agent_coder_test")
    
    # Fake Observation with Metadata
    domain = CodingDomain()
    meta = domain.get_metadata("StringReverse")
    
    obs = Observation(
        experiment_history=[],
        performance_trends={},
        novelty_landscape={},
        knowledge_summary="",
        env_metadata={"StringReverse": meta}
    )
    
    # 1. Ask agent to generate a solution
    print(f"\nAsking Agent to solve: {meta['description']}")
    try:
        config = agent._generate_code_solution("StringReverse", obs)
        code = config.hyperparameters.get("code", "")
        print("\n--- Generated Code ---")
        print(code)
        print("----------------------")
        
        if "def solution" in code:
            print("[PASS] Method signature detected.")
        else:
            print("[FAIL] Method signature missing.")
            
    except Exception as e:
        print(f"[FAIL] Exception during generation: {e}")

if __name__ == "__main__":
    main()
