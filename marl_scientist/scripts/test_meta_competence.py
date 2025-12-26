
import sys
import os
import random
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marl_scientist.agents.researcher import ResearcherAgent
from marl_scientist.core import Observation, ExperimentResult, ExperimentConfig
from marl_scientist.env.domains import CodingDomain, RLDomain

def main():
    print("Testing Meta-Brain Competence Tracking...")
    
    agent = ResearcherAgent("agent_meta_test")
    
    # Mock Environment
    rl_domain = RLDomain()
    code_domain = CodingDomain()
    
    env_meta = {}
    for t in rl_domain.get_tasks(tier=0):
        env_meta[t] = rl_domain.get_metadata(t)
    for t in code_domain.get_tasks(tier=0):
        env_meta[t] = code_domain.get_metadata(t)
        
    obs = Observation(
        experiment_history=[],
        performance_trends={},
        novelty_landscape={},
        knowledge_summary="",
        env_metadata=env_meta
    )
    
    print(f"Initial Competence: {agent.competence}")
    
    # Run loop
    for i in range(20):
        config = agent.propose_experiment(obs)
        domain = config.domain
        
        # Simulate Result
        # Let's say Agent is a Genius at Coding (1.0) but bad at RL (0.2)
        score = 0.0
        if domain == "coding":
            score = 0.95 + random.uniform(-0.05, 0.05)
        else:
            score = 0.2 + random.uniform(-0.1, 0.1)
            
        result = ExperimentResult(
            config=config,
            final_mean_reward=0.0, # Irrelevant for update logic which uses performance_score
            training_curve=[],
            metrics={},
            performance_score=score
        )
        
        agent.update_knowledge(result)
        
        if i % 5 == 0:
            print(f"Items: {i} | Domain Chosen: {domain} | Result: {score:.2f}")
            print(f"  Competence: RL={agent.competence['rl']:.2f}, Coding={agent.competence['coding']:.2f}")
            
    print("\nFinal Competence (Should be high for Coding, low for RL):")
    print(agent.competence)
    
    if agent.competence["coding"] > agent.competence["rl"] + 0.3:
        print("[PASS] Agent correctly learned it is better at Coding.")
    else:
        print("[FAIL] Agent failed to differentiate skills significantly.")

if __name__ == "__main__":
    main()
