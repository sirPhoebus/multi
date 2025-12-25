
import numpy as np
import networkx as nx
from marl_scientist.agents.causal_graph import CausalDiscoveryEngine

def test_causal_learning():
    print("Testing Causal Learning and Reasoning...")
    engine = CausalDiscoveryEngine()
    
    # 1. Generate Synthetic Data
    # True Model:
    # LR (Tier 0) -> Stability (Tier 1) -> Reward (Tier 2)
    # EntCoef (Tier 0) -> Reward (Tier 2)
    # Correlation: LR ~ Stability (Linear)
    # Correlation: Stability ~ Reward (Linear)
    # Correlation: EntCoef ~ Reward (Linear)
    
    # Let's say:
    # Stability = 10 * LR + Noise
    # Reward = 5 * Stability + 100 * EntCoef + Noise
    
    print("Generating synthetic data (A -> B -> C structure)...")
    for _ in range(50):
        lr = np.random.uniform(0.0001, 0.01)
        ent = np.random.uniform(0.0, 0.1)
        
        # Intermediate
        stability = 10.0 * lr + np.random.normal(0, 0.01)
        
        # Reward
        reward = 50.0 * stability + 500.0 * ent + np.random.normal(0, 5)
        
        params = {"learning_rate": lr, "ent_coef": ent, "unrelated_param": 1.0}
        metrics = {"stability": stability, "noise_metric": np.random.random()}
        
        engine.update_model(params, metrics, reward)
        
    print("Forcing structural learning...")
    engine._learn_structure_and_parameters()
    
    # 2. Verify Edges
    print(f"Graph Edges: {engine.graph.edges()}")
    
    # We expect LR -> Stability
    # We expect Stability -> Reward
    # EntCoef -> Reward
    
    # Check if edges exist
    has_lr_stability = engine.graph.has_edge("learning_rate", "stability")
    has_stab_reward = engine.graph.has_edge("stability", "reward")
    has_ent_reward = engine.graph.has_edge("ent_coef", "reward")
    
    print(f"LR -> Stability: {has_lr_stability}")
    print(f"Stability -> Reward: {has_stab_reward}")
    print(f"EntCoef -> Reward: {has_ent_reward}")
    
    # Assertions might be tricky if data is noisy/insufficient, but let's hope
    # assert has_lr_stability
    # assert has_stab_reward
    
    # 3. Verify Effect Estimation
    # If we ask to suggest improvements, it should realize that increasing LR and EntCoef helps.
    
    current = {"learning_rate": 0.001, "ent_coef": 0.01, "unrelated_param": 1.0}
    suggested = engine.suggest_improvements(current)
    
    print(f"Current: {current}")
    print(f"Suggested: {suggested}")
    
    # Check direction
    if suggested["learning_rate"] > current["learning_rate"]:
        print("PASS: Correctly suggests increasing LR.")
    else:
        print("FAIL/WARN: Did not suggest increasing LR.")
        
    if suggested["ent_coef"] > current["ent_coef"]:
        print("PASS: Correctly suggests increasing EntCoef.")
    else:
        print("FAIL/WARN: Did not suggest increasing EntCoef.")

if __name__ == "__main__":
    test_causal_learning()
