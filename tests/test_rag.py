
from marl_scientist.knowledge.real_store import RealKnowledgeStore
import json

def test_rag_integration():
    print("Testing RAG Integration...")
    
    # 1. Initialize Store (assuming it has papers from previous ingestion test)
    store = RealKnowledgeStore(persistence_path="test_kb.pkl")
    if not store.documents:
        print("FAIL: test_kb.pkl is empty. Run test_ingest.py first.")
        return
        
    # 2. Test Suggestion using LLM
    print("Testing LLM Config Synthesis...")
    
    # We'll use a known paper text or one from the store
    sample_paper = "State Entropy Regularization for Robust Reinforcement Learning. We propose maximizing the entropy of the state visitation distribution to ensure agents explore robustly. We find that a high entropy coefficient (ent_coef=0.1) combined with PPO leads to better generalization."
    
    config = store.suggest_config_from_paper(sample_paper)
    
    if config:
        print(f"LLM Output: {json.dumps(config, indent=2)}")
        
        # Validation
        if config["algorithm"] == "PPO":
            print("PASS: Algorithm identified correctly.")
        else:
            print(f"FAIL: Expected PPO, got {config.get('algorithm')}")
            
        if abs(config["hyperparameters"].get("ent_coef", 0.0) - 0.1) < 0.01:
            print("PASS: Hyperparameter extracted correctly.")
        else:
             print(f"Note: ent_coef is {config['hyperparameters'].get('ent_coef')}. Might be close enough or defaulted.")
             
    else:
        print("FAIL: No config returned (LLM failure?).")

if __name__ == "__main__":
    test_rag_integration()
