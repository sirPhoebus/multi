import os
import shutil
import torch
from marl_scientist.knowledge.real_store import RealKnowledgeStore
from marl_scientist.agents.neural_researcher import NeuralResearcherAgent
from marl_scientist.env.lab_env import LabEnvironment
from marl_scientist.core import Observation

def main():
    print("=== Short End-to-End Neural Test ===")
    
    # 1. Setup Dummy Knowledge
    test_dir = "test_knowledge_dir"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    with open(os.path.join(test_dir, "ppo_expert_opinion.txt"), "w") as f:
        f.write("I believe PPO works best with high entropy coefficients (around 0.05) on complex control tasks.")
    
    with open(os.path.join(test_dir, "dqn_efficiency.txt"), "w") as f:
        f.write("DQN requires a large buffer size for LunarLander, at least 100,000 steps.")
    
    # 2. Ingest Knowledge
    print("\n[Step 1] Ingesting Folder...")
    kb = RealKnowledgeStore(persistence_path="test_neural_kb.pkl")
    
    # CHECK LLM CONNECTION
    if not kb.client.check_connection():
        print("[!] LLM Server not detected. Mocking embeddings for this test...")
        # Monkey patch the client to return random vectors instead of failing
        import numpy as np
        kb.client.get_embedding = lambda x: np.random.randn(768).astype(np.float32)
        kb.client.get_embeddings_batch = lambda texts: np.random.randn(len(texts), 768).astype(np.float32)

    kb.ingest_folder(test_dir)
    print(f"Knowledge Base populated with {len(kb.documents)} documents.")
    
    # 3. Initialize Agent
    print("\n[Step 2] Initializing Neural Agent...")
    agent = NeuralResearcherAgent(agent_id="NeuralTestBot")
    agent.set_knowledge_store(kb)
    
    # 4. Propose Experiment
    print("\n[Step 3] Requesting Proposal...")
    # Mock observation
    lab = LabEnvironment()
    obs = lab.get_observation()
    
    config = agent.propose_experiment(obs)
    print(f"\n[SUCCESS] Agent Proposed: {config.algorithm} on {config.env_id}")
    print(f"Hyperparameters: {config.hyperparameters}")
    
    # 5. Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    if os.path.exists("test_neural_kb.pkl"):
        os.remove("test_neural_kb.pkl")
        
    print("\n=== Test Finished ===")

if __name__ == "__main__":
    main()
