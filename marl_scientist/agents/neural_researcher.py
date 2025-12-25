import torch
import numpy as np
from typing import Dict, Any, Optional, List
from marl_scientist.core import Researcher, Observation, ExperimentConfig, ExperimentResult
from marl_scientist.agents.brain import MetaBrain, BrainEncoder
from marl_scientist.agents.memory import EpisodicMemory

class NeuralResearcherAgent(Researcher):
    """
    An AI Scientist that uses a Neural Network 'Brain' to decide experiments.
    Learns how to do research over time.
    """
    def __init__(self, agent_id: str, model_path: Optional[str] = None):
        self.agent_id = agent_id
        self.encoder = BrainEncoder()
        self.memory = EpisodicMemory()
        
        # Initialize Brain
        self.brain = MetaBrain()
        if model_path:
            try:
                self.brain.load_state_dict(torch.load(model_path))
                print(f"[NeuralResearcher] Loaded brain from {model_path}")
            except:
                print(f"[NeuralResearcher] Failed to load brain, using random init.")
        
        self.brain.eval() # Deployment mode by default
        
        # Recurrent state
        self.hidden_state = None
        self.knowledge_store = None
        
        self.best_performance = -float('inf')
        
        # [NEW] Self-Modification Sandbox
        from marl_scientist.sandbox.sandbox import CodeSandbox
        self.sandbox = CodeSandbox(sandbox_dir=f"sandbox_{agent_id}")

    def attempt_self_modification(self, module_name: str, new_code: str, test_script: str) -> bool:
        """
        Attempts to rewrite a part of the agent's code.
        1. Validates syntax.
        2. Runs unit tests in sandbox.
        3. If passed, applies the patch to the actual file.
        """
        import os
        
        # Security: Only allow modifying specific files in a specific directory
        # For MVP, let's say we can only modify a "strategy.py" or similar helper
        # But for this prototype, we'll allow modifying 'proposer_utils.py' (fictional) or itself 'neural_researcher.py' (DANGEROUS but fun)
        
        # Check syntax
        valid, msg = self.sandbox.validate_syntax(new_code)
        if not valid:
            print(f"[Self-Mod] Syntax Error: {msg}")
            return False
            
        # Test in sandbox
        passed, out = self.sandbox.run_unit_test(new_code, test_script)
        if not passed:
            print(f"[Self-Mod] Test Failed: {out}")
            return False
            
        # Apply Patch
        # Resolving path relative to this agent's codebase
        # WARNING: In a real deployment, this path should be very carefully controlled
        target_path = os.path.abspath(f"marl_scientist/agents/{module_name}")
        
        if not os.path.exists(target_path):
             print(f"[Self-Mod] Target {target_path} does not exist.")
             return False
             
        print(f"[Self-Mod] SUCCESS! Applying patch to {module_name}...")
        return self.sandbox.apply_patch(target_path, new_code)
        
    def set_knowledge_store(self, store):
        self.knowledge_store = store

    def propose_experiment(self, observation: Observation) -> ExperimentConfig:
        """
        Queries the Brain to get a new experiment configuration.
        """
        # 1. Literature Review for Embeddings
        # We query the knowledge store for relevant papers to get context
        knowledge_vec = self._get_knowledge_embedding(observation)
        
        # 2. Encode Observation
        inputs = self.encoder.encode_observation(observation, self.agent_id, knowledge_vec)
        
        # 3. Brain Inference
        with torch.no_grad():
            if "allowed_envs" in observation.env_metadata:
                allowed_names = observation.env_metadata["allowed_envs"]
            else:
                allowed_names = [] 

            # [NEW] Memory Retrieval (Context Injection)
            target_env_context = self.encoder.ENVS[0]
            if allowed_names:
                target_env_context = allowed_names[-1] 
                
            mem_vec = self.memory.retrieve_vector(target_env_context)
            mem_tensor = torch.tensor(mem_vec, dtype=torch.float32).unsqueeze(0) # [1, 6]
            
            # Forward pass with Context
            algo_logits, env_logits, hp_means, value, new_hidden = self.brain(
                inputs["history"],
                inputs["trends"],
                inputs["knowledge"],
                mem_tensor, 
                self.hidden_state
            )
            
            # Curriculum Masking
            if allowed_names:
                mask = torch.full_like(env_logits, -float('inf'))
                for name in allowed_names:
                    if name in self.encoder.ENVS:
                        idx = self.encoder.ENVS.index(name)
                        mask[0, idx] = 0.0 
                env_logits = env_logits + mask
            
            self.hidden_state = new_hidden
            
            # 1. Select Environment First
            env_idx = torch.argmax(env_logits, dim=-1).item()
            selected_env = self.encoder.ENVS[env_idx]
            
            # 2. Compatibility Masking
            discrete_envs = ["CartPole-v1", "LunarLander-v3", "Acrobot-v1"]
            continuous_envs = ["Pendulum-v1", "MountainCarContinuous-v0"]
            
            algo_mask = torch.zeros_like(algo_logits)
            
            if selected_env in discrete_envs:
                if "SAC" in self.encoder.ALGOS:
                    sac_idx = self.encoder.ALGOS.index("SAC")
                    algo_mask[0, sac_idx] = -float('inf')
                    
            elif selected_env in continuous_envs:
                if "DQN" in self.encoder.ALGOS:
                    dqn_idx = self.encoder.ALGOS.index("DQN")
                    algo_mask[0, dqn_idx] = -float('inf')
            
            algo_logits = algo_logits + algo_mask
            
            # 3. Select Algorithm
            algo_idx = torch.argmax(algo_logits, dim=-1).item()
            hp_vals = hp_means.squeeze(0).cpu().numpy()
            
        # 4. Decode to Config
        config_dict = self.encoder.decode_action(algo_idx, env_idx, hp_vals)
        
        return ExperimentConfig(
            algorithm=config_dict["algorithm"],
            hyperparameters=config_dict["hyperparameters"],
            env_id=config_dict["env_id"]
        )

    def update_knowledge(self, result: ExperimentResult):
        """
        Neural agent doesn't do manual causal updates, 
        it learns via the PPO meta-training loop.
        However, it can still publish results to the journal.
        """
        self.best_performance = max(self.best_performance, result.final_mean_reward)
        
        # [NEW] Episodic Memory Update
        self.memory.add(result)
        
        if self.knowledge_store and result.final_mean_reward > 400.0:
            paper = self.knowledge_store.synthesize_new_paper(result, self.agent_id)
            self.knowledge_store.add_paper(paper)

    def _get_knowledge_embedding(self, observation: Observation) -> np.ndarray:
        """
        Fetches the top paper embedding from the knowledge store.
        """
        if not self.knowledge_store:
            return np.zeros(768)
            
        # Formulate a simple query based on current trends
        query = "State of the art reinforcement learning hyperparameters stability"
        results = self.knowledge_store.search(query, k=1)
        
        if results:
            # We need to get the actual embedding. 
            # In KnowledgeShard/RealStore, the search returns metadata but not the vector.
            # However, we can re-embed the text or the store could return it.
            # For now, let's assume the store has a way to get the embedding of the hit.
            # RealKnowledgeStore has self.embeddings.
            
            # Since we are using LLMClient, we can just re-embed the top hit's text 
            # (cached in LLM if needed, otherwise small cost)
            hit_text = results[0]["text"]
            try:
                # Accessing LLMClient via parent_store or shard
                client = getattr(self.knowledge_store, "client", None)
                if client:
                    return client.get_embedding(hit_text)
            except:
                pass
        
        return np.zeros(768)

    def save(self, filepath: str):
        """Save the brain and metadata."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Use brain save helper but also could save other metadata
        torch.save({
            "agent_id": self.agent_id,
            "brain_state": self.brain.state_dict(),
        }, filepath)
        
        # [NEW] Save Memory
        mem_path = filepath.replace(".pkl", "_memory.json")
        self.memory.save(mem_path)
        
        print(f"[NeuralResearcher {self.agent_id}] State saved to {filepath}")

    def load(self, filepath: str):
        """Load the brain and metadata."""
        import os
        if not os.path.exists(filepath):
            return
        try:
            # Fix: PyTorch 2.4+ defaults weights_only=True which breaks dict loading unless whitelisted
            # We use weights_only=False for local dev convenience, assuming trusted source
            checkpoint = torch.load(filepath, weights_only=False)
            self.agent_id = checkpoint.get("agent_id", self.agent_id)
            self.brain.load_state_dict(checkpoint["brain_state"])
            
            # [NEW] Load Memory
            mem_path = filepath.replace(".pkl", "_memory.json")
            if os.path.exists(mem_path):
                self.memory.load(mem_path)
                
            print(f"[NeuralResearcher {self.agent_id}] State loaded from {filepath}")
        except Exception as e:
            print(f"[NeuralResearcher {self.agent_id}] Load failed: {e}")

    def save_brain(self, path: str):
        torch.save(self.brain.state_dict(), path)
        print(f"[NeuralResearcher] Brain saved to {path}")
