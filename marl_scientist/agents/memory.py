import json
import os
import numpy as np
from typing import Dict, List, Any, Optional
from marl_scientist.core import ExperimentResult, ExperimentConfig

class EpisodicMemory:
    """
    Long-Term Episodic Memory.
    Stores the 'Best Known Configuration' for each environment.
    This allows agents to 'remember' what worked best in previous sessions
    or earlier in the current life.
    """
    def __init__(self, capacity_per_env: int = 5):
        self.capacity = capacity_per_env
        # Struct: env_id -> list of results (sorted by reward desc)
        self.memories: Dict[str, List[Dict]] = {}
        
    def add(self, result: ExperimentResult):
        """Add an experiment result to memory if it's good."""
        env_id = result.config.env_id
        if env_id not in self.memories:
            self.memories[env_id] = []
            
        # We store a lightweight dict instead of full object for easier serialization
        import time
        entry = {
            "algorithm": result.config.algorithm,
            "hyperparameters": result.config.hyperparameters,
            "reward": result.final_mean_reward,
            "timestamp": getattr(result, "start_time", time.time())
        }
        
        self.memories[env_id].append(entry)
        
        # Sort by reward descending (Best first)
        self.memories[env_id].sort(key=lambda x: x["reward"], reverse=True)
        
        # Keep top K
        if len(self.memories[env_id]) > self.capacity:
            self.memories[env_id] = self.memories[env_id][:self.capacity]
            
    def retrieve(self, env_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the best memory for a specific environment.
        Returns None if no memory exists.
        """
        if env_id in self.memories and len(self.memories[env_id]) > 0:
            return self.memories[env_id][0] # Return top 1
        return None
    
    def retrieve_vector(self, env_id: str) -> np.ndarray:
        """
        Returns a flat vector representation of the best config.
        Format matches the 6 HP values used in BrainEncoder output.
        Order: [lr, gamma, ent_coef, gae_lambda, param_4, param_5]
        """
        best = self.retrieve(env_id)
        if best is None:
            return np.zeros(6, dtype=np.float32)
            
        hps = best["hyperparameters"]
        
        # We need to map back to the 6 raw values roughly.
        # Note: Ideally we'd store the raw 'action' values, but we only have decoded HPs here.
        # We will just pass the raw HP values. The BrainEncoder will stick them into an MLP 
        # so exact scaling normalization is less critical than consistency.
        
        # Mapping:
        # 0: Learning Rate
        # 1: Gamma
        # 2: Ent Coef
        # 3: GAE Lambda
        # 4: Batch/Steps/Tau (Algo specific)
        # 5: Batch/Starts/TotalSteps (Algo specific)
        
        vec = np.zeros(6, dtype=np.float32)
        vec[0] = hps.get("learning_rate", 0.0)
        vec[1] = hps.get("gamma", 0.99)
        vec[2] = hps.get("ent_coef", 0.0)
        vec[3] = hps.get("gae_lambda", 0.95)
        
        # Algo specific slots for 4 and 5
        algo = best["algorithm"]
        if algo == "PPO":
            vec[4] = hps.get("n_steps", 0)
            vec[5] = hps.get("batch_size", 0)
        elif algo == "DQN":
            vec[4] = hps.get("batch_size", 0)
            vec[5] = hps.get("learning_starts", 0)
        elif algo == "SAC":
            vec[4] = hps.get("batch_size", 0)
            vec[5] = hps.get("tau", 0)
            
        return vec

    def save(self, path: str):
        """Save memory to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.memories, f, indent=2)
            
    def load(self, path: str):
        """Load memory from disk."""
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r') as f:
                self.memories = json.load(f)
        except Exception as e:
            print(f"[Memory] Failed to load {path}: {e}")
