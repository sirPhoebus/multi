from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from marl_scientist.llm.client import LLMClient
from marl_scientist.core import ExperimentConfig, ExperimentResult

class TrajectoryStore:
    """
    Stores and retrieves experiment trajectories as vectors.
    Uses LLM embeddings for semantic search over experiment outcomes.
    """
    def __init__(self, persistence_path="trajectory_db.pkl"):
        self.client = LLMClient()
        self.persistence_path = persistence_path
        self.summaries: List[str] = []
        self.configs: List[Dict[str, Any]] = []
        self.outcomes: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.load()

    def add_trajectory(self, result: ExperimentResult):
        """
        Summarizes an experiment result, embeds it, and stores it.
        """
        # 1. Summarize
        summary = self._summarize_result(result)
        
        # 2. Embed
        try:
            vector = self.client.get_embedding(summary).reshape(1, -1)
        except Exception as e:
            print(f"[TrajectoryStore] Embedding failed: {e}")
            return

        # 3. Store
        self.summaries.append(summary)
        self.configs.append({
            "algorithm": result.config.algorithm,
            "hyperparameters": result.config.hyperparameters,
            "env_id": result.config.env_id
        })
        self.outcomes.append({
            "reward": result.final_mean_reward,
            "metrics": result.metrics,
            "success": result.final_mean_reward > 450.0 # Heuristic success
        })

        if self.embeddings is None:
            self.embeddings = vector
        else:
            self.embeddings = np.vstack([self.embeddings, vector])

        self.save()

    def search_similar(self, config: ExperimentConfig, k: int = 5, return_embeddings: bool = False) -> List[Dict[str, Any]]:
        """
        Finds the top k similar past trajectories.
        If return_embeddings is True, includes the raw embedding vectors.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
            
        try:
            # Get query embedding
            query_text = self._summarize_config(config)
            query_vec = self.client.get_embedding(query_text).reshape(1, -1)
            
            # Calculate cosine similarities
            stored_embeddings = self.embeddings 
            similarities = cosine_similarity(query_vec, stored_embeddings)[0]
            
            # Get top k
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                res = {
                    "summary": self.summaries[idx],
                    "config": self.configs[idx],
                    "outcome": self.outcomes[idx],
                    "score": float(similarities[idx])
                }
                if return_embeddings:
                    res["embedding"] = self.embeddings[idx]
                results.append(res)
                
            return results
        except Exception as e:
            print(f"[TrajectoryStore] Search failed: {e}")
            return []

    def _summarize_result(self, result: ExperimentResult) -> str:
        """
        Creates a text summary of the experiment for embedding.
        """
        hp_str = ", ".join([f"{k}={v}" for k, v in result.config.hyperparameters.items()])
        summary = (
            f"Algorithm {result.config.algorithm} on {result.config.env_id}. "
            f"Hyperparameters: {hp_str}. "
            f"Result: {result.final_mean_reward:.1f} mean reward. "
            f"Performance: {result.performance_score:.2f}, "
            f"Stability: {result.stability_score:.2f}, "
            f"Efficiency: {result.efficiency_score:.2f}."
        )
        return summary

    def _summarize_config(self, config: ExperimentConfig) -> str:
        """
        Summarizes a configuration for similarity search.
        """
        hp_str = ", ".join([f"{k}={v}" for k, v in config.hyperparameters.items()])
        return f"Experiment using {config.algorithm} on {config.env_id} with HPs: {hp_str}"

    def save(self):
        try:
            with open(self.persistence_path, 'wb') as f:
                pickle.dump({
                    "summaries": self.summaries,
                    "configs": self.configs,
                    "outcomes": self.outcomes,
                    "embeddings": self.embeddings
                }, f)
        except Exception as e:
            print(f"[TrajectoryStore] Save failed: {e}")

    def load(self):
        if not os.path.exists(self.persistence_path):
            return
        try:
            with open(self.persistence_path, 'rb') as f:
                data = pickle.load(f)
            self.summaries = data.get("summaries", [])
            self.configs = data.get("configs", [])
            self.outcomes = data.get("outcomes", [])
            self.embeddings = data.get("embeddings", None)
            print(f"[TrajectoryStore] Loaded {len(self.summaries)} trajectories.")
        except Exception as e:
            print(f"[TrajectoryStore] Load failed: {e}")
