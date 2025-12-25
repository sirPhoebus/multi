import numpy as np
from typing import List, Dict, Any
from marl_scientist.core import ExperimentConfig, ExperimentResult

class NoveltyCalculator:
    """Calculates how novel a proposed configuration is compared to history."""
    
    def __init__(self):
        self.history: List[ExperimentConfig] = []
        
    def calculate_novelty(self, config: ExperimentConfig) -> float:
        """
        Calculates novelty score (0.0 to 1.0).
        Simple implementation: Max cosine distance from existing configs in a feature space.
        """
        if not self.history:
            return 1.0
            
        # Simplified Feature extraction from config (e.g. just hyperparameters)
        current_features = self._extract_features(config)
        history_features = [self._extract_features(c) for c in self.history]
        
        # Calculate distances
        distances = []
        for h_f in history_features:
            dist = np.linalg.norm(np.array(current_features) - np.array(h_f))
            distances.append(dist)
            
        # Novelty is proportional to the minimum distance to any previous point
        # (If it's far from everything, it's novel)
        min_dist = min(distances) if distances else 1.0
        return min(min_dist, 1.0) # Cap at 1.0
        
    def add_to_history(self, config: ExperimentConfig):
        self.history.append(config)
        
    def _extract_features(self, config: ExperimentConfig) -> List[float]:
        # Quick hack: linearize known numeric params
        feats = []
        params = config.hyperparameters
        feats.append(float(params.get("learning_rate", 0.001)) * 1000) # Scale up
        feats.append(float(params.get("gamma", 0.99)) * 10)
        
        ent_coef = params.get("ent_coef", 0.0)
        if isinstance(ent_coef, (int, float)):
            feats.append(float(ent_coef) * 100)
        else:
            # Handle 'auto' or other strings
            feats.append(0.5) # Arbitrary fixed value for 'auto'
        return feats

class CausalGraph:
    """Stores assumed relationships between params and performance."""
    def __init__(self):
        self.edges = []
        
    def update(self, config: ExperimentConfig, result: ExperimentResult):
        # Stub: Record simple correlation
        pass
