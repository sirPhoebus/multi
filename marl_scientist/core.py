import dataclasses
from typing import List, Dict, Any, Optional, Protocol, Tuple, Union
import numpy as np

@dataclasses.dataclass
class ExperimentConfig:
    """The 'Action' proposed by a Researcher Agent.
    
    Represents a configuration for an RL training run.
    """
    algorithm: str  # "PPO", "A2C", "DQN"
    hyperparameters: Dict[str, Union[float, int, str]]
    # Future: architecture_graph: Dict...

@dataclasses.dataclass
class ExperimentResult:
    """The result of running an experiment."""
    config: ExperimentConfig
    final_mean_reward: float
    training_curve: List[float]  # Reward over time
    metrics: Dict[str, float] # e.g., sample_efficiency, stability_score

@dataclasses.dataclass
class Observation:
    """What the Researcher Agent sees."""
    experiment_history: List[ExperimentResult]
    performance_trends: Dict[str, float]  # e.g., "improvement_rate", "stability"
    novelty_landscape: Dict[str, float]   # e.g., "unexplored_ratio", "cluster_density"
    knowledge_summary: str                # Text summary of related papers
    # In the future: current_causal_graph: Any
    # In the future: paper_embeddings: np.ndarray

class Researcher(Protocol):
    """Interface for a Researcher Agent."""
    
    def propose_experiment(self, observation: Observation) -> ExperimentConfig:
        ...
        
    def update_knowledge(self, result: ExperimentResult):
        ...

class MetaEnvironment(Protocol):
    """Interface for the 'Lab' Environment."""
    
    def step(self, configs: List[ExperimentConfig]) -> Tuple[List[ExperimentResult], Any]:
        ...
