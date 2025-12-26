import dataclasses
from typing import List, Dict, Any, Optional, Protocol, Tuple, Union
import numpy as np

@dataclasses.dataclass
class ExperimentConfig:
    """The 'Action' proposed by a Researcher Agent.
    
    Represents a configuration for an RL training run.
    """
    algorithm: str  # "PPO", "A2C", "DQN", "LLM-Zero-Shot"
    hyperparameters: Dict[str, Union[float, int, str]]
    env_id: str = "CartPole-v1" # Default env or Task ID
    domain: str = "rl" # "rl", "coding", "reasoning", "vision"
    # Future: architecture_graph: Dict...

@dataclasses.dataclass
class ExperimentResult:
    """The result of running an experiment."""
    config: ExperimentConfig
    final_mean_reward: float
    training_curve: List[float]  # Reward over time
    metrics: Dict[str, float] # e.g., sample_efficiency, stability_score
    
    # [NEW] Temporal Metrics
    duration_seconds: float = 0.0
    start_time: float = 0.0
    total_env_steps: int = 0
    
    # [NEW] Multi-Task Scores (0 to 1)
    performance_score: float = 0.0
    efficiency_score: float = 0.0
    stability_score: float = 0.0
    
    visual_snapshot: Optional[str] = None # Path to image
    final_model_path: Optional[str] = None
    info: Dict[str, Any] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class Observation:
    """What the Researcher Agent sees."""
    experiment_history: List[ExperimentResult]
    performance_trends: Dict[str, float]  # e.g., "improvement_rate", "stability"
    novelty_landscape: Dict[str, float]   # e.g., "unexplored_ratio", "cluster_density"
    knowledge_summary: str                # Text summary of related papers
    env_metadata: Dict[str, Any] = dataclasses.field(default_factory=dict) # Metadata for each benchmark
    # In the future: current_causal_graph: Any
    # In the future: paper_embeddings: np.ndarray

class Researcher(Protocol):
    """Interface for a Researcher Agent."""
    
    def propose_experiment(self, observation: Observation) -> Union[ExperimentConfig, List[ExperimentConfig]]:
        # Existing neural proposal logic...
        base_config = self.neural_brain(observation)  # your RNN/LSTM output

        # Trigger slow reasoning on high uncertainty or stagnation
        if observation.novelty_landscape.get("unexplored_ratio", 0.0) > 0.7 or self.competence_low():
            hypothesis = self._generate_hypothesis(observation)
            configs = self._hypothesis_to_configs(hypothesis, observation)
            return configs  # Could be multiple from one hypothesis

        return base_config

    def _generate_hypothesis(self, observation: Observation) -> str:
        prompt = f"""
            You are an RL researcher analyzing past experiments.

            Recent trends: {observation.performance_trends}
            Knowledge summary: {observation.knowledge_summary[:2000]}  # truncate for token limit

            Propose a novel, testable hypothesis to improve performance.
            Focus on hyperparameters, architecture changes, or domain transfer.
            Output only the hypothesis as natural language.
            """
        return LLMClient.chat(prompt)

    def _hypothesis_to_configs(self, hypothesis: str, observation: Observation) -> List[ExperimentConfig]:
        prompt = f"""
            Translate this hypothesis into 1-3 concrete ExperimentConfig JSON objects.

            Hypothesis: {hypothesis}

            Available domains: rl, coding, vision
            Use valid JSON format with keys from ExperimentConfig.

            Example:
            [{"algorithm": "PPO", "hyperparameters": {"learning_rate": 0.0003}, "env_id": "LunarLander-v3", "domain": "rl"}]
            """
        response = LLMClient.chat(prompt)
        try:
            configs = json.loads(response)
            return [ExperimentConfig(**c) for c in configs]
        except:
            # Fallback to base
            return [self._default_config()]    
    def update_knowledge(self, result: ExperimentResult):
        ...

class MetaEnvironment(Protocol):
    """Interface for the 'Lab' Environment."""
    
    def step(self, configs: List[ExperimentConfig]) -> Tuple[List[ExperimentResult], Any]:
        ...
