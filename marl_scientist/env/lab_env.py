from typing import List, Tuple, Dict, Any
import numpy as np
from marl_scientist.core import MetaEnvironment, ExperimentConfig, ExperimentResult, Observation
from marl_scientist.evaluation.sb3_runner import SB3ExperimentRunner
from marl_scientist.evaluation.metrics import NoveltyCalculator

class LabEnvironment(MetaEnvironment):
    """
    The 'Lab' where Researcher Agents submit their experiment configurations.
    """
    def __init__(self, authorized_benchmarks: List[str] = ["CartPole-v1"]):
        self.benchmarks = authorized_benchmarks
        self.runner = SB3ExperimentRunner(benchmark_env_id=self.benchmarks[0])
        self.novelty_calc = NoveltyCalculator()
        self.history: List[ExperimentResult] = []
        
    def step(self, configs: List[ExperimentConfig]) -> Tuple[List[ExperimentResult], List[float]]:
        """
        Runs one step of the meta-environment.
        Args:
            configs: List of experiments proposed by agents (one per agent).
        Returns:
            results: List of ExperimentResult objects.
            rewards: List of float rewards for each agent.
        """
        results = []
        rewards = []
        
        for config in configs:
            # 1. Run the experiment (Inner Loop)
            # In a real distributed system, this would be async/parallel
            result = self.runner.run(config)
            
            # 2. Calculate Reward
            # Base reward: Normalized performance (clipped)
            # Cartpole max is 500. Normalize to 0-1.
            performance_score = min(result.final_mean_reward / 500.0, 1.0)
            if performance_score < 0: performance_score = 0.0 # Handle crash penalty
            
            # Novelty bonus
            novelty_score = self.novelty_calc.calculate_novelty(config)
            
            # Total Reward
            # Weighted sum: mostly performance, some novelty to drive exploration
            total_reward = (0.8 * performance_score) + (0.2 * novelty_score)
            
            results.append(result)
            rewards.append(total_reward)
            
            # 3. Update State
            self.history.append(result)
            self.novelty_calc.add_to_history(config)
            
        return results, rewards

    def get_observation(self) -> Observation:
        """Returns the global state (public knowledge)."""
        
        # 1. Performance Trends
        trends = {}
        if len(self.history) > 0:
            rewards = [r.final_mean_reward for r in self.history]
            trends["mean_reward_all"] = float(np.mean(rewards))
            trends["max_reward_all"] = float(np.max(rewards))
            
            # Recent trends (last 10 vs last 50)
            recent = rewards[-10:]
            trends["mean_reward_recent"] = float(np.mean(recent))
            
            if len(rewards) > 1:
                # Simple derivative: slope of linear fit to last 10 points
                y = np.array(recent)
                x = np.arange(len(y))
                if len(y) > 1:
                    slope, _ = np.polyfit(x, y, 1)
                    trends["improvement_rate"] = float(slope)
                else:
                    trends["improvement_rate"] = 0.0
                    
                # Stability (Inverse variance)
                std = np.std(recent)
                trends["stability"] = 1.0 / (std + 1e-6)
            else:
                trends["improvement_rate"] = 0.0
                trends["stability"] = 0.0
        else:
            trends = {"mean_reward_all": 0.0, "improvement_rate": 0.0, "stability": 0.0}
            
        # 2. Novelty Landscape (Placeholder / Simple Heuristic)
        # In a full version, we'd query the NoveltyCalculator for coverage metrics
        novelty_stats = {}
        novelty_stats["explored_ratio"] = min(1.0, len(self.history) / 500.0) # Dummy density
        novelty_stats["archive_size"] = float(len(self.novelty_calc.history))
        
        # 3. Knowledge Summary (Placeholder)
        # This would come from the Knowledge Base (RAG)
        know_summary = "Literature suggests that PPO with Clipping 0.2 is stable. Ent_coef > 0.01 aids exploration."

        return Observation(
            experiment_history=self.history,
            performance_trends=trends,
            novelty_landscape=novelty_stats,
            knowledge_summary=know_summary
        )
