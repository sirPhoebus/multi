from typing import List, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from concurrent.futures import ProcessPoolExecutor, as_completed
from marl_scientist.core import MetaEnvironment, ExperimentConfig, ExperimentResult, Observation
from marl_scientist.evaluation.sb3_runner import SB3ExperimentRunner
from marl_scientist.evaluation.metrics import NoveltyCalculator
from marl_scientist.evaluation.parallel_runner import run_experiment_task
from marl_scientist.utils.logger import setup_logger

class LabEnvironment(MetaEnvironment):
    """
    The 'Lab' where Researcher Agents submit their experiment configurations.
    """
    def __init__(self, authorized_benchmarks: List[str] = ["CartPole-v1"]):
        self.log = setup_logger()
        self.benchmarks = authorized_benchmarks
        # Cache metadata for benchmarks
        self.env_metadata = {b: self._get_env_metadata(b) for b in self.benchmarks}
        
        # We assume first benchmark for simple runner init
        self.runner = SB3ExperimentRunner(benchmark_env_id=self.benchmarks[0])
        self.novelty_calc = NoveltyCalculator()
        self.history: List[ExperimentResult] = []
        self.best_reward = -float('inf')
        self.best_config = None
        self.step_counter = 0
        
    def _get_env_metadata(self, env_id: str) -> Dict[str, Any]:
        """Extracts observation and action space info from gymnasium."""
        try:
            temp_env = gym.make(env_id)
            meta = {
                "env_id": env_id,
                "observation_space_type": str(type(temp_env.observation_space).__name__),
                "action_space_type": str(type(temp_env.action_space).__name__),
                "is_discrete": isinstance(temp_env.action_space, gym.spaces.Discrete),
                "is_continuous": isinstance(temp_env.action_space, gym.spaces.Box),
            }
            temp_env.close()
            return meta
        except Exception as e:
            return {"env_id": env_id, "error": str(e)}
        
    def step(self, actions: Dict[str, ExperimentConfig]) -> Tuple[Dict[str, ExperimentResult], Dict[str, float]]:
        """
        Runs one step of the meta-environment.
        Args:
            actions: Dictionary of experiments proposed by agents (agent_id -> ExperimentConfig).
        Returns:
            results: Dictionary of ExperimentResult objects (agent_id -> Result).
            rewards: Dictionary of float rewards (agent_id -> reward).
        """
        results_out = {}
        rewards = {}
        
        # Run Experiments in Parallel
        # We use a ProcessPoolExecutor to truly parallelize the gym loops
        
        results_map = {}
        futures = []
        
        # Determine max workers
        max_workers = min(len(actions), 8)
        
        self.log.info(f"[Lab] Submitting {len(actions)} experiments to pool (workers={max_workers})...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for agent_id, config in actions.items():
                self.log.info(f"[Lab] Scheduling {agent_id} on {config.env_id} ({config.algorithm})...")
                futures.append(executor.submit(run_experiment_task, config, agent_id))
            
            # Collect results
            self.log.info(f"[Lab] Waiting for completion...")
            for future in as_completed(futures):
                a_id, result, error = future.result()
                if result:
                    results_map[a_id] = result
                    self.log.info(f"[Lab] {a_id} finished: {result.final_mean_reward:.1f}")
                else:
                    self.log.error(f"[Lab] {a_id} FAILED: {error}")
                    # Handle failure gracefully?
                    pass

        # Post-Processing
        self.step_counter += 1
        
        # Adaptive Weights
        # Decay exploration weight from 0.4 to 0.1 over 20 steps
        # Increase performance weight from 0.6 to 0.9
        progress = min(1.0, self.step_counter / 20.0)
        w_novelty = 0.4 - (0.3 * progress)
        w_perf = 1.0 - w_novelty
        
        for agent_id, result in results_map.items():
            if not result: continue
            
            # 1. Base Performance (0.0 to 1.0)
            performance_score = min(result.final_mean_reward / 500.0, 1.0)
            if performance_score < 0: performance_score = 0.0 
            
            # 2. Stability Penalty
            # metrics['std_reward'] comes from SB3ExperimentRunner (if available)
            # Default to 0 if missing
            std_reward = result.metrics.get("std_reward", 0.0)
            # Normalize std: if std is 50 (10% of max), penalty is high.
            # Let's say we penalize 0.002 per unit of std
            stability_penalty = min(0.2, std_reward * 0.002)
            
            adjusted_perf = max(0.0, performance_score - stability_penalty)
            
            # 3. Novelty Bonus
            novelty_score = self.novelty_calc.calculate_novelty(result.config)
            
            # 4. Total Meta-Reward
            total_reward = (w_perf * adjusted_perf) + (w_novelty * novelty_score)
            
            rewards[agent_id] = total_reward
            self.history.append(result)
            results_out[agent_id] = result
            self.novelty_calc.add_to_history(result.config)
            
            # Check for best
            if result.final_mean_reward > self.best_reward:
                self.best_reward = result.final_mean_reward
                self.best_config = result.config
                print(f"!!! New Global Best found by {agent_id}: {self.best_reward:.1f} !!!")
                
        return results_out, rewards

    def get_observation(self) -> Observation:
        """Returns the global state (public knowledge)."""
        
        # 1. Performance Trends
        trends = {}
        if len(self.history) > 0:
            reward_vals = [r.final_mean_reward for r in self.history]
            trends["mean_reward_all"] = float(np.mean(reward_vals))
            trends["max_reward_all"] = float(np.max(reward_vals))
            
            # Recent trends (last 10)
            recent = reward_vals[-10:]
            trends["mean_reward_recent"] = float(np.mean(recent))
            
            if len(reward_vals) > 1:
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
            
        # 2. Novelty Landscape
        novelty_stats = {}
        novelty_stats["explored_ratio"] = min(1.0, len(self.history) / 500.0) 
        novelty_stats["archive_size"] = float(len(self.novelty_calc.history))
        
        # 3. Knowledge Summary
        know_summary = "Literature suggests that PPO with Clipping 0.2 is stable. Ent_coef > 0.01 aids exploration."

        return Observation(
            experiment_history=self.history,
            performance_trends=trends,
            novelty_landscape=novelty_stats,
            knowledge_summary=know_summary,
            env_metadata=self.env_metadata
        )
