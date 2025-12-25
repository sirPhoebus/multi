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
        
        # [NEW] Curriculum Tiers
        self.tier = 0
        self.tiers = {
            0: ["CartPole-v1", "Pendulum-v1"],
            1: ["LunarLander-v3", "Acrobot-v1"],
            2: ["MountainCarContinuous-v0"]
        }
        # Threshold to UNLOCK next tier
        self.tier_thresholds = {
            0: 450.0, # CartPole/Pendulum solved
            1: 200.0, # LunarLander solved
            2: float('inf')
        }
    
    @property
    def allowed_envs(self) -> List[str]:
        # Return all envs up to current tier
        envs = []
        for t in range(self.tier + 1):
            envs.extend(self.tiers.get(t, []))
        return envs
        
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
        import time
        results_out = {}
        rewards = {}
        
        # Run Experiments in Parallel
        futures_map = {}
        
        # Determine max workers
        max_workers = min(len(actions), 8)
        
        self.log.info(f"[Lab] Submitting {len(actions)} experiments to pool (workers={max_workers})...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for agent_id, config in actions.items():
                if config is None: continue
                self.log.info(f"[Lab] Scheduling {agent_id} on {config.env_id} ({config.algorithm})...")
                fut = executor.submit(run_experiment_task, config, agent_id)
                futures_map[fut] = (agent_id, time.time())
            
            # Collect results
            self.log.info(f"[Lab] Waiting for completion with 60s timeout...")
            try:
                for future in as_completed(futures_map, timeout=60):
                    a_id, t_start = futures_map[future]
                    try:
                        a_id_returned, result, error = future.result()
                        t_end = time.time()
                        
                        if result:
                            # Success
                            result.duration_seconds = t_end - t_start
                            result.total_env_steps = result.config.hyperparameters.get("n_timesteps", 0)
                            
                            self.log.info(f"[Lab] {a_id_returned} finished: {result.final_mean_reward:.1f} (Time: {result.duration_seconds:.1f}s)")
                            
                            # Post-Processing for meta-reward calculation
                            self.step_counter += 1
                            
                            # Environment-Specific Normalization Ranges
                            norms = {
                                "CartPole-v1": {"min": 0, "max": 500},
                                "Acrobot-v1": {"min": -500, "max": -100},
                                "Pendulum-v1": {"min": -2000, "max": -150},
                                "LunarLander-v3": {"min": -500, "max": 200},
                            }
                            
                            # 1. Base Performance
                            env_id = result.config.env_id
                            spec = norms.get(env_id, {"min": -1000, "max": 0}) 
                            
                            raw_reward = result.final_mean_reward
                            performance_score = (raw_reward - spec["min"]) / (spec["max"] - spec["min"])
                            performance_score = max(0.0, min(1.0, performance_score))
                            
                            # 2. Stability Penalty
                            std_reward = result.metrics.get("std_reward", 0.0)
                            stability_penalty = min(0.2, std_reward * 0.002)
                            
                            adjusted_perf = max(0.0, performance_score - stability_penalty)
                            
                            # 3. Novelty Bonus
                            novelty_score = self.novelty_calc.calculate_novelty(result.config)
                            
                            # Adaptive Weights
                            progress = min(1.0, self.step_counter / 20.0)
                            w_novelty = 0.4 - (0.3 * progress)
                            w_perf = 1.0 - w_novelty
                            
                            # 4. Total Meta-Reward
                            weighted_score = (w_perf * adjusted_perf) + (w_novelty * novelty_score)

                            # 5. Time Penalty
                            time_penalty = 0.01 * result.duration_seconds
                            final_meta_reward = weighted_score - time_penalty
                            
                            rewards[a_id_returned] = final_meta_reward
                            self.history.append(result)
                            results_out[a_id_returned] = result
                            self.novelty_calc.add_to_history(result.config)
                            
                            # Check for best
                            if result.final_mean_reward > self.best_reward:
                                self.best_reward = result.final_mean_reward
                                self.best_config = result.config
                                self.log.info(f"!!! New Global Best found by {a_id_returned}: {self.best_reward:.1f} !!!")
                                
                            # [NEW] Check Promotion
                            if self.tier < 2:
                                thresh = self.tier_thresholds[self.tier]
                                if result.final_mean_reward >= thresh:
                                    self.tier += 1
                                    self.log.info(f"\n[bold green]>>> CURRICULUM PROMOTION! Unlocked Tier {self.tier} Envs: {self.tiers[self.tier]} <<<[/bold green]\n")

                        else:
                            # Failure returned by worker
                            self.log.error(f"[Lab] {a_id_returned} FAILED: {error}")
                            rewards[a_id_returned] = -1.0 # Penalty for failure
                            results_out[a_id_returned] = None
                            
                    except Exception as e:
                        self.log.error(f"[Lab] Inner Loop Error: {e}")

            except TimeoutError:
                self.log.warning("[Lab] !!! TIMEOUT: Experiments exceeding 60s limit were dropped !!!")
                # Any agents not in results_out get marked as failed/timed out
                pass

        return results_out, rewards

    def get_observation(self) -> Observation:
        """Returns the global state (public knowledge)."""
        
        # 1. Performance Trends
        trends = {}
        if len(self.history) > 0:
            reward_vals = [res.final_mean_reward for res in self.history]
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
                if len(recent) > 1:
                    trends["stability"] = 1.0 / (np.std(recent) + 1e-6)
                else:
                    trends["stability"] = 0.0
        else:
            trends = {"mean_reward_all": 0.0, "improvement_rate": 0.0, "stability": 0.0}
            
        # 2. Novelty Landscape
        novelty_stats = {}
        novelty_stats["explored_ratio"] = min(1.0, len(self.history) / 500.0) 
        # [NEW] Leaderboard
        # We assume the caller (main.py) will inject/use this, 
        # or we calculate a simple one here for the observation
        # For Observation, we just need the 'novelty landscape' basically.
        # But let's add a "leaderboard_stats" to env_metadata for now if needed.
        
        # Inject curriculum info
        meta = self.env_metadata.copy()
        meta["allowed_envs"] = self.allowed_envs
        meta["current_tier"] = self.tier
        
        obs = Observation(
            experiment_history=self.history[-50:], # Truncate for prompt
            performance_trends=trends,
            novelty_landscape={
                "cluster_density": 0.5, # Placeholder
                "unexplored_ratio": 0.8
            },
            knowledge_summary="Extracted from vector DB...",
            env_metadata=meta
        )
        return obs
