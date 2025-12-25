from typing import List, Tuple, Dict, Any, Optional
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
    def __init__(self, authorized_benchmarks: List[str] = ["CartPole-v1"], max_workers: int = 8, visual: bool = False):
        self.log = setup_logger()
        self.benchmarks = authorized_benchmarks
        self.visual = visual
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
            2: ["MountainCarContinuous-v0"],
            3: ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
        }
        # Threshold to UNLOCK next tier
        self.tier_thresholds = {
            0: 450.0, # CartPole/Pendulum solved
            1: 200.0, # LunarLander solved
            2: 90.0,  # MountainCar solved (or progressed)
            3: float('inf')
        }
    
        # [NEW] Async Executor
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.futures_map = {} # future -> (agent_id, start_time)
        self.running_experiments = {} # agent_id -> config
        
        # [NEW] Global Benchmark Metadata for Dispatch Logic
        self.env_metadata = {}
        self._prepopulate_metadata()
        
    def _prepopulate_metadata(self):
        """Pre-fetches essential metadata for all benchmarks to assist in dispatching."""
        import gymnasium as gym
        for tier_envs in self.tiers.values():
            for env_id in tier_envs:
                try:
                    # Create a dummy env to extract space info
                    temp_env = gym.make(env_id)
                    self.env_metadata[env_id] = {
                        "is_discrete": isinstance(temp_env.action_space, gym.spaces.Discrete),
                        "is_continuous": isinstance(temp_env.action_space, (gym.spaces.Box, gym.spaces.Dict)),
                    }
                    temp_env.close()
                except:
                    # Fallback for common ones if make fails
                    if "CartPole" in env_id or "Acrobot" in env_id or "LunarLander-v3" == env_id:
                        self.env_metadata[env_id] = {"is_discrete": True, "is_continuous": False}
                    else:
                        self.env_metadata[env_id] = {"is_discrete": False, "is_continuous": True}
        
    def close(self):
        if self.executor:
            # Try to cancel pending work and shutdown
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python < 3.9 doesn't support cancel_futures
                self.executor.shutdown(wait=False)
            
            # Note: On Windows, workers might still hang if they are blocked in a C-extension.
            # We've done what we can to signal termination.

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
            import gymnasium as gym
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

    def submit_experiment(self, agent_id: str, config: ExperimentConfig):
        """Non-blocking submission of an experiment."""
        if config is None: return
        
        self.log.info(f"[Lab] Scheduling {agent_id} on {config.env_id} ({config.algorithm})...")
        fut = self.executor.submit(run_experiment_task, config, agent_id, self.visual)
        
        import time
        self.futures_map[fut] = (agent_id, time.time())
        self.running_experiments[agent_id] = config

    def poll_results(self, agent_preferences: Optional[Dict[str, np.ndarray]] = None) -> Tuple[Dict[str, ExperimentResult], Dict[str, float]]:
        """Checks for completed experiments without blocking."""
        import time
        results_out = {}
        rewards = {}
        
        # Check completed futures
        # We use a list to avoid modifying dict while iterating if we remove keys
        done_futures = [f for f in self.futures_map if f.done()]
        
        if not done_futures:
            return {}, {}
            
        for future in done_futures:
            agent_id, t_start = self.futures_map.pop(future)
            if agent_id in self.running_experiments:
                del self.running_experiments[agent_id]
                
            try:
                # Get result immediately since it is done
                a_id_returned, result, error = future.result()
                t_end = time.time()
                
                if result:
                    # Success
                    result.duration_seconds = t_end - t_start
                    result.start_time = t_start
                    # total_env_steps logic handled in result usually, but ensure it's there
                    result.total_env_steps = result.config.hyperparameters.get("total_timesteps", 0)
                    
                    self.log.info(f"[Lab] {a_id_returned} finished: {result.final_mean_reward:.1f} (Time: {result.duration_seconds:.1f}s)")
                    
                    # Post-Processing
                    self.step_counter += 1
                    
                    # Apply Agent-Specific Preference if available
                    pref = None
                    if agent_preferences and a_id_returned in agent_preferences:
                        pref = agent_preferences[a_id_returned]
                        
                    meta_reward = self._calculate_meta_reward(result, preference=pref)
                    
                    rewards[a_id_returned] = meta_reward
                    self.history.append(result)
                    results_out[a_id_returned] = result
                    self.novelty_calc.add_to_history(result.config)
                    
                    # Check for best
                    if result.final_mean_reward > self.best_reward:
                        self.best_reward = result.final_mean_reward
                        self.best_config = result.config
                        self.log.info(f"!!! New Global Best found by {a_id_returned}: {self.best_reward:.1f} !!!")
                        
                    # Check Promotion
                    self._check_promotion(result)

                else:
                    # Failure
                    self.log.error(f"[Lab] {a_id_returned} FAILED: {error}")
                    rewards[a_id_returned] = -1.0
                    results_out[a_id_returned] = None
                    
            except Exception as e:
                self.log.error(f"[Lab] Poll Error: {e}")
                
        return results_out, rewards

    def _calculate_meta_reward(self, result: ExperimentResult, preference: Optional[np.ndarray] = None) -> float:
        """
        Calculates a meta-reward based on multiple objectives.
        preference: [Perf, Efficiency, Stability] - sums to 1.0 (roughly)
        """
        if preference is None:
            preference = np.array([0.6, 0.2, 0.2]) # Default: Focus on Perf
            
        norms = {
            "CartPole-v1": {"min": 0, "max": 500},
            "Acrobot-v1": {"min": -500, "max": -100},
            "Pendulum-v1": {"min": -2000, "max": -150},
            "LunarLander-v3": {"min": -500, "max": 200},
            "MountainCarContinuous-v0": {"min": -100, "max": 100},
            "Hopper-v4": {"min": 0, "max": 3000},
            "Walker2d-v4": {"min": 0, "max": 3500},
            "HalfCheetah-v4": {"min": 0, "max": 6000}
        }
        
        # 1. Base Performance (0 to 1)
        env_id = result.config.env_id
        spec = norms.get(env_id, {"min": -1000, "max": 0}) 
        raw_reward = result.final_mean_reward
        performance_score = (raw_reward - spec["min"]) / (spec["max"] - spec["min"])
        result.performance_score = max(0.0, min(1.0, performance_score))
        
        # 2. Stability Score (0 to 1)
        std_reward = result.metrics.get("std_reward", 0.0)
        # Low std -> high stability. Normalize 0 to 50 -> 1 to 0
        result.stability_score = max(0.0, 1.0 - (std_reward / 50.0))
        
        # 3. Efficiency Score (Speed) (0 to 1)
        # Non-linear penalty: sqrt(penalty) makes it gentler initially.
        duration_ratio = min(1.0, result.duration_seconds / 150.0)
        eff_score = max(0.0, 1.0 - np.sqrt(duration_ratio))
        
        # [NEW] Failure-Aware Capping
        # If the agent failed the task (performance < 10%), we don't reward speed.
        if result.performance_score < 0.1:
            eff_score = min(eff_score, 0.1)
            
        result.efficiency_score = eff_score
        
        # 4. Novelty Score (0 to 1)
        novelty_score = self.novelty_calc.calculate_novelty(result.config)
        
        # --- Weighted Sum ---
        w_perf, w_effic, w_stable = preference
        
        # Give performance a slight boost if it's very high (>0.8)
        perf_boost = 1.2 if result.performance_score > 0.8 else 1.0
        
        final_meta_reward = (
            (w_perf * result.performance_score * perf_boost) + 
            (w_effic * result.efficiency_score) + 
            (w_stable * result.stability_score) +
            (0.1 * novelty_score) # Reduced novelty weight as swarm matures
        )
        
        return final_meta_reward

    def _check_promotion(self, result: ExperimentResult):
        if self.tier < 2:
            thresh = self.tier_thresholds[self.tier]
            if result.final_mean_reward >= thresh:
                self.tier += 1
                self.log.info(f"\n[bold green]>>> CURRICULUM PROMOTION! Unlocked Tier {self.tier} Envs: {self.tiers[self.tier]} <<<[/bold green]\n")

    def step(self, actions: Dict[str, ExperimentConfig]) -> Tuple[Dict[str, ExperimentResult], Dict[str, float]]:
        """Deprecated synchronous step."""
        for aid, cfg in actions.items():
            self.submit_experiment(aid, cfg)
            
        # Blocking wait for all
        import time
        while self.futures_map:
            res, rew = self.poll_results()
            if res: return res, rew # This is a broken synchronous shim, but main.py won't use it.
            time.sleep(0.5)
        return {}, {}

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
