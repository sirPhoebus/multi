import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from marl_scientist.core import Researcher, Observation, ExperimentConfig, ExperimentResult
from marl_scientist.agents.brain import MetaBrain, BrainEncoder
from marl_scientist.agents.memory import EpisodicMemory, TrajectoryMemory
from marl_scientist.utils.logger import setup_logger

from marl_scientist.agents.researcher import ResearcherAgent

class NeuralResearcherAgent(ResearcherAgent):
    """
    An AI Scientist that uses a Neural Network 'Brain' to decide experiments.
    Learns how to do research over time.
    """
    def __init__(self, agent_id: str, model_path: Optional[str] = None, 
                 knowledge_store: Optional[Any] = None, 
                 trajectory_memory: Optional[TrajectoryMemory] = None,
                 validator: Optional[Any] = None):
        super().__init__(agent_id, knowledge_store, trajectory_memory)
        self.encoder = BrainEncoder()
        self.memory = EpisodicMemory()
        self.log = setup_logger()
        
        # Initialize Brain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = MetaBrain().to(self.device)
        if model_path:
            try:
                self.brain.load_state_dict(torch.load(model_path))
                print(f"[NeuralResearcher] Loaded brain from {model_path}")
            except:
                print(f"[NeuralResearcher] Failed to load brain, using random init.")
        
        self.brain.eval() # Deployment mode by default
        
        # [NEW] Multi-Task Preference [Performance, Efficiency, Stability]
        self.preference_vec = np.array([0.5, 0.25, 0.25]) 
        
        # [NEW] Strategic Intent Persistence
        self.last_goal = None
        self.goal_persistence_counter = 0
        self.max_goal_persistence = 3 # Experiments per goal
        
        # Recurrent state
        self.hidden_state = None
        self.horizon_active = False
        self.horizon_buffer: List[ExperimentConfig] = []
        self.horizon_results: List[ExperimentResult] = []
        self.horizon_target_idx = 5 # Number of sub-proposals
        self.validator = validator # [PHASE 2]
        
        self.best_performance = -float('inf')
        
        # [PHASE 4] Competency Tracking & Mastery
        self.competency_scores: Dict[str, float] = {} # env_id -> max_reward_ever
        
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

    def set_trajectory_memory(self, memory):
        self.trajectory_memory = memory

    def set_validator(self, validator):
        self.validator = validator

    async def propose_experiment(self, observation: Observation) -> Tuple[Union[ExperimentConfig, List[ExperimentConfig]], Dict[str, Any]]:
        """
        Queries the Brain to get a new experiment configuration (Async).
        Returns (Config, TrainingData)
        """
        # 1. Literature Review for Embeddings
        knowledge_vec = await self._get_knowledge_embedding(observation)
        
        # 2. Latent Retrieval [PHASE 3]
        latent_vec = await self._get_latent_embedding(observation)
        
        # 3. Encode Observation
        inputs = self.encoder.encode_observation(observation, self.agent_id, knowledge_vec, latent_vec)
        
        # [PHASE 4] Horizon Consolidation
        if self.horizon_results and not self.horizon_active:
            self.log.info(f"[{self.agent_id}] Selecting best proposal from Horizon results...")
            best_res = max(self.horizon_results, key=lambda r: r.final_mean_reward)
            best_config = best_res.config
            best_config.hyperparameters.pop("is_horizon", None)
            best_config.hyperparameters.pop("parent_agent", None)
            best_config.hyperparameters.pop("sub_id", None)
            
            self.horizon_results = []
            self.last_goal = 2 # Force REFINE
            self.goal_persistence_counter = 1
            
            return best_config, {} 

        # 4. Brain Inference
        with torch.no_grad():
            if "allowed_envs" in observation.env_metadata:
                allowed_names = observation.env_metadata["allowed_envs"]
            else:
                allowed_names = [] 

            target_env_context = self.encoder.ENVS[0]
            if allowed_names:
                target_env_context = allowed_names[-1] 

            # Prepare Inputs
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            
            if self.hidden_state is not None:
                self.hidden_state = self.hidden_state.to(self.device)
            
            pref_tensor = torch.tensor(self.preference_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            mem_vec = self.memory.retrieve_vector(target_env_context)
            mem_tensor = torch.tensor(mem_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            intents = ["EXPLORE", "EXPLOIT", "REFINE"]
            is_persistent = self.last_goal is not None and self.goal_persistence_counter > 0
            forced_g_idx = self.last_goal if is_persistent else None
            
            # Forward pass
            algo_logits, env_logits, hp_means, value, new_hidden, goal_logits = self.brain(
                inputs["history"],
                inputs["trends"],
                inputs["knowledge"],
                inputs["latent"],
                mem_tensor, 
                pref_tensor, 
                self.hidden_state,
                forced_goal=forced_g_idx
            )
            
            # Goal selection
            if is_persistent:
                goal_idx = self.last_goal
                self.goal_persistence_counter -= 1
                self.log.info(f"[{self.agent_id}] Strategy Persistent: {intents[goal_idx]} ({self.goal_persistence_counter} left)")
            else:
                goal_idx = torch.argmax(goal_logits, dim=-1).item()
                
                # [EXPLORATION] Occasional Override
                if random.random() < 0.10:
                    goal_idx = 0 # Force EXPLORE
                    self.log.info(f"[{self.agent_id}] Strategy Override: Forcing EXPLORE for diversity!")
                
                self.last_goal = goal_idx
                self.goal_persistence_counter = self.max_goal_persistence - 1
                self.log.info(f"[{self.agent_id}] New Strategic Intent: {intents[goal_idx]}")
            
            # Curriculum Masking for Env
            if allowed_names:
                mask = torch.full_like(env_logits, -float('inf'))
                for name in allowed_names:
                    if name in self.encoder.ENVS:
                        idx = self.encoder.ENVS.index(name)
                        mask[0, idx] = 0.0 
                env_logits = env_logits + mask
            
            env_idx = torch.argmax(env_logits, dim=-1).item()
            selected_env = self.encoder.ENVS[env_idx]
            
            # Compatibility Masking for Algo
            discrete_envs = ["CartPole-v1", "LunarLander-v3", "Acrobot-v1"]
            continuous_envs = ["Pendulum-v1", "MountainCarContinuous-v0", "Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
            algo_mask = torch.zeros_like(algo_logits)
            if selected_env in discrete_envs:
                if "SAC" in self.encoder.ALGOS:
                    algo_mask[0, self.encoder.ALGOS.index("SAC")] = -float('inf')
            elif selected_env in continuous_envs:
                if "DQN" in self.encoder.ALGOS:
                    algo_mask[0, self.encoder.ALGOS.index("DQN")] = -float('inf')
            algo_logits = algo_logits + algo_mask
            
            algo_idx = torch.argmax(algo_logits, dim=-1).item()
            hp_vals = hp_means.squeeze(0).cpu().numpy()
            
            # Capture Train Data
            from torch.distributions import Categorical, Normal
            log_prob = Categorical(logits=goal_logits).log_prob(torch.tensor([goal_idx], device=self.device)) + \
                       Categorical(logits=algo_logits).log_prob(torch.tensor([algo_idx], device=self.device)) + \
                       Categorical(logits=env_logits).log_prob(torch.tensor([env_idx], device=self.device)) + \
                       Normal(hp_means, torch.exp(self.brain.hp_logstd)).log_prob(torch.tensor(hp_vals, device=self.device)).sum(dim=-1)
            
            train_data = {
                "inputs": {k: v.cpu() for k, v in inputs.items()},
                "hidden": self.hidden_state.cpu() if self.hidden_state is not None else None,
                "goal_idx": goal_idx,
                "algo_idx": algo_idx,
                "env_idx": env_idx,
                "hp_vals": hp_vals,
                "log_prob": log_prob.item(),
                "value": value.item()
            }
            
            self.hidden_state = new_hidden

        # 5. Decode to Config
        config_dict = self.encoder.decode_action(algo_idx, env_idx, hp_vals)
        config = ExperimentConfig(
            algorithm=config_dict["algorithm"],
            hyperparameters=config_dict["hyperparameters"],
            env_id=config_dict["env_id"]
        )
        
        # [REFINEMENT] Algorithm Mutation
        # Force diversity: If PPO is dominant, small chance to try SAC
        if config.algorithm == "PPO" and random.random() < 0.1:
            # Check if SAC is valid for this env
            if config.env_id not in ["CartPole-v1", "LunarLander-v3", "Acrobot-v1"]: # Discrete
                 config.algorithm = "SAC"
                 self.log.info(f"[{self.agent_id}] [Mutation] Force-switched PPO -> SAC for diversity.")
        
        # [PHASE 4] Horizon Mode Trigger
        probs = F.softmax(goal_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
        
        if intents[goal_idx] == "EXPLORE" and entropy > 0.8 and not self.horizon_active:
            # 30% chance to use LLM Slow Reasoning, 70% Horizon Mode
            if random.random() < 0.3:
                self.log.info(f"[{self.agent_id}] High Uncertainty (H={entropy:.2f}). Triggering Slow Reasoning (LLM Hypothesis)!")
                llm_config, _ = await super().propose_experiment(observation)
                if llm_config:
                    return llm_config, train_data

            self.log.info(f"[{self.agent_id}] High Uncertainty (H={entropy:.2f}). Triggering Horizon Mode Lite!")
            self.horizon_active = True
            
            # [PHASE 4] Hard Env Expansion
            hard_envs = ["MountainCarContinuous-v0", "Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
            scout_count = 10 if selected_env in hard_envs else self.horizon_target_idx
            scout_steps = 3000 if selected_env in hard_envs else 5000
            
            sub_configs = []
            for _ in range(scout_count):
                jittered_hp = hp_vals + np.random.normal(0, 0.1, size=hp_vals.shape)
                c_dict = self.encoder.decode_action(algo_idx, env_idx, jittered_hp)
                sub_config = ExperimentConfig(algorithm=c_dict["algorithm"], hyperparameters=c_dict["hyperparameters"], env_id=c_dict["env_id"])
                sub_config.hyperparameters["is_horizon"] = True
                sub_config.hyperparameters["parent_agent"] = self.agent_id
                sub_config.hyperparameters["total_timesteps"] = scout_steps
                sub_configs.append(sub_config)
            
            return sub_configs, train_data 

        return config, train_data

    def propose_system_change(self) -> Optional[Dict]:
        """Supports the new self-modification mechanism."""
        if self.best_performance < 400:
            return None
        return {
            "target": "update_interval",
            "value": 20,
            "reason": "Neural convergence suggests higher batch size."
        }

    async def update_knowledge(self, result: ExperimentResult):
        """Neural agent learns via meta-loop and hypothesis-driven insights."""
        # [PHASE 4] Handle Horizon Results
        if result.config.hyperparameters.get("is_horizon"):
            self.horizon_results.append(result)
            if len(self.horizon_results) >= self.horizon_target_idx:
                self.log.info(f"[{self.agent_id}] Horizon Mode Complete. Consolidating {len(self.horizon_results)} sub-results.")
                self.horizon_active = False
            return 

        # [PHASE 1] Base class handles insights, causal updates, and trajectory memory
        await super().update_knowledge(result)

        # [PHASE 4] Curriculum Mastery Update
        env_id = result.config.env_id
        old_comp = self.competency_scores.get(env_id, -1000.0)
        self.competency_scores[env_id] = max(old_comp, result.final_mean_reward)
        
        if result.final_mean_reward > old_comp + 1.0:
            self.log.info(f"[{self.agent_id}] New Mastery Level on {env_id}: {result.final_mean_reward:.1f}")
        
        # Episodic Memory for Brain Encoding
        self.memory.add(result)

    def harvest_weights(self, competitor_brain: MetaBrain, tau: float = 0.1):
        """
        Soft-copy weights from a successful competitor.
        L_new = (1 - tau) * L_old + tau * L_competitor
        """
        with torch.no_grad():
            for target_param, source_param in zip(self.brain.parameters(), competitor_brain.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + source_param.data * tau
                )
        print(f"[NeuralResearcher {self.agent_id}] Harvested weights from competitor (tau={tau})")

    async def _get_knowledge_embedding(self, observation: Observation) -> np.ndarray:
        """
        Fetches the top paper embedding from the knowledge store (Async).
        """
        query = "State of the art reinforcement learning hyperparameters stability"
        results = await self.knowledge_store.search(query, k=1)
        
        if results:
            return results[0].get("embedding", np.zeros(768)) 
        return np.zeros(768)

    async def _get_latent_embedding(self, observation: Observation) -> np.ndarray:
        """
        Retrieves similar past trajectory embeddings from the latent store (Async).
        """
        if not self.trajectory_memory:
            return np.zeros(768, dtype=np.float32)
        
        if observation.experiment_history:
            last_res = observation.experiment_history[-1]
            return await self.trajectory_memory.retrieve_context(last_res.config, k=3)
                
        return np.zeros(768, dtype=np.float32)

    def save(self, filepath: str):
        """Save the brain and metadata."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "agent_id": self.agent_id,
            "brain_state": self.brain.state_dict(),
        }, filepath)
        
        mem_path = filepath.replace(".pkl", "_memory.json")
        self.memory.save(mem_path)
        
        print(f"[NeuralResearcher {self.agent_id}] State saved to {filepath}")

    def load(self, filepath: str):
        """Load the brain and metadata."""
        import os
        if not os.path.exists(filepath):
            return
        try:
            checkpoint = torch.load(filepath, weights_only=False)
            self.agent_id = checkpoint.get("agent_id", self.agent_id)
            self.brain.load_state_dict(checkpoint["brain_state"])
            
            mem_path = filepath.replace(".pkl", "_memory.json")
            if os.path.exists(mem_path):
                self.memory.load(mem_path)
                
            print(f"[NeuralResearcher {self.agent_id}] State loaded from {filepath}")
        except Exception as e:
            print(f"[NeuralResearcher {self.agent_id}] Load failed: {e}")

    def save_brain(self, path: str):
        torch.save(self.brain.state_dict(), path)
        print(f"[NeuralResearcher] Brain saved to {path}")
