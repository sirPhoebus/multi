import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Union
from marl_scientist.core import Researcher, Observation, ExperimentConfig, ExperimentResult
from marl_scientist.agents.brain import MetaBrain, BrainEncoder
from marl_scientist.agents.memory import EpisodicMemory, TrajectoryMemory
from marl_scientist.utils.logger import setup_logger

class NeuralResearcherAgent(Researcher):
    """
    An AI Scientist that uses a Neural Network 'Brain' to decide experiments.
    Learns how to do research over time.
    """
    def __init__(self, agent_id: str, model_path: Optional[str] = None):
        self.agent_id = agent_id
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
        self.knowledge_store = None
        self.trajectory_memory: Optional[TrajectoryMemory] = None
        
        # [PHASE 4] Horizon Mode State
        self.horizon_active = False
        self.horizon_buffer: List[ExperimentConfig] = []
        self.horizon_results: List[ExperimentResult] = []
        self.horizon_target_idx = 5 # Number of sub-proposals
        self.validator = None # [PHASE 2]
        
        self.best_performance = -float('inf')
        
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

    def propose_experiment(self, observation: Observation) -> ExperimentConfig:
        """
        Queries the Brain to get a new experiment configuration.
        """
        # 1. Literature Review for Embeddings
        knowledge_vec = self._get_knowledge_embedding(observation)
        
        # 2. Latent Retrieval [PHASE 3]
        latent_vec = self._get_latent_embedding(observation)
        
        # 3. Encode Observation
        inputs = self.encoder.encode_observation(observation, self.agent_id, knowledge_vec, latent_vec)
        
        # 3. Brain Inference
        with torch.no_grad():
            if "allowed_envs" in observation.env_metadata:
                allowed_names = observation.env_metadata["allowed_envs"]
            else:
                allowed_names = [] 

            # [NEW] Memory Retrieval (Context Injection)
            target_env_context = self.encoder.ENVS[0]
            if allowed_names:
                target_env_context = allowed_names[-1] 
                
            # [PHASE 4] Horizon Consolidation
            if self.horizon_results and not self.horizon_active:
                self.log.info(f"[{self.agent_id}] Selecting best proposal from Horizon results...")
                best_res = max(self.horizon_results, key=lambda r: r.final_mean_reward)
                best_config = best_res.config
                # Remove horizon markers
                best_config.hyperparameters.pop("is_horizon", None)
                best_config.hyperparameters.pop("sub_id", None)
                best_config.hyperparameters.pop("parent_agent", None)
                self.horizon_results = [] # Reset
                # We skip the rest of the brain inference and return this "voted" champion
                return best_config

            # 1. Prepare Inputs
            # Prepare inputs on correct device
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            
            if self.hidden_state is not None:
                self.hidden_state = self.hidden_state.to(self.device)
            
            pref_tensor = torch.tensor(self.preference_vec, dtype=torch.float32, device=self.device).unsqueeze(0) # [1, 3]
            
            # Memory Context (Matches HP space)
            mem_vec = self.memory.retrieve_vector(target_env_context)
            mem_tensor = torch.tensor(mem_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            intents = ["EXPLORE", "EXPLOIT", "REFINE"]
            is_persistent = self.last_goal is not None and self.goal_persistence_counter > 0
            forced_g_idx = self.last_goal if is_persistent else None
            
            # Forward pass with Context (Hierarchical)
            algo_logits, env_logits, hp_means, value, new_hidden, goal_logits = self.brain(
                inputs["history"],
                inputs["trends"],
                inputs["knowledge"],
                inputs["latent"], # [PHASE 3]
                mem_tensor, 
                pref_tensor, 
                self.hidden_state,
                forced_goal=forced_g_idx # [NEW]
            )
            
            if is_persistent:
                goal_idx = self.last_goal
                self.goal_persistence_counter -= 1
                intent_str = intents[goal_idx]
                self.log.info(f"[{self.agent_id}] Strategy Persistent: {intent_str} ({self.goal_persistence_counter} left)")
            else:
                goal_idx = torch.argmax(goal_logits, dim=-1).item()
                intent_str = intents[goal_idx]
                self.last_goal = goal_idx
                self.goal_persistence_counter = self.max_goal_persistence - 1
                self.log.info(f"[{self.agent_id}] New Strategic Intent: {intent_str}")
                
            # Curriculum Masking
            if allowed_names:
                mask = torch.full_like(env_logits, -float('inf'))
                for name in allowed_names:
                    if name in self.encoder.ENVS:
                        idx = self.encoder.ENVS.index(name)
                        mask[0, idx] = 0.0 
                env_logits = env_logits + mask
            
            self.hidden_state = new_hidden
            
            # 1. Select Environment First
            env_idx = torch.argmax(env_logits, dim=-1).item()
            selected_env = self.encoder.ENVS[env_idx]
            
            # 2. Compatibility Masking
            discrete_envs = ["CartPole-v1", "LunarLander-v3", "Acrobot-v1"]
            continuous_envs = [
                "Pendulum-v1", "MountainCarContinuous-v0", 
                "Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"
            ]
            
            algo_mask = torch.zeros_like(algo_logits)
            
            if selected_env in discrete_envs:
                if "SAC" in self.encoder.ALGOS:
                    sac_idx = self.encoder.ALGOS.index("SAC")
                    algo_mask[0, sac_idx] = -float('inf')
                    
            elif selected_env in continuous_envs:
                if "DQN" in self.encoder.ALGOS:
                    dqn_idx = self.encoder.ALGOS.index("DQN")
                    algo_mask[0, dqn_idx] = -float('inf')
            
            algo_logits = algo_logits + algo_mask
            
            # 3. Select Algorithm
            algo_idx = torch.argmax(algo_logits, dim=-1).item()
            hp_vals = hp_means.squeeze(0).cpu().numpy()
            
        # 4. Decode to Config
        config_dict = self.encoder.decode_action(algo_idx, env_idx, hp_vals)
        
        config = ExperimentConfig(
            algorithm=config_dict["algorithm"],
            hyperparameters=config_dict["hyperparameters"],
            env_id=config_dict["env_id"]
        )
        
        # [PHASE 4] Horizon Mode Trigger
        # Measure uncertainty via entropy of goal_logits
        probs = F.softmax(goal_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
        
        # If we are in EXPLORE mode and entropy is high (> 0.8), trigger Horizon Mode
        if intent_str == "EXPLORE" and entropy > 0.8 and not self.horizon_active:
            self.log.info(f"[{self.agent_id}] High Uncertainty (H={entropy:.2f}). Triggering Horizon Mode Lite!")
            self.horizon_active = True
            self.horizon_buffer = []
            
            # Generate K sub-proposals (stochastic sampling)
            sub_configs = []
            for _ in range(self.horizon_target_idx):
                # Sample again from logits (stochastic)
                # For simplicity, we just use the existing algo/env and jitter HPs
                # Real implementation should sample from distribution
                jittered_hp = hp_vals + np.random.normal(0, 0.1, size=hp_vals.shape)
                c_dict = self.encoder.decode_action(algo_idx, env_idx, jittered_hp)
                sub_config = ExperimentConfig(
                    algorithm=c_dict["algorithm"],
                    hyperparameters=c_dict["hyperparameters"],
                    env_id=c_dict["env_id"]
                )
                # Mark as horizon for main.py
                sub_config.hyperparameters["is_horizon"] = True
                sub_config.hyperparameters["parent_agent"] = self.agent_id
                sub_configs.append(sub_config)
            
            return sub_configs

        # [PHASE 1] Basic De-duplication
        if self.trajectory_memory:
            similar = self.trajectory_memory.check_similarity(config)
            if similar:
                print(f"[NeuralResearcher {self.agent_id}] Proposal similar to past failure. Warning only.")
        
        # [PHASE 2] Self-Correction (Redundant Check)
        if self.validator:
            valid, violations = self.validator.validate(config)
            if not valid:
                 print(f"[NeuralResearcher {self.agent_id}] Brain proposed invalid config: {violations}. Forcing re-roll.")
                 # In a perfect world we would re-run the brain with a penalty, 
                 # but for now we just log it and let main.py handle the rejection loop.
                 # The snapping in BrainEncoder should prevent this anyway.

        return config

    def update_knowledge(self, result: ExperimentResult):
        """
        Neural agent doesn't do manual causal updates, 
        it learns via the PPO meta-training loop.
        However, it can still publish results to the journal.
        """
        # [PHASE 4] Handle Horizon Results
        if result.config.hyperparameters.get("is_horizon"):
            self.horizon_results.append(result)
            if len(self.horizon_results) >= self.horizon_target_idx:
                self.log.info(f"[{self.agent_id}] Horizon Mode Complete. Consolidating {len(self.horizon_results)} sub-results.")
                # We don't return anything here, but the NEXT propose_experiment will use these
                # Actually, we could select the best one here and store it.
                self.horizon_active = False
            return # Don't update main weights with dry-runs? 
                   # Or maybe we DO but with lower weighting. For now, skip.

        self.best_performance = max(self.best_performance, result.final_mean_reward)
        
        # [NEW] Episodic Memory Update
        self.memory.add(result)
        
        # [PHASE 1] Trajectory Compression
        if self.trajectory_memory:
            self.trajectory_memory.add(result)
        
        if self.knowledge_store and result.final_mean_reward > 400.0:
            paper = self.knowledge_store.synthesize_new_paper(result, self.agent_id)
            self.knowledge_store.add_paper(paper)

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

    def _get_knowledge_embedding(self, observation: Observation) -> np.ndarray:
        """
        Fetches the top paper embedding from the knowledge store.
        """
        # Formulate a simple query based on current trends
        query = "State of the art reinforcement learning hyperparameters stability"
        results = self.knowledge_store.search(query, k=1)
        
        if results:
            return results[0].get("embedding", np.zeros(768)) # Search returns dicts now
        return np.zeros(768)

    def _get_latent_embedding(self, observation: Observation) -> np.ndarray:
        """
        Retrieves similar past trajectory embeddings from the latent store. [PHASE 3]
        """
        if not self.trajectory_memory:
            return np.zeros(768)
        
        # Use the latest experiment config as a query, if any
        if observation.experiment_history:
            last_res = observation.experiment_history[-1]
            # Fetch similar trajectories WITH embeddings
            results = self.trajectory_memory.store.search_similar(last_res.config, k=3, return_embeddings=True)
            
            if results:
                # Average the embeddings of the top k hits
                embs = [r["embedding"] for r in results]
                return np.mean(embs, axis=0)
                
        return np.zeros(768)

    def save(self, filepath: str):
        """Save the brain and metadata."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Use brain save helper but also could save other metadata
        torch.save({
            "agent_id": self.agent_id,
            "brain_state": self.brain.state_dict(),
        }, filepath)
        
        # [NEW] Save Memory
        mem_path = filepath.replace(".pkl", "_memory.json")
        self.memory.save(mem_path)
        
        print(f"[NeuralResearcher {self.agent_id}] State saved to {filepath}")

    def load(self, filepath: str):
        """Load the brain and metadata."""
        import os
        if not os.path.exists(filepath):
            return
        try:
            # Fix: PyTorch 2.4+ defaults weights_only=True which breaks dict loading unless whitelisted
            # We use weights_only=False for local dev convenience, assuming trusted source
            checkpoint = torch.load(filepath, weights_only=False)
            self.agent_id = checkpoint.get("agent_id", self.agent_id)
            self.brain.load_state_dict(checkpoint["brain_state"])
            
            # [NEW] Load Memory
            mem_path = filepath.replace(".pkl", "_memory.json")
            if os.path.exists(mem_path):
                self.memory.load(mem_path)
                
            print(f"[NeuralResearcher {self.agent_id}] State loaded from {filepath}")
        except Exception as e:
            print(f"[NeuralResearcher {self.agent_id}] Load failed: {e}")

    def save_brain(self, path: str):
        torch.save(self.brain.state_dict(), path)
        print(f"[NeuralResearcher] Brain saved to {path}")
